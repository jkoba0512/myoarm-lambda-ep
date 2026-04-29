"""
実験 F3: エファレンスコピー操作（仮説 H3 検証）

仮説 H3:
  エファレンスコピーを遮断・劣化させると小脳の外乱適応が遅くなる。
  エファレンスコピーなしでは予測できないため補正トルクが不正確になり、
  外乱後の MAE_post 増加・τ_adapt（1/e 収束時間）の延長が生じる。

実験設計:
  AnatomicalController の efcopy_enabled / efcopy_noise_std のみを変化させる。
  prop_delay_steps=10, cereb_delay_steps=15 (生体正常値) で固定。
  D0 条件（フル PD ゲイン、外乱あり 60 Nm）を使用。

事前指定評価指標 (pre-specified, H3):
  - MAE_post [mrad]  : 外乱後の定常追従誤差（エファレンスコピー劣化で増加する）
  - tau_adapt [s]    : 外乱後 1/e 収束時間（MAE_post_peak から 1/e 減衰するまで）

条件:
  f3_full      : efcopy_enabled=True,  noise_std=0.0  （正常）
  f3_noisy_low : efcopy_enabled=True,  noise_std=5.0  （軽ノイズ）
  f3_noisy_hi  : efcopy_enabled=True,  noise_std=20.0 （重ノイズ）
  f3_removed   : efcopy_enabled=False               （完全遮断）

出力:
  results/experiment_franka_f3/
    {f3_full,f3_noisy_low,f3_noisy_hi,f3_removed}/seed{0..9}/metrics.json
    f3_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch as _torch

from common.franka_env import FrankaEnv, N_JOINTS
from methodF import AnatomicalController, AnatomicalConfig

DEVICE       = "cuda" if _torch.cuda.is_available() else "cpu"
SIM_DURATION = 6.0
DIST_T       = 3.0
DIST_STEPS   = 20
DIST_JOINT   = 1
DIST_TORQUE  = 60.0

Q_OFFSET   = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.5, 0.0])
KP_DEFAULT = np.array([50., 50., 50., 50., 10., 10., 10.])
KD_DEFAULT = np.array([ 7.,  7.,  7.,  7.,  1.5, 1.5, 1.5])

PROP_DELAY_FIXED  = 10
CEREB_DELAY_FIXED = 15

CONDITIONS = {
    "f3_full":       {"efcopy_enabled": True,  "efcopy_noise_std": 0.0},
    "f3_noisy_low":  {"efcopy_enabled": True,  "efcopy_noise_std": 5.0},
    "f3_noisy_hi":   {"efcopy_enabled": True,  "efcopy_noise_std": 20.0},
    "f3_removed":    {"efcopy_enabled": False, "efcopy_noise_std": 0.0},
}

E1_DIR      = ROOT / "results" / "experiment_franka_e1"
RESULTS_DIR = ROOT / "results" / "experiment_franka_f3"
SEEDS_DEFAULT = list(range(10))


def _tau_adapt(t_post: np.ndarray, err_post: np.ndarray) -> float:
    """外乱後誤差の 1/e 収束時間 [s] を推定する。"""
    if len(err_post) < 4:
        return float("nan")
    peak_val = float(err_post.max())
    target   = peak_val * (1.0 / np.e)
    idx      = np.where(err_post <= target)[0]
    if len(idx) == 0:
        return float("nan")
    peak_t   = float(t_post[np.argmax(err_post)])
    return float(t_post[idx[0]] - peak_t)


def _oscillation_freq(err_1d: np.ndarray, dt: float) -> float:
    if len(err_1d) < 8:
        return float("nan")
    n = len(err_1d)
    fft = np.abs(np.fft.rfft(err_1d - err_1d.mean()))
    freqs = np.fft.rfftfreq(n, d=dt)
    if len(freqs) < 2:
        return float("nan")
    return float(freqs[int(np.argmax(fft[1:])) + 1])


def compute_metrics(t_arr, q_arr, tau_arr, dt, io_stats=None) -> dict:
    err_arr   = np.abs(q_arr - Q_OFFSET)
    mask_pre  = t_arr < DIST_T
    mask_post = t_arr > DIST_T

    mae_pre  = float(err_arr[mask_pre].mean())  if mask_pre.any()  else float("nan")
    mae_post = float(err_arr[mask_post].mean()) if mask_post.any() else float("nan")

    peak_err = float("nan")
    recovery_time = None
    tau_adapt_s = float("nan")
    if mask_post.any():
        post_err_j = err_arr[mask_post, DIST_JOINT]
        post_t     = t_arr[mask_post]
        peak_err   = float(post_err_j.max())
        rec_idx    = np.where(post_err_j < 0.1)[0]
        if len(rec_idx):
            recovery_time = float(post_t[rec_idx[0]] - DIST_T)
        tau_adapt_s = _tau_adapt(post_t, post_err_j)

    pre_steady      = float(err_arr[mask_pre, DIST_JOINT].mean()) if mask_pre.any() else 1e-9
    overshoot_ratio = (peak_err / (pre_steady + 1e-9)) if not np.isnan(peak_err) else float("nan")
    osc_freq        = _oscillation_freq(err_arr[mask_post, DIST_JOINT], dt) if mask_post.any() else float("nan")

    m: dict = {
        "MAE_pre":          round(mae_pre  * 1000, 4),
        "MAE_post":         round(mae_post * 1000, 4),
        "peak_err_rad":     round(peak_err, 6),
        "recovery_time_s":  recovery_time,
        "tau_adapt_s":      round(tau_adapt_s, 4) if not np.isnan(tau_adapt_s) else None,
        "overshoot_ratio":  round(float(overshoot_ratio), 4),
        "oscillation_freq": round(osc_freq, 4),
        "energy_J":         round(float(np.sum(tau_arr ** 2) * dt), 4),
    }
    if io_stats:
        m["io_fire_count"]         = io_stats.get("io_fire_count", 0)
        m["io_fire_rate_hz"]       = round(io_stats.get("io_fire_rate_hz", 0.0), 4)
        m["io_fire_interval_mean"] = round(io_stats.get("io_fire_interval_mean", 0.0), 4)
    return m


def run_faithful(ctrl, env, seed, dist_torque) -> dict:
    rng = np.random.default_rng(seed)
    actual_torque = float(np.clip(dist_torque * (1.0 + rng.normal(0.0, 0.25)), 20.0, 120.0))
    q, dq = env.reset(q0=Q_OFFSET.copy())
    ctrl.reset()
    dt = env.dt
    t_log, q_log, tau_log = [], [], []
    dist_applied = False

    while env.time < SIM_DURATION:
        t = env.time
        q, dq = env.get_state()
        if not dist_applied and t >= DIST_T:
            td = np.zeros(N_JOINTS)
            td[DIST_JOINT] = actual_torque
            env.apply_disturbance(td, duration_steps=DIST_STEPS)
            dist_applied = True
            continue
        tau, _ = ctrl.step(q, dq, Q_OFFSET)
        env.step(tau)
        q_actual, _ = env.get_state()
        ctrl.update_cerebellum(q_actual)
        t_log.append(t)
        q_log.append(q.copy())
        tau_log.append(tau.copy())

    return compute_metrics(np.array(t_log), np.array(q_log), np.array(tau_log), dt,
                           io_stats=ctrl.get_io_stats())


def make_ctrl(efcopy_enabled, efcopy_noise_std, env_seed, cfc_path, ctrl_seed):
    env = FrankaEnv(rng=np.random.default_rng(env_seed), obs_noise_std=0.002)
    cfg = AnatomicalConfig(
        prop_delay_steps=PROP_DELAY_FIXED,
        cereb_delay_steps=CEREB_DELAY_FIXED,
        kp=KP_DEFAULT.copy(), kd=KD_DEFAULT.copy(),
        io_mode="sparse", io_firing_rate_hz=1.0, io_gain=5.0,
        inverse_model_loc="m1",
        efcopy_enabled=efcopy_enabled,
        efcopy_noise_std=efcopy_noise_std,
        cfc_hidden_units=64, cfc_device=DEVICE,
    )
    ctrl = AnatomicalController(cfg, seed=ctrl_seed)
    if cfc_path and efcopy_enabled:
        ctrl.load_cfc(cfc_path)
    return ctrl, env


def run_seed(seed: int) -> None:
    fwd_path = E1_DIR / f"seed{seed}" / "cfc_forward.pt"
    fwd_path_str = str(fwd_path) if fwd_path.exists() else None
    print(f"\nseed={seed}")

    for cond_name, cond_kw in CONDITIONS.items():
        out_dir = RESULTS_DIR / cond_name / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        ctrl, env = make_ctrl(
            efcopy_enabled=cond_kw["efcopy_enabled"],
            efcopy_noise_std=cond_kw["efcopy_noise_std"],
            env_seed=seed * 100,
            cfc_path=fwd_path_str,
            ctrl_seed=seed,
        )
        m = run_faithful(ctrl, env, seed, DIST_TORQUE)
        m.update({
            "condition":         cond_name,
            "efcopy_enabled":    cond_kw["efcopy_enabled"],
            "efcopy_noise_std":  cond_kw["efcopy_noise_std"],
            "prop_delay_steps":  PROP_DELAY_FIXED,
            "cereb_delay_steps": CEREB_DELAY_FIXED,
            "seed":              seed,
        })

        with open(out_dir / "metrics.json", "w") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)

        rt  = "n/a" if m["recovery_time_s"] is None else f"{m['recovery_time_s']:.3f}s"
        tau = "n/a" if m["tau_adapt_s"] is None else f"{m['tau_adapt_s']:.3f}s"
        print(
            f"  {cond_name:15s}  MAE_post={m['MAE_post']:7.2f} mrad  "
            f"tau_adapt={tau}  RT={rt}"
        )


def build_summary(seeds: list[int]) -> dict:
    summary: dict = {
        "experiment": "f3",
        "hypothesis": "H3: エファレンスコピー遮断 → 外乱適応遅延",
        "seeds": seeds,
        "conditions": {},
    }
    for cond_name in CONDITIONS:
        maes, taus, rts = [], [], []
        for seed in seeds:
            p = RESULTS_DIR / cond_name / f"seed{seed}" / "metrics.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            if not np.isnan(d.get("MAE_post", float("nan"))):
                maes.append(d["MAE_post"])
            if d.get("tau_adapt_s") is not None:
                taus.append(d["tau_adapt_s"])
            if d.get("recovery_time_s") is not None:
                rts.append(d["recovery_time_s"])
        summary["conditions"][cond_name] = {
            "MAE_post_mean":    round(float(np.mean(maes)), 3) if maes else None,
            "MAE_post_std":     round(float(np.std(maes)),  3) if maes else None,
            "tau_adapt_mean":   round(float(np.mean(taus)), 4) if taus else None,
            "tau_adapt_std":    round(float(np.std(taus)),  4) if taus else None,
            "recovery_time_mean": round(float(np.mean(rts)), 4) if rts else None,
            "n_seeds":          len(maes),
        }
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F3: エファレンスコピー実験 (仮説 H3)")
    p.add_argument("--seed",  type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    p.add_argument("--check-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    seeds = [args.seed] if args.seed is not None else args.seeds

    if not args.check_only:
        print(f"F3 実験開始  seeds={seeds}  device={DEVICE}")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            run_seed(seed)

    summary = build_summary(seeds)
    out = RESULTS_DIR / "f3_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 65)
    print("F3 集計結果")
    print("=" * 65)
    print(f"{'condition':>16}  {'MAE_post':>12}  {'tau_adapt':>10}  {'RT':>8}")
    for cname, data in summary["conditions"].items():
        mae_s = f"{data['MAE_post_mean']:7.2f}±{data['MAE_post_std']:.2f}" if data["MAE_post_mean"] else "  N/A"
        tau_s = f"{data['tau_adapt_mean']:.3f}s" if data["tau_adapt_mean"] else "  N/A"
        rt_s  = f"{data['recovery_time_mean']:.3f}s" if data["recovery_time_mean"] else "  N/A"
        print(f"  {cname:>14}  {mae_s:>12}  {tau_s:>10}  {rt_s:>8}")
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
