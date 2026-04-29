"""
実験 F5: 逆モデル所在（仮説 H5 検証）

仮説 H5:
  逆モデルが M1 のみにある（解剖学的忠実）場合に最も高い外乱適応柔軟性を持つ。
  小脳が逆モデルを担う場合（Kawato 解釈）は外乱後の回復が遅い。
  M1 と小脳の両方が逆モデルを担う場合（MOSAIC 的）は冗長で中間的な性能になる。

実験設計:
  inverse_model_loc のみを変化させる。
  prop_delay_steps=10, cereb_delay_steps=15 で固定。
  D0 条件（フル PD ゲイン、外乱あり 60 Nm）を使用。
  F0-abstract（EF-full）をベースラインとして再掲する。

事前指定評価指標 (pre-specified, H5):
  - MAE_post [mrad]      : 外乱後の定常追従誤差
  - recovery_time [s]    : 外乱後 0.1 rad 以内への復帰時間
  - adaptation_trajectory: MAE の時系列（外乱後 3 s 分）を記録

条件:
  f5_m1only    : inverse_model_loc="m1"          （解剖学的忠実・デフォルト）
  f5_cerebonly : inverse_model_loc="cerebellum"  （Kawato 極端解釈）
  f5_both      : inverse_model_loc="both"        （MOSAIC 的実装）
  f5_abstract  : F0-abstract (EF-full) を再掲   （ベースライン比較）

出力:
  results/experiment_franka_f5/
    {f5_m1only,f5_cerebonly,f5_both}/seed{0..9}/metrics.json
    f5_summary.json
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
ADAPT_WINDOW_S    = 3.0   # 適応軌跡記録ウィンドウ [s]（外乱後）

CONDITIONS = {
    "f5_m1only":    {"inverse_model_loc": "m1"},
    "f5_cerebonly": {"inverse_model_loc": "cerebellum"},
    "f5_both":      {"inverse_model_loc": "both"},
}

E1_DIR       = ROOT / "results" / "experiment_franka_e1"
F0_ABS_DIR   = ROOT / "results" / "experiment_franka_f0" / "f0_abstract" / "d0"
RESULTS_DIR  = ROOT / "results" / "experiment_franka_f5"
SEEDS_DEFAULT = list(range(10))


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
    adapt_traj: list[float] = []

    if mask_post.any():
        post_err_j = err_arr[mask_post, DIST_JOINT]
        post_t     = t_arr[mask_post]
        peak_err   = float(post_err_j.max())
        rec_idx    = np.where(post_err_j < 0.1)[0]
        if len(rec_idx):
            recovery_time = float(post_t[rec_idx[0]] - DIST_T)
        # 適応軌跡（外乱後 ADAPT_WINDOW_S 秒分）
        win_mask = post_t - DIST_T <= ADAPT_WINDOW_S
        adapt_traj = [round(float(v * 1000), 4) for v in post_err_j[win_mask]]

    pre_steady      = float(err_arr[mask_pre, DIST_JOINT].mean()) if mask_pre.any() else 1e-9
    overshoot_ratio = (peak_err / (pre_steady + 1e-9)) if not np.isnan(peak_err) else float("nan")
    osc_freq        = _oscillation_freq(err_arr[mask_post, DIST_JOINT], dt) if mask_post.any() else float("nan")

    m: dict = {
        "MAE_pre":           round(mae_pre  * 1000, 4),
        "MAE_post":          round(mae_post * 1000, 4),
        "peak_err_rad":      round(peak_err, 6),
        "recovery_time_s":   recovery_time,
        "overshoot_ratio":   round(float(overshoot_ratio), 4),
        "oscillation_freq":  round(osc_freq, 4),
        "energy_J":          round(float(np.sum(tau_arr ** 2) * dt), 4),
        "adaptation_traj":   adapt_traj[:150],  # 最大 150 点（0.3 s @ 500 Hz）
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


def make_ctrl(inverse_model_loc, env_seed, cfc_path, ctrl_seed):
    env = FrankaEnv(rng=np.random.default_rng(env_seed), obs_noise_std=0.002)
    cfg = AnatomicalConfig(
        prop_delay_steps=PROP_DELAY_FIXED,
        cereb_delay_steps=CEREB_DELAY_FIXED,
        kp=KP_DEFAULT.copy(), kd=KD_DEFAULT.copy(),
        io_mode="sparse", io_firing_rate_hz=1.0, io_gain=5.0,
        inverse_model_loc=inverse_model_loc,
        efcopy_enabled=True,
        cfc_hidden_units=64, cfc_device=DEVICE,
    )
    ctrl = AnatomicalController(cfg, seed=ctrl_seed)
    if cfc_path:
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
            inverse_model_loc=cond_kw["inverse_model_loc"],
            env_seed=seed * 100,
            cfc_path=fwd_path_str,
            ctrl_seed=seed,
        )
        m = run_faithful(ctrl, env, seed, DIST_TORQUE)
        m.update({
            "condition":          cond_name,
            "inverse_model_loc":  cond_kw["inverse_model_loc"],
            "prop_delay_steps":   PROP_DELAY_FIXED,
            "cereb_delay_steps":  CEREB_DELAY_FIXED,
            "seed":               seed,
        })

        with open(out_dir / "metrics.json", "w") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)

        rt = "n/a" if m["recovery_time_s"] is None else f"{m['recovery_time_s']:.3f}s"
        print(
            f"  {cond_name:15s}  MAE_post={m['MAE_post']:7.2f} mrad  "
            f"overshoot={m['overshoot_ratio']:5.2f}  RT={rt}"
        )


def build_summary(seeds: list[int]) -> dict:
    summary: dict = {
        "experiment": "f5",
        "hypothesis": "H5: 逆モデル所在 M1 のみが最高適応性",
        "seeds": seeds,
        "conditions": {},
    }

    # F0-abstract 結果を再掲
    abs_maes = []
    for seed in seeds:
        p = F0_ABS_DIR / f"seed{seed}" / "metrics.json"
        if p.exists():
            d = json.loads(p.read_text())
            v = d.get("MAE_post", float("nan"))
            if not np.isnan(v):
                abs_maes.append(v)
    summary["conditions"]["f5_abstract_efull"] = {
        "source":          "f0_abstract/d0 (reuse)",
        "MAE_post_mean":   round(float(np.mean(abs_maes)), 3) if abs_maes else None,
        "MAE_post_std":    round(float(np.std(abs_maes)),  3) if abs_maes else None,
        "n_seeds":         len(abs_maes),
    }

    for cond_name in CONDITIONS:
        maes, rts, overs = [], [], []
        for seed in seeds:
            p = RESULTS_DIR / cond_name / f"seed{seed}" / "metrics.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            if not np.isnan(d.get("MAE_post", float("nan"))):
                maes.append(d["MAE_post"])
            if d.get("recovery_time_s") is not None:
                rts.append(d["recovery_time_s"])
            ov = d.get("overshoot_ratio", float("nan"))
            if not np.isnan(ov):
                overs.append(ov)

        summary["conditions"][cond_name] = {
            "MAE_post_mean":        round(float(np.mean(maes)),  3) if maes  else None,
            "MAE_post_std":         round(float(np.std(maes)),   3) if maes  else None,
            "recovery_time_mean":   round(float(np.mean(rts)),   4) if rts   else None,
            "overshoot_ratio_mean": round(float(np.mean(overs)), 4) if overs else None,
            "n_seeds":              len(maes),
        }

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F5: 逆モデル所在実験 (仮説 H5)")
    p.add_argument("--seed",  type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    p.add_argument("--check-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    seeds = [args.seed] if args.seed is not None else args.seeds

    if not args.check_only:
        print(f"F5 実験開始  seeds={seeds}  device={DEVICE}")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            run_seed(seed)

    summary = build_summary(seeds)
    out = RESULTS_DIR / "f5_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("F5 集計結果")
    print("=" * 70)
    print(f"{'condition':>25}  {'MAE_post':>12}  {'recovery':>10}  {'overshoot':>10}")
    for cname, data in summary["conditions"].items():
        mae_s = (f"{data['MAE_post_mean']:7.2f}±{data['MAE_post_std']:.2f}"
                 if data.get("MAE_post_mean") else "  N/A")
        rt_s  = f"{data.get('recovery_time_mean', 0):.3f}s" if data.get("recovery_time_mean") else "  N/A"
        ov_s  = f"{data.get('overshoot_ratio_mean', 0):.3f}" if data.get("overshoot_ratio_mean") else "  N/A"
        print(f"  {cname:>23}  {mae_s:>12}  {rt_s:>10}  {ov_s:>10}")
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
