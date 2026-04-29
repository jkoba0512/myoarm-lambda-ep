"""
実験 F1: 感覚遅延スイープ（仮説 H1 検証）

仮説 H1:
  固有受容遅延（prop_delay_steps）が増加すると振動・不安定化が生じる。
  生体正常値（20 ms = 10 steps）付近で動作が安定し、それを超えると劣化する。

実験設計:
  AnatomicalController の prop_delay_steps のみを変化させる。
  cereb_delay_steps は常に 15 (30 ms) で固定。
  D0 条件（フル PD ゲイン、外乱あり 60 Nm）を使用。

事前指定評価指標 (pre-specified, H1):
  - oscillation_freq [Hz] : 外乱後誤差の支配周波数（遅延増加で上昇→振動化を示す）
  - instability_ratio     : MAE_post > 500 mrad を不安定と定義したシード比率
  - MAE_increase_rate     : prop_delay_steps が 1 増えるあたりの MAE_post 増加 [mrad/step]
  （上記は実験完了後の集計スクリプト f1_summary で算出）

出力:
  results/experiment_franka_f1/
    delay{0,5,10,20,30,50}/
      seed{0..9}/metrics.json
    f1_summary.json
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

# ── 定数 ─────────────────────────────────────────────────────────────────
DEVICE       = "cuda" if _torch.cuda.is_available() else "cpu"
SIM_DURATION = 6.0
DIST_T       = 3.0
DIST_STEPS   = 20
DIST_JOINT   = 1
DIST_TORQUE  = 60.0
INSTABILITY_THR_MRAD = 500.0  # 不安定判定閾値 [mrad]

Q_OFFSET   = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.5, 0.0])
KP_DEFAULT = np.array([50., 50., 50., 50., 10., 10., 10.])
KD_DEFAULT = np.array([ 7.,  7.,  7.,  7.,  1.5, 1.5, 1.5])

# 小脳ループ遅延は F1 では固定（F2 で変化させる）
CEREB_DELAY_STEPS_FIXED = 15

# 感覚遅延スイープ条件
PROP_DELAY_SWEEP = [0, 5, 10, 20, 30, 50]

E1_DIR      = ROOT / "results" / "experiment_franka_e1"
RESULTS_DIR = ROOT / "results" / "experiment_franka_f1"

SEEDS_DEFAULT = list(range(10))


# ── メトリクス計算（F0 と共通） ────────────────────────────────────────────

def _jerk_rms(qa: np.ndarray, dt: float) -> float:
    if len(qa) < 4:
        return float("nan")
    d3q = np.diff(qa, n=3, axis=0) / (dt ** 3)
    return float(np.sqrt(np.mean(d3q ** 2)))


def _oscillation_freq(err_1d: np.ndarray, dt: float) -> float:
    if len(err_1d) < 8:
        return float("nan")
    n = len(err_1d)
    fft = np.abs(np.fft.rfft(err_1d - err_1d.mean()))
    freqs = np.fft.rfftfreq(n, d=dt)
    if len(freqs) < 2:
        return float("nan")
    peak_idx = int(np.argmax(fft[1:])) + 1
    return float(freqs[peak_idx])


def compute_metrics(
    t_arr:  np.ndarray,
    q_arr:  np.ndarray,
    tau_arr: np.ndarray,
    dt:     float,
    io_stats: dict | None = None,
) -> dict:
    err_arr   = np.abs(q_arr - Q_OFFSET)
    mask_pre  = t_arr < DIST_T
    mask_post = t_arr > DIST_T

    mae_pre  = float(err_arr[mask_pre].mean())  if mask_pre.any()  else float("nan")
    mae_post = float(err_arr[mask_post].mean()) if mask_post.any() else float("nan")

    peak_err = float("nan")
    recovery_time = None
    if mask_post.any():
        post_err_j = err_arr[mask_post, DIST_JOINT]
        post_t     = t_arr[mask_post]
        peak_err   = float(post_err_j.max())
        rec_idx    = np.where(post_err_j < 0.1)[0]
        if len(rec_idx):
            recovery_time = float(post_t[rec_idx[0]] - DIST_T)

    pre_steady    = float(err_arr[mask_pre, DIST_JOINT].mean()) if mask_pre.any() else 1e-9
    overshoot_ratio = (peak_err / (pre_steady + 1e-9)) if not np.isnan(peak_err) else float("nan")

    osc_freq = _oscillation_freq(err_arr[mask_post, DIST_JOINT], dt) if mask_post.any() else float("nan")
    energy   = float(np.sum(tau_arr ** 2) * dt)
    energy_post = float(np.sum(tau_arr[mask_post] ** 2) * dt) if mask_post.any() else float("nan")
    jerk_rms = _jerk_rms(q_arr, dt)

    m: dict = {
        "MAE_pre":          round(mae_pre  * 1000, 4),
        "MAE_post":         round(mae_post * 1000, 4),
        "peak_err_rad":     round(peak_err, 6),
        "recovery_time_s":  recovery_time,
        "overshoot_ratio":  round(float(overshoot_ratio), 4),
        "oscillation_freq": round(osc_freq, 4),
        "energy_J":         round(energy, 4),
        "energy_post_J":    round(energy_post, 4),
        "jerk_rms":         round(jerk_rms, 6),
        "unstable":         mae_post * 1000 > INSTABILITY_THR_MRAD,
    }

    if io_stats:
        m["io_fire_count"]         = io_stats.get("io_fire_count", 0)
        m["io_fire_rate_hz"]       = round(io_stats.get("io_fire_rate_hz", 0.0), 4)
        m["io_fire_interval_mean"] = round(io_stats.get("io_fire_interval_mean", 0.0), 4)
    else:
        m["io_fire_count"] = m["io_fire_rate_hz"] = m["io_fire_interval_mean"] = None

    return m


# ── エピソード実行 ──────────────────────────────────────────────────────────

def run_faithful(
    ctrl: AnatomicalController,
    env:  FrankaEnv,
    seed: int,
    dist_torque: float,
) -> dict:
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

    return compute_metrics(
        np.array(t_log), np.array(q_log), np.array(tau_log), dt,
        io_stats=ctrl.get_io_stats(),
    )


# ── コントローラ生成 ────────────────────────────────────────────────────────

def make_ctrl(
    prop_delay_steps: int,
    env_seed:         int,
    cfc_path:         str | None,
    ctrl_seed:        int,
) -> tuple[AnatomicalController, FrankaEnv]:
    env = FrankaEnv(rng=np.random.default_rng(env_seed), obs_noise_std=0.002)
    cfg = AnatomicalConfig(
        prop_delay_steps=prop_delay_steps,
        cereb_delay_steps=CEREB_DELAY_STEPS_FIXED,
        kp=KP_DEFAULT.copy(),
        kd=KD_DEFAULT.copy(),
        io_mode="sparse",
        io_firing_rate_hz=1.0,
        io_gain=5.0,
        inverse_model_loc="m1",
        efcopy_enabled=True,
        cfc_hidden_units=64,
        cfc_device=DEVICE,
    )
    ctrl = AnatomicalController(cfg, seed=ctrl_seed)
    if cfc_path:
        ctrl.load_cfc(cfc_path)
    return ctrl, env


# ── seed 単位のメイン処理 ────────────────────────────────────────────────────

def run_seed(seed: int) -> None:
    fwd_path = E1_DIR / f"seed{seed}" / "cfc_forward.pt"
    fwd_path_str = str(fwd_path) if fwd_path.exists() else None
    if not fwd_path_str:
        print(f"  WARNING: CfC モデルが見つかりません: {fwd_path}")

    print(f"\nseed={seed}")

    for prop_steps in PROP_DELAY_SWEEP:
        delay_ms = prop_steps * 2
        label    = f"delay{prop_steps}"

        out_dir = RESULTS_DIR / label / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        ctrl, env = make_ctrl(
            prop_delay_steps=prop_steps,
            env_seed=seed * 100,
            cfc_path=fwd_path_str,
            ctrl_seed=seed,
        )
        m = run_faithful(ctrl, env, seed, DIST_TORQUE)
        m.update({
            "condition":          label,
            "prop_delay_steps":   prop_steps,
            "prop_delay_ms":      delay_ms,
            "cereb_delay_steps":  CEREB_DELAY_STEPS_FIXED,
            "seed":               seed,
        })

        with open(out_dir / "metrics.json", "w") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)

        unstable_mark = " *** UNSTABLE ***" if m["unstable"] else ""
        rt = "n/a" if m["recovery_time_s"] is None else f"{m['recovery_time_s']:.3f}s"
        print(
            f"  delay={prop_steps:2d} ({delay_ms:3d}ms)  "
            f"MAE_post={m['MAE_post']:7.2f} mrad  "
            f"osc={m['oscillation_freq']:5.2f}Hz  RT={rt}{unstable_mark}"
        )


# ── サマリ集計 ────────────────────────────────────────────────────────────────

def build_summary(seeds: list[int]) -> dict:
    summary: dict = {
        "experiment":         "f1",
        "hypothesis":         "H1: 感覚遅延増加 → 振動・不安定化",
        "seeds":              seeds,
        "cereb_delay_fixed":  CEREB_DELAY_STEPS_FIXED,
        "instability_thr_mrad": INSTABILITY_THR_MRAD,
        "sweep":              {},
    }

    mae_by_delay: dict[int, list[float]] = {s: [] for s in PROP_DELAY_SWEEP}
    osc_by_delay: dict[int, list[float]] = {s: [] for s in PROP_DELAY_SWEEP}
    unstable_cnt: dict[int, int]         = {s: 0  for s in PROP_DELAY_SWEEP}

    for prop_steps in PROP_DELAY_SWEEP:
        label = f"delay{prop_steps}"
        for seed in seeds:
            p = RESULTS_DIR / label / f"seed{seed}" / "metrics.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            mae = d.get("MAE_post", float("nan"))
            osc = d.get("oscillation_freq", float("nan"))
            if not np.isnan(mae):
                mae_by_delay[prop_steps].append(mae)
            if not np.isnan(osc):
                osc_by_delay[prop_steps].append(osc)
            if d.get("unstable", False):
                unstable_cnt[prop_steps] += 1

    for prop_steps in PROP_DELAY_SWEEP:
        n = len(seeds)
        maes = mae_by_delay[prop_steps]
        oscs = osc_by_delay[prop_steps]
        summary["sweep"][str(prop_steps)] = {
            "prop_delay_ms":      prop_steps * 2,
            "MAE_post_mean":      round(float(np.mean(maes)),  3) if maes else None,
            "MAE_post_std":       round(float(np.std(maes)),   3) if maes else None,
            "oscillation_freq_mean": round(float(np.mean(oscs)), 3) if oscs else None,
            "oscillation_freq_std":  round(float(np.std(oscs)),  3) if oscs else None,
            "instability_ratio":  unstable_cnt[prop_steps] / n if n > 0 else None,
            "n_seeds":            len(maes),
        }

    # MAE_increase_rate: 線形回帰 (delay_steps vs MAE_post_mean)
    valid_steps = [s for s in PROP_DELAY_SWEEP if mae_by_delay[s]]
    if len(valid_steps) >= 2:
        x = np.array(valid_steps, dtype=float)
        y = np.array([np.mean(mae_by_delay[s]) for s in valid_steps])
        slope, _ = np.polyfit(x, y, 1)
        summary["MAE_increase_rate_mrad_per_step"] = round(float(slope), 4)
    else:
        summary["MAE_increase_rate_mrad_per_step"] = None

    # instability_threshold: 初めて不安定率 > 50% となる delay_steps
    thr = None
    for s in PROP_DELAY_SWEEP:
        ratio = unstable_cnt[s] / len(seeds) if seeds else 0
        if ratio > 0.5:
            thr = s
            break
    summary["instability_threshold_steps"] = thr

    return summary


# ── argparse ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="F1: 感覚遅延スイープ実験 (仮説 H1)"
    )
    p.add_argument("--seed",  type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    p.add_argument("--check-only", action="store_true",
                   help="集計・完了条件チェックのみ実行")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    seeds = [args.seed] if args.seed is not None else args.seeds

    if not args.check_only:
        print(f"F1 実験開始  seeds={seeds}  device={DEVICE}")
        print(f"prop_delay sweep: {PROP_DELAY_SWEEP} steps")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            run_seed(seed)

    summary = build_summary(seeds)
    out = RESULTS_DIR / "f1_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("F1 集計結果")
    print("=" * 60)
    print(f"{'delay_steps':>12}  {'delay_ms':>8}  {'MAE_post':>10}  {'osc_freq':>9}  {'unstable%':>9}")
    for s, data in summary["sweep"].items():
        mae_s   = f"{data['MAE_post_mean']:7.2f}±{data['MAE_post_std']:.2f}" if data["MAE_post_mean"] else "  N/A"
        osc_s   = f"{data['oscillation_freq_mean']:6.2f}" if data["oscillation_freq_mean"] else "  N/A"
        unst_s  = f"{data['instability_ratio']*100:5.1f}%" if data["instability_ratio"] is not None else "  N/A"
        print(f"{s:>12}  {data['prop_delay_ms']:>8}  {mae_s:>10}  {osc_s:>9}  {unst_s:>9}")

    print(f"\nMAE increase rate : {summary['MAE_increase_rate_mrad_per_step']} mrad/step")
    print(f"Instability thresh: {summary['instability_threshold_steps']} steps")
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
