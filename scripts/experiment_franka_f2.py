"""
実験 F2: 小脳ループ遅延スイープ（仮説 H2 検証）

仮説 H2:
  小脳サイドループ遅延（cereb_delay_steps）が増加するとオーバーシュート・振動が増加する。
  遅延 0 ms（フィードバックなし同期補正）で最高性能だが、過補正による振動が起きる。
  生体正常値（30 ms = 15 steps）付近でオーバーシュートと振動のバランスが最良になる。

実験設計:
  AnatomicalController の cereb_delay_steps のみを変化させる。
  prop_delay_steps は常に 10 (20 ms) で固定（生体正常値）。
  ablated 条件（小脳補正を完全にゼロにする）も含む。
  D0 条件（フル PD ゲイン、外乱あり 60 Nm）を使用。

事前指定評価指標 (pre-specified, H2):
  - overshoot_ratio       : ピーク誤差 / 定常誤差（オーバーシュート度合い）
  - oscillation_freq [Hz] : 外乱後誤差の支配周波数
  - decomposition_index   : (MAE_post - MAE_pre_ablated) / MAE_pre_ablated（小脳寄与指標）
  - recovery_time [s]     : 外乱後 0.1 rad 以内への復帰時間

出力:
  results/experiment_franka_f2/
    loop{0,8,15,30}/seed{0..9}/metrics.json
    ablated/seed{0..9}/metrics.json
    f2_summary.json
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
INSTABILITY_THR_MRAD = 500.0

Q_OFFSET   = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.5, 0.0])
KP_DEFAULT = np.array([50., 50., 50., 50., 10., 10., 10.])
KD_DEFAULT = np.array([ 7.,  7.,  7.,  7.,  1.5, 1.5, 1.5])

# 固有受容遅延は F2 では固定（F1 で変化させた）
PROP_DELAY_STEPS_FIXED = 10

# 小脳ループ遅延スイープ条件
# None は ablated（小脳補正ゼロ）を表す
CEREB_DELAY_SWEEP: list[int | None] = [0, 8, 15, 30, None]

E1_DIR      = ROOT / "results" / "experiment_franka_e1"
RESULTS_DIR = ROOT / "results" / "experiment_franka_f2"

SEEDS_DEFAULT = list(range(10))


# ── メトリクス計算 ────────────────────────────────────────────────────────

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
    t_arr:    np.ndarray,
    q_arr:    np.ndarray,
    tau_arr:  np.ndarray,
    dt:       float,
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

    osc_freq    = _oscillation_freq(err_arr[mask_post, DIST_JOINT], dt) if mask_post.any() else float("nan")
    energy      = float(np.sum(tau_arr ** 2) * dt)
    energy_post = float(np.sum(tau_arr[mask_post] ** 2) * dt) if mask_post.any() else float("nan")
    jerk_rms    = _jerk_rms(q_arr, dt)

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
    cereb_delay_steps: int | None,
    env_seed:          int,
    cfc_path:          str | None,
    ctrl_seed:         int,
) -> tuple[AnatomicalController, FrankaEnv]:
    env = FrankaEnv(rng=np.random.default_rng(env_seed), obs_noise_std=0.002)

    # ablated: cereb_delay_steps=None → cereb_delay_steps=0 + efcopy_enabled=False
    # で小脳補正を実質無効化する（CfC は存在するが出力は efcopy=0 → 補正=0）
    actual_cereb = cereb_delay_steps if cereb_delay_steps is not None else 0

    cfg = AnatomicalConfig(
        prop_delay_steps=PROP_DELAY_STEPS_FIXED,
        cereb_delay_steps=actual_cereb,
        kp=KP_DEFAULT.copy(),
        kd=KD_DEFAULT.copy(),
        io_mode="sparse",
        io_firing_rate_hz=1.0,
        io_gain=5.0,
        inverse_model_loc="m1",
        efcopy_enabled=(cereb_delay_steps is not None),  # ablated で efcopy を遮断
        cfc_hidden_units=64,
        cfc_device=DEVICE,
    )
    ctrl = AnatomicalController(cfg, seed=ctrl_seed)
    if cfc_path and cereb_delay_steps is not None:
        ctrl.load_cfc(cfc_path)
    return ctrl, env


# ── seed 単位のメイン処理 ────────────────────────────────────────────────────

def run_seed(seed: int) -> None:
    fwd_path = E1_DIR / f"seed{seed}" / "cfc_forward.pt"
    fwd_path_str = str(fwd_path) if fwd_path.exists() else None
    if not fwd_path_str:
        print(f"  WARNING: CfC モデルが見つかりません: {fwd_path}")

    print(f"\nseed={seed}")

    for cereb_steps in CEREB_DELAY_SWEEP:
        label    = "ablated" if cereb_steps is None else f"loop{cereb_steps}"
        delay_ms = 0 if cereb_steps is None else cereb_steps * 2

        out_dir = RESULTS_DIR / label / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        ctrl, env = make_ctrl(
            cereb_delay_steps=cereb_steps,
            env_seed=seed * 100,
            cfc_path=fwd_path_str,
            ctrl_seed=seed,
        )
        m = run_faithful(ctrl, env, seed, DIST_TORQUE)
        m.update({
            "condition":         label,
            "prop_delay_steps":  PROP_DELAY_STEPS_FIXED,
            "cereb_delay_steps": cereb_steps,
            "cereb_delay_ms":    delay_ms,
            "ablated":           cereb_steps is None,
            "seed":              seed,
        })

        with open(out_dir / "metrics.json", "w") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)

        unstable_mark = " *** UNSTABLE ***" if m["unstable"] else ""
        rt = "n/a" if m["recovery_time_s"] is None else f"{m['recovery_time_s']:.3f}s"
        ablated_mark = " [ablated]" if cereb_steps is None else ""
        print(
            f"  cereb={str(cereb_steps):>4} ({delay_ms:3d}ms){ablated_mark}  "
            f"MAE_post={m['MAE_post']:7.2f} mrad  "
            f"overshoot={m['overshoot_ratio']:5.2f}  RT={rt}{unstable_mark}"
        )


# ── サマリ集計 ────────────────────────────────────────────────────────────────

def build_summary(seeds: list[int]) -> dict:
    summary: dict = {
        "experiment":        "f2",
        "hypothesis":        "H2: 小脳ループ遅延増加 → オーバーシュート・振動",
        "seeds":             seeds,
        "prop_delay_fixed":  PROP_DELAY_STEPS_FIXED,
        "sweep":             {},
    }

    # ablated MAE を decomposition_index の基準として使う
    ablated_maes: list[float] = []
    for seed in seeds:
        p = RESULTS_DIR / "ablated" / f"seed{seed}" / "metrics.json"
        if p.exists():
            d = json.loads(p.read_text())
            v = d.get("MAE_post", float("nan"))
            if not np.isnan(v):
                ablated_maes.append(v)
    mae_ablated_mean = float(np.mean(ablated_maes)) if ablated_maes else float("nan")

    for cereb_steps in CEREB_DELAY_SWEEP:
        label = "ablated" if cereb_steps is None else f"loop{cereb_steps}"
        key   = "ablated" if cereb_steps is None else str(cereb_steps)

        maes, osr, rts, oscs = [], [], [], []
        for seed in seeds:
            p = RESULTS_DIR / label / f"seed{seed}" / "metrics.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            for lst, key2 in [(maes, "MAE_post"), (osr, "overshoot_ratio"),
                               (oscs, "oscillation_freq")]:
                v = d.get(key2, float("nan"))
                if not np.isnan(v):
                    lst.append(v)
            rt = d.get("recovery_time_s")
            if rt is not None:
                rts.append(rt)

        mae_mean = float(np.mean(maes)) if maes else float("nan")
        dec_idx  = ((mae_ablated_mean - mae_mean) / (mae_ablated_mean + 1e-9)
                    if not np.isnan(mae_ablated_mean) and not np.isnan(mae_mean) else float("nan"))

        summary["sweep"][key] = {
            "cereb_delay_ms":        (cereb_steps or 0) * 2,
            "ablated":               cereb_steps is None,
            "MAE_post_mean":         round(mae_mean, 3) if not np.isnan(mae_mean) else None,
            "MAE_post_std":          round(float(np.std(maes)), 3) if maes else None,
            "overshoot_ratio_mean":  round(float(np.mean(osr)),  3) if osr  else None,
            "oscillation_freq_mean": round(float(np.mean(oscs)), 3) if oscs else None,
            "recovery_time_mean":    round(float(np.mean(rts)),  3) if rts  else None,
            "decomposition_index":   round(dec_idx, 4) if not np.isnan(dec_idx) else None,
            "n_seeds":               len(maes),
        }

    return summary


# ── argparse ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="F2: 小脳ループ遅延スイープ実験 (仮説 H2)"
    )
    p.add_argument("--seed",  type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    p.add_argument("--check-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    seeds = [args.seed] if args.seed is not None else args.seeds

    if not args.check_only:
        print(f"F2 実験開始  seeds={seeds}  device={DEVICE}")
        print(f"cereb_delay sweep: {CEREB_DELAY_SWEEP} steps (None=ablated)")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            run_seed(seed)

    summary = build_summary(seeds)
    out = RESULTS_DIR / "f2_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("F2 集計結果")
    print("=" * 70)
    print(f"{'cereb_steps':>12}  {'delay_ms':>8}  {'MAE_post':>12}  {'overshoot':>9}  {'decomp_idx':>11}")
    for key, data in summary["sweep"].items():
        mae_s  = (f"{data['MAE_post_mean']:7.2f}±{data['MAE_post_std']:.2f}"
                  if data["MAE_post_mean"] else "  N/A")
        ov_s   = f"{data['overshoot_ratio_mean']:6.2f}" if data["overshoot_ratio_mean"] else "  N/A"
        dec_s  = f"{data['decomposition_index']:+.4f}" if data["decomposition_index"] is not None else "  N/A"
        ab_s   = " [ablated]" if data["ablated"] else ""
        print(f"{key:>12}  {data['cereb_delay_ms']:>8}  {mae_s:>12}  {ov_s:>9}  {dec_s:>11}{ab_s}")

    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
