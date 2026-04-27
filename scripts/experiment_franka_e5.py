"""
実験 E5: 評価指標の拡張

目的:
  E1–E4 の主要条件を再評価し、mae 以外の多面的指標を追加する。
  エネルギー効率・軌道滑らかさ・コ・コントラクション率を比較することで
  「ヒト神経運動制御に倣った実装が省エネ・滑らかかどうか」を検証する。

比較条件（D0/D1/D5 × 4条件 × 11 seed）:
  E5-PD       : PD のみ（ベースライン）
  E5-E3       : CC + Ia/Ib 反射弧（E3 ベスト）
  E5-MCA      : MCA + CC（E4-MCA）
  E5-full     : MCA + CC + Ia/Ib 反射弧（E4-full）

評価指標:
  - mae_post_mrad   : 外乱後 MAE [mrad]（E2–E4 と比較可能）
  - energy_J        : 総エネルギー消費 ∫τ²dt [Nm²·s]
  - jerk            : 軌道の粗さ ∫(d³q/dt³)²dt [rad²/s⁵]（全関節和）
  - cc_ratio        : 同時収縮率 = mean(τ_cc) / (mean(|τ|) + mean(τ_cc) + ε)
  - recovery_time_s : 外乱回復時間 [s]
  - peak_err_rad    : 外乱後ピーク誤差 [rad]

出力:
  results/experiment_franka_e5/{sweep_name}/seed<N>/metrics.json
  results/franka_e5_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.franka_env import FrankaEnv, N_JOINTS
from common.franka_neural_controller import FrankaNeuralController

RESULTS_DIR = ROOT / "results" / "experiment_franka_e5"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

Q_OFFSET   = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.5, 0.0])
CPG_PARAMS = dict(tau=0.3, tau_r=0.6, beta=2.5, w=2.0, amplitude=0.0)

SIM_DURATION = 6.0
DIST_T       = 3.0
DIST_JOINT   = 1
DIST_TORQUE  = 60.0
DIST_STEPS   = 20

D_CONDITIONS = {
    "D0": {},
    "D1": {"kp_scale": 0.5},
    "D5": {"torque_saturation": 30.0},
}

KP_DEFAULT = np.array([50., 50., 50., 50., 10., 10., 10.])
KD_DEFAULT = np.array([7.,  7.,  7.,  7.,  1.5, 1.5, 1.5])


def run_episode(
    ctrl, env,
    do_disturb: bool  = True,
    dist_torque: float = DIST_TORQUE,
) -> dict:
    q, dq = env.reset(q0=Q_OFFSET.copy())
    ctrl.reset()
    dt = env.dt

    t_log, q_log, dq_log, tau_log, tau_cc_log = [], [], [], [], []
    dist_applied = False

    while env.time < SIM_DURATION:
        t = env.time
        q, dq = env.get_state()

        if do_disturb and not dist_applied and t >= DIST_T:
            td = np.zeros(N_JOINTS)
            td[DIST_JOINT] = dist_torque
            env.apply_disturbance(td, duration_steps=DIST_STEPS)
            dist_applied = True
            continue

        tau, info = ctrl.step(q, dq, Q_OFFSET)
        env.step(tau)

        t_log.append(t)
        q_log.append(q.copy())
        dq_log.append(dq.copy())
        tau_log.append(tau.copy())
        tau_cc_log.append(
            info["tau_cc"].copy() if "tau_cc" in info and info["tau_cc"] is not None
            else np.zeros(N_JOINTS)
        )

    ta   = np.array(t_log)
    qa   = np.array(q_log)       # (T, n)
    taua = np.array(tau_log)     # (T, n)
    tau_cc_a = np.array(tau_cc_log)  # (T, n)
    err  = np.abs(qa - Q_OFFSET)

    # ── 基本指標 ──────────────────────────────────────────────────────
    result: dict = {}
    result["mae"] = float(err.mean())

    if do_disturb:
        mask     = ta > DIST_T
        post_err = err[mask, DIST_JOINT] if mask.any() else np.array([])
        rec_i    = np.where(post_err < 0.1)[0]
        result["peak_err"]      = float(post_err.max()) if len(post_err) else float("nan")
        result["recovery_time"] = float(ta[mask][rec_i[0]] - DIST_T) if len(rec_i) else None
        result["mae_post"]      = float(post_err.mean()) if len(post_err) else float("nan")

    # ── エネルギー消費 ∫τ²dt ────────────────────────────────────────
    # 全関節の二乗トルク積分 [Nm²·s]
    result["energy_J"] = float(np.sum(taua ** 2) * dt)

    # 外乱後エネルギー（回復コスト）
    if do_disturb and mask.any():
        result["energy_post_J"] = float(np.sum(taua[mask] ** 2) * dt)
    else:
        result["energy_post_J"] = float("nan")

    # ── 軌道滑らかさ（jerk）∫(d³q/dt³)²dt ───────────────────────────
    # 3階差分で近似: d³q ≈ diff³(q) / dt³
    if len(qa) >= 4:
        d3q  = np.diff(qa, n=3, axis=0) / (dt ** 3)   # (T-3, n)
        result["jerk"] = float(np.sum(d3q ** 2) * dt)
    else:
        result["jerk"] = float("nan")

    # 外乱後 jerk
    if do_disturb and len(qa) >= 4:
        post_idx = np.where(ta > DIST_T)[0]
        if len(post_idx) >= 4:
            qa_post = qa[post_idx]
            d3_post = np.diff(qa_post, n=3, axis=0) / (dt ** 3)
            result["jerk_post"] = float(np.sum(d3_post ** 2) * dt)
        else:
            result["jerk_post"] = float("nan")
    else:
        result["jerk_post"] = float("nan")

    # ── コ・コントラクション率 ─────────────────────────────────────────
    # cc_ratio = mean(τ_cc) / (mean(|τ|) + mean(τ_cc) + ε)
    # τ_cc=0 の条件（PD など）は 0 になる
    mean_tau_cc  = float(np.mean(tau_cc_a))
    mean_abs_tau = float(np.mean(np.abs(taua)))
    result["cc_ratio"] = mean_tau_cc / (mean_abs_tau + mean_tau_cc + 1e-8)

    return result


def make_ctrl(
    kp_scale:   float        = 1.0,
    torque_sat: float | None = None,
    use_cc:     bool         = False,
    use_ia_ib:  bool         = False,
    use_mca:    bool         = False,
    env_seed:   int          = 9999,
) -> tuple[FrankaNeuralController, FrankaEnv]:
    kp = KP_DEFAULT * kp_scale
    env_kw: dict = {}
    if torque_sat is not None:
        env_kw["torque_saturation"] = torque_sat

    ctrl_env = FrankaEnv(
        rng=np.random.default_rng(env_seed),
        obs_noise_std=0.002,
        **env_kw,
    )
    ctrl = FrankaNeuralController(
        dt=ctrl_env.dt,
        q_range=ctrl_env.ctrl_range,
        cpg_params=CPG_PARAMS,
        kp=kp, kd=KD_DEFAULT.copy(),
        use_proprioceptor=False,
        use_reflex=use_ia_ib,
        use_ia_ib_reflex=use_ia_ib,
        use_cerebellum=False,
        use_cocontraction=use_cc,
        use_motor_cortex=use_mca,
        cpg_alpha_fb=0.0,
    )
    return ctrl, ctrl_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--sweep-name", type=str, default="default")
    return p.parse_args()


def main():
    args   = parse_args()
    seed   = args.seed
    base   = RESULTS_DIR if args.sweep_name == "default" else RESULTS_DIR / args.sweep_name
    outdir = base / f"seed{seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    print(f"seed={seed}  out={outdir}")

    conds = [
        ("E5-PD",   dict(use_cc=False, use_ia_ib=False, use_mca=False)),
        ("E5-E3",   dict(use_cc=True,  use_ia_ib=True,  use_mca=False)),
        ("E5-MCA",  dict(use_cc=True,  use_ia_ib=False, use_mca=True)),
        ("E5-full", dict(use_cc=True,  use_ia_ib=True,  use_mca=True)),
    ]

    results: dict = {}

    for d_idx, (d_name, d_kw) in enumerate(D_CONDITIONS.items()):
        kp_scale   = d_kw.get("kp_scale", 1.0)
        torque_sat = d_kw.get("torque_saturation", None)
        results[d_name] = {}

        d_rng = np.random.default_rng(seed * 100 + d_idx)
        actual_dist_torque = float(np.clip(
            DIST_TORQUE * (1.0 + d_rng.normal(0.0, 0.25)), 20.0, 120.0
        ))
        d_env_seed = seed * 100 + d_idx

        for cname, ckw in conds:
            ctrl, env = make_ctrl(
                kp_scale=kp_scale, torque_sat=torque_sat,
                env_seed=d_env_seed, **ckw,
            )
            res = run_episode(ctrl, env, do_disturb=True, dist_torque=actual_dist_torque)
            results[d_name][cname] = res

            rt = res.get("recovery_time")
            rt_s = f"{rt:.3f}s" if rt else "未回復"
            print(
                f"  [{d_name}] {cname:10s}  "
                f"MAE_post={res.get('mae_post', float('nan'))*1000:7.1f} mrad  "
                f"E={res['energy_J']:8.1f} Nm²s  "
                f"jerk={res['jerk']:.2e}  "
                f"cc={res['cc_ratio']:.3f}  RT={rt_s}"
            )

    # ── JSON 保存 ────────────────────────────────────────────────────
    metrics_out = {
        "experiment": "e5",
        "seed": seed,
        "sweep_name": args.sweep_name,
        "results": {
            d: {
                c: {
                    "mae_mrad":        float(r["mae"] * 1000),
                    "mae_post_mrad":   float(r.get("mae_post", float("nan")) or float("nan")) * 1000,
                    "peak_err_rad":    float(r.get("peak_err", float("nan"))),
                    "recovery_time_s": r.get("recovery_time"),
                    "energy_J":        float(r["energy_J"]),
                    "energy_post_J":   float(r.get("energy_post_J", float("nan"))),
                    "jerk":            float(r["jerk"]),
                    "jerk_post":       float(r.get("jerk_post", float("nan"))),
                    "cc_ratio":        float(r["cc_ratio"]),
                }
                for c, r in cond_res.items()
            }
            for d, cond_res in results.items()
        },
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)

    # ── プロット ─────────────────────────────────────────────────────
    cond_names = ["E5-PD", "E5-E3", "E5-MCA", "E5-full"]
    colors     = ["tab:gray", "tab:red", "tab:blue", "tab:purple"]
    metrics_plot = [
        ("mae_post",    "MAE_post [mrad]",  1000.0),
        ("energy_J",    "Energy [Nm²·s]",   1.0),
        ("jerk",        "Jerk [rad²/s⁵]",   1.0),
        ("cc_ratio",    "CC ratio",          1.0),
    ]

    fig, axes = plt.subplots(len(metrics_plot), len(D_CONDITIONS), figsize=(16, 12))
    fig.suptitle(f"Experiment E5: Extended Metrics  [seed={seed}]", fontsize=12)

    for col, d_name in enumerate(D_CONDITIONS):
        for row, (metric_key, ylabel, scale) in enumerate(metrics_plot):
            ax = axes[row][col]
            vals = [results[d_name].get(c, {}).get(metric_key, float("nan")) for c in cond_names]
            vals = [v * scale if not np.isnan(v) else float("nan") for v in vals]

            bars = ax.bar(range(len(cond_names)), vals, color=colors, alpha=0.8)
            ax.set_xticks(range(len(cond_names)))
            ax.set_xticklabels(cond_names, fontsize=7, rotation=15, ha="right")
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(f"{d_name}: {ylabel}", fontsize=8)
            valid = [v for v in vals if not np.isnan(v)]
            vmax  = max(valid) if valid else 1.0
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    fmt = ".2e" if metric_key == "jerk" else ".1f"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + vmax * 0.02,
                        f"{v:{fmt}}", ha="center", va="bottom", fontsize=6,
                    )
            ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = outdir / "plot_franka_e5.png"
    plt.savefig(str(plot_path), dpi=150)
    plt.close()
    print(f"\nプロット保存: {plot_path}")

    print("\n=== E5 結果サマリ ===")
    for d_name in D_CONDITIONS:
        print(f"\n  {d_name}:")
        for cname in cond_names:
            r = results[d_name][cname]
            rt = r.get("recovery_time")
            rt_s = f"{rt:.3f}s" if rt else "未回復"
            print(
                f"    {cname:10s}  "
                f"MAE_post={r.get('mae_post', float('nan'))*1000:7.1f} mrad  "
                f"E={r['energy_J']:8.1f} Nm²s  "
                f"jerk={r['jerk']:.2e}  "
                f"cc={r['cc_ratio']:.3f}  RT={rt_s}"
            )


if __name__ == "__main__":
    main()
