"""
実験 E3: 真の Ia/Ib 反射弧の評価（Phase D 条件）

目的:
  IaIbReflexArc による脊髄反射が、E2（仮想コ・コントラクション）より
  さらに優れた外乱回復を示すかを検証する。

比較条件:
  E3-PD       : PD のみ（ベースライン）
  E3-cc       : 仮想コ・コントラクションのみ（E2 ベスト）
  E3-ia_ib    : Ia/Ib 反射弧のみ（reflex なし）
  E3-cc+ia_ib : 仮想コ・コントラクション + Ia/Ib 反射弧

評価条件（Phase D から抽出）:
  D0: baseline（ノイズなし）
  D1: 低ゲイン kp×0.5（旧実装で悪化 d=-0.420）
  D5: トルク飽和 30 Nm（旧実装で悪化 d=-0.185）

成功判定:
  1. Ia 反射潜時 ≤ 30 ms（≤ 15 steps @ 500 Hz）
  2. D1・D5 の両方で E3-ia_ib または E3-cc+ia_ib の Cohen's d > E2 の E2-cc 値

出力:
  results/experiment_franka_e3/{sweep_name}/seed<N>/
    ├── metrics.json
    └── plot_franka_e3.png
  results/franka_e3_summary.json
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

RESULTS_DIR = ROOT / "results" / "experiment_franka_e3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

Q_OFFSET   = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.5, 0.0])
CPG_PARAMS = dict(tau=0.3, tau_r=0.6, beta=2.5, w=2.0, amplitude=0.0)

SIM_DURATION = 6.0
DIST_T       = 3.0
DIST_JOINT   = 1
DIST_TORQUE  = 60.0
DIST_STEPS   = 20

# Phase D 条件
D_CONDITIONS = {
    "D0": {},
    "D1": {"kp_scale": 0.5},
    "D5": {"torque_saturation": 30.0},
}

KP_DEFAULT = np.array([50., 50., 50., 50., 10., 10., 10.])
KD_DEFAULT = np.array([7.,  7.,  7.,  7.,  1.5, 1.5, 1.5])


def run_episode(ctrl, env, do_disturb: bool = True,
                dist_torque: float = DIST_TORQUE) -> dict:
    q, dq = env.reset(q0=Q_OFFSET.copy())
    ctrl.reset()
    t_log, q_log = [], []
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

    ta  = np.array(t_log)
    qa  = np.array(q_log)
    err = np.abs(qa - Q_OFFSET)

    result = {"mae": err.mean()}
    if do_disturb:
        mask = ta > DIST_T
        post_err = err[mask, DIST_JOINT] if mask.any() else np.array([])
        rec_i = np.where(post_err < 0.1)[0]
        result["peak_err"]      = float(post_err.max()) if len(post_err) else np.nan
        result["recovery_time"] = float(ta[mask][rec_i[0]] - DIST_T) if len(rec_i) else None
        result["mae_post"]      = float(post_err.mean()) if len(post_err) else np.nan
    return result


def make_ctrl(
    env, kp_scale: float = 1.0, torque_sat: float | None = None,
    use_cc: bool = False, use_ia_ib: bool = False,
    env_seed: int = 9999,
) -> tuple[FrankaNeuralController, FrankaEnv]:
    kp = KP_DEFAULT * kp_scale
    env_kw: dict = {}
    if torque_sat is not None:
        env_kw["torque_saturation"] = torque_sat

    # obs_noise_std=0.002 rad (2 mrad) で観測ノイズを追加してシード間の分散を確保
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

    # Ia/Ib 反射潜時を確認（成功基準 1: ≤ 30 ms）
    from methodE.ia_ib_reflex import IaIbReflexArc
    dummy_reflex = IaIbReflexArc(n_joints=N_JOINTS, dt=0.002)
    latency_ms = dummy_reflex.get_reflex_latency_ms()
    print(f"Ia 反射潜時: {latency_ms:.1f} ms  (基準: ≤ 30 ms → {'OK' if latency_ms <= 30.0 else 'NG'})")

    base_env = FrankaEnv(rng=np.random.default_rng(seed + 1000))

    results: dict = {}

    for d_idx, (d_name, d_kw) in enumerate(D_CONDITIONS.items()):
        kp_scale   = d_kw.get("kp_scale", 1.0)
        torque_sat = d_kw.get("torque_saturation", None)

        results[d_name] = {}

        conds = [
            ("E3-PD",       dict(use_cc=False, use_ia_ib=False)),
            ("E3-cc",       dict(use_cc=True,  use_ia_ib=False)),
            ("E3-ia_ib",    dict(use_cc=False, use_ia_ib=True)),
            ("E3-cc+ia_ib", dict(use_cc=True,  use_ia_ib=True)),
        ]

        d_rng = np.random.default_rng(seed * 100 + d_idx)
        actual_dist_torque = float(np.clip(DIST_TORQUE * (1.0 + d_rng.normal(0.0, 0.25)), 20.0, 120.0))
        d_env_seed = seed * 100 + d_idx
        for cname, ckw in conds:
            ctrl, env = make_ctrl(base_env, kp_scale=kp_scale,
                                  torque_sat=torque_sat, env_seed=d_env_seed, **ckw)
            res = run_episode(ctrl, env, do_disturb=True, dist_torque=actual_dist_torque)
            results[d_name][cname] = res
            rt = res.get("recovery_time")
            rt_s = f"{rt:.3f}s" if rt else "未回復"
            print(f"  [{d_name}] {cname:14s}  MAE={res['mae']*1000:.1f} mrad"
                  f"  Peak={res.get('peak_err', np.nan):.4f} rad  RT={rt_s}")

    # ── JSON 保存 ────────────────────────────────────────────────────────
    metrics_out = {
        "experiment": "e3",
        "seed": seed,
        "sweep_name": args.sweep_name,
        "ia_latency_ms": latency_ms,
        "latency_ok": bool(latency_ms <= 30.0),
        "results": {
            d: {
                c: {
                    "mae_mrad":        float(r["mae"] * 1000),
                    "peak_err_rad":    float(r.get("peak_err", np.nan)),
                    "recovery_time_s": r.get("recovery_time"),
                    "mae_post_mrad":   float(r.get("mae_post", np.nan) or np.nan) * 1000,
                }
                for c, r in cond_res.items()
            }
            for d, cond_res in results.items()
        },
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)

    # ── プロット ─────────────────────────────────────────────────────────
    cond_names = ["E3-PD", "E3-cc", "E3-ia_ib", "E3-cc+ia_ib"]
    colors     = ["tab:gray", "tab:blue", "tab:red", "tab:purple"]
    fig, axes  = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Experiment E3: Ia/Ib Reflex Arc  [seed={seed}  Ia latency={latency_ms:.0f}ms]",
        fontsize=12,
    )

    for col, d_name in enumerate(D_CONDITIONS):
        for row, metric in enumerate(["mae", "peak_err"]):
            ax = axes[row][col]
            vals = [results[d_name].get(c, {}).get(metric, np.nan) for c in cond_names]
            scale = 1000.0 if metric == "mae" else 1.0
            vals  = [v * scale if not np.isnan(v) else np.nan for v in vals]
            unit  = "mrad" if metric == "mae" else "rad"
            label = f"Static MAE [{unit}]" if metric == "mae" else f"Peak error [{unit}]"

            bars = ax.bar(range(len(cond_names)), vals, color=colors, alpha=0.8)
            ax.set_xticks(range(len(cond_names)))
            ax.set_xticklabels(cond_names, fontsize=7, rotation=15, ha="right")
            ax.set_ylabel(label)
            ax.set_title(f"{d_name}: {label}")
            valid = [v for v in vals if not np.isnan(v)]
            vmax  = max(valid) if valid else 1.0
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + vmax * 0.02,
                            f"{v:.1f}", ha="center", va="bottom", fontsize=7)
            ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = outdir / "plot_franka_e3.png"
    plt.savefig(str(plot_path), dpi=150)
    plt.close()
    print(f"\nプロット保存: {plot_path}")

    print("\n=== E3 結果サマリ ===")
    for d_name in D_CONDITIONS:
        print(f"\n  {d_name}:")
        for cname in cond_names:
            r = results[d_name][cname]
            rt = r.get("recovery_time")
            rt_s = f"{rt:.3f}s" if rt else "未回復"
            print(f"    {cname:14s}  MAE={r['mae']*1000:7.1f} mrad"
                  f"  Peak={r.get('peak_err', np.nan):.4f} rad  RT={rt_s}")


if __name__ == "__main__":
    main()
