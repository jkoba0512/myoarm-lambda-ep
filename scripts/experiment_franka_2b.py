"""
実験 2-B (Franka Panda): 外乱耐性評価（反射弓の効果）

生物学的根拠:
  静止保持中に外乱が加わった場合、脊髄反射弓（Ia求心性線維→抑制性介在ニューロン）が
  20ms 以内に高速補正する。CPG は休止したまま。

比較条件:
  1. PD + 小脳 CfC（反射弓なし）
  2. PD + 小脳 CfC + 反射弓（Full 提案手法）

外乱: J2 に外力トルク（軽/中/重 の 3 段階）

出力:
  results/experiment_franka_2b/
    ├── metrics.npz
    └── plot_franka_2b.png
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
import torch as _torch

from common.franka_env import FrankaEnv, N_JOINTS
from common.franka_neural_controller import FrankaNeuralController

RESULTS_DIR = ROOT / "results" / "experiment_franka_2b"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE       = "cuda" if _torch.cuda.is_available() else "cpu"
SIM_DURATION = 6.0
Q_OFFSET     = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.5, 0.0])

# CPG オフ（静止保持タスク）
CPG_PARAMS = dict(tau=0.3, tau_r=0.6, beta=2.5, w=2.0, amplitude=0.0)

DISTURBANCE_LEVELS = {
    "light_30Nm":  (30.0, 10),
    "medium_60Nm": (60.0, 20),
    "heavy_87Nm":  (87.0, 40),
}

print(f"使用デバイス: {DEVICE}")


# ──────────────────────────────────────────────────────────────────────
def run_episode(
    controller:  FrankaNeuralController,
    env:         FrankaEnv,
    dist_torque: float,
    dist_steps:  int,
    dist_t:      float,
    dist_joint:  int,
    q_init:      np.ndarray | None = None,
) -> dict:
    q0 = (q_init if q_init is not None else Q_OFFSET).copy()
    q, dq = env.reset(q0=q0)
    controller.reset()

    t_log, q_log, tau_sys_log, reflex_log = [], [], [], []
    dist_applied = False

    while env.time < SIM_DURATION:
        t = env.time
        q, dq = env.get_state()

        if not dist_applied and t >= dist_t:
            tau_dist = np.zeros(N_JOINTS)
            tau_dist[dist_joint] = dist_torque
            env.apply_disturbance(tau_dist, duration_steps=dist_steps)
            dist_applied = True
            continue

        tau_total, info = controller.step(q, dq, Q_OFFSET)
        env.step(tau_total)

        t_log.append(t)
        q_log.append(q.copy())
        if info["tau_sys"] is not None:
            tau_sys_log.append(info["tau_sys"])
        reflex_log.append(info["tau_reflex"].copy())

    t_arr    = np.array(t_log)
    q_arr    = np.array(q_log)
    err_arr  = np.abs(q_arr - Q_OFFSET)
    post_mask = t_arr > dist_t

    recovery_time = None
    if post_mask.any():
        post_err = err_arr[post_mask, dist_joint]
        post_t   = t_arr[post_mask]
        rec_idx  = np.where(post_err < 0.1)[0]
        if len(rec_idx):
            recovery_time = post_t[rec_idx[0]] - dist_t

    return {
        "t":             t_arr,
        "q":             q_arr,
        "err":           err_arr,
        "tau_sys":       np.array(tau_sys_log) if tau_sys_log else None,
        "reflex":        np.array(reflex_log),
        "peak_err":      err_arr[post_mask, dist_joint].max() if post_mask.any() else np.nan,
        "mae_post":      err_arr[post_mask].mean() if post_mask.any() else np.nan,
        "recovery_time": recovery_time,
    }


# ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42, help="乱数シード")
    p.add_argument("--disturbance-set", choices=["default", "custom"], default="default",
                   help="default は light/medium/heavy を実行。custom は単一条件だけ実行。")
    p.add_argument("--disturbance-name", type=str, default="custom",
                   help="custom 条件名")
    p.add_argument("--disturbance-torque", type=float, default=60.0,
                   help="custom 外乱トルク [Nm]")
    p.add_argument("--disturbance-steps", type=int, default=20,
                   help="custom 外乱印加ステップ数")
    p.add_argument("--disturbance-time", type=float, default=3.0,
                   help="外乱印加時刻 [s]")
    p.add_argument("--disturbance-joint", type=int, default=1,
                   help="外乱を入れる関節インデックス")
    p.add_argument("--q-init-noise", type=float, default=0.0,
                   help="初期関節角ガウスノイズ標準偏差 [rad]")
    p.add_argument("--sweep-name", type=str, default="default",
                   help="default 以外では results/experiment_franka_2b/<sweep-name>/seed*/ に保存")
    return p.parse_args()


def main():
    args   = parse_args()
    seed   = args.seed
    base_dir = RESULTS_DIR if args.sweep_name == "default" else RESULTS_DIR / args.sweep_name
    outdir = base_dir / f"seed{seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    cfc_path = ROOT / "results" / "experiment_franka_2a" / f"seed{seed}" / "cfc_cerebellum.pt"
    if not cfc_path.exists():
        print(f"ERROR: {cfc_path} が見つかりません。先に experiment_franka_2a.py --seed {seed} を実行してください。")
        return

    print(f"seed={seed}  out={outdir}")

    rng = np.random.default_rng(seed)
    q_init = Q_OFFSET.copy()
    if args.q_init_noise > 0.0:
        q_init = q_init + rng.normal(0.0, args.q_init_noise, size=q_init.shape)

    env     = FrankaEnv()
    q_range = env.ctrl_range

    def make_ctrl(use_reflex: bool) -> FrankaNeuralController:
        ctrl = FrankaNeuralController(
            dt=env.dt, q_range=q_range,
            cpg_params=CPG_PARAMS,
            use_proprioceptor=False,
            use_reflex=use_reflex,
            use_cerebellum=True,
            cpg_alpha_fb=0.0,
            device=DEVICE,
        )
        ctrl.load_cerebellum(str(cfc_path))
        return ctrl

    conditions = [("PD+CfC", False), ("Full", True)]
    results    = {}
    if args.disturbance_set == "default":
        disturbance_levels = DISTURBANCE_LEVELS
    else:
        disturbance_levels = {
            args.disturbance_name: (args.disturbance_torque, args.disturbance_steps)
        }
    print("=== 実験 2-B (Franka): 外乱耐性評価（静止保持） ===")

    for dist_label, (dist_torque, dist_steps) in disturbance_levels.items():
        print(f"\n  外乱強度: {dist_label}")
        results[dist_label] = {}

        for label, use_reflex in conditions:
            ctrl = make_ctrl(use_reflex)
            log  = run_episode(
                ctrl, env, dist_torque, dist_steps,
                dist_t=args.disturbance_time,
                dist_joint=args.disturbance_joint,
                q_init=q_init,
            )
            results[dist_label][label] = log
            rt     = log["recovery_time"]
            rt_str = f"{rt:.3f} s" if rt else "未回復"
            print(f"    {label:8s}  ピーク誤差: {log['peak_err']:.4f} rad  "
                  f"外乱後MAE: {log['mae_post']:.4f} rad  回復時間: {rt_str}")

    # ── 標準化 JSON 保存 ──────────────────────────────────────────
    summary: dict = {
        "experiment": "2b",
        "seed": seed,
        "sweep_name": args.sweep_name,
        "disturbance_time_s": args.disturbance_time,
        "disturbance_joint": args.disturbance_joint,
        "q_init_noise": args.q_init_noise,
        "disturbance_levels": {},
    }
    for dist_label, dlogs in results.items():
        summary["disturbance_levels"][dist_label] = {}
        for label, log in dlogs.items():
            rt = log["recovery_time"]
            summary["disturbance_levels"][dist_label][label] = {
                "peak_err_rad":    float(log["peak_err"]),
                "mae_post_mrad":   float(log["mae_post"] * 1000),
                "recovery_time_s": float(rt) if rt is not None else None,
            }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── 保存（npz） ───────────────────────────────────────────────
    save_dict = {}
    for dl, dlogs in results.items():
        for label, log in dlogs.items():
            for k, v in log.items():
                if v is not None and isinstance(v, np.ndarray):
                    save_dict[f"{dl}_{label}_{k}"] = v
    np.savez(str(outdir / "metrics.npz"), **save_dict)

    # ── プロット ──────────────────────────────────────────────────
    colors    = {"PD+CfC": "tab:orange", "Full": "tab:green"}
    n_levels  = len(disturbance_levels)
    fig, axes = plt.subplots(n_levels, 2, figsize=(12, 4 * n_levels))
    if n_levels == 1:
        axes = np.array([axes])
    fig.suptitle(f"Experiment 2-B (Franka Panda): Disturbance Rejection  [seed={seed}]\n"
                 "Static holding — PD+CfC vs Full (with Izhikevich Reflex Arc)", fontsize=11)

    for row, (dist_label, _) in enumerate(disturbance_levels.items()):
        ax = axes[row, 0]
        for label, log in results[dist_label].items():
            ax.plot(log["t"], log["err"][:, args.disturbance_joint],
                    label=label, color=colors[label])
        ax.axvspan(args.disturbance_time, SIM_DURATION, alpha=0.06, color="red")
        ax.axvline(args.disturbance_time, color="red", lw=1.5, ls="--", alpha=0.8)
        ax.set_ylabel(f"J{args.disturbance_joint + 1} joint error [rad]")
        ax.set_title(f"{dist_label}: J{args.disturbance_joint + 1} error")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time [s]")

        ax = axes[row, 1]
        peaks = [results[dist_label][l]["peak_err"] for l, _ in conditions]
        maes  = [results[dist_label][l]["mae_post"]  for l, _ in conditions]
        x = np.arange(len(conditions))
        b1 = ax.bar(x - 0.2, peaks, 0.35, label="Peak error [rad]",
                    color=[colors[l] for l, _ in conditions], alpha=0.8)
        ax.bar(x + 0.2, maes, 0.35, label="Post-dist MAE [rad]",
               color=[colors[l] for l, _ in conditions], alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([l for l, _ in conditions], fontsize=8)
        ax.set_ylabel("Error [rad]")
        ax.set_title(f"{dist_label}: Error summary")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        for bar in b1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = str(outdir / "plot_franka_2b.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nプロット保存: {path}")


if __name__ == "__main__":
    main()
