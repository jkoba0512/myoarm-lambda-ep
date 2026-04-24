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
CFC_PATH    = ROOT / "results" / "experiment_franka_2a" / "cfc_cerebellum.pt"

DEVICE       = "cuda" if _torch.cuda.is_available() else "cpu"
SIM_DURATION = 6.0
DIST_T       = 3.0
Q_OFFSET     = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.5, 0.0])

# CPG オフ（静止保持タスク）
CPG_PARAMS = dict(tau=0.3, tau_r=0.6, beta=2.5, w=2.0, amplitude=0.0)

DIST_JOINT = 1  # J2（肩ピッチ）
DISTURBANCE_LEVELS = {
    "軽度 (30Nm×10steps)": (30.0, 10),
    "中度 (60Nm×20steps)": (60.0, 20),
    "重度 (87Nm×40steps)": (87.0, 40),
}

print(f"使用デバイス: {DEVICE}")


# ──────────────────────────────────────────────────────────────────────
def run_episode(
    controller:  FrankaNeuralController,
    env:         FrankaEnv,
    dist_torque: float,
    dist_steps:  int,
) -> dict:
    q, dq = env.reset(q0=Q_OFFSET.copy())
    controller.reset()

    t_log, q_log, tau_sys_log, reflex_log = [], [], [], []
    dist_applied = False

    while env.time < SIM_DURATION:
        t = env.time
        q, dq = env.get_state()

        if not dist_applied and t >= DIST_T:
            tau_dist = np.zeros(N_JOINTS)
            tau_dist[DIST_JOINT] = dist_torque
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
    post_mask = t_arr > DIST_T

    recovery_time = None
    if post_mask.any():
        post_err = err_arr[post_mask, DIST_JOINT]
        post_t   = t_arr[post_mask]
        rec_idx  = np.where(post_err < 0.1)[0]
        if len(rec_idx):
            recovery_time = post_t[rec_idx[0]] - DIST_T

    return {
        "t":             t_arr,
        "q":             q_arr,
        "err":           err_arr,
        "tau_sys":       np.array(tau_sys_log) if tau_sys_log else None,
        "reflex":        np.array(reflex_log),
        "peak_err":      err_arr[post_mask, DIST_JOINT].max() if post_mask.any() else np.nan,
        "mae_post":      err_arr[post_mask].mean() if post_mask.any() else np.nan,
        "recovery_time": recovery_time,
    }


# ──────────────────────────────────────────────────────────────────────
def main():
    env     = FrankaEnv()
    q_range = env.ctrl_range

    if not CFC_PATH.exists():
        print(f"ERROR: {CFC_PATH} が見つかりません。先に experiment_franka_2a.py を実行してください。")
        return

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
        ctrl.load_cerebellum(str(CFC_PATH))
        return ctrl

    results = {}
    print("=== 実験 2-B (Franka): 外乱耐性評価（静止保持） ===")

    for dist_label, (dist_torque, dist_steps) in DISTURBANCE_LEVELS.items():
        print(f"\n  外乱強度: {dist_label}")
        results[dist_label] = {}

        for label, use_reflex in [("PD+小脳", False), ("Full（提案）", True)]:
            ctrl = make_ctrl(use_reflex)
            log  = run_episode(ctrl, env, dist_torque, dist_steps)
            results[dist_label][label] = log
            rt     = log["recovery_time"]
            rt_str = f"{rt:.3f} s" if rt else "未回復"
            print(f"    {label:12s}  ピーク誤差: {log['peak_err']:.4f} rad  "
                  f"外乱後MAE: {log['mae_post']:.4f} rad  回復時間: {rt_str}")

    # ── 保存 ──────────────────────────────────────────────────────
    save_dict = {}
    for dl, dlogs in results.items():
        key = dl.split("(")[0].strip().replace(" ", "_")
        for label, log in dlogs.items():
            lk = label.replace("（", "").replace("）", "").replace(" ", "_")
            for k, v in log.items():
                if v is not None and isinstance(v, np.ndarray):
                    save_dict[f"{key}_{lk}_{k}"] = v
    np.savez(str(RESULTS_DIR / "metrics.npz"), **save_dict)

    # ── プロット ──────────────────────────────────────────────────
    n_levels = len(DISTURBANCE_LEVELS)
    fig, axes = plt.subplots(n_levels, 2, figsize=(12, 4 * n_levels))
    fig.suptitle("Experiment 2-B (Franka Panda): Disturbance Rejection\n"
                 "Static holding — PD+Cerebellum vs Full (with Izhikevich Reflex Arc)", fontsize=11)

    colors = {"PD+小脳": "tab:orange", "Full（提案）": "tab:green"}

    for row, (dist_label, _) in enumerate(DISTURBANCE_LEVELS.items()):
        ax = axes[row, 0]
        for label, log in results[dist_label].items():
            ax.plot(log["t"], log["err"][:, DIST_JOINT],
                    label=label, color=colors[label])
        ax.axvspan(DIST_T, SIM_DURATION, alpha=0.06, color="red")
        ax.axvline(DIST_T, color="red", lw=1.5, ls="--", alpha=0.8)
        ax.set_ylabel("J2 joint error [rad]")
        ax.set_title(f"{dist_label}: J2 error")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time [s]")

        ax = axes[row, 1]
        labels_bar = list(results[dist_label].keys())
        peaks = [results[dist_label][l]["peak_err"] for l in labels_bar]
        maes  = [results[dist_label][l]["mae_post"]  for l in labels_bar]
        x = np.arange(len(labels_bar))
        b1 = ax.bar(x - 0.2, peaks, 0.35, label="Peak error [rad]",
                    color=[colors[l] for l in labels_bar], alpha=0.8)
        ax.bar(x + 0.2, maes, 0.35, label="Post-dist MAE [rad]",
               color=[colors[l] for l in labels_bar], alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_bar, fontsize=8)
        ax.set_ylabel("Error [rad]")
        ax.set_title(f"{dist_label}: Error summary")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        for bar in b1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = str(RESULTS_DIR / "plot_franka_2b.png")
    plt.savefig(path, dpi=150)
    print(f"\nプロット保存: {path}")


if __name__ == "__main__":
    main()
