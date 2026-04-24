"""
実験 2-D (Franka Panda): 統合評価（静止保持 + 外乱）

生物学的根拠:
  静止位置保持タスク（CPG 休止）における全コンポーネントの統合効果を評価。
  - 小脳（CfC）: 重力・コリオリ補償 → 定常追従精度の向上
  - 反射弓（Izhikevich）: 外乱時の高速トルク補正 → ピーク誤差・回復時間の短縮
  - τ_sys 可視化: CfC の適応レートが外乱時に増加する（神経調節の仮説検証）

比較条件:
  1. PD のみ（ベースライン）
  2. PD + 小脳 CfC
  3. PD + 小脳 CfC + 反射弓（Full 提案手法）

評価:
  - 外乱なし: MAE [rad]
  - 外乱あり（中度: 60Nm×20steps）: ピーク誤差・回復時間
  - τ_sys 時系列可視化
  - 反射弓トルク出力可視化

出力:
  results/experiment_franka_2d/
    ├── metrics.npz
    └── plot_franka_2d.png
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

RESULTS_DIR = ROOT / "results" / "experiment_franka_2d"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CFC_PATH    = ROOT / "results" / "experiment_franka_2a" / "cfc_cerebellum.pt"

DEVICE       = "cuda" if _torch.cuda.is_available() else "cpu"
SIM_DURATION = 8.0
DIST_T       = 4.0
DIST_TORQUE  = 60.0
DIST_STEPS   = 20
DIST_JOINT   = 1
Q_OFFSET     = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.5, 0.0])

# CPG オフ（静止保持タスク）
CPG_PARAMS = dict(tau=0.3, tau_r=0.6, beta=2.5, w=2.0, amplitude=0.0)

JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "J7"]

print(f"使用デバイス: {DEVICE}")


# ──────────────────────────────────────────────────────────────────────
def run_episode(
    controller: FrankaNeuralController,
    env:        FrankaEnv,
    with_dist:  bool = True,
) -> dict:
    q, dq = env.reset(q0=Q_OFFSET.copy())
    controller.reset()

    t_log, q_log = [], []
    tau_sys_log, reflex_log, reflex_active_log = [], [], []
    dist_applied = False

    while env.time < SIM_DURATION:
        t = env.time
        q, dq = env.get_state()

        if with_dist and not dist_applied and t >= DIST_T:
            tau_dist = np.zeros(N_JOINTS)
            tau_dist[DIST_JOINT] = DIST_TORQUE
            env.apply_disturbance(tau_dist, duration_steps=DIST_STEPS)
            dist_applied = True
            continue

        tau_total, info = controller.step(q, dq, Q_OFFSET)
        env.step(tau_total)

        t_log.append(t)
        q_log.append(q.copy())
        reflex_log.append(info["tau_reflex"].copy())
        reflex_active_log.append(info["reflex_active"].copy())
        if info["tau_sys"] is not None:
            tau_sys_log.append(info["tau_sys"])

    t_arr     = np.array(t_log)
    q_arr     = np.array(q_log)
    err_arr   = np.abs(q_arr - Q_OFFSET)
    post_mask = t_arr > DIST_T

    recovery_time = None
    if with_dist and post_mask.any():
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
        "reflex_active": np.array(reflex_active_log),
        "mae":           err_arr.mean(),
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

    def make_ctrl(label: str) -> FrankaNeuralController:
        if label == "PD のみ":
            return FrankaNeuralController(
                dt=env.dt, q_range=q_range,
                cpg_params=CPG_PARAMS,
                use_proprioceptor=False, use_reflex=False, use_cerebellum=False,
                cpg_alpha_fb=0.0, device=DEVICE,
            )
        elif label == "PD + 小脳":
            ctrl = FrankaNeuralController(
                dt=env.dt, q_range=q_range,
                cpg_params=CPG_PARAMS,
                use_proprioceptor=False, use_reflex=False, use_cerebellum=True,
                cpg_alpha_fb=0.0, device=DEVICE,
            )
            ctrl.load_cerebellum(str(CFC_PATH))
            return ctrl
        else:  # Full
            ctrl = FrankaNeuralController(
                dt=env.dt, q_range=q_range,
                cpg_params=CPG_PARAMS,
                use_proprioceptor=False, use_reflex=True, use_cerebellum=True,
                cpg_alpha_fb=0.0, device=DEVICE,
            )
            ctrl.load_cerebellum(str(CFC_PATH))
            return ctrl

    labels = ["PD のみ", "PD + 小脳", "Full（提案）"]
    colors = {"PD のみ": "tab:gray", "PD + 小脳": "tab:blue", "Full（提案）": "tab:green"}

    print("=== 実験 2-D (Franka): 統合評価（静止保持） ===")
    results_free = {}
    results_dist = {}

    for label in labels:
        print(f"  {label} 実行中 (外乱なし)...")
        results_free[label] = run_episode(make_ctrl(label), env, with_dist=False)
        print(f"  {label} 実行中 (外乱あり)...")
        results_dist[label] = run_episode(make_ctrl(label), env, with_dist=True)

    print("\n=== 結果サマリ ===")
    for label in labels:
        rf = results_free[label]
        rd = results_dist[label]
        rt = rd["recovery_time"]
        rt_str = f"{rt:.3f} s" if rt else "未回復"
        print(f"  {label:12s}  外乱なしMAE: {rf['mae']*1000:.2f} mrad  "
              f"ピーク誤差: {rd['peak_err']:.4f} rad  回復時間: {rt_str}")

    # ── 保存 ──────────────────────────────────────────────────────
    np.savez(
        str(RESULTS_DIR / "metrics.npz"),
        **{f"free_{label.replace(' ', '_')}_{k}": v
           for label, log in results_free.items()
           for k, v in log.items()
           if v is not None and isinstance(v, np.ndarray)},
        **{f"dist_{label.replace(' ', '_')}_{k}": v
           for label, log in results_dist.items()
           for k, v in log.items()
           if v is not None and isinstance(v, np.ndarray)},
    )

    # ── プロット ──────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    fig.suptitle("Experiment 2-D (Franka Panda): Integrated Evaluation\n"
                 "Static holding — PD vs PD+Cerebellum vs Full Neural System", fontsize=11)

    def shade(ax):
        ax.axvspan(DIST_T, SIM_DURATION, alpha=0.06, color="red")
        ax.axvline(DIST_T, color="red", lw=1.5, ls="--", alpha=0.8)

    # 上左: 外乱なし 平均関節誤差
    ax = axes[0, 0]
    for label in labels:
        ax.plot(results_free[label]["t"],
                results_free[label]["err"].mean(axis=1) * 1000,
                label=label, color=colors[label])
    ax.set_ylabel("Mean joint error [mrad]")
    ax.set_title("No disturbance: joint error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time [s]")

    # 上右: 外乱あり J2 誤差
    ax = axes[0, 1]
    for label in labels:
        ax.plot(results_dist[label]["t"],
                results_dist[label]["err"][:, DIST_JOINT],
                label=label, color=colors[label])
    shade(ax)
    ax.set_ylabel("J2 joint error [rad]")
    ax.set_title(f"Disturbance ({DIST_TORQUE:.0f}Nm×{DIST_STEPS}steps): J2 error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time [s]")

    # 中左: τ_sys（Full のみ、外乱あり）
    ax = axes[1, 0]
    log_full = results_dist["Full（提案）"]
    if log_full["tau_sys"] is not None:
        t_ts = log_full["t"][:len(log_full["tau_sys"])]
        ax.plot(t_ts, log_full["tau_sys"], color="purple", lw=1.2)
        shade(ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("t_interp (τ_sys)")
        ax.set_title("Full: CfC τ_sys dynamics\n"
                     "↑ disturbance → faster adaptation")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time [s]")

    # 中右: 反射弓トルク（Full のみ、J1-J4）
    ax = axes[1, 1]
    for i in range(4):
        ax.plot(log_full["t"], log_full["reflex"][:, i],
                label=JOINT_NAMES[i], alpha=0.8)
    shade(ax)
    ax.set_ylabel("Reflex torque [Nm]")
    ax.set_title("Full: Izhikevich reflex arc output (J1-J4)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time [s]")

    # 下左: ピーク誤差・回復時間
    ax = axes[2, 0]
    peaks = [results_dist[l]["peak_err"] for l in labels]
    maes  = [results_dist[l]["mae_post"]  for l in labels]
    x = np.arange(len(labels))
    b1 = ax.bar(x - 0.2, peaks, 0.35, label="Peak error [rad]",
                color=[colors[l] for l in labels], alpha=0.8)
    ax.bar(x + 0.2, maes, 0.35, label="Post-dist MAE [rad]",
           color=[colors[l] for l in labels], alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=10, ha="right")
    ax.set_ylabel("Error [rad]")
    ax.set_title("Post-disturbance error comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    # 下右: MAE サマリ（外乱なし）
    ax = axes[2, 1]
    free_maes = [results_free[l]["mae"] * 1000 for l in labels]
    bars = ax.bar(range(len(labels)), free_maes,
                  color=[colors[l] for l in labels], alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=10, ha="right")
    ax.set_ylabel("MAE [mrad]")
    ax.set_title("No-disturbance MAE summary")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, free_maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = str(RESULTS_DIR / "plot_franka_2d.png")
    plt.savefig(path, dpi=150)
    print(f"\nプロット保存: {path}")


if __name__ == "__main__":
    main()
