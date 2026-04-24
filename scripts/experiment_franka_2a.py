"""
実験 2-A (Franka Panda): コンポーネント別アブレーション（静止保持）

生物学的根拠:
  ヒトが静止位置を保持する際、CPGは休止しトニックな皮質指令が支配する。
  → CPG振幅=0（静止指令）で小脳・反射弓の純粋な寄与を評価する。

比較条件:
  1. PD のみ（ベースライン）
  2. PD + 小脳 CfC（重力・コリオリ補償）
  3. PD + 小脳 CfC + 反射弓（Full 提案手法）

評価: 外乱なし静止保持 6s の MAE [rad]

出力:
  results/experiment_franka_2a/
    ├── cfc_cerebellum.pt
    ├── metrics.npz
    └── plot_franka_2a.png
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

RESULTS_DIR = ROOT / "results" / "experiment_franka_2a"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE       = "cuda" if _torch.cuda.is_available() else "cpu"
SIM_DURATION = 6.0

Q_OFFSET = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.5, 0.0])

N_TRAJECTORIES = 500
SEQ_LEN        = 50
N_EPOCHS       = 300

# CPG オフ（amplitude=0）: 静止保持タスクでは CPG は使わない
CPG_PARAMS = dict(tau=0.3, tau_r=0.6, beta=2.5, w=2.0, amplitude=0.0)

print(f"使用デバイス: {DEVICE}")


# ──────────────────────────────────────────────────────────────────────
def run_episode(controller: FrankaNeuralController, env: FrankaEnv) -> dict:
    q, dq = env.reset(q0=Q_OFFSET.copy())
    controller.reset()

    t_log, q_log, tau_sys_log = [], [], []

    while env.time < SIM_DURATION:
        t = env.time
        q, dq = env.get_state()

        tau_total, info = controller.step(q, dq, Q_OFFSET)
        env.step(tau_total)

        t_log.append(t)
        q_log.append(q.copy())
        if info["tau_sys"] is not None:
            tau_sys_log.append(info["tau_sys"])

    t_arr   = np.array(t_log)
    q_arr   = np.array(q_log)
    err_arr = np.abs(q_arr - Q_OFFSET)

    return {
        "t":       t_arr,
        "q":       q_arr,
        "err":     err_arr,
        "mae":     err_arr.mean(),
        "tau_sys": np.array(tau_sys_log) if tau_sys_log else None,
    }


# ──────────────────────────────────────────────────────────────────────
SEED = 42


def main():
    # 再現性確保: モデル初期化・DataLoader shuffle を固定
    _torch.manual_seed(SEED)
    np.random.seed(SEED)

    env     = FrankaEnv()
    q_range = env.ctrl_range

    # ── 小脳 CfC を事前訓練 ───────────────────────────────────────
    print("=== 小脳 CfC 訓練 ===")
    ctrl_train = FrankaNeuralController(
        dt=env.dt, q_range=q_range,
        cpg_params=CPG_PARAMS,
        use_proprioceptor=False,
        use_reflex=False,
        use_cerebellum=True,
        cfc_hidden_units=64,
        device=DEVICE,
    )
    loss_hist = ctrl_train.train_cerebellum(
        env,
        n_trajectories=N_TRAJECTORIES,
        seq_len=SEQ_LEN,
        n_epochs=N_EPOCHS,
        verbose=True,
    )
    cfc_path = str(RESULTS_DIR / "cfc_cerebellum.pt")
    ctrl_train.save_cerebellum(cfc_path)
    print(f"小脳モデル保存: {cfc_path}")

    # ── アブレーション評価 ────────────────────────────────────────
    # CPG オフのため固有受容器は CPG への FB パスが無効 → 除外
    conditions = [
        ("PD のみ",           dict(use_proprioceptor=False, use_reflex=False, use_cerebellum=False)),
        ("PD + 小脳",         dict(use_proprioceptor=False, use_reflex=False, use_cerebellum=True)),
        ("PD + 小脳 + 反射弓", dict(use_proprioceptor=False, use_reflex=True,  use_cerebellum=True)),
    ]

    results = {}
    print("\n=== アブレーション評価（静止保持） ===")
    for label, flags in conditions:
        ctrl = FrankaNeuralController(
            dt=env.dt, q_range=q_range,
            cpg_params=CPG_PARAMS,
            cpg_alpha_fb=0.0,
            device=DEVICE,
            **flags,
        )
        if flags["use_cerebellum"]:
            ctrl.load_cerebellum(cfc_path)

        log = run_episode(ctrl, env)
        results[label] = log
        print(f"  {label:22s}  MAE: {log['mae']*1000:.2f} mrad")

    # ── 保存 ──────────────────────────────────────────────────────
    np.savez(
        str(RESULTS_DIR / "metrics.npz"),
        loss_hist=np.array(loss_hist),
        **{f"{label.replace(' ', '_')}_mae": results[label]["mae"]  for label, _ in conditions},
        **{f"{label.replace(' ', '_')}_err": results[label]["err"]  for label, _ in conditions},
    )

    # ── プロット ──────────────────────────────────────────────────
    colors = ["tab:gray", "tab:blue", "tab:green"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Experiment 2-A (Franka Panda): Ablation Study\n"
                 "Static holding task — CPG disabled (biologically accurate)", fontsize=11)

    ax = axes[0]
    labels = [l for l, _ in conditions]
    maes   = [results[l]["mae"] * 1000 for l in labels]
    bars   = ax.bar(range(len(labels)), maes, color=colors, alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=12, ha="right")
    ax.set_ylabel("MAE [mrad]")
    ax.set_title("Mean Absolute Error")
    for bar, v in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    for (label, _), color in zip(conditions, colors):
        log = results[label]
        ax.plot(log["t"], log["err"].mean(axis=1) * 1000, label=label, color=color)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mean joint error [mrad]")
    ax.set_title("Joint error over time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.semilogy(loss_hist, color="tab:blue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Cerebellum CfC training loss")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = str(RESULTS_DIR / "plot_franka_2a.png")
    plt.savefig(path, dpi=150)
    print(f"\nプロット保存: {path}")

    print("\n=== 結果サマリ ===")
    for label, _ in conditions:
        print(f"  {label:22s}  MAE: {results[label]['mae']*1000:.2f} mrad")


if __name__ == "__main__":
    main()
