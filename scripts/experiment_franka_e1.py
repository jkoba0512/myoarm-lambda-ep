"""
実験 E1 (Franka Panda): 小脳 Forward model の評価

目的:
  旧 CfCGravityCompensator（逆動力学）を CfCForwardModel（順動力学）に
  置き換えることで、制御性能と適応速度が改善するかを検証する。

比較条件:
  E1-A0   : 旧 CfC（逆動力学, ts=1.0 固定）         ← Phase A0 baseline
  E1-MLP  : MLP 補償器（逆動力学）                   ← Phase A3 baseline
  E1-fwd  : CfC Forward model（順動力学, 補正ゲイン） ← 新実装
  E1-fwd-online: E1-fwd + オンライン重み更新

評価:
  静止保持 MAE [mrad]
  外乱後ピーク誤差・回復時間
  予測誤差 RMSE [mrad]（Forward model 固有）
  外乱後の予測誤差収束速度（適応速度）

出力:
  results/experiment_franka_e1/seed<N>/
    ├── cfc_forward.pt
    ├── cfc_inverse.pt    (E1-A0 用, Phase A から流用可)
    ├── mlp.pt            (E1-MLP 用, Phase A から流用可)
    ├── metrics.json
    └── plot_franka_e1.png
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
from methodB.cfc_compensator import CfCGravityCompensator, MLPCompensator
from methodB.cfc_forward_model import CfCForwardModel

RESULTS_DIR = ROOT / "results" / "experiment_franka_e1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE       = "cuda" if _torch.cuda.is_available() else "cpu"
SIM_DURATION = 6.0
Q_OFFSET     = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.5, 0.0])
CPG_PARAMS   = dict(tau=0.3, tau_r=0.6, beta=2.5, w=2.0, amplitude=0.0)

N_TRAJECTORIES = 500
SEQ_LEN        = 50
N_EPOCHS       = 300

DIST_TORQUE = 60.0   # [Nm]
DIST_STEPS  = 20
DIST_T      = 3.0
DIST_JOINT  = 1

print(f"使用デバイス: {DEVICE}")


# ──────────────────────────────────────────────────────────────────────
def run_episode_static(
    controller: FrankaNeuralController,
    env: FrankaEnv,
    use_fwd: bool = False,
) -> dict:
    """静止保持タスク（2A 相当）。"""
    q, dq = env.reset(q0=Q_OFFSET.copy())
    controller.reset()

    t_log, q_log, pred_err_log = [], [], []

    while env.time < SIM_DURATION:
        t = env.time
        q, dq = env.get_state()

        tau_total, info = controller.step(q, dq, Q_OFFSET)
        env.step(tau_total)

        if use_fwd:
            q_next, _ = env.get_state()
            controller.update_cerebellum(q_next)

        t_log.append(t)
        q_log.append(q.copy())
        if info["pred_error"] is not None:
            pred_err_log.append(np.abs(info["pred_error"]).mean() * 1000)

    t_arr   = np.array(t_log)
    q_arr   = np.array(q_log)
    err_arr = np.abs(q_arr - Q_OFFSET)

    return {
        "t":        t_arr,
        "q":        q_arr,
        "err":      err_arr,
        "mae":      err_arr.mean(),
        "pred_err": np.array(pred_err_log) if pred_err_log else None,
    }


def run_episode_disturb(
    controller: FrankaNeuralController,
    env: FrankaEnv,
    use_fwd: bool = False,
) -> dict:
    """外乱耐性タスク（2B 相当）。"""
    q, dq = env.reset(q0=Q_OFFSET.copy())
    controller.reset()

    t_log, q_log, pred_err_log = [], [], []
    dist_applied = False

    while env.time < SIM_DURATION:
        t = env.time
        q, dq = env.get_state()

        if not dist_applied and t >= DIST_T:
            tau_dist = np.zeros(N_JOINTS)
            tau_dist[DIST_JOINT] = DIST_TORQUE
            env.apply_disturbance(tau_dist, duration_steps=DIST_STEPS)
            dist_applied = True
            continue

        tau_total, info = controller.step(q, dq, Q_OFFSET)
        env.step(tau_total)

        if use_fwd:
            q_next, _ = env.get_state()
            controller.update_cerebellum(q_next)

        t_log.append(t)
        q_log.append(q.copy())
        if info["pred_error"] is not None:
            pred_err_log.append(np.abs(info["pred_error"]).mean() * 1000)

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

    # 予測誤差の外乱後収束速度（外乱後 0.5 s 以内に 1/e 未満になるまでの時間）
    adapt_time = None
    if pred_err_log:
        pred_arr  = np.array(pred_err_log)
        post_pidx = np.where(t_arr > DIST_T)[0]
        if len(post_pidx):
            base   = pred_arr[post_pidx[0]]
            thresh = base / np.e
            done   = np.where(pred_arr[post_pidx] < thresh)[0]
            if len(done):
                adapt_time = t_arr[post_pidx[done[0]]] - DIST_T

    return {
        "t":            t_arr,
        "q":            q_arr,
        "err":          err_arr,
        "peak_err":     err_arr[post_mask, DIST_JOINT].max() if post_mask.any() else np.nan,
        "mae_post":     err_arr[post_mask].mean() if post_mask.any() else np.nan,
        "recovery_time": recovery_time,
        "pred_err":     np.array(pred_err_log) if pred_err_log else None,
        "adapt_time":   adapt_time,
    }


# ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--sweep-name", type=str,  default="default")
    p.add_argument("--n-epochs",  type=int,   default=N_EPOCHS,
                   help="訓練エポック数（スモークテスト時は短縮）")
    p.add_argument("--n-traj",    type=int,   default=N_TRAJECTORIES,
                   help="訓練軌道数")
    p.add_argument("--online-lr", type=float, default=1e-4,
                   help="Forward model オンライン学習率（0=無効）")
    return p.parse_args()


def main():
    args   = parse_args()
    seed   = args.seed
    base_dir = RESULTS_DIR if args.sweep_name == "default" \
               else RESULTS_DIR / args.sweep_name
    outdir = base_dir / f"seed{seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    _torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"seed={seed}  out={outdir}")

    env     = FrankaEnv(rng=np.random.default_rng(seed + 1000))
    q_range = env.ctrl_range

    # ── 旧 CfC (E1-A0) ───────────────────────────────────────────
    inv_path = outdir / "cfc_inverse.pt"
    phase_a_path = (ROOT / "results" / "experiment_franka_2a"
                    / f"seed{seed}" / "cfc_cerebellum.pt")

    if phase_a_path.exists():
        print(f"[E1-A0] Phase A モデルを流用: {phase_a_path}")
        inv_path_use = str(phase_a_path)
    else:
        print("[E1-A0] 旧 CfC を新規訓練")
        ctrl_inv = FrankaNeuralController(
            dt=env.dt, q_range=q_range, cpg_params=CPG_PARAMS,
            use_proprioceptor=False, use_reflex=False,
            use_cerebellum=True, use_forward_model=False,
            cfc_hidden_units=64, device=DEVICE,
        )
        _torch.manual_seed(seed); np.random.seed(seed)
        ctrl_inv.train_cerebellum(env, n_trajectories=args.n_traj,
                                  seq_len=SEQ_LEN, n_epochs=args.n_epochs,
                                  verbose=True)
        ctrl_inv.save_cerebellum(str(inv_path))
        inv_path_use = str(inv_path)

    # ── MLP (E1-MLP) ────────────────────────────────────────────
    mlp_path = outdir / "mlp.pt"
    phase_a_mlp = (ROOT / "results" / "experiment_franka_2a"
                   / "condition_sweep_phase_a" / "A3_mlp"
                   / f"seed{seed}" / "cfc_cerebellum.pt")

    if phase_a_mlp.exists():
        print(f"[E1-MLP] Phase A MLP を流用: {phase_a_mlp}")
        mlp_path_use = str(phase_a_mlp)
        mlp_obj = MLPCompensator(n_joints=N_JOINTS, hidden_units=64, device=DEVICE)
        mlp_obj.load(mlp_path_use)
    else:
        print("[E1-MLP] MLP を新規訓練")
        mlp_obj = MLPCompensator(n_joints=N_JOINTS, hidden_units=64, device=DEVICE)
        _torch.manual_seed(seed); np.random.seed(seed)
        rng = np.random.default_rng(seed)
        q_seqs, dq_seqs, tau_seqs = CfCGravityCompensator.collect_sequence_data(
            env, n_trajectories=args.n_traj, seq_len=SEQ_LEN, rng=rng)
        mlp_obj.fit(q_seqs, dq_seqs, tau_seqs,
                    n_epochs=args.n_epochs, verbose=True, seed=seed)
        mlp_obj.save(str(mlp_path))
        mlp_path_use = str(mlp_path)

    # ── CfC Forward model (E1-fwd) ─────────────────────────────
    fwd_path = outdir / "cfc_forward.pt"
    print("\n=== CfC Forward model 訓練 ===")
    ctrl_fwd_train = FrankaNeuralController(
        dt=env.dt, q_range=q_range, cpg_params=CPG_PARAMS,
        use_proprioceptor=False, use_reflex=False,
        use_cerebellum=True, use_forward_model=True,
        cfc_hidden_units=64, device=DEVICE,
    )
    _torch.manual_seed(seed); np.random.seed(seed)
    loss_fwd = ctrl_fwd_train.train_cerebellum_forward(
        env, n_trajectories=args.n_traj, seq_len=SEQ_LEN,
        n_epochs=args.n_epochs, verbose=True, seed=seed,
    )
    ctrl_fwd_train.save_cerebellum(str(fwd_path))
    print(f"Forward model 保存: {fwd_path}")

    # ── 評価 ─────────────────────────────────────────────────────
    def make_inv_ctrl(cerebellum_obj=None) -> FrankaNeuralController:
        c = FrankaNeuralController(
            dt=env.dt, q_range=q_range, cpg_params=CPG_PARAMS,
            use_proprioceptor=False, use_reflex=False,
            use_cerebellum=True, use_forward_model=False,
            cpg_alpha_fb=0.0, device=DEVICE,
        )
        if cerebellum_obj is not None:
            c.cerebellum = cerebellum_obj
            c.cerebellum.reset()
        else:
            c.load_cerebellum(inv_path_use)
        return c

    def make_fwd_ctrl(online_lr: float = 0.0) -> FrankaNeuralController:
        c = FrankaNeuralController(
            dt=env.dt, q_range=q_range, cpg_params=CPG_PARAMS,
            use_proprioceptor=False, use_reflex=False,
            use_cerebellum=True, use_forward_model=True,
            cpg_alpha_fb=0.0, cfc_hidden_units=64,
            online_lr=online_lr, device=DEVICE,
        )
        c.load_cerebellum(str(fwd_path))
        return c

    conditions = [
        ("E1-A0",         make_inv_ctrl,        False),
        ("E1-MLP",        lambda: make_inv_ctrl(mlp_obj), False),
        ("E1-fwd",        lambda: make_fwd_ctrl(0.0),     True),
        ("E1-fwd-online", lambda: make_fwd_ctrl(args.online_lr), True),
    ]

    print("\n=== 静止保持評価 ===")
    static_res = {}
    for label, factory, use_fwd in conditions:
        ctrl = factory()
        log  = run_episode_static(ctrl, env, use_fwd=use_fwd)
        static_res[label] = log
        pe_str = (f"  PredErr={log['pred_err'].mean():.2f} mrad"
                  if log["pred_err"] is not None else "")
        print(f"  {label:18s}  MAE={log['mae']*1000:.2f} mrad{pe_str}")

    print("\n=== 外乱耐性評価 ===")
    dist_res = {}
    for label, factory, use_fwd in conditions:
        ctrl = factory()
        log  = run_episode_disturb(ctrl, env, use_fwd=use_fwd)
        dist_res[label] = log
        rt_str = f"{log['recovery_time']:.3f} s" if log["recovery_time"] else "未回復"
        at_str = (f"  AdaptT={log['adapt_time']:.3f} s"
                  if log["adapt_time"] is not None else "")
        print(f"  {label:18s}  Peak={log['peak_err']:.4f} rad"
              f"  RT={rt_str}{at_str}")

    # ── JSON 保存 ────────────────────────────────────────────────
    summary = {
        "experiment": "e1",
        "seed": seed,
        "sweep_name": args.sweep_name,
        "static": {
            label: {
                "mae_mrad": float(static_res[label]["mae"] * 1000),
                "pred_err_mean_mrad": (
                    float(static_res[label]["pred_err"].mean())
                    if static_res[label]["pred_err"] is not None else None
                ),
            }
            for label, _, _ in conditions
        },
        "disturbance": {
            label: {
                "peak_err_rad":    float(dist_res[label]["peak_err"]),
                "mae_post_mrad":   float(dist_res[label]["mae_post"] * 1000),
                "recovery_time_s": (float(dist_res[label]["recovery_time"])
                                    if dist_res[label]["recovery_time"] is not None
                                    else None),
                "adapt_time_s":    (float(dist_res[label]["adapt_time"])
                                    if dist_res[label]["adapt_time"] is not None
                                    else None),
            }
            for label, _, _ in conditions
        },
        "training": {
            "n_epochs":     args.n_epochs,
            "n_traj":       args.n_traj,
            "fwd_final_loss": float(loss_fwd[-1]),
        },
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── プロット ─────────────────────────────────────────────────
    colors = {
        "E1-A0":         "tab:gray",
        "E1-MLP":        "tab:orange",
        "E1-fwd":        "tab:blue",
        "E1-fwd-online": "tab:green",
    }
    labels_plot = [l for l, _, _ in conditions]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Experiment E1: CfC Forward model vs Baselines  [seed={seed}]\n"
        "Forward model: (q, dq, τ_efference) → Δq_hat  |  "
        "Correction: τ_cereb = K·(Δq_actual − Δq_hat)",
        fontsize=11,
    )

    # 静止保持 MAE 棒グラフ
    ax = axes[0, 0]
    maes = [static_res[l]["mae"] * 1000 for l in labels_plot]
    bars = ax.bar(range(len(labels_plot)), maes,
                  color=[colors[l] for l in labels_plot], alpha=0.8)
    ax.set_xticks(range(len(labels_plot)))
    ax.set_xticklabels(labels_plot, fontsize=8, rotation=12, ha="right")
    ax.set_ylabel("MAE [mrad]")
    ax.set_title("Static Holding MAE")
    for bar, v in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # 静止保持 誤差時系列
    ax = axes[0, 1]
    for label, _, _ in conditions:
        log = static_res[label]
        ax.plot(log["t"], log["err"].mean(axis=1) * 1000,
                label=label, color=colors[label])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mean joint error [mrad]")
    ax.set_title("Static Holding: error over time")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 予測誤差時系列（Forward model のみ）
    ax = axes[0, 2]
    for label, _, use_fwd in conditions:
        if not use_fwd:
            continue
        pe = static_res[label]["pred_err"]
        if pe is not None:
            ax.plot(static_res[label]["t"][:len(pe)], pe,
                    label=label, color=colors[label])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Prediction error [mrad]")
    ax.set_title("Forward model: prediction error (static)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 外乱耐性 誤差時系列
    ax = axes[1, 0]
    for label, _, _ in conditions:
        log = dist_res[label]
        ax.plot(log["t"], log["err"][:, DIST_JOINT],
                label=label, color=colors[label])
    ax.axvline(DIST_T, color="red", lw=1.5, ls="--", alpha=0.8)
    ax.axvspan(DIST_T, SIM_DURATION, alpha=0.04, color="red")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"J{DIST_JOINT+1} error [rad]")
    ax.set_title(f"Disturbance Rejection (J{DIST_JOINT+1}, {DIST_TORQUE} Nm)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 外乱耐性 ピーク誤差棒グラフ
    ax = axes[1, 1]
    peaks = [dist_res[l]["peak_err"] for l in labels_plot]
    bars  = ax.bar(range(len(labels_plot)), peaks,
                   color=[colors[l] for l in labels_plot], alpha=0.8)
    ax.set_xticks(range(len(labels_plot)))
    ax.set_xticklabels(labels_plot, fontsize=8, rotation=12, ha="right")
    ax.set_ylabel("Peak error [rad]")
    ax.set_title("Peak Error after Disturbance")
    for bar, v in zip(bars, peaks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Forward model 訓練損失
    ax = axes[1, 2]
    ax.semilogy(loss_fwd, color="tab:blue", label="CfC Forward model")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Forward model training loss")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = str(outdir / "plot_franka_e1.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nプロット保存: {plot_path}")

    print("\n=== 結果サマリ ===")
    print(f"{'条件':20s}  {'静止MAE':>10s}  {'ピーク誤差':>12s}  {'回復時間':>10s}")
    for label, _, _ in conditions:
        mae  = static_res[label]["mae"] * 1000
        peak = dist_res[label]["peak_err"]
        rt   = dist_res[label]["recovery_time"]
        rt_s = f"{rt:.3f} s" if rt else "未回復"
        print(f"  {label:18s}  {mae:8.2f} mrad  {peak:10.4f} rad  {rt_s:>10s}")


if __name__ == "__main__":
    main()
