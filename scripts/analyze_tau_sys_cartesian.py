"""
分析スクリプト: τ_sys 定量分析 + Cartesian 誤差評価

目的:
  1. τ_sys 定量分析
       CfC 小脳の適応レート t_interp が外乱時に増加するか統計的に検証。
       t_interp = σ(t_a·dt + t_b) ∈ (0,1)
         大: 高速応答（外乱・把持時）/ 小: 緩慢応答（定常時）

  2. Cartesian 誤差評価 [mm]
       関節角誤差 [mrad] に加え、エンドエフェクタ位置誤差を mm 単位で評価。
       参照 EE 位置は FK(Q_OFFSET) から取得。
       実際の EE 位置は env.get_ee_pos() から取得。

設定: 実験 2-D と同じ（静止保持 + 中度外乱 60 Nm × 20 steps on J2）

出力:
  results/analyze_tau_sys_cartesian/
    ├── metrics.npz
    └── plot_tau_sys_cartesian.png
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch as _torch

from common.franka_env import FrankaEnv, N_JOINTS
from common.franka_neural_controller import FrankaNeuralController

RESULTS_DIR = ROOT / "results" / "analyze_tau_sys_cartesian"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CFC_PATH = ROOT / "results" / "experiment_franka_2a" / "cfc_cerebellum.pt"

DEVICE       = "cuda" if _torch.cuda.is_available() else "cpu"
SIM_DURATION = 8.0
DIST_T       = 4.0
DIST_TORQUE  = 60.0
DIST_STEPS   = 20
DIST_JOINT   = 1
Q_OFFSET     = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.5, 0.0])

CPG_PARAMS = dict(tau=0.3, tau_r=0.6, beta=2.5, w=2.0, amplitude=0.0)
JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "J7"]


# ──────────────────────────────────────────────────────────────────────
def fk_ee(env: FrankaEnv, q_ref: np.ndarray) -> np.ndarray:
    """FK: q_ref から EE 位置 [m] を計算（一時的な MjData を使用）。"""
    data_tmp = mujoco.MjData(env.model)
    data_tmp.qpos[:N_JOINTS] = q_ref
    mujoco.mj_forward(env.model, data_tmp)
    try:
        site_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        return data_tmp.site_xpos[site_id].copy()
    except Exception:
        body_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, "link7"
        )
        return data_tmp.xpos[body_id].copy()


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """2群の Cohen's d（効果量）。"""
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return (b.mean() - a.mean()) / (pooled_std + 1e-12)


def welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch の t 検定（scipy なしの実装）。"""
    na, nb = len(a), len(b)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    t_stat = (mb - ma) / (se + 1e-12)
    # 自由度（Welch–Satterthwaite）
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1) + 1e-12
    )
    # p 値: 正規近似（df > 30 で良好）
    from math import erfc, sqrt
    p_approx = erfc(abs(t_stat) / sqrt(2))
    return float(t_stat), float(p_approx)


# ──────────────────────────────────────────────────────────────────────
def run_episode(
    controller: FrankaNeuralController,
    env:        FrankaEnv,
    ee_ref:     np.ndarray,
) -> dict:
    q, dq = env.reset(q0=Q_OFFSET.copy())
    controller.reset()

    t_log, q_log, ee_log, tau_sys_log = [], [], [], []
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

        t_log.append(t)
        q_log.append(q.copy())
        ee_log.append(env.get_ee_pos())

        if info["tau_sys"] is not None:
            tau_sys_log.append(info["tau_sys"])

    t_arr   = np.array(t_log)
    q_arr   = np.array(q_log)
    ee_arr  = np.array(ee_log)  # (T, 3) [m]
    err_arr = np.abs(q_arr - Q_OFFSET)

    # Cartesian 誤差 [mm]
    cart_err = np.linalg.norm(ee_arr - ee_ref, axis=1) * 1000.0

    post_mask = t_arr > DIST_T
    pre_mask  = t_arr <= DIST_T

    tau_sys_arr = np.array(tau_sys_log) if tau_sys_log else None

    # τ_sys 統計（pre vs post disturbance）
    tau_stats = {}
    if tau_sys_arr is not None and len(tau_sys_arr) == len(t_arr):
        ts_pre  = tau_sys_arr[pre_mask]
        ts_post = tau_sys_arr[post_mask]
        t_stat, p_val = welch_t(ts_pre, ts_post)
        d = cohen_d(ts_pre, ts_post)
        tau_stats = {
            "pre_mean":  float(ts_pre.mean()),
            "pre_std":   float(ts_pre.std()),
            "post_mean": float(ts_post.mean()),
            "post_std":  float(ts_post.std()),
            "t_stat":    t_stat,
            "p_val":     p_val,
            "cohen_d":   d,
        }

    return {
        "t":           t_arr,
        "q":           q_arr,
        "err":         err_arr,
        "ee":          ee_arr,
        "cart_err":    cart_err,
        "tau_sys":     tau_sys_arr,
        "tau_stats":   tau_stats,
        "mae_joint_pre":  float(err_arr[pre_mask].mean()) if pre_mask.any() else np.nan,
        "mae_joint_post": float(err_arr[post_mask].mean()) if post_mask.any() else np.nan,
        "cart_err_pre":   float(cart_err[pre_mask].mean()) if pre_mask.any() else np.nan,
        "cart_err_post":  float(cart_err[post_mask].mean()) if post_mask.any() else np.nan,
        "peak_cart_err":  float(cart_err[post_mask].max()) if post_mask.any() else np.nan,
    }


# ──────────────────────────────────────────────────────────────────────
def main():
    env     = FrankaEnv()
    q_range = env.ctrl_range

    if not CFC_PATH.exists():
        print(f"ERROR: {CFC_PATH} が見つかりません。先に experiment_franka_2a.py を実行してください。")
        return

    # 参照 EE 位置（FK(Q_OFFSET) = 定常目標の EE 座標）
    ee_ref = fk_ee(env, Q_OFFSET)
    print(f"参照 EE 位置: x={ee_ref[0]:.4f}m  y={ee_ref[1]:.4f}m  z={ee_ref[2]:.4f}m")

    # ── コントローラ定義 ──────────────────────────────────────────────
    def make_ctrl(use_reflex: bool, use_cerebellum: bool) -> FrankaNeuralController:
        ctrl = FrankaNeuralController(
            dt=env.dt, q_range=q_range,
            cpg_params=CPG_PARAMS,
            use_proprioceptor=False,
            use_reflex=use_reflex,
            use_cerebellum=use_cerebellum,
            cpg_alpha_fb=0.0,
            device=DEVICE,
        )
        if use_cerebellum:
            ctrl.load_cerebellum(str(CFC_PATH))
        return ctrl

    conditions = {
        "PD のみ":      make_ctrl(use_reflex=False, use_cerebellum=False),
        "PD + 小脳":    make_ctrl(use_reflex=False, use_cerebellum=True),
        "Full（提案）": make_ctrl(use_reflex=True,  use_cerebellum=True),
    }
    colors = {
        "PD のみ":      "tab:gray",
        "PD + 小脳":    "tab:blue",
        "Full（提案）": "tab:green",
    }

    print(f"\n=== τ_sys 定量分析 + Cartesian 誤差評価 ===")
    print(f"  設定: 静止保持 + 外乱 {DIST_TORQUE:.0f} Nm × {DIST_STEPS} steps at t={DIST_T}s")

    results = {}
    for label, ctrl in conditions.items():
        print(f"  実行中: {label} ...")
        results[label] = run_episode(ctrl, env, ee_ref)

    # ── サマリ表示 ────────────────────────────────────────────────────
    print("\n=== 結果サマリ ===")
    print(f"{'条件':20s}  {'関節MAE pre':>12s}  {'関節MAE post':>12s}  "
          f"{'Cart pre':>10s}  {'Cart post':>10s}  {'Cart peak':>10s}")
    for label, log in results.items():
        print(f"  {label:18s}  "
              f"{log['mae_joint_pre']*1000:10.2f} mrad  "
              f"{log['mae_joint_post']*1000:10.2f} mrad  "
              f"{log['cart_err_pre']:8.2f} mm  "
              f"{log['cart_err_post']:8.2f} mm  "
              f"{log['peak_cart_err']:8.2f} mm")

    # τ_sys 統計（Full のみ）
    label_full = "Full（提案）"
    ts = results[label_full]["tau_stats"]
    if ts:
        print(f"\n=== τ_sys 統計 ({label_full}) ===")
        print(f"  外乱前: mean={ts['pre_mean']:.4f}  std={ts['pre_std']:.4f}")
        print(f"  外乱後: mean={ts['post_mean']:.4f}  std={ts['post_std']:.4f}")
        print(f"  変化量: Δmean={ts['post_mean']-ts['pre_mean']:+.4f}  "
              f"({(ts['post_mean']-ts['pre_mean'])/ts['pre_mean']*100:+.2f}%)")
        print(f"  Welch t={ts['t_stat']:.3f}  p≈{ts['p_val']:.4f}  "
              f"Cohen's d={ts['cohen_d']:.3f}")
        sig = "**有意差あり**" if ts["p_val"] < 0.05 else "有意差なし"
        print(f"  → {sig} (p < 0.05 基準)")

    # ── 保存 ──────────────────────────────────────────────────────────
    save_dict = {}
    for label, log in results.items():
        key = label.replace("（", "").replace("）", "").replace(" ", "_")
        for k, v in log.items():
            if isinstance(v, np.ndarray):
                save_dict[f"{key}_{k}"] = v
            elif isinstance(v, float):
                save_dict[f"{key}_{k}"] = np.array([v])
    if ts:
        for k, v in ts.items():
            save_dict[f"tau_stats_{k}"] = np.array([v])
    np.savez(str(RESULTS_DIR / "metrics.npz"), **save_dict)

    # ── プロット ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "tau_sys Quantitative Analysis + Cartesian Error Evaluation\n"
        f"Static holding, disturbance: {DIST_TORQUE:.0f}Nm x {DIST_STEPS}steps at t={DIST_T}s",
        fontsize=11,
    )

    def shade(ax):
        ax.axvspan(DIST_T, SIM_DURATION, alpha=0.06, color="red")
        ax.axvline(DIST_T, color="red", lw=1.5, ls="--", alpha=0.8, label="disturbance")

    # [0,0] J2 関節誤差（全条件）
    ax = axes[0, 0]
    for label, log in results.items():
        ax.plot(log["t"], log["err"][:, DIST_JOINT] * 1000,
                label=label, color=colors[label], lw=1.2)
    shade(ax)
    ax.set_ylabel("J2 joint error [mrad]")
    ax.set_title("J2 tracking error (all conditions)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time [s]")

    # [0,1] Cartesian 誤差（全条件）
    ax = axes[0, 1]
    for label, log in results.items():
        ax.plot(log["t"], log["cart_err"],
                label=label, color=colors[label], lw=1.2)
    shade(ax)
    ax.set_ylabel("EE Cartesian error [mm]")
    ax.set_title("End-effector Cartesian error (all conditions)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time [s]")

    # [0,2] Cartesian 誤差 pre/post バー比較
    ax = axes[0, 2]
    labels_list = list(results.keys())
    x = np.arange(len(labels_list))
    pre_vals  = [results[l]["cart_err_pre"]  for l in labels_list]
    post_vals = [results[l]["cart_err_post"] for l in labels_list]
    bar_colors = [colors[l] for l in labels_list]
    b1 = ax.bar(x - 0.2, pre_vals,  0.35, label="Pre-dist [mm]",
                color=bar_colors, alpha=0.4)
    b2 = ax.bar(x + 0.2, post_vals, 0.35, label="Post-dist [mm]",
                color=bar_colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, fontsize=8, rotation=10, ha="right")
    ax.set_ylabel("Cartesian error [mm]")
    ax.set_title("EE Cartesian error summary")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    # [1,0] τ_sys 時系列（Full のみ）
    ax = axes[1, 0]
    log_full = results[label_full]
    if log_full["tau_sys"] is not None:
        t_ts = log_full["t"][:len(log_full["tau_sys"])]
        ts_arr = log_full["tau_sys"]
        ax.plot(t_ts, ts_arr, color="purple", lw=1.0, alpha=0.8, label="t_interp")
        # 外乱前後の平均線
        pre_mask  = t_ts <= DIST_T
        post_mask = t_ts > DIST_T
        if pre_mask.any():
            ax.axhline(ts_arr[pre_mask].mean(), color="blue",
                       ls=":", lw=1.5, alpha=0.7, label=f"pre mean={ts_arr[pre_mask].mean():.4f}")
        if post_mask.any():
            ax.axhline(ts_arr[post_mask].mean(), color="red",
                       ls=":", lw=1.5, alpha=0.7, label=f"post mean={ts_arr[post_mask].mean():.4f}")
        shade(ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("t_interp (tau_sys proxy)")
        ax.set_title("CfC tau_sys over time (Full condition)\n"
                     "Higher = faster adaptation")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time [s]")

    # [1,1] τ_sys 分布比較（violin）
    ax = axes[1, 1]
    if log_full["tau_sys"] is not None and len(log_full["tau_sys"]) == len(log_full["t"]):
        t_full   = log_full["t"]
        ts_arr   = log_full["tau_sys"]
        ts_pre   = ts_arr[t_full <= DIST_T]
        ts_post  = ts_arr[t_full >  DIST_T]
        vp = ax.violinplot([ts_pre, ts_post], positions=[0, 1],
                           showmedians=True, showextrema=True)
        vp["bodies"][0].set_facecolor("blue")
        vp["bodies"][1].set_facecolor("red")
        for body in vp["bodies"]:
            body.set_alpha(0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Pre-dist\n(t<{DIST_T}s)", f"Post-dist\n(t>{DIST_T}s)"])
        ax.set_ylabel("t_interp distribution")
        ax.set_title("tau_sys distribution: pre vs post disturbance")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        if ts:
            stat_text = (
                f"t={ts['t_stat']:.3f}, p={ts['p_val']:.4f}\n"
                f"Cohen's d={ts['cohen_d']:.3f}"
            )
            ax.text(0.5, 0.05, stat_text, transform=ax.transAxes,
                    ha="center", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    # [1,2] EE 軌道 XY 平面
    ax = axes[1, 2]
    ax.scatter(ee_ref[0] * 1000, ee_ref[1] * 1000,
               s=120, marker="*", color="black", zorder=5, label="Reference EE")
    for label, log in results.items():
        ax.plot(log["ee"][:, 0] * 1000, log["ee"][:, 1] * 1000,
                label=label, color=colors[label], lw=1.0, alpha=0.8)
    ax.set_xlabel("EE X [mm]")
    ax.set_ylabel("EE Y [mm]")
    ax.set_title("End-effector trajectory (XY plane)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    path = str(RESULTS_DIR / "plot_tau_sys_cartesian.png")
    plt.savefig(path, dpi=150)
    print(f"\nプロット保存: {path}")


if __name__ == "__main__":
    main()
