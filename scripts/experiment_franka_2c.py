"""
実験 2-C (Franka Panda): 2点間サイクリック動作における固有受容器フィードバック評価

生物学的根拠:
  繰り返しリーチング（2点間往復）は脊髄 CPG が制御するリズム運動の典型例。
  筋紡錘（LIF 固有受容器）は関節角度・速度をスパイク発火率に変換し、
  CPG の入力電流 I を変調して位相・振幅を自動補正する。

  t=LOAD_T で持続負荷トルクを印加（物体を把持した状況を模擬）。
  負荷により到達点がずれる → LIF FB がずれを検知 → CPG ドライブを補正
  → 到達精度が回復するかを評価。

タスク:
  J2 (肩ピッチ) が Q_CENTER ± amplitude を往復
  center:    Q_OFFSET = [0.0, -0.3, 0.0, -1.5, 0.0, 1.5, 0.0]
  amplitude: 0.3 rad → J2 が [-0.6, 0.0] rad を往復

比較条件:
  1. CPG + 小脳（固有受容器 FB なし）
  2. CPG + 小脳 + LIF 固有受容器 FB

評価指標:
  - エンドポイント到達誤差（各サイクルのピーク/トラフで評価）
  - 負荷前後の平均追従誤差
  - 収束時間（到達誤差が負荷前レベルの 120% 以内に戻るまで）

出力:
  results/experiment_franka_2c/
    ├── metrics.npz
    └── plot_franka_2c.png
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

RESULTS_DIR = ROOT / "results" / "experiment_franka_2c"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE       = "cuda" if _torch.cuda.is_available() else "cpu"
SIM_DURATION = 9.0   # 3s 安定 + 6s 負荷応答を観察
LOAD_T       = 3.0   # 持続負荷印加タイミング
Q_OFFSET     = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.5, 0.0])

# CPG: J2 が ±0.3 rad を往復（2点間サイクリック動作）
CPG_PARAMS = dict(tau=0.3, tau_r=0.6, beta=2.5, w=2.0, amplitude=0.3)

# 持続負荷トルク: J2 に -25 Nm（重力方向 = アーム下方向）
TAU_LOAD = np.zeros(N_JOINTS)
TAU_LOAD[1] = -25.0

ENDPOINT_JOINT = 1   # 評価対象関節（J2）
RECOVERY_THR   = 1.2 # 回復判定: 負荷前エンドポイント誤差の 120% 以内

JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "J7"]


# ──────────────────────────────────────────────────────────────────────
def detect_peaks(signal: np.ndarray, dt: float, min_interval: float = 0.2) -> np.ndarray:
    """信号のピーク（極大・極小）インデックスを返す。"""
    min_steps = max(1, int(min_interval / dt))
    peaks = []
    for i in range(min_steps, len(signal) - min_steps):
        window = signal[i - min_steps: i + min_steps + 1]
        if signal[i] == window.max() or signal[i] == window.min():
            if not peaks or i - peaks[-1] >= min_steps:
                peaks.append(i)
    return np.array(peaks, dtype=int)


def run_episode(
    controller: FrankaNeuralController,
    env:        FrankaEnv,
) -> dict:
    q, dq = env.reset(q0=Q_OFFSET.copy())
    controller.reset()

    t_log, q_log, q_ref_log, r_q_log, tau_sys_log = [], [], [], [], []

    while env.time < SIM_DURATION:
        t = env.time
        q, dq = env.get_state()

        # 持続負荷: LOAD_T 以降は毎ステップ外力を設定
        if t >= LOAD_T:
            env.data.qfrc_applied[:N_JOINTS] = TAU_LOAD
        else:
            env.data.qfrc_applied[:N_JOINTS] = 0.0

        tau_total, info = controller.step(q, dq, Q_OFFSET)
        env.step(tau_total)

        t_log.append(t)
        q_log.append(q.copy())
        q_ref_log.append(info["q_target"].copy())
        r_q_log.append(info["r_q"].copy())
        if info["tau_sys"] is not None:
            tau_sys_log.append(info["tau_sys"])

    # セッション終了後に外力をリセット
    env.data.qfrc_applied[:N_JOINTS] = 0.0

    t_arr    = np.array(t_log)
    q_arr    = np.array(q_log)
    qref_arr = np.array(q_ref_log)
    dt       = env.dt

    # エンドポイント到達誤差: 各サイクルのピーク/トラフで評価
    ref_j2  = qref_arr[:, ENDPOINT_JOINT]
    act_j2  = q_arr[:,   ENDPOINT_JOINT]
    peaks   = detect_peaks(ref_j2, dt)

    pre_mask  = t_arr < LOAD_T
    post_mask = t_arr >= LOAD_T

    ep_err_pre  = np.mean(np.abs(act_j2[peaks[t_arr[peaks] < LOAD_T]]
                                  - ref_j2[peaks[t_arr[peaks] < LOAD_T]])) \
                  if any(t_arr[peaks] < LOAD_T) else np.nan
    ep_err_post = np.mean(np.abs(act_j2[peaks[t_arr[peaks] >= LOAD_T]]
                                  - ref_j2[peaks[t_arr[peaks] >= LOAD_T]])) \
                  if any(t_arr[peaks] >= LOAD_T) else np.nan

    # 収束時間: エンドポイント誤差が負荷前レベルの RECOVERY_THR 倍以内に戻る時刻
    recovery_time = None
    if not np.isnan(ep_err_pre):
        thr = ep_err_pre * RECOVERY_THR
        for pk in peaks[t_arr[peaks] >= LOAD_T]:
            err = abs(act_j2[pk] - ref_j2[pk])
            if err <= thr:
                recovery_time = t_arr[pk] - LOAD_T
                break

    return {
        "t":           t_arr,
        "q":           q_arr,
        "q_ref":       qref_arr,
        "r_q":         np.array(r_q_log),
        "tau_sys":     np.array(tau_sys_log) if tau_sys_log else None,
        "peaks":       peaks,
        "ep_err_pre":  ep_err_pre,
        "ep_err_post": ep_err_post,
        "mae_pre":     np.abs(q_arr - qref_arr)[pre_mask].mean()  if pre_mask.any()  else np.nan,
        "mae_post":    np.abs(q_arr - qref_arr)[post_mask].mean() if post_mask.any() else np.nan,
        "recovery_time": recovery_time,
    }


# ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42, help="乱数シード")
    return p.parse_args()


def main():
    args   = parse_args()
    seed   = args.seed
    outdir = RESULTS_DIR / f"seed{seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    cfc_path = ROOT / "results" / "experiment_franka_2a" / f"seed{seed}" / "cfc_cerebellum.pt"
    if not cfc_path.exists():
        print(f"ERROR: {cfc_path} が見つかりません。先に experiment_franka_2a.py --seed {seed} を実行してください。")
        return

    print(f"seed={seed}  out={outdir}")

    env     = FrankaEnv()
    q_range = env.ctrl_range

    conditions = [
        ("CPG+CfC",        dict(use_proprioceptor=False, cpg_alpha_fb=0.0)),
        ("CPG+CfC+LIF_FB", dict(use_proprioceptor=True,  cpg_alpha_fb=0.3)),
    ]

    results = {}
    print("=== 実験 2-C (Franka): 2点間サイクリック動作 + LIF 固有受容器 FB 評価 ===")
    print(f"  タスク: J2 が ±{CPG_PARAMS['amplitude']} rad を往復（CPG 振動）")
    print(f"  持続負荷: t={LOAD_T}s 以降 J2 に {TAU_LOAD[1]:.0f} Nm を印加")

    for label, flags in conditions:
        ctrl = FrankaNeuralController(
            dt=env.dt, q_range=q_range,
            cpg_params=CPG_PARAMS,
            use_reflex=False,
            use_cerebellum=True,
            device=DEVICE,
            **flags,
        )
        ctrl.load_cerebellum(str(cfc_path))
        log = run_episode(ctrl, env)
        results[label] = log

        rt     = log["recovery_time"]
        rt_str = f"{rt:.3f} s" if rt else "未回復"
        print(f"\n  {label}")
        print(f"    エンドポイント誤差 負荷前: {log['ep_err_pre']*1000:.2f} mrad  "
              f"負荷後: {log['ep_err_post']*1000:.2f} mrad")
        print(f"    追従MAE 負荷前: {log['mae_pre']*1000:.2f} mrad  "
              f"負荷後: {log['mae_post']*1000:.2f} mrad  収束時間: {rt_str}")

    # ── 改善率 ────────────────────────────────────────────────────
    ep_no = results["CPG+CfC"]["ep_err_post"]
    ep_fb = results["CPG+CfC+LIF_FB"]["ep_err_post"]
    if not np.isnan(ep_no) and ep_no > 0:
        print(f"\n  LIF FB によるエンドポイント誤差改善率（負荷後）: "
              f"{(ep_no - ep_fb) / ep_no * 100:.2f}%  "
              f"({ep_no*1000:.2f} → {ep_fb*1000:.2f} mrad)")

    # ── 標準化 JSON 保存 ──────────────────────────────────────────
    summary: dict = {"experiment": "2c", "seed": seed, "conditions": {}}
    for label, log in results.items():
        rt = log["recovery_time"]
        summary["conditions"][label] = {
            "ep_err_pre_mrad":  float(log["ep_err_pre"]  * 1000),
            "ep_err_post_mrad": float(log["ep_err_post"] * 1000),
            "mae_post_mrad":    float(log["mae_post"]    * 1000),
            "recovery_time_s":  float(rt) if rt is not None else None,
        }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── 保存（npz） ───────────────────────────────────────────────
    np.savez(
        str(outdir / "metrics.npz"),
        **{f"{label}_{k}": v
           for label, log in results.items()
           for k, v in log.items()
           if v is not None and isinstance(v, (np.ndarray, float, int))},
    )

    # ── プロット ──────────────────────────────────────────────────
    colors = {"CPG+CfC": "tab:orange", "CPG+CfC+LIF_FB": "tab:green"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Experiment 2-C (Franka Panda): 2-Point Cyclic Motion  [seed={seed}]\n"
                 "LIF Proprioceptor FB under sustained load "
                 f"({TAU_LOAD[1]:.0f} Nm on J2 at t={LOAD_T}s)", fontsize=11)

    def shade(ax):
        ax.axvspan(0, LOAD_T, alpha=0.04, color="blue", label="pre-load")
        ax.axvspan(LOAD_T, SIM_DURATION, alpha=0.04, color="red", label="post-load")
        ax.axvline(LOAD_T, color="red", lw=1.5, ls="--", alpha=0.8)

    log_fb = results["CPG+CfC+LIF_FB"]

    ax = axes[0, 0]
    ax.plot(log_fb["t"], log_fb["q_ref"][:, ENDPOINT_JOINT], "--",
            color="black", alpha=0.5, lw=1.0, label="CPG ref")
    for label, log in results.items():
        ax.plot(log["t"], log["q"][:, ENDPOINT_JOINT],
                label=label, color=colors[label], lw=1.2)
    shade(ax)
    ax.set_ylabel("J2 angle [rad]")
    ax.set_title("J2 joint tracking (cyclic motion)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time [s]")

    ax = axes[0, 1]
    for label, log in results.items():
        err = np.abs(log["q"][:, ENDPOINT_JOINT] - log["q_ref"][:, ENDPOINT_JOINT])
        ax.plot(log["t"], err * 1000, label=label, color=colors[label], lw=1.0)
    shade(ax)
    ax.set_ylabel("J2 tracking error [mrad]")
    ax.set_title("J2 tracking error over time")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time [s]")

    ax = axes[1, 0]
    for i in range(4):
        ax.plot(log_fb["t"], log_fb["r_q"][:, i],
                label=JOINT_NAMES[i], alpha=0.8)
    shade(ax)
    ax.set_ylabel("LIF firing rate r_q ∈ [-1, 1]")
    ax.set_title("Proprioceptor firing rates (J1-J4)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time [s]")

    ax = axes[1, 1]
    cond_keys = list(results.keys())
    ep_pre  = [results[l]["ep_err_pre"]  * 1000 for l in cond_keys]
    ep_post = [results[l]["ep_err_post"] * 1000 for l in cond_keys]
    x = np.arange(len(cond_keys))
    ax.bar(x - 0.2, ep_pre,  0.35, label="Pre-load [mrad]",
           color=[colors[l] for l in cond_keys], alpha=0.4)
    b2 = ax.bar(x + 0.2, ep_post, 0.35, label="Post-load [mrad]",
                color=[colors[l] for l in cond_keys], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_keys, fontsize=7, rotation=10, ha="right")
    ax.set_ylabel("Endpoint error [mrad]")
    ax.set_title("Endpoint reaching error\n(peak/trough of each cycle)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = str(outdir / "plot_franka_2c.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nプロット保存: {path}")


if __name__ == "__main__":
    main()
