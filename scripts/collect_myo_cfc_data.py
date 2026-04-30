"""
collect_myo_cfc_data.py — MyoArm 動態データ収集（CfC 前向きモデル学習用）。

収集内容:
  各ステップで (q_t, dq_t, a_t, q_{t+1}) を記録する。
  CfC 前向きモデルは (q, dq, a[:n_joints]) → Δq = q_{t+1} - q_t を学習する。

探索方針:
  - ランダム筋活性化でエピソードを実行（広い状態空間をカバー）
  - 安定化のため [0.05, 0.6] の範囲でランダムサンプリング

出力:
  results/myo_cfc_data/train_data.npz
    q   : (N, 20)  — 関節角 [rad]
    dq  : (N, 20)  — 関節速度 [rad/s]
    a   : (N, 34)  — 筋活性化
    q_next : (N, 20)  — 次ステップ関節角
"""

from __future__ import annotations

import sys
from pathlib import Path

import myosuite  # noqa
import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "results" / "myo_cfc_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def collect(
    n_episodes:    int   = 200,
    max_steps:     int   = 300,
    act_low:       float = 0.05,
    act_high:      float = 0.60,
    smooth_alpha:  float = 0.85,  # 滑らかなランダムウォーク
    seed:          int   = 42,
) -> None:
    rng = np.random.default_rng(seed)
    env = gym.make("myoArmReachFixed-v0")
    n_j = 20
    n_m = env.action_space.shape[0]  # 34

    q_list:      list[np.ndarray] = []
    dq_list:     list[np.ndarray] = []
    a_list:      list[np.ndarray] = []
    q_next_list: list[np.ndarray] = []

    total_steps = 0
    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31))
        obs, _ = env.reset(seed=ep_seed)
        uw = env.unwrapped
        a_prev = rng.uniform(act_low, act_high, n_m).astype(np.float32)

        for _ in range(max_steps):
            d = uw.obs_dict
            q  = np.array(d["qpos"])
            dq = np.array(d["qvel"])

            # 滑らかなランダムウォーク
            a_new = rng.uniform(act_low, act_high, n_m).astype(np.float32)
            a = (smooth_alpha * a_prev + (1.0 - smooth_alpha) * a_new).clip(0.0, 1.0)
            a_prev = a

            q_list.append(q)
            dq_list.append(dq)
            a_list.append(a)

            obs, _, term, trunc, _ = env.step(a)

            d2 = uw.obs_dict
            q_next = np.array(d2["qpos"])
            q_next_list.append(q_next)
            total_steps += 1

            if term or trunc:
                break

        if (ep + 1) % 20 == 0:
            print(f"  ep {ep+1:4d}/{n_episodes}  total_steps={total_steps:6d}")

    env.close()

    Q      = np.stack(q_list,      axis=0).astype(np.float32)
    DQ     = np.stack(dq_list,     axis=0).astype(np.float32)
    A      = np.stack(a_list,      axis=0).astype(np.float32)
    Q_NEXT = np.stack(q_next_list, axis=0).astype(np.float32)

    out_path = OUT_DIR / "train_data.npz"
    np.savez_compressed(out_path, q=Q, dq=DQ, a=A, q_next=Q_NEXT)
    print(f"\nSaved {total_steps} steps → {out_path}")
    print(f"  q shape: {Q.shape}, a shape: {A.shape}")


if __name__ == "__main__":
    print("=== MyoArm CfC データ収集 ===")
    collect(n_episodes=200, max_steps=300)
