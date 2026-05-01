"""
collect_myo_cfc_lambda_data.py — λ-EP virtual trajectory 動態データ収集
                                  (CfC 前向きモデル再訓練用)

既存 CfC は myoArmReachFixed-v0 でランダム活性化から訓練されており、
λ-EP 動態と分布乖離が大きく「CfC が効いていない」原因。
λ-EP virtual trajectory rollouts で収集しなおし、より自然な
reach 動態を CfC に学習させる。

設計:
  env       : myoArmReachRandom-v0 + deterministic_reset
  controller: λ-EP virtual trajectory c=20 sg=1.2 (F9 best biological)
  K_cereb=0 : CfC 補正なしで純粋な λ-EP 動態を収集
  seeds     : 100..299 から reach_dist<0.85m を満たす最初 100 (test 0..49 と分離)
  max_steps : 200/episode (early termination 許容)

出力:
  results/myo_cfc_data_lambda/train_data.npz
    q   : (N, 20)  関節角
    dq  : (N, 20)  関節速度
    a   : (N, 34)  λ-EP で生成された筋活性化
    q_next : (N, 20)  次ステップ関節角
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import myosuite  # noqa
import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myoarm.myo_controller import MyoArmController, MyoArmConfig
from myoarm.env_utils import deterministic_reset

OUT_DIR = ROOT / "results" / "myo_cfc_data_lambda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DT          = 0.020
MAX_REACH_M = 0.85
N_TRAIN     = 100
SEED_POOL   = list(range(100, 300))


def make_lambda_cfg() -> MyoArmConfig:
    """K_cereb=0 で λ-EP virtual trajectory のみ動作させる設定。"""
    return MyoArmConfig(
        Kp_ee=80.0, Kd_ee=15.0, Ki_ee=2.0, act_bias=0.15,
        K_cereb=0.0, K_ia=0.05, K_ib=0.03, K_ri=0.5,  # 反射は通常通り (実体験動態)
        io_mode="sparse", io_firing_rate_hz=0.0,
        traj_speed_gain=1.2, traj_dt=DT, vel_scale_min=0.10,
        Kd_traj=50.0, use_traj_plan=False,
        control_mode="lambda_ep", c_lambda=20.0, lambda_offset=0.005,
        lambda_trajectory=True, lambda_traj_speed_gain=1.2,
    )


def find_reachable_seeds(env: gym.Env, pool: list[int], n: int) -> list[int]:
    out = []
    for s in pool:
        deterministic_reset(env, s)
        od = env.unwrapped.obs_dict
        d = float(np.linalg.norm(np.array(od["reach_err"])))
        if d < MAX_REACH_M:
            out.append(s)
        if len(out) >= n:
            break
    return out


def collect(max_steps: int = 200) -> None:
    print("=== λ-EP virtual trajectory データ収集 ===")
    env = gym.make("myoArmReachRandom-v0")
    deterministic_reset(env, 0)
    uw = env.unwrapped
    m = uw.mj_model
    muscle_names = [m.actuator(i).name for i in range(m.nu)]

    seeds = find_reachable_seeds(env, SEED_POOL, N_TRAIN)
    print(f"  reachable train seeds (n={len(seeds)}): {seeds[:5]}...{seeds[-5:]}")

    cfg = make_lambda_cfg()
    ctrl = MyoArmController(config=cfg, muscle_names=muscle_names)

    q_list: list[np.ndarray]      = []
    dq_list: list[np.ndarray]     = []
    a_list: list[np.ndarray]      = []
    q_next_list: list[np.ndarray] = []

    total_steps = 0
    t0 = time.time()
    for ep_i, seed in enumerate(seeds):
        obs, _ = deterministic_reset(env, seed)
        d = uw.mj_data
        ctrl.reset(); ctrl.initialize(m, d)

        for step in range(max_steps):
            od = uw.obs_dict
            q  = np.array(od["qpos"]).astype(np.float32)
            dq = np.array(od["qvel"]).astype(np.float32)
            reach_err = np.array(od["reach_err"])
            tip_pos   = np.array(od["tip_pos"])

            a_total, _ = ctrl.step(
                q=q, dq=dq, reach_err=reach_err, tip_pos=tip_pos,
                muscle_vel=d.actuator_velocity.copy(),
                muscle_force=d.actuator_force.copy(),
                m=m, d=d,
            )
            obs, _, term, trunc, _ = env.step(a_total)
            q_next = np.array(uw.obs_dict["qpos"]).astype(np.float32)

            q_list.append(q)
            dq_list.append(dq)
            a_list.append(a_total.astype(np.float32))
            q_next_list.append(q_next)
            total_steps += 1

            if term or trunc:
                break

        if (ep_i + 1) % 10 == 0:
            print(f"  ep {ep_i+1:3d}/{len(seeds)}  total_steps={total_steps:6d}  "
                  f"elapsed={time.time()-t0:.1f}s")

    env.close()

    Q      = np.stack(q_list,      axis=0)
    DQ     = np.stack(dq_list,     axis=0)
    A      = np.stack(a_list,      axis=0)
    Q_NEXT = np.stack(q_next_list, axis=0)

    out_path = OUT_DIR / "train_data.npz"
    np.savez_compressed(out_path, q=Q, dq=DQ, a=A, q_next=Q_NEXT, seeds=np.array(seeds))
    print(f"\nSaved {total_steps} steps → {out_path}")
    print(f"  q shape: {Q.shape}, a shape: {A.shape}")
    print(f"  q range: [{Q.min():.2f}, {Q.max():.2f}]  a range: [{A.min():.3f}, {A.max():.3f}]")
    print(f"  total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    collect()
