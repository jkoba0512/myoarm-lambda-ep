"""
experiment_myo_p12_kinematics.py — Phase 1-2: 軌跡記録と最小ジャーク比較。

目的:
  myoArmReachRandom-v0 環境で MyoArmController を動かし、手先軌跡の運動学的
  特性を記録する。Flash & Hogan (1985) の最小ジャーク原理が予測するヒト到達
  運動の特徴と比較し、一致・不一致を定量化する。

事前指定の比較指標（Phase 1-2 の事前登録原則に基づき実験前に確定）:
  1. jerk_rms       = √(mean(|d³r/dt³|²))  [m/s³]  — 軌跡の滑らかさ
  2. vel_peak_ratio = t_peak / T_move        [0,1]   — 速度ピークの相対タイミング
                      最小ジャーク予測値: 0.50（対称ベル型）
  3. straightness   = D_straight / L_path   [0,1]   — 直線性（1=完全直線）
  4. movement_time  = T_move                [s]     — 到達時間
  5. peak_speed     = max|dr/dt|            [m/s]   — ピーク手先速度

比較基準:
  - 最小ジャーク モデル (Flash & Hogan 1985): 解析的に軌跡を生成し同じ指標を計算
  - 先行研究の典型値:
      vel_peak_ratio ≈ 0.45–0.55（ヒト到達運動、対称ベル型）
      straightness   ≈ 0.85–0.95（ほぼ直線）
      movement_time  ≈ 0.5–1.5 s（目標距離依存）

出力:
  results/experiment_myo_p12/kinematics_summary.json
  results/experiment_myo_p12/trajectories.npz  (生データ)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import myosuite  # noqa
import gymnasium as gym
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myoarm.myo_controller import MyoArmController, MyoArmConfig

RESULTS_DIR    = ROOT / "results" / "experiment_myo_p12"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CFC_MODEL_PATH = ROOT / "results" / "myo_cfc_data" / "cfc_model.pt"

DT = 0.020   # myoArm 実制御周期 [s] (frame_skip=10 × mj_timestep=0.002s)


# ──────────────────────────────────────────────────────────────────────
# 最小ジャーク モデル (Flash & Hogan 1985)
# ──────────────────────────────────────────────────────────────────────

def minimum_jerk_trajectory(
    p_start: np.ndarray,
    p_end:   np.ndarray,
    T:       float,
    dt:      float = DT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    最小ジャーク軌跡を生成する。

    p(τ) = p0 + (pf-p0) * (10τ³ - 15τ⁴ + 6τ⁵)  where τ = t/T

    Returns
    -------
    pos  : (N, 3) 手先位置
    vel  : (N, 3) 手先速度
    jerk : (N, 3) 手先ジャーク（3階微分）
    """
    n_steps = max(int(round(T / dt)) + 1, 10)
    t_arr   = np.linspace(0.0, T, n_steps)
    tau     = t_arr / T

    # 5次多項式
    s   = 10*tau**3 - 15*tau**4 + 6*tau**5
    ds  = (30*tau**2 - 60*tau**3 + 30*tau**4) / T
    # 3階微分 (ジャーク)
    d3s = (60 - 360*tau + 360*tau**2) / T**3

    pos  = p_start + np.outer(s,   p_end - p_start)
    vel  = np.outer(ds,  p_end - p_start)
    jerk = np.outer(d3s, p_end - p_start)

    return pos, vel, jerk


# ──────────────────────────────────────────────────────────────────────
# 運動学指標の計算
# ──────────────────────────────────────────────────────────────────────

def compute_kinematics_metrics(
    positions: np.ndarray,   # (N, 3)  手先位置 [m]
    dt:        float = DT,
    onset_speed_threshold: float = 0.02,  # 運動開始/終了の速度閾値 [m/s]
) -> dict:
    """
    手先位置時系列から運動学指標を計算する。

    Returns
    -------
    dict containing: jerk_rms, vel_peak_ratio, straightness,
                     movement_time, peak_speed, onset_step, offset_step
    """
    if len(positions) < 10:
        return {k: float("nan") for k in
                ["jerk_rms", "vel_peak_ratio", "straightness",
                 "movement_time", "peak_speed"]}

    vel   = np.diff(positions, axis=0) / dt          # (N-1, 3)
    accel = np.diff(vel,       axis=0) / dt          # (N-2, 3)
    jerk  = np.diff(accel,     axis=0) / dt          # (N-3, 3)

    speed = np.linalg.norm(vel, axis=1)               # (N-1,)

    # 運動開始・終了を速度閾値で検出
    above = np.where(speed > onset_speed_threshold)[0]
    if len(above) < 5:
        onset, offset = 0, len(speed) - 1
    else:
        onset, offset = above[0], above[-1]

    # 運動区間に限定
    pos_move  = positions[onset:offset+2]
    vel_move  = vel[onset:offset+1]
    speed_move = speed[onset:offset+1]
    jerk_move  = jerk[max(0, onset-2):offset]

    if len(speed_move) < 5:
        return {k: float("nan") for k in
                ["jerk_rms", "vel_peak_ratio", "straightness",
                 "movement_time", "peak_speed"]}

    # 1. Jerk RMS
    jerk_rms = float(np.sqrt(np.mean(jerk_move**2))) if len(jerk_move) > 0 else float("nan")

    # 2. 速度ピーク相対タイミング
    peak_idx       = int(np.argmax(speed_move))
    T_move_steps   = len(speed_move)
    vel_peak_ratio = peak_idx / max(T_move_steps - 1, 1)

    # 3. 直線性
    L_path    = float(np.sum(np.linalg.norm(np.diff(pos_move, axis=0), axis=1)))
    D_straight = float(np.linalg.norm(pos_move[-1] - pos_move[0]))
    straightness = D_straight / max(L_path, 1e-6)

    # 4. 到達時間
    movement_time = T_move_steps * dt

    # 5. ピーク速度
    peak_speed = float(np.max(speed_move))

    return {
        "jerk_rms":       jerk_rms,
        "vel_peak_ratio": float(vel_peak_ratio),
        "straightness":   float(straightness),
        "movement_time":  float(movement_time),
        "peak_speed":     float(peak_speed),
        "onset_step":     int(onset),
        "offset_step":    int(offset),
        "n_move_steps":   int(T_move_steps),
    }


# ──────────────────────────────────────────────────────────────────────
# 1 エピソード実行
# ──────────────────────────────────────────────────────────────────────

def run_episode(
    env:      gym.Env,
    ctrl:     MyoArmController,
    max_steps: int = 800,
    seed:     int  = 0,
) -> dict:
    obs, _ = env.reset(seed=seed)
    ctrl.reset()

    uw   = env.unwrapped
    m, d = uw.mj_model, uw.mj_data
    ctrl.initialize(m, d)

    od = uw.obs_dict
    target_pos = np.array(od["target_pos"])

    positions:    list[np.ndarray] = []
    solved_steps: list[int]        = []

    for step in range(max_steps):
        od = uw.obs_dict
        q            = np.array(od["qpos"])
        dq           = np.array(od["qvel"])
        reach_err    = np.array(od["reach_err"])
        tip_pos      = np.array(od["tip_pos"])
        muscle_vel   = d.actuator_velocity.copy()
        muscle_force = d.actuator_force.copy()

        positions.append(tip_pos.copy())

        a_total, _ = ctrl.step(
            q=q, dq=dq,
            reach_err=reach_err, tip_pos=tip_pos,
            muscle_vel=muscle_vel, muscle_force=muscle_force,
            m=m, d=d,
        )

        _, _, terminated, truncated, info = env.step(a_total)

        od2    = uw.obs_dict
        q_next = np.array(od2["qpos"])
        ctrl.update_cerebellum(q_next, m, d)

        if info.get("solved", False):
            solved_steps.append(step)

        if terminated or truncated:
            break

    pos_arr = np.array(positions)  # (N, 3)

    # 運動学指標の計算
    metrics = compute_kinematics_metrics(pos_arr, dt=DT)

    # 最小ジャーク参照との比較（同じ距離・到達時間）
    D = float(np.linalg.norm(target_pos - pos_arr[0]))
    mt = metrics.get("movement_time", 0.6)
    T_mj = float(mt) if (mt is not None and not np.isnan(mt)) else 0.6
    T_mj = max(T_mj, 0.2)
    _, vel_mj, jerk_mj = minimum_jerk_trajectory(pos_arr[0], target_pos, T=T_mj)
    speed_mj = np.linalg.norm(vel_mj, axis=1)
    mj_metrics = {
        "jerk_rms":       float(np.sqrt(np.mean(jerk_mj**2))),
        "vel_peak_ratio": float(np.argmax(speed_mj) / max(len(speed_mj)-1, 1)),
        "straightness":   1.0,   # 最小ジャークは完全な直線
        "peak_speed":     float(np.max(speed_mj)),
    }

    # 誤差（モデル vs 最小ジャーク）
    comparison = {}
    for key in ["jerk_rms", "vel_peak_ratio", "straightness"]:
        m_val  = metrics.get(key, float("nan"))
        mj_val = mj_metrics.get(key, float("nan"))
        comparison[f"{key}_model"]        = m_val
        comparison[f"{key}_min_jerk"]     = mj_val
        comparison[f"{key}_diff"]         = m_val - mj_val
        comparison[f"{key}_ratio"]        = (m_val / mj_val
                                              if mj_val != 0 else float("nan"))

    final_err = float(np.linalg.norm(np.array(uw.obs_dict["reach_err"])))

    return {
        "seed":            seed,
        "n_steps":         step + 1,
        "target_pos":      target_pos.tolist(),
        "reach_distance_m": D,
        "final_err_mm":    final_err * 1000,
        "solved":          len(solved_steps) > 0,
        "model_metrics":   metrics,
        "min_jerk_ref":    mj_metrics,
        "comparison":      comparison,
        "positions":       pos_arr,   # (N, 3) — npz 保存用
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    n_seeds   = 20
    max_steps = 800

    print("=== Phase 1-2: 軌跡記録 & 最小ジャーク比較 ===")
    print(f"env: myoArmReachRandom-v0  seeds: {n_seeds}  max_steps: {max_steps}")

    env = gym.make("myoArmReachRandom-v0")
    uw  = env.unwrapped
    env.reset(seed=0)
    m = uw.mj_model
    muscle_names = [m.actuator(i).name for i in range(m.nu)]

    cfg  = MyoArmConfig(Kp_ee=80.0, Kd_ee=15.0, Ki_ee=2.0, act_bias=0.15, K_cereb=0.2)
    ctrl = MyoArmController(config=cfg, muscle_names=muscle_names)
    if CFC_MODEL_PATH.exists():
        ctrl.load_cfc(CFC_MODEL_PATH)

    all_results = []
    all_positions = []
    t0 = time.time()

    for seed in range(n_seeds):
        r = run_episode(env, ctrl, max_steps=max_steps, seed=seed)
        pos = r.pop("positions")
        all_results.append(r)
        all_positions.append(pos)

        cm = r["comparison"]
        print(
            f"  seed {seed:2d}: "
            f"dist={r['reach_distance_m']*100:.0f}cm  "
            f"T={r['model_metrics'].get('movement_time', 0):.2f}s  "
            f"jerk={cm['jerk_rms_model']:.2f}(mj={cm['jerk_rms_min_jerk']:.2f})  "
            f"peak_ratio={cm['vel_peak_ratio_model']:.2f}(mj={cm['vel_peak_ratio_min_jerk']:.2f})  "
            f"straight={cm['straightness_model']:.3f}  "
            f"{'✓' if r['solved'] else '—'}"
        )
    env.close()
    elapsed = time.time() - t0

    # 集計
    def agg(key):
        vals = [r["comparison"][key] for r in all_results
                if not np.isnan(r["comparison"].get(key, float("nan")))]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else {}

    n_solved = sum(r["solved"] for r in all_results)
    aggregate = {
        "n_seeds":    n_seeds,
        "solve_rate": n_solved / n_seeds,
        "jerk_rms":           agg("jerk_rms_model"),
        "jerk_rms_min_jerk":  agg("jerk_rms_min_jerk"),
        "vel_peak_ratio":     agg("vel_peak_ratio_model"),
        "straightness":       agg("straightness_model"),
        "movement_time_s":    {
            "mean": float(np.mean([r["model_metrics"].get("movement_time", np.nan)
                                   for r in all_results])),
        },
    }

    summary = {
        "phase":      "1-2",
        "env":        "myoArmReachRandom-v0",
        "controller": "MyoArmController",
        "n_seeds":    n_seeds,
        "elapsed_s":  round(elapsed, 1),
        "aggregate":  aggregate,
        "pre_registered_metrics": {
            "jerk_rms":       "√(mean(|d³r/dt³|²)) [m/s³] — 小さいほど滑らか",
            "vel_peak_ratio": "t_peak/T_move ∈ [0,1] — ヒト予測値 ≈ 0.50",
            "straightness":   "D_straight/L_path — ヒト予測値 ≈ 0.85-0.95",
            "movement_time":  "T_move [s] — ヒト典型値 0.5-1.5 s",
        },
        "per_seed": [{k: v for k, v in r.items() if k != "positions"}
                     for r in all_results],
    }

    # JSON 保存
    out_json = RESULTS_DIR / "kinematics_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else x)

    # 軌跡生データ保存
    max_len = max(p.shape[0] for p in all_positions)
    pos_padded = np.full((n_seeds, max_len, 3), np.nan)
    for i, p in enumerate(all_positions):
        pos_padded[i, :p.shape[0]] = p
    np.savez_compressed(RESULTS_DIR / "trajectories.npz", positions=pos_padded)

    # サマリー表示
    print()
    print("=== 集計結果 ===")
    print(f"  solve rate      : {n_solved}/{n_seeds}")
    print(f"  jerk_rms        : {aggregate['jerk_rms']['mean']:.2f} m/s³  "
          f"(min_jerk ref: {aggregate['jerk_rms_min_jerk']['mean']:.2f})")
    print(f"  vel_peak_ratio  : {aggregate['vel_peak_ratio']['mean']:.3f}  "
          f"(human ~0.50, ratio=1.0が完全一致)")
    print(f"  straightness    : {aggregate['straightness']['mean']:.3f}  "
          f"(human ~0.85-0.95)")
    print(f"  movement_time   : {aggregate['movement_time_s']['mean']:.2f} s  "
          f"(human ~0.5-1.5 s)")
    print(f"  elapsed         : {elapsed:.1f} s")
    print(f"  saved → {out_json}")


if __name__ == "__main__":
    main()
