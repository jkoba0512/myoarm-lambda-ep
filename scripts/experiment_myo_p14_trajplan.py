"""
experiment_myo_p14_trajplan.py — Phase 1-4: 最小ジャーク軌跡追従制御の評価。

MyoArmController に MinimumJerkPlanner を統合した軌跡追従制御を評価し、
Phase 1-2 の純反応型制御との運動学的指標を比較する。

期待される改善:
  - vel_peak_ratio: 0.05 → ~0.50 (ベル形速度プロファイル)
  - straightness  : 0.46 → ~0.70 以上
  - jerk_rms      : 251,000 → < 5,000 m/s³

出力:
  results/experiment_myo_p14/trajplan_summary.json
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

RESULTS_DIR = ROOT / "results" / "experiment_myo_p14"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CFC_MODEL_PATH = ROOT / "results" / "myo_cfc_data" / "cfc_model.pt"


# ---------------------------------------------------------------------------

def compute_kinematics_metrics(
    positions: np.ndarray, dt: float = 0.005
) -> dict:
    """tip_pos 軌跡から運動学的指標を計算する。"""
    if len(positions) < 5:
        return {k: float("nan") for k in
                ["jerk_rms", "vel_peak_ratio", "straightness", "movement_time", "peak_speed"]}

    vel = np.diff(positions, axis=0) / dt        # (T-1, 3)
    speed = np.linalg.norm(vel, axis=1)          # (T-1,)
    acc = np.diff(vel, axis=0) / dt              # (T-2, 3)
    jerk = np.diff(acc, axis=0) / dt             # (T-3, 3)

    thresh = 0.02
    onset = None
    for i, s in enumerate(speed):
        if s > thresh:
            onset = i
            break
    if onset is None:
        return {"jerk_rms": float("nan"), "vel_peak_ratio": float("nan"),
                "straightness": float("nan"), "movement_time": float("nan"),
                "peak_speed": float(np.max(speed))}

    offset = None
    for i in range(onset + 5, len(speed)):
        if speed[i] < thresh:
            offset = i
            break
    if offset is None:
        offset = len(speed) - 1

    movement_speed = speed[onset:offset+1]
    T_actual = (offset - onset) * dt

    peak_idx      = int(np.argmax(movement_speed))
    vel_peak_ratio = peak_idx / max(len(movement_speed) - 1, 1)

    if offset - 1 < len(jerk):
        jerk_seg = jerk[onset:offset-1] if offset-1 < len(jerk) else jerk[onset:]
    else:
        jerk_seg = jerk[onset:]
    jerk_rms = float(np.sqrt(np.mean(np.sum(jerk_seg**2, axis=1)))) if len(jerk_seg) > 0 else float("nan")

    total_path  = float(np.sum(np.linalg.norm(np.diff(positions[onset:offset+1], axis=0), axis=1)))
    direct_dist = float(np.linalg.norm(positions[offset] - positions[onset]))
    straightness = direct_dist / total_path if total_path > 1e-6 else float("nan")

    return {
        "jerk_rms":       jerk_rms,
        "vel_peak_ratio": float(vel_peak_ratio),
        "straightness":   float(straightness),
        "movement_time":  float(T_actual),
        "peak_speed":     float(np.max(movement_speed)),
        "onset_step":     int(onset),
        "offset_step":    int(offset),
    }


def run_episode(
    env: gym.Env,
    ctrl: MyoArmController,
    seed: int,
    max_steps: int = 800,
    dt: float = 0.005,
) -> dict:
    obs, _ = env.reset(seed=seed)
    uw = env.unwrapped
    m, d = uw.mj_model, uw.mj_data

    ctrl.reset()
    ctrl.initialize(m, d)

    positions: list[np.ndarray] = []
    tip_errors: list[float] = []
    solved = False

    for step in range(max_steps):
        od = uw.obs_dict
        q            = np.array(od["qpos"])
        dq           = np.array(od["qvel"])
        reach_err    = np.array(od["reach_err"])
        tip_pos      = np.array(od["tip_pos"])
        muscle_vel   = d.actuator_velocity.copy()
        muscle_force = d.actuator_force.copy()

        positions.append(tip_pos.copy())
        tip_errors.append(float(np.linalg.norm(reach_err)))

        a_total, _ = ctrl.step(
            q=q, dq=dq,
            reach_err=reach_err, tip_pos=tip_pos,
            muscle_vel=muscle_vel, muscle_force=muscle_force,
            m=m, d=d,
        )

        obs, _, terminated, truncated, info = env.step(a_total)

        od2    = uw.obs_dict
        q_next = np.array(od2["qpos"])
        ctrl.update_cerebellum(q_next, m, d)

        if info.get("solved", False):
            solved = True

        if terminated or truncated:
            break

    traj = np.array(positions)

    # 初期距離
    env.reset(seed=seed)
    od0 = env.unwrapped.obs_dict
    dist = float(np.linalg.norm(np.array(od0["reach_err"])))

    km = compute_kinematics_metrics(traj, dt=dt)

    return {
        "seed":              seed,
        "dist_m":            dist,
        "solved":            solved,
        "tip_err_final_mm":  tip_errors[-1] * 1000 if tip_errors else float("nan"),
        "tip_err_min_mm":    float(np.min(tip_errors)) * 1000 if tip_errors else float("nan"),
        **km,
        "n_steps":           len(positions),
    }


# ---------------------------------------------------------------------------

def run_condition(
    env: gym.Env,
    muscle_names: list,
    use_traj_plan: bool,
    n_seeds: int,
    max_steps: int,
    dt: float,
) -> tuple[list, dict]:
    """条件ごとに n_seeds エピソードを実行して結果を返す。"""
    cfg = MyoArmConfig(
        Kp_ee=80.0, Kd_ee=15.0, Ki_ee=2.0, act_bias=0.15,
        K_cereb=0.2,
        io_mode="sparse", io_firing_rate_hz=1.0,
        use_traj_plan=use_traj_plan,
        traj_speed_gain=1.2,
        traj_dt=0.020,
        vel_scale_min=0.10,
    )
    ctrl = MyoArmController(config=cfg, muscle_names=muscle_names)
    if CFC_MODEL_PATH.exists():
        ctrl.load_cfc(CFC_MODEL_PATH)

    results = []
    for seed in range(n_seeds):
        r = run_episode(env, ctrl, seed=seed, max_steps=max_steps, dt=dt)
        results.append(r)

    valid = [r for r in results if r.get("vel_peak_ratio") is not None
             and not np.isnan(r.get("vel_peak_ratio", float("nan")))]

    def smean(vals):
        v = [x for x in vals if x is not None and not np.isnan(x)]
        return float(np.mean(v)) if v else float("nan")

    agg = {
        "n_valid": len(valid),
        "n_solved": sum(r["solved"] for r in results),
        "solve_rate": sum(r["solved"] for r in results) / n_seeds,
        "tip_err_min_mm_mean":   smean([r["tip_err_min_mm"]   for r in valid]),
        "jerk_rms_mean":         smean([r.get("jerk_rms") for r in valid]),
        "vel_peak_ratio_mean":   smean([r.get("vel_peak_ratio") for r in valid]),
        "straightness_mean":     smean([r.get("straightness") for r in valid]),
        "movement_time_mean":    smean([r.get("movement_time") for r in valid]),
        "peak_speed_mean":       smean([r.get("peak_speed") for r in valid]),
    }
    return results, agg


def main() -> None:
    n_seeds   = 20
    max_steps = 600
    # myoArm: frame_skip=10 × mj_dt=0.002s = 0.020s 実制御周期
    dt        = 0.020

    print("=== Phase 1-4: 最小ジャーク軌跡追従制御（ベースライン比較） ===")
    print(f"env: myoArmReachFixed-v0  seeds: {n_seeds}  max_steps: {max_steps}  dt={dt}s")

    env = gym.make("myoArmReachFixed-v0")
    uw  = env.unwrapped
    env.reset(seed=0)
    m = uw.mj_model
    muscle_names = [m.actuator(i).name for i in range(m.nu)]

    # --- ベースライン (軌跡計画なし) ---
    print("\n[Condition A] Baseline (no trajectory planning)")
    t0 = time.time()
    res_base, agg_base = run_condition(env, muscle_names, use_traj_plan=False,
                                       n_seeds=n_seeds, max_steps=max_steps, dt=dt)
    t_base = time.time() - t0

    # --- 軌跡計画あり ---
    print("[Condition B] MinimumJerk trajectory planning")
    t0 = time.time()
    res_traj, agg_traj = run_condition(env, muscle_names, use_traj_plan=True,
                                       n_seeds=n_seeds, max_steps=max_steps, dt=dt)
    t_traj = time.time() - t0

    env.close()

    summary = {
        "phase": "1-4",
        "env": "myoArmReachFixed-v0",
        "dt_note": "dt=0.020s (frame_skip=10 × mj_dt=0.002s). Correct control interval.",
        "cfc_pretrained": CFC_MODEL_PATH.exists(),
        "elapsed_s": round(t_base + t_traj, 1),
        "condition_A_baseline": {
            "label": "Neural controller (no trajectory planning)",
            "elapsed_s": round(t_base, 1),
            "aggregate": agg_base,
        },
        "condition_B_trajplan": {
            "label": "Neural controller + MinimumJerk velocity scaling",
            "elapsed_s": round(t_traj, 1),
            "aggregate": agg_traj,
        },
        "human_reference": {
            "vel_peak_ratio": 0.50,
            "straightness":   0.85,
            "jerk_rms_note":  "~5-20 m/s³ at dt=0.020s",
        },
        "per_seed_baseline": res_base,
        "per_seed_trajplan":  res_traj,
    }

    out = RESULTS_DIR / "trajplan_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)

    print()
    print("=== Phase 1-4 結果比較 (myoArmReachFixed-v0, dt=0.020s) ===")
    print(f"{'Metric':<25} {'Baseline':>12} {'TrajPlan':>12} {'Human':>10}")
    print("-" * 62)
    metrics = [
        ("vel_peak_ratio",   "vel_peak_ratio_mean",   0.50),
        ("straightness",     "straightness_mean",     0.85),
        ("jerk_rms (m/s³)", "jerk_rms_mean",          5.0),
        ("min_err (mm)",     "tip_err_min_mm_mean",    None),
        ("peak_speed (m/s)", "peak_speed_mean",        1.0),
    ]
    for label, key, human in metrics:
        b = agg_base.get(key, float("nan"))
        t = agg_traj.get(key, float("nan"))
        h = f"~{human}" if human else "—"
        print(f"  {label:<23} {b:>12.3f} {t:>12.3f} {h:>10}")
    print(f"\n  solve_rate      : A={agg_base['n_solved']}/{n_seeds}  B={agg_traj['n_solved']}/{n_seeds}")
    print(f"  elapsed         : {t_base:.1f}s + {t_traj:.1f}s")
    print(f"  saved → {out}")


if __name__ == "__main__":
    main()
