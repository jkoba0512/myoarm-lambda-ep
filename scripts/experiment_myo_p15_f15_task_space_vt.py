"""
experiment_myo_p15_f15_task_space_vt.py — F15: task-space virtual trajectory

F14 で straightness 0.6-0.7 (ヒト 0.85+) と判明。原因仮説:
  λ-EP 既存版は **筋空間で min-jerk 補間** → task 空間で必ずしも straight でない
  ヒトは task 空間で min-jerk → IK で関節 → 筋 → straight な手先軌跡

F15: task 空間 min-jerk 軌跡を pre-compute、各 waypoint で IK→λ 配列。
  期待: approach straightness が 0.66 → 0.80+

設計:
  test seeds: 0..49 reachable subset (n=20)
  conditions:
    1. F12 best (pure λ visuo)              ← F12 main result
    2. + task_space_VT                      ← straightness 改善期待
    3. + task_space_VT + reflexes
    4. + task_space_VT + reflexes + visuomotor
    5. + reflexes (F14 best biological)
    6. endpoint_pd reference

  指標 : kinematic invariance (vpr, straight) + accuracy (min_err, dir_err)
  検定 : task-space あり vs なしで Welch's t-test、各成分 vs F12 best
  期待 : task_space で straightness が 0.7→0.85 に近づく

出力: results/experiment_myo_p15/f15_task_space_vt.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import myosuite  # noqa
import gymnasium as gym
import numpy as np
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myoarm.myo_controller import MyoArmController, MyoArmConfig
from myoarm.env_utils import deterministic_reset
from myoarm.exp_utils import (
    find_reachable_seeds, welch_test, stats_for_results, sig_marker, DEFAULT_DT,
)

RESULTS_DIR     = ROOT / "results" / "experiment_myo_p15"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CFC_OLD_PATH    = ROOT / "results" / "myo_cfc_data"        / "cfc_model.pt"

DT          = DEFAULT_DT
N_REACHABLE = 20


def compute_full_kinematics(positions: np.ndarray, dt: float = DT) -> dict:
    if len(positions) < 5:
        return {k: float("nan") for k in
                ["jerk_rms", "vel_peak_ratio", "straightness", "movement_time",
                 "peak_speed", "skewness", "approach_straightness"]}
    vel = np.diff(positions, axis=0) / dt
    speed = np.linalg.norm(vel, axis=1)
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    thresh = 0.02
    onset = next((i for i, s in enumerate(speed) if s > thresh), None)
    if onset is None:
        return {"jerk_rms": float("nan"), "vel_peak_ratio": float("nan"),
                "straightness": float("nan"), "movement_time": float("nan"),
                "peak_speed": float(np.max(speed)) if len(speed) > 0 else float("nan"),
                "skewness": float("nan"), "approach_straightness": float("nan")}
    offset = next((i for i in range(onset+5, len(speed)) if speed[i] < thresh),
                  len(speed)-1)
    movement_speed = speed[onset:offset+1]
    n_samples = len(movement_speed)
    T_actual = (offset - onset) * dt
    peak_idx = int(np.argmax(movement_speed))
    vpr = peak_idx / max(n_samples - 1, 1)

    skew = float(sp_stats.skew(movement_speed)) if n_samples >= 5 else float("nan")
    jerk_seg = jerk[onset:offset-1] if offset-1 < len(jerk) else jerk[onset:]
    jerk_rms = (float(np.sqrt(np.mean(np.sum(jerk_seg**2, axis=1))))
                if len(jerk_seg) > 0 else float("nan"))

    full_seg = positions[onset:offset+2]
    L_full = float(np.sum(np.linalg.norm(np.diff(full_seg, axis=0), axis=1)))
    D_full = float(np.linalg.norm(full_seg[-1] - full_seg[0]))
    straightness = D_full / max(L_full, 1e-6)

    # approach phase: peak velocity の地点までの軌跡 (target に向かう前半)
    approach_idx = onset + peak_idx
    approach_seg = positions[onset:approach_idx+2]
    if len(approach_seg) >= 2:
        L_app = float(np.sum(np.linalg.norm(np.diff(approach_seg, axis=0), axis=1)))
        D_app = float(np.linalg.norm(approach_seg[-1] - approach_seg[0]))
        approach_straightness = D_app / max(L_app, 1e-6)
    else:
        approach_straightness = float("nan")

    return {
        "jerk_rms":             jerk_rms,
        "vel_peak_ratio":       float(vpr),
        "straightness":         float(straightness),
        "approach_straightness":float(approach_straightness),
        "movement_time":        float(T_actual),
        "peak_speed":           float(np.max(movement_speed)),
        "skewness":             skew,
    }


def run_episode(env, ctrl, seed, max_steps=600) -> dict:
    obs, _ = deterministic_reset(env, seed)
    uw = env.unwrapped
    m, d = uw.mj_model, uw.mj_data
    ctrl.reset(); ctrl.initialize(m, d)

    od0 = uw.obs_dict
    tip0 = np.array(od0["tip_pos"])
    target = tip0 + np.array(od0["reach_err"])
    target_dir = target - tip0
    target_norm = float(np.linalg.norm(target_dir))
    target_dir_unit = target_dir / max(target_norm, 1e-9)

    positions, errs = [], []
    solved = False
    for _ in range(max_steps):
        od = uw.obs_dict
        positions.append(np.array(od["tip_pos"]))
        errs.append(float(np.linalg.norm(np.array(od["reach_err"]))))
        a_total, _ = ctrl.step(
            q=np.array(od["qpos"]), dq=np.array(od["qvel"]),
            reach_err=np.array(od["reach_err"]),
            tip_pos=np.array(od["tip_pos"]),
            muscle_vel=d.actuator_velocity.copy(),
            muscle_force=d.actuator_force.copy(),
            m=m, d=d,
        )
        obs, _, term, trunc, info = env.step(a_total)
        ctrl.update_cerebellum(np.array(uw.obs_dict["qpos"]), m, d)
        if info.get("solved", False): solved = True
        if term or trunc: break

    positions = np.array(positions)
    final_tip = positions[-1] if len(positions) else tip0
    travel = final_tip - tip0
    travel_norm = float(np.linalg.norm(travel))
    travel_unit = travel / max(travel_norm, 1e-9)
    cos_a = float(np.clip(np.dot(travel_unit, target_dir_unit), -1.0, 1.0))
    direction_error_deg = float(np.degrees(np.arccos(cos_a)))
    progress_ratio = float(np.dot(travel, target_dir_unit)) / max(target_norm, 1e-9)

    km = compute_full_kinematics(positions, dt=DT)
    return {
        "seed": seed, "solved": solved,
        "tip_err_min_mm":      min(errs)*1000 if errs else float("nan"),
        "tip_err_final_mm":    errs[-1]*1000 if errs else float("nan"),
        "direction_error_deg": direction_error_deg,
        "progress_ratio":      progress_ratio,
        "target_dist_m":       target_norm,
        **km,
    }


def make_cfg(*,
             control_mode: str = "lambda_ep",
             lambda_trajectory: bool = True,
             task_space_trajectory: bool = False,
             visuomotor: bool = True,
             reflexes: bool = False,
             K_cereb: float = 0.0) -> MyoArmConfig:
    return MyoArmConfig(
        Kp_ee=80.0, Kd_ee=15.0, Ki_ee=2.0, act_bias=0.15,
        K_cereb=K_cereb,
        K_ia=0.05 if reflexes else 0.0,
        K_ib=0.03 if reflexes else 0.0,
        K_ri=0.5  if reflexes else 0.0,
        io_mode="sparse",
        io_firing_rate_hz=1.0 if reflexes else 0.0,
        control_mode=control_mode,
        c_lambda=20.0, lambda_offset=0.005,
        lambda_trajectory=lambda_trajectory,
        task_space_trajectory=task_space_trajectory,
        lambda_traj_speed_gain=1.2,
        visuomotor_feedback=visuomotor,
        visuomotor_period_steps=10,
        traj_dt=DT, use_traj_plan=False,
    )


def run_condition(env, muscle_names, cfg, seeds, cfc_path):
    ctrl = MyoArmController(config=cfg, muscle_names=muscle_names)
    if cfc_path is not None and cfc_path.exists():
        ctrl.load_cfc(cfc_path)
    results = [run_episode(env, ctrl, seed=s) for s in seeds]
    keys = ["tip_err_min_mm", "tip_err_final_mm",
            "vel_peak_ratio", "straightness", "approach_straightness",
            "jerk_rms", "peak_speed", "skewness",
            "direction_error_deg", "progress_ratio"]
    return {
        "n_solved": sum(r["solved"] for r in results),
        "stats":    stats_for_results(results, keys),
        "per_seed": results,
    }


def main():
    print("=== F15: task-space virtual trajectory ===")
    env = gym.make("myoArmReachRandom-v0")
    deterministic_reset(env, 0)
    uw = env.unwrapped
    m = uw.mj_model
    muscle_names = [m.actuator(i).name for i in range(m.nu)]
    seeds = find_reachable_seeds(env, pool=range(50), n=N_REACHABLE)
    print(f"  test seeds: {seeds}")

    conditions = [
        ("F12 best (musc-VT)",
         make_cfg(),  # default = task_space=False, visuomotor=True, no reflex
         None),
        ("musc-VT + reflexes",
         make_cfg(reflexes=True),
         None),
        ("task-VT (no reflex)",
         make_cfg(task_space_trajectory=True),
         None),
        ("task-VT + reflexes",
         make_cfg(task_space_trajectory=True, reflexes=True),
         None),
        ("task-VT + reflexes − visuomotor",
         make_cfg(task_space_trajectory=True, reflexes=True, visuomotor=False),
         None),
        ("endpoint_pd reference",
         make_cfg(control_mode="endpoint_pd", lambda_trajectory=False,
                  visuomotor=False, reflexes=True, K_cereb=0.2),
         CFC_OLD_PATH),
    ]

    t0 = time.time()
    results = {}
    for name, cfg, cfc in conditions:
        agg = run_condition(env, muscle_names, cfg, seeds, cfc)
        results[name] = agg
        s = agg["stats"]
        print(f"\n  [{name}]")
        print(f"    solve={agg['n_solved']:2d}/{len(seeds)}  "
              f"min_err={s['tip_err_min_mm']['mean']:5.1f}mm  "
              f"final_err={s['tip_err_final_mm']['mean']:5.1f}mm  "
              f"dir_err={s['direction_error_deg']['mean']:5.1f}°")
        print(f"    straight_full={s['straightness']['mean']:.3f}  "
              f"straight_appr={s['approach_straightness']['mean']:.3f}  "
              f"vpr={s['vel_peak_ratio']['mean']:.3f}  "
              f"peak_v={s['peak_speed']['mean']:.2f}m/s")
    env.close()

    # vs F12 best (musc-VT)
    print("\n=== Welch's t-test: vs F12 best (musc-VT) ===")
    base_per = results["F12 best (musc-VT)"]["per_seed"]
    test_results = {}
    for name in [n for n, _, _ in conditions[1:]]:
        cond_per = results[name]["per_seed"]
        for key in ["straightness", "approach_straightness",
                    "vel_peak_ratio", "tip_err_min_mm", "direction_error_deg",
                    "peak_speed"]:
            t = welch_test([r[key] for r in cond_per], [r[key] for r in base_per])
            test_results.setdefault(key, {})[name] = t
            if t["p_value"] is not None:
                d = f"{t['cohens_d']:+.2f}" if t['cohens_d'] is not None else "n/a"
                print(f"  {key:<22} {name:<32} t={t['t_stat']:+6.2f}  "
                      f"p={t['p_value']:.4g}{sig_marker(t['p_value']):<4} d={d}")

    # task-VT vs musc-VT (一番興味ある対比)
    print("\n=== task-VT vs musc-VT (両方 +reflexes ありの比較) ===")
    task_per = results["task-VT + reflexes"]["per_seed"]
    musc_per = results["musc-VT + reflexes"]["per_seed"]
    for key in ["straightness", "approach_straightness", "tip_err_min_mm",
                "direction_error_deg", "vel_peak_ratio"]:
        t = welch_test([r[key] for r in task_per], [r[key] for r in musc_per])
        if t["p_value"] is not None:
            d = f"{t['cohens_d']:+.2f}" if t['cohens_d'] is not None else "n/a"
            print(f"  {key:<22} task vs musc  t={t['t_stat']:+6.2f}  "
                  f"p={t['p_value']:.4g}{sig_marker(t['p_value']):<4}  d={d}")

    summary = {
        "phase": "1-6 F15", "purpose": "task-space virtual trajectory",
        "n_seeds": len(seeds), "seeds": seeds,
        "elapsed_s": round(time.time()-t0, 1),
        "conditions": {n: {"n_solved": r["n_solved"], "stats": r["stats"]}
                       for n, r in results.items()},
        "stat_tests_vs_F12_best": test_results,
        "raw_per_seed": {n: r["per_seed"] for n, r in results.items()},
    }
    out = RESULTS_DIR / "f15_task_space_vt.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print(f"\n  elapsed: {time.time()-t0:.1f}s, saved → {out}")


if __name__ == "__main__":
    main()
