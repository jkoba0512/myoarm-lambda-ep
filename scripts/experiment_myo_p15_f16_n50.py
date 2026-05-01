"""
experiment_myo_p15_f16_n50.py — F16: n=50 拡張で主結果の effect size 確定

F4-F15 まで n=20 で進めてきた。検出力的に d≥0.5 で約 80% で検出可能、d=0.3 では
underpowered。n=50 で d=0.3 の効果も p<0.05 で検出 (~70% power) → 主結果の堅牢
性を確認する。

設計:
  test seeds: pool 0..149 から reachable 50 を取得 (拡張)
  conditions: F4-F14 で重要だった 6 条件のみに絞る
    1. neural endpoint_pd (reference)
    2. λ-traj baseline (no visuomotor) — null condition
    3. F12 best (pure λ visuo)
    4. F12 best + reflexes (F14 で最 biological と判明)
    5. F12 best + reflexes + K_cereb=0.2 joint
    6. F12 best + reflexes + K_cereb_λ=0.5

  指標 : tip_err_min/final, dir_err, progress, vpr, straightness, peak_v, jerk
  検定 : 各条件 vs F12 best、各条件 vs endpoint_pd、F12+reflex vs F12 best

出力: results/experiment_myo_p15/f16_n50.json
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
CFC_LAMBDA_PATH = ROOT / "results" / "myo_cfc_data_lambda" / "cfc_model.pt"
CFC_OLD_PATH    = ROOT / "results" / "myo_cfc_data"        / "cfc_model.pt"

DT          = DEFAULT_DT
N_REACHABLE = 50  # 拡張


def compute_kinematics(positions: np.ndarray, dt: float = DT) -> dict:
    if len(positions) < 5:
        return {k: float("nan") for k in
                ["jerk_rms", "vel_peak_ratio", "straightness", "peak_speed",
                 "movement_time", "skewness"]}
    vel = np.diff(positions, axis=0) / dt
    speed = np.linalg.norm(vel, axis=1)
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    thresh = 0.02
    onset = next((i for i, s in enumerate(speed) if s > thresh), None)
    if onset is None:
        return {"jerk_rms": float("nan"), "vel_peak_ratio": float("nan"),
                "straightness": float("nan"), "peak_speed": float("nan"),
                "movement_time": float("nan"), "skewness": float("nan")}
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
    seg = positions[onset:offset+2]
    L_path = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))
    D = float(np.linalg.norm(seg[-1] - seg[0]))
    straightness = D / max(L_path, 1e-6)
    return {
        "jerk_rms": jerk_rms, "vel_peak_ratio": float(vpr),
        "straightness": float(straightness),
        "peak_speed": float(np.max(movement_speed)),
        "movement_time": float(T_actual), "skewness": skew,
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

    km = compute_kinematics(positions, dt=DT)
    return {
        "seed": seed, "solved": solved,
        "tip_err_min_mm": min(errs)*1000 if errs else float("nan"),
        "tip_err_final_mm": errs[-1]*1000 if errs else float("nan"),
        "direction_error_deg": direction_error_deg,
        "progress_ratio": progress_ratio,
        "target_dist_m": target_norm,
        **km,
    }


def make_cfg(*,
             control_mode: str = "lambda_ep",
             lambda_trajectory: bool = True,
             visuomotor: bool = True,
             reflexes: bool = False,
             K_cereb: float = 0.0,
             cereb_target: str = "joint",
             K_cereb_lambda: float = 0.0) -> MyoArmConfig:
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
        lambda_traj_speed_gain=1.2,
        visuomotor_feedback=visuomotor,
        visuomotor_period_steps=10,
        cereb_correction_target=cereb_target,
        K_cereb_lambda=K_cereb_lambda,
        traj_dt=DT, use_traj_plan=False,
    )


def run_condition(env, muscle_names, cfg, seeds, cfc_path):
    ctrl = MyoArmController(config=cfg, muscle_names=muscle_names)
    if cfc_path is not None and cfc_path.exists():
        ctrl.load_cfc(cfc_path)
    results = [run_episode(env, ctrl, seed=s) for s in seeds]
    keys = ["tip_err_min_mm", "tip_err_final_mm",
            "vel_peak_ratio", "straightness", "jerk_rms", "peak_speed",
            "movement_time", "skewness", "direction_error_deg", "progress_ratio"]
    return {
        "n_solved": sum(r["solved"] for r in results),
        "stats": stats_for_results(results, keys),
        "per_seed": results,
    }


def main():
    print(f"=== F16: n={N_REACHABLE} 拡張で主結果 effect size 確定 ===")
    env = gym.make("myoArmReachRandom-v0")
    deterministic_reset(env, 0)
    uw = env.unwrapped
    m = uw.mj_model
    muscle_names = [m.actuator(i).name for i in range(m.nu)]
    seeds = find_reachable_seeds(env, pool=range(150), n=N_REACHABLE)
    print(f"  test seeds (n={len(seeds)}): {seeds[:10]}...{seeds[-5:]}")

    conditions = [
        ("endpoint_pd",
         make_cfg(control_mode="endpoint_pd", lambda_trajectory=False,
                  visuomotor=False, reflexes=True, K_cereb=0.2),
         CFC_OLD_PATH),
        ("λ-traj baseline (no visuo)",
         make_cfg(visuomotor=False),  # 神経成分なし
         None),
        ("F12 best (pure λ visuo)",
         make_cfg(),  # default
         None),
        ("F12 best + reflexes",
         make_cfg(reflexes=True),
         None),
        ("F12 best + reflex + K_cereb=0.2 j",
         make_cfg(reflexes=True, K_cereb=0.2, cereb_target="joint"),
         CFC_LAMBDA_PATH),
        ("F12 best + reflex + K_cereb_λ=0.5",
         make_cfg(reflexes=True, cereb_target="lambda", K_cereb_lambda=0.5),
         CFC_LAMBDA_PATH),
    ]

    t0 = time.time()
    results = {}
    for name, cfg, cfc in conditions:
        agg = run_condition(env, muscle_names, cfg, seeds, cfc)
        results[name] = agg
        s = agg["stats"]
        print(f"\n  [{name}]")
        print(f"    solve={agg['n_solved']:2d}/{len(seeds)}  "
              f"min_err={s['tip_err_min_mm']['mean']:5.1f}±{s['tip_err_min_mm']['std']:5.1f}mm  "
              f"final_err={s['tip_err_final_mm']['mean']:5.1f}mm")
        print(f"    progress={s['progress_ratio']['mean']:+.3f}  "
              f"dir_err={s['direction_error_deg']['mean']:5.1f}°  "
              f"straight={s['straightness']['mean']:.3f}  vpr={s['vel_peak_ratio']['mean']:.3f}  "
              f"peak_v={s['peak_speed']['mean']:.2f}  jerk={s['jerk_rms']['mean']:.0f}")
    env.close()

    # 重要な比較
    print("\n=== Welch's t-test: 各条件 vs F12 best ===")
    base = results["F12 best (pure λ visuo)"]["per_seed"]
    test_results = {}
    metric_keys = ["tip_err_min_mm", "tip_err_final_mm", "direction_error_deg",
                   "vel_peak_ratio", "straightness", "peak_speed", "jerk_rms",
                   "progress_ratio"]
    for name in [n for n, _, _ in conditions if n != "F12 best (pure λ visuo)"]:
        per = results[name]["per_seed"]
        for key in metric_keys:
            t = welch_test([r[key] for r in per], [r[key] for r in base])
            test_results.setdefault(key, {})[name] = t
            if t["p_value"] is not None:
                d_str = f"{t['cohens_d']:+.2f}" if t['cohens_d'] is not None else "n/a"
                print(f"  {key:<22} {name:<35} t={t['t_stat']:+6.2f}  "
                      f"p={t['p_value']:.4g}{sig_marker(t['p_value']):<4} d={d_str}")

    # F12 best + reflexes vs endpoint_pd (best biological config の検定)
    print("\n=== F12 best + reflexes vs endpoint_pd ===")
    biolog = results["F12 best + reflexes"]["per_seed"]
    pdref  = results["endpoint_pd"]["per_seed"]
    for key in metric_keys:
        t = welch_test([r[key] for r in biolog], [r[key] for r in pdref])
        if t["p_value"] is not None:
            d_str = f"{t['cohens_d']:+.2f}" if t['cohens_d'] is not None else "n/a"
            print(f"  {key:<22} biolog vs pd  t={t['t_stat']:+6.2f}  "
                  f"p={t['p_value']:.4g}{sig_marker(t['p_value']):<4}  d={d_str}")

    summary = {
        "phase": "1-6 F16", "purpose": f"n={N_REACHABLE} expansion",
        "n_seeds": len(seeds), "seeds": seeds,
        "elapsed_s": round(time.time()-t0, 1),
        "conditions": {n: {"n_solved": r["n_solved"], "stats": r["stats"]}
                       for n, r in results.items()},
        "stat_tests_vs_F12_best": test_results,
        "raw_per_seed": {n: r["per_seed"] for n, r in results.items()},
    }
    out = RESULTS_DIR / "f16_n50.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print(f"\n  elapsed: {time.time()-t0:.1f}s, saved → {out}")


if __name__ == "__main__":
    main()
