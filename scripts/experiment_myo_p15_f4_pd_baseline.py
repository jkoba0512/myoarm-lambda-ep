"""
experiment_myo_p15_f4_pd_baseline.py — F4: 純PDをRandom env baselineとして再評価。

Phase 1-5 E1 (diagnose_random) で純PDが神経制御より良い (124mm vs 294mm) と
判明。論文の主張を「神経成分が改善」から「Fixed では改善するが Random では純PDが
良い → structural regularization (筋シナジー) が必要」に再フレーミングするため、
純PDの Random 性能を完全な kinematics 込みで確立し、神経 default と統計検定する。

設計（事前宣言）:
  env       : myoArmReachRandom-v0
  seeds     : reach_dist < 0.85m を満たす 0..49 から最初の 20 シード
              (diagnose_random / random_validation と同一プール)
  条件      :
              1. pure_PD                : K_cereb=0, reflex/RI=0, IO=0, no traj_plan
              2. pure_PD + vel_scale    : 上 + use_traj_plan=True (traj_mode=vel_scale)
              3. pure_PD + feedforward  : 上 + feedforward (K_ff=4, Kp_t=20  Fixed env best)
              4. neural_default (P1-1)  : 神経成分 default (K_cereb=0.2 ほか, 比較用)
  指標      : tip_err_min/final_mm, vel_peak_ratio, straightness, jerk_rms,
              peak_speed, movement_time, direction_error_deg, progress_ratio
  検定      : 各 pure_PD 条件 vs neural_default で Welch's t-test
              主要指標 = tip_err_min_mm  (Cohen's d も併記)
  α         : 0.05

出力:
  results/experiment_myo_p15/f4_pd_baseline.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import myosuite  # noqa
import gymnasium as gym
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myoarm.myo_controller import MyoArmController, MyoArmConfig
from myoarm.env_utils import deterministic_reset

RESULTS_DIR    = ROOT / "results" / "experiment_myo_p15"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CFC_MODEL_PATH = ROOT / "results" / "myo_cfc_data" / "cfc_model.pt"

DT          = 0.020
MAX_REACH_M = 0.85
N_REACHABLE = 20
SEED_POOL   = list(range(50))


# ──────────────────────────────────────────────────────────────────────
# 運動学 (random_validation と同一)
# ──────────────────────────────────────────────────────────────────────

def compute_kinematics(positions: np.ndarray, dt: float = DT) -> dict:
    if len(positions) < 5:
        return {k: float("nan") for k in
                ["jerk_rms", "vel_peak_ratio", "straightness", "movement_time", "peak_speed"]}
    vel   = np.diff(positions, axis=0) / dt
    speed = np.linalg.norm(vel, axis=1)
    acc   = np.diff(vel, axis=0) / dt
    jerk  = np.diff(acc, axis=0) / dt
    thresh = 0.02
    onset = next((i for i, s in enumerate(speed) if s > thresh), None)
    if onset is None:
        return {"jerk_rms": float("nan"), "vel_peak_ratio": float("nan"),
                "straightness": float("nan"), "movement_time": float("nan"),
                "peak_speed": float(np.max(speed)) if len(speed) > 0 else float("nan")}
    offset = next((i for i in range(onset+5, len(speed)) if speed[i] < thresh), len(speed)-1)
    movement_speed = speed[onset:offset+1]
    T_actual = (offset - onset) * dt
    peak_idx = int(np.argmax(movement_speed))
    vpr = peak_idx / max(len(movement_speed) - 1, 1)
    jerk_seg = jerk[onset:offset-1] if offset-1 < len(jerk) else jerk[onset:]
    jerk_rms = float(np.sqrt(np.mean(np.sum(jerk_seg**2, axis=1)))) if len(jerk_seg) > 0 else float("nan")
    seg = positions[onset:offset+2]
    L_path = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))
    D = float(np.linalg.norm(seg[-1] - seg[0]))
    straightness = D / max(L_path, 1e-6)
    return {
        "jerk_rms": jerk_rms,
        "vel_peak_ratio": float(vpr),
        "straightness": float(straightness),
        "movement_time": float(T_actual),
        "peak_speed": float(np.max(movement_speed)),
    }


# ──────────────────────────────────────────────────────────────────────
# シード収集
# ──────────────────────────────────────────────────────────────────────

def find_reachable_seeds(env: gym.Env, pool=SEED_POOL, n=N_REACHABLE) -> list[int]:
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


# ──────────────────────────────────────────────────────────────────────
# 1 エピソード
# ──────────────────────────────────────────────────────────────────────

def run_episode(env, ctrl, seed, max_steps=600) -> dict:
    obs, _ = deterministic_reset(env, seed)
    uw = env.unwrapped
    m, d = uw.mj_model, uw.mj_data
    ctrl.reset(); ctrl.initialize(m, d)

    od0 = uw.obs_dict
    tip0   = np.array(od0["tip_pos"])
    target = tip0 + np.array(od0["reach_err"])
    target_dir = target - tip0
    target_norm = float(np.linalg.norm(target_dir))
    target_dir_unit = target_dir / max(target_norm, 1e-9)

    positions = []
    errs = []
    solved = False
    for step in range(max_steps):
        od = uw.obs_dict
        q, dq      = np.array(od["qpos"]), np.array(od["qvel"])
        reach_err  = np.array(od["reach_err"])
        tip_pos    = np.array(od["tip_pos"])
        muscle_vel   = d.actuator_velocity.copy()
        muscle_force = d.actuator_force.copy()
        positions.append(tip_pos.copy())
        errs.append(float(np.linalg.norm(reach_err)))

        a_total, _ = ctrl.step(q=q, dq=dq, reach_err=reach_err, tip_pos=tip_pos,
                               muscle_vel=muscle_vel, muscle_force=muscle_force, m=m, d=d)
        obs, _, term, trunc, info = env.step(a_total)
        ctrl.update_cerebellum(np.array(uw.obs_dict["qpos"]), m, d)
        if info.get("solved", False):
            solved = True
        if term or trunc:
            break

    positions = np.array(positions)
    final_tip = positions[-1] if len(positions) else tip0
    travel = final_tip - tip0
    travel_norm = float(np.linalg.norm(travel))
    travel_unit = travel / max(travel_norm, 1e-9)
    cos_angle = float(np.clip(np.dot(travel_unit, target_dir_unit), -1.0, 1.0))
    direction_error_deg = float(np.degrees(np.arccos(cos_angle)))
    progress_m = float(np.dot(travel, target_dir_unit))
    progress_ratio = progress_m / max(target_norm, 1e-9)

    km = compute_kinematics(positions, dt=DT)
    return {
        "seed": seed, "solved": solved,
        "tip_err_min_mm":   min(errs)*1000 if errs else float("nan"),
        "tip_err_final_mm": errs[-1]*1000 if errs else float("nan"),
        "direction_error_deg": direction_error_deg,
        "progress_ratio": progress_ratio,
        "target_dist_m": target_norm,
        **km,
    }


# ──────────────────────────────────────────────────────────────────────
# 1 条件まとめ
# ──────────────────────────────────────────────────────────────────────

def run_condition(env, muscle_names, cfg: MyoArmConfig, seeds: list[int],
                  load_cfc: bool = True) -> dict:
    ctrl = MyoArmController(config=cfg, muscle_names=muscle_names)
    if load_cfc and CFC_MODEL_PATH.exists():
        ctrl.load_cfc(CFC_MODEL_PATH)
    results = [run_episode(env, ctrl, seed=s) for s in seeds]

    metric_keys = ["tip_err_min_mm", "tip_err_final_mm",
                   "vel_peak_ratio", "straightness", "jerk_rms",
                   "peak_speed", "movement_time",
                   "direction_error_deg", "progress_ratio"]

    def stats_for(key):
        vals = [r[key] for r in results
                if r.get(key) is not None and not np.isnan(r.get(key, float("nan")))]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {"mean": float(np.mean(vals)),
                "std":  float(np.std(vals, ddof=1)),
                "n":    len(vals)}

    n_solved = sum(r["solved"] for r in results)
    return {
        "n_solved": n_solved,
        "solve_rate": n_solved / len(seeds),
        "stats": {k: stats_for(k) for k in metric_keys},
        "per_seed": results,
    }


# ──────────────────────────────────────────────────────────────────────
# 統計検定
# ──────────────────────────────────────────────────────────────────────

def welch_test(a: list[float], b: list[float]) -> dict:
    """a vs b で Welch's t-test と Cohen's d を計算 (lower-is-better 主義の指標を想定)。"""
    a = [x for x in a if x is not None and not np.isnan(x)]
    b = [x for x in b if x is not None and not np.isnan(x)]
    if len(a) < 2 or len(b) < 2:
        return {"t_stat": None, "p_value": None, "cohens_d": None,
                "significant": False, "n_a": len(a), "n_b": len(b)}
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    cohens_d = (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else float("nan")
    return {"t_stat": float(t_stat), "p_value": float(p_value),
            "cohens_d": float(cohens_d) if not np.isnan(cohens_d) else None,
            "significant": bool(p_value < 0.05),
            "n_a": len(a), "n_b": len(b)}


# ──────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Phase 1-5 F4: 純PDをRandom env baselineとして再評価 ===")
    print(f"env: myoArmReachRandom-v0  reachable_threshold={MAX_REACH_M}m  n={N_REACHABLE}")

    env = gym.make("myoArmReachRandom-v0")
    uw  = env.unwrapped
    env.reset(seed=0)
    m = uw.mj_model
    muscle_names = [m.actuator(i).name for i in range(m.nu)]

    seeds = find_reachable_seeds(env)
    print(f"  reachable seeds: {seeds}")

    # ── 4 条件 ──
    pd_kwargs = dict(
        Kp_ee=80.0, Kd_ee=15.0, Ki_ee=2.0, act_bias=0.15,
        K_cereb=0.0, K_ia=0.0, K_ib=0.0, K_ri=0.0,
        io_mode="sparse", io_firing_rate_hz=0.0,
        traj_speed_gain=1.2, traj_dt=DT, vel_scale_min=0.10,
        Kd_traj=50.0,
    )
    neural_kwargs = dict(
        Kp_ee=80.0, Kd_ee=15.0, Ki_ee=2.0, act_bias=0.15,
        K_cereb=0.2,  # 神経成分有効
        io_mode="sparse", io_firing_rate_hz=1.0,
        traj_speed_gain=1.2, traj_dt=DT, vel_scale_min=0.10,
        Kd_traj=50.0,
    )
    conditions = [
        ("pure_PD",
         MyoArmConfig(**pd_kwargs, use_traj_plan=False),
         False),  # CfC 不要 (K_cereb=0)
        ("pure_PD + vel_scale",
         MyoArmConfig(**pd_kwargs, use_traj_plan=True, traj_mode="vel_scale"),
         False),
        ("pure_PD + feedforward",
         MyoArmConfig(**pd_kwargs, use_traj_plan=True, traj_mode="feedforward",
                      K_ff=4.0, Kp_traj=20.0),
         False),
        ("neural_default",
         MyoArmConfig(**neural_kwargs, use_traj_plan=False),
         True),
    ]

    t0 = time.time()
    results = {}
    for name, cfg, load_cfc in conditions:
        agg = run_condition(env, muscle_names, cfg, seeds=seeds, load_cfc=load_cfc)
        results[name] = agg
        s = agg["stats"]
        print(f"\n  [{name}]")
        print(f"    solve={agg['n_solved']:2d}/{len(seeds)}  "
              f"min_err={s['tip_err_min_mm']['mean']:5.1f}±{s['tip_err_min_mm']['std']:5.1f}mm  "
              f"progress={s['progress_ratio']['mean']:+.2f}±{s['progress_ratio']['std']:.2f}  "
              f"dir_err={s['direction_error_deg']['mean']:5.1f}°")
        print(f"    vpr={s['vel_peak_ratio']['mean']:.3f}±{s['vel_peak_ratio']['std']:.3f}  "
              f"straight={s['straightness']['mean']:.3f}±{s['straightness']['std']:.3f}  "
              f"jerk={s['jerk_rms']['mean']:.0f}±{s['jerk_rms']['std']:.0f}  "
              f"peak_v={s['peak_speed']['mean']:.2f}m/s")

    env.close()
    elapsed = time.time() - t0

    # ── 統計検定: 各 PD 条件 vs neural_default ──
    print("\n=== Welch's t-test: 各 pure_PD 条件 vs neural_default ===")
    test_results = {}
    nd_per_seed = results["neural_default"]["per_seed"]
    for key in ["tip_err_min_mm", "progress_ratio", "direction_error_deg"]:
        nd_vals = [r[key] for r in nd_per_seed]
        for name in ["pure_PD", "pure_PD + vel_scale", "pure_PD + feedforward"]:
            cond_vals = [r[key] for r in results[name]["per_seed"]]
            t = welch_test(cond_vals, nd_vals)
            test_results.setdefault(key, {})[name] = t
            if t["p_value"] is not None:
                sig = ("***" if t["p_value"] < 0.001 else
                       "**"  if t["p_value"] < 0.01  else
                       "*"   if t["p_value"] < 0.05  else "")
                d = f"{t['cohens_d']:+.2f}" if t['cohens_d'] is not None else "n/a"
                print(f"  {key:<22} {name:<25} t={t['t_stat']:+6.2f}  "
                      f"p={t['p_value']:.4g}{sig:<4}  d={d}")
            else:
                print(f"  {key:<22} {name:<25} (insufficient data)")

    summary = {
        "phase": "1-5 F4",
        "purpose": "純PDをRandom env baseline化 + neural_default との統計検定",
        "env": "myoArmReachRandom-v0",
        "dt": DT,
        "reachable_threshold_m": MAX_REACH_M,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "elapsed_s": round(elapsed, 1),
        "conditions": {
            name: {
                "n_solved":   r["n_solved"],
                "solve_rate": r["solve_rate"],
                "stats":      r["stats"],
            }
            for name, r in results.items()
        },
        "stat_tests_vs_neural_default": test_results,
        "raw_per_seed": {name: r["per_seed"] for name, r in results.items()},
    }

    out = RESULTS_DIR / "f4_pd_baseline.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)

    print(f"\n  elapsed: {elapsed:.1f}s, saved → {out}")


if __name__ == "__main__":
    main()
