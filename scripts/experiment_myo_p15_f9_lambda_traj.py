"""
experiment_myo_p15_f9_lambda_traj.py — F9: virtual trajectory λ-model
                                       (Feldman 1998 完全版理論)

F8 (静的 λ) で得た所見:
  - overshoot は構造的に減少 (c_λ=20 で progress 1.19→1.07, p=0.018, d=-0.79)
  - **しかし direction_error が有意悪化** (5.7°→9.9°, p<0.005)
  - 原因: 筋空間で活性化 → q_target に向かう自然なダイナミクスは task 空間で straight でない

F9 で試す: virtual trajectory hypothesis (Feldman 1998 review):
  下行性指令は静的な λ_target でなく **λ(t) の時系列**。
  λ(t) = λ_start + s(τ) × (λ_target - λ_start)、s(τ) は最小ジャーク補間。
  各時点で平衡点が滑らかに移動 → 軌跡が straight になり、kinematic
  invariance (Morasso 1981) を再現する仮説。

設計（事前宣言）:
  env       : myoArmReachRandom-v0 + deterministic_reset
  seeds     : 0..49 から reach_dist<0.85m の最初 20
  c_lambda  : 20.0 固定 (F8 best)
  traj_T    : dist × speed_gain / 0.5 [s] で自動計算 (clip [0.3, 2.5])

  conditions:
    1. neural endpoint_pd                      (baseline)
    2. neural λ static c=20                    (F8 best, virtual=False)
    3. neural λ traj c=20 speed_gain=0.8       (速い)
    4. neural λ traj c=20 speed_gain=1.2       (中庸)
    5. neural λ traj c=20 speed_gain=2.0       (遅い)
    6. neural λ traj c=10 speed_gain=1.2       (低 c_λ)
    7. pure λ traj c=20 speed_gain=1.2         (神経成分なし)

  指標      : 同 F8 + traj_progress, traj_T_actual
  検定      : 各 traj 条件 vs neural endpoint_pd, λ traj vs λ static
  期待      : direction_error が λ static より改善し、overshoot は維持

出力:
  results/experiment_myo_p15/f9_lambda_traj.json
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
    return {"jerk_rms": jerk_rms, "vel_peak_ratio": float(vpr),
            "straightness": float(straightness), "movement_time": float(T_actual),
            "peak_speed": float(np.max(movement_speed))}


def find_reachable_seeds(env: gym.Env) -> list[int]:
    out = []
    for s in SEED_POOL:
        deterministic_reset(env, s)
        od = env.unwrapped.obs_dict
        d = float(np.linalg.norm(np.array(od["reach_err"])))
        if d < MAX_REACH_M:
            out.append(s)
        if len(out) >= N_REACHABLE:
            break
    return out


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
    traj_T_set = None
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

        a_total, info = ctrl.step(q=q, dq=dq, reach_err=reach_err, tip_pos=tip_pos,
                                  muscle_vel=muscle_vel, muscle_force=muscle_force, m=m, d=d)
        if traj_T_set is None and ctrl._traj_T is not None:
            traj_T_set = float(ctrl._traj_T)

        obs, _, term, trunc, info_env = env.step(a_total)
        ctrl.update_cerebellum(np.array(uw.obs_dict["qpos"]), m, d)
        if info_env.get("solved", False):
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
        "traj_T_set":    traj_T_set,
        **km,
    }


def make_cfg(*, control_mode: str, c_lambda: float = 5.0,
             lambda_offset: float = 0.005,
             lambda_trajectory: bool = False,
             lambda_traj_speed_gain: float = 1.2,
             neural: bool = True) -> MyoArmConfig:
    base = dict(
        Kp_ee=80.0, Kd_ee=15.0, Ki_ee=2.0, act_bias=0.15,
        traj_speed_gain=1.2, traj_dt=DT, vel_scale_min=0.10,
        Kd_traj=50.0, use_traj_plan=False,
        control_mode=control_mode,
        c_lambda=c_lambda, lambda_offset=lambda_offset,
        lambda_trajectory=lambda_trajectory,
        lambda_traj_speed_gain=lambda_traj_speed_gain,
    )
    if neural:
        base.update(K_cereb=0.2, K_ia=0.05, K_ib=0.03, K_ri=0.5,
                    io_mode="sparse", io_firing_rate_hz=1.0)
    else:
        base.update(K_cereb=0.0, K_ia=0.0, K_ib=0.0, K_ri=0.0,
                    io_mode="sparse", io_firing_rate_hz=0.0)
    return MyoArmConfig(**base)


def run_condition(env, muscle_names, cfg: MyoArmConfig, seeds: list[int],
                  load_cfc: bool) -> dict:
    ctrl = MyoArmController(config=cfg, muscle_names=muscle_names)
    if load_cfc and CFC_MODEL_PATH.exists():
        ctrl.load_cfc(CFC_MODEL_PATH)
    results = [run_episode(env, ctrl, seed=s) for s in seeds]

    metric_keys = ["tip_err_min_mm", "tip_err_final_mm",
                   "vel_peak_ratio", "straightness", "jerk_rms",
                   "peak_speed", "movement_time",
                   "direction_error_deg", "progress_ratio",
                   "traj_T_set"]

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


def welch_test(a: list[float], b: list[float]) -> dict:
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


def main() -> None:
    print("=== Phase 1-6 F9: virtual trajectory λ-model (Feldman 完全版) ===")
    print(f"env: myoArmReachRandom-v0 + deterministic_reset  n={N_REACHABLE}")

    env = gym.make("myoArmReachRandom-v0")
    uw  = env.unwrapped
    deterministic_reset(env, 0)
    m = uw.mj_model
    muscle_names = [m.actuator(i).name for i in range(m.nu)]

    seeds = find_reachable_seeds(env)
    print(f"  reachable seeds: {seeds}")

    conditions = [
        ("neural endpoint_pd",
         make_cfg(control_mode="endpoint_pd"), True),
        ("neural λ static c=20",
         make_cfg(control_mode="lambda_ep", c_lambda=20.0,
                  lambda_trajectory=False), True),
        ("neural λ traj c=20 sg=0.8",
         make_cfg(control_mode="lambda_ep", c_lambda=20.0,
                  lambda_trajectory=True, lambda_traj_speed_gain=0.8), True),
        ("neural λ traj c=20 sg=1.2",
         make_cfg(control_mode="lambda_ep", c_lambda=20.0,
                  lambda_trajectory=True, lambda_traj_speed_gain=1.2), True),
        ("neural λ traj c=20 sg=2.0",
         make_cfg(control_mode="lambda_ep", c_lambda=20.0,
                  lambda_trajectory=True, lambda_traj_speed_gain=2.0), True),
        ("neural λ traj c=10 sg=1.2",
         make_cfg(control_mode="lambda_ep", c_lambda=10.0,
                  lambda_trajectory=True, lambda_traj_speed_gain=1.2), True),
        ("pure λ traj c=20 sg=1.2",
         make_cfg(control_mode="lambda_ep", c_lambda=20.0,
                  lambda_trajectory=True, lambda_traj_speed_gain=1.2,
                  neural=False), False),
    ]

    t0 = time.time()
    results = {}
    for name, cfg, load_cfc in conditions:
        agg = run_condition(env, muscle_names, cfg, seeds=seeds, load_cfc=load_cfc)
        results[name] = agg
        s = agg["stats"]
        T_str = f"T={s['traj_T_set']['mean']:.2f}s" if not np.isnan(s['traj_T_set']['mean']) else ""
        print(f"\n  [{name}]  {T_str}")
        print(f"    solve={agg['n_solved']:2d}/{len(seeds)}  "
              f"min_err={s['tip_err_min_mm']['mean']:5.1f}±{s['tip_err_min_mm']['std']:5.1f}mm  "
              f"final_err={s['tip_err_final_mm']['mean']:5.1f}mm  "
              f"progress={s['progress_ratio']['mean']:+.3f}±{s['progress_ratio']['std']:.3f}")
        print(f"    dir_err={s['direction_error_deg']['mean']:5.1f}°  "
              f"straight={s['straightness']['mean']:.3f}  "
              f"jerk={s['jerk_rms']['mean']:.0f}  peak_v={s['peak_speed']['mean']:.2f}m/s")

    env.close()
    elapsed = time.time() - t0

    # ── 統計検定 ──
    print("\n=== Welch's t-test: vs neural endpoint_pd ===")
    test_results_vs_pd = {}
    base_per_seed = results["neural endpoint_pd"]["per_seed"]
    for key in ["progress_ratio", "tip_err_min_mm", "direction_error_deg", "tip_err_final_mm"]:
        base_vals = [r[key] for r in base_per_seed]
        for name in [n for n, _, _ in conditions if n != "neural endpoint_pd"]:
            cond_vals = [r[key] for r in results[name]["per_seed"]]
            t = welch_test(cond_vals, base_vals)
            test_results_vs_pd.setdefault(key, {})[name] = t
            if t["p_value"] is not None:
                sig = ("***" if t["p_value"] < 0.001 else
                       "**"  if t["p_value"] < 0.01  else
                       "*"   if t["p_value"] < 0.05  else "")
                d = f"{t['cohens_d']:+.2f}" if t['cohens_d'] is not None else "n/a"
                print(f"  {key:<22} {name:<32} t={t['t_stat']:+6.2f}  p={t['p_value']:.4g}{sig:<4} d={d}")

    # ── λ traj sg=1.2 vs λ static ──
    print("\n=== Welch's t-test: λ traj (sg=1.2) vs λ static (両方 c=20) ===")
    test_results_traj_vs_static = {}
    static_per_seed = results["neural λ static c=20"]["per_seed"]
    traj_per_seed   = results["neural λ traj c=20 sg=1.2"]["per_seed"]
    for key in ["progress_ratio", "tip_err_min_mm", "direction_error_deg", "tip_err_final_mm",
                "straightness"]:
        a = [r[key] for r in traj_per_seed]
        b = [r[key] for r in static_per_seed]
        t = welch_test(a, b)
        test_results_traj_vs_static[key] = t
        if t["p_value"] is not None:
            sig = ("***" if t["p_value"] < 0.001 else
                   "**"  if t["p_value"] < 0.01  else
                   "*"   if t["p_value"] < 0.05  else "")
            d = f"{t['cohens_d']:+.2f}" if t['cohens_d'] is not None else "n/a"
            print(f"  {key:<22} traj vs static  t={t['t_stat']:+6.2f}  p={t['p_value']:.4g}{sig:<4} d={d}")

    summary = {
        "phase": "1-6 F9",
        "purpose": "virtual trajectory λ-model (Feldman 1998 完全版)",
        "env": "myoArmReachRandom-v0 + deterministic_reset",
        "dt": DT,
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
        "stat_tests_vs_endpoint_pd": test_results_vs_pd,
        "stat_tests_traj_vs_static": test_results_traj_vs_static,
        "raw_per_seed": {name: r["per_seed"] for name, r in results.items()},
    }

    out = RESULTS_DIR / "f9_lambda_traj.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)

    print(f"\n  elapsed: {elapsed:.1f}s, saved → {out}")


if __name__ == "__main__":
    main()
