"""
experiment_myo_p15_f10b_kcereb_sweep.py — F10b: K_cereb 高ゲイン掃引

F10 で K_cereb ∈ {0, 0.2, 0.5, 1.0} が全て bit-identical な結果。
事後分析で K_cereb=1.0 でも `tau_cereb ≈ 0.20 Nm`、J_act⁺ 経由の muscle
補正は ~0.04 (a_base ≈ 0.5 の 8% 程度) と推定。
ゲインを 1-3 桁上げて behavioral 変化が出るかを確認、ゲイン問題 (A1) か
経路問題 (A2) かを切り分ける。

設計:
  env       : myoArmReachRandom-v0 + deterministic_reset
  test seeds: 0..49 reachable subset (n=20)
  λ params  : c_λ=20, sg=1.2 (F9 best biological)
  CfC       : new_CfC (λ-EP 再訓練)
  conditions: λ-traj + new_CfC で K_cereb ∈ {0, 1, 10, 100, 500}

  指標: F9/F10 同様 + cerebellar correction の強さ (a_cereb_norm)
  期待:
    - K=10-100 で behavioral 差が出る (ゲイン問題) → tuning で解決
    - K=500 でも no change (経路問題) → A2 (λ 直接補正) が必要

出力:
  results/experiment_myo_p15/f10b_kcereb_sweep.json
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

RESULTS_DIR     = ROOT / "results" / "experiment_myo_p15"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CFC_LAMBDA_PATH = ROOT / "results" / "myo_cfc_data_lambda" / "cfc_model.pt"

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
    a_cereb_norms = []
    pred_err_norms = []
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
        a_cereb_norms.append(float(np.linalg.norm(info["a_cereb_delayed"])))

        obs, _, term, trunc, info_env = env.step(a_total)
        ctrl.update_cerebellum(np.array(uw.obs_dict["qpos"]), m, d)
        pred_err_norms.append(float(np.linalg.norm(ctrl.get_pred_error())))
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
        "a_cereb_norm_mean": float(np.mean(a_cereb_norms)) if a_cereb_norms else 0.0,
        "a_cereb_norm_max":  float(np.max(a_cereb_norms))  if a_cereb_norms else 0.0,
        "pred_err_mean": float(np.mean(pred_err_norms)) if pred_err_norms else 0.0,
        **km,
    }


def make_lambda_cfg(*, K_cereb: float) -> MyoArmConfig:
    return MyoArmConfig(
        Kp_ee=80.0, Kd_ee=15.0, Ki_ee=2.0, act_bias=0.15,
        K_cereb=K_cereb, K_ia=0.05, K_ib=0.03, K_ri=0.5,
        io_mode="sparse", io_firing_rate_hz=1.0,
        traj_speed_gain=1.2, traj_dt=DT, vel_scale_min=0.10,
        Kd_traj=50.0, use_traj_plan=False,
        control_mode="lambda_ep", c_lambda=20.0, lambda_offset=0.005,
        lambda_trajectory=True, lambda_traj_speed_gain=1.2,
    )


def run_condition(env, muscle_names, cfg, seeds):
    ctrl = MyoArmController(config=cfg, muscle_names=muscle_names)
    if CFC_LAMBDA_PATH.exists():
        ctrl.load_cfc(CFC_LAMBDA_PATH)
    results = [run_episode(env, ctrl, seed=s) for s in seeds]
    keys = ["tip_err_min_mm", "tip_err_final_mm",
            "vel_peak_ratio", "straightness", "jerk_rms", "peak_speed",
            "direction_error_deg", "progress_ratio",
            "a_cereb_norm_mean", "a_cereb_norm_max", "pred_err_mean"]
    def s(k):
        v = [r[k] for r in results
             if r.get(k) is not None and not np.isnan(r.get(k, float("nan")))]
        if not v: return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "n": len(v)}
    return {
        "n_solved": sum(r["solved"] for r in results),
        "stats": {k: s(k) for k in keys},
        "per_seed": results,
    }


def welch(a, b):
    a = [x for x in a if x is not None and not np.isnan(x)]
    b = [x for x in b if x is not None and not np.isnan(x)]
    if len(a) < 2 or len(b) < 2: return None
    t, p = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    d = (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0
    return {"t": float(t), "p": float(p), "d": float(d)}


def main():
    print("=== F10b: K_cereb 高ゲイン掃引 ===")
    env = gym.make("myoArmReachRandom-v0")
    deterministic_reset(env, 0)
    uw = env.unwrapped
    m = uw.mj_model
    muscle_names = [m.actuator(i).name for i in range(m.nu)]
    seeds = find_reachable_seeds(env)
    print(f"  test seeds: {seeds}")

    K_values = [0.0, 1.0, 10.0, 100.0, 500.0]
    results = {}
    t0 = time.time()
    for K in K_values:
        name = f"K={K}"
        agg = run_condition(env, muscle_names, make_lambda_cfg(K_cereb=K), seeds)
        results[name] = agg
        s = agg["stats"]
        print(f"\n  [{name:<10}] solve={agg['n_solved']:2d}/{len(seeds)}  "
              f"min_err={s['tip_err_min_mm']['mean']:5.1f}mm  "
              f"progress={s['progress_ratio']['mean']:+.3f}  "
              f"dir_err={s['direction_error_deg']['mean']:5.1f}°")
        print(f"               a_cereb_max={s['a_cereb_norm_max']['mean']:.3f}  "
              f"a_cereb_mean={s['a_cereb_norm_mean']['mean']:.4f}  "
              f"pred_err={s['pred_err_mean']['mean']:.3f}  "
              f"jerk={s['jerk_rms']['mean']:.0f}  peak_v={s['peak_speed']['mean']:.2f}")
    env.close()

    # 検定: 各 K vs K=0
    print("\n=== Welch's t-test: 各 K vs K=0 ===")
    base = results["K=0.0"]["per_seed"]
    test_results = {}
    for K in K_values[1:]:
        name = f"K={K}"
        per = results[name]["per_seed"]
        for key in ["direction_error_deg", "tip_err_min_mm", "progress_ratio", "tip_err_final_mm"]:
            t = welch([r[key] for r in per], [r[key] for r in base])
            test_results.setdefault(key, {})[name] = t
            if t:
                sig = "***" if t["p"]<0.001 else "**" if t["p"]<0.01 else "*" if t["p"]<0.05 else ""
                print(f"  {key:<22} {name:<10} t={t['t']:+6.2f}  p={t['p']:.4g}{sig:<4}  d={t['d']:+.2f}")

    summary = {
        "phase": "1-6 F10b",
        "purpose": "K_cereb 高ゲイン掃引で gain vs path 切り分け",
        "n_seeds": len(seeds), "seeds": seeds, "elapsed_s": round(time.time()-t0, 1),
        "K_values": K_values,
        "conditions": {n: {"n_solved": r["n_solved"], "stats": r["stats"]}
                       for n, r in results.items()},
        "stat_tests_vs_K0": test_results,
        "raw_per_seed": {n: r["per_seed"] for n, r in results.items()},
    }
    out = RESULTS_DIR / "f10b_kcereb_sweep.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print(f"\n  elapsed: {time.time()-t0:.1f}s, saved → {out}")


if __name__ == "__main__":
    main()
