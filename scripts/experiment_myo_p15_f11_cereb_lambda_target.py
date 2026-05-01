"""
experiment_myo_p15_f11_cereb_lambda_target.py — F11: A2 λ-space cerebellar correction

F10/F10b で「joint 空間 → J_act⁺ → muscle 活性化」経路は λ-EP と整合せず、
ゲインを上げると数値発散することを確認。

A2 の中身:
  CfC の delta_q_err (joint 空間, rad) を moment_arm 経由で muscle 空間 (m) に
  変換し、**λ_target を直接補正**する。これは Kawato/Wolpert/Ito の小脳逆モデル
  理論で「小脳出力が下行性指令 (λ) を直接修正する」描像と整合。

  Δλ = -K_cereb_lambda × R @ delta_q_err  (R = dL/dq, moment_arm)
  λ_eff = λ + Δλ_delayed  (cereb_delay_steps で遅延)
  stretch = max(L_now - λ_eff, 0) → activation

  動機:
    - delta_q_err ~ 0.2 rad, R ~ 0.05 m/rad → R@e ~ 10mm
    - lambda_offset=5mm, stretch ~30mm と同じ桁 → 直接 stretch を変調できる
    - J_act⁺ 経由のような不適切な intermediate transformation を避ける

設計:
  env       : myoArmReachRandom-v0 + deterministic_reset
  test seeds: 0..49 reachable subset (n=20)
  base      : λ-traj c=20 sg=1.2 (F9 best biological) + new_CfC
  conditions:
    1. λ-traj joint K=0      (CfC なし、baseline)
    2. λ-traj joint K=0.2    (F10 既存経路、効果なし参照)
    3. λ-traj lambda K_cl=0.3
    4. λ-traj lambda K_cl=1.0
    5. λ-traj lambda K_cl=3.0
    6. λ-traj lambda K_cl=10.0

  指標 : F9/F10 同様 + lambda_cereb_norm
  検定 : 各 lambda 条件 vs joint K=0 baseline
  期待 : direction_error が 9.9° → 6° 付近に低下、min_err 改善

出力:
  results/experiment_myo_p15/f11_cereb_lambda_target.json
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
    lambda_cereb_norms = []
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
        lambda_cereb_norms.append(info.get("lambda_cereb_norm", 0.0))

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
        "lambda_cereb_norm_max":  float(np.max(lambda_cereb_norms))  if lambda_cereb_norms else 0.0,
        "lambda_cereb_norm_mean": float(np.mean(lambda_cereb_norms)) if lambda_cereb_norms else 0.0,
        "pred_err_mean": float(np.mean(pred_err_norms)) if pred_err_norms else 0.0,
        **km,
    }


def make_cfg(*, target: str, K_cereb: float = 0.0, K_cereb_lambda: float = 0.0) -> MyoArmConfig:
    return MyoArmConfig(
        Kp_ee=80.0, Kd_ee=15.0, Ki_ee=2.0, act_bias=0.15,
        K_cereb=K_cereb, K_ia=0.05, K_ib=0.03, K_ri=0.5,
        io_mode="sparse", io_firing_rate_hz=1.0,
        traj_speed_gain=1.2, traj_dt=DT, vel_scale_min=0.10,
        Kd_traj=50.0, use_traj_plan=False,
        control_mode="lambda_ep", c_lambda=20.0, lambda_offset=0.005,
        lambda_trajectory=True, lambda_traj_speed_gain=1.2,
        cereb_correction_target=target, K_cereb_lambda=K_cereb_lambda,
    )


def run_condition(env, muscle_names, cfg, seeds, load_cfc=True):
    ctrl = MyoArmController(config=cfg, muscle_names=muscle_names)
    if load_cfc and CFC_LAMBDA_PATH.exists():
        ctrl.load_cfc(CFC_LAMBDA_PATH)
    results = [run_episode(env, ctrl, seed=s) for s in seeds]
    keys = ["tip_err_min_mm", "tip_err_final_mm",
            "vel_peak_ratio", "straightness", "jerk_rms", "peak_speed",
            "direction_error_deg", "progress_ratio",
            "lambda_cereb_norm_max", "lambda_cereb_norm_mean", "pred_err_mean"]
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
    print("=== F11: A2 λ-space cerebellar correction ===")
    env = gym.make("myoArmReachRandom-v0")
    deterministic_reset(env, 0)
    uw = env.unwrapped
    m = uw.mj_model
    muscle_names = [m.actuator(i).name for i in range(m.nu)]
    seeds = find_reachable_seeds(env)
    print(f"  test seeds: {seeds}")

    conditions = [
        ("joint K=0       ",        make_cfg(target="joint",  K_cereb=0.0)),
        ("joint K=0.2     ",        make_cfg(target="joint",  K_cereb=0.2)),
        ("lambda K_cl=0.3 ",        make_cfg(target="lambda", K_cereb_lambda=0.3)),
        ("lambda K_cl=1.0 ",        make_cfg(target="lambda", K_cereb_lambda=1.0)),
        ("lambda K_cl=3.0 ",        make_cfg(target="lambda", K_cereb_lambda=3.0)),
        ("lambda K_cl=10.0",        make_cfg(target="lambda", K_cereb_lambda=10.0)),
    ]

    t0 = time.time()
    results = {}
    for name, cfg in conditions:
        agg = run_condition(env, muscle_names, cfg, seeds)
        results[name] = agg
        s = agg["stats"]
        print(f"\n  [{name}] solve={agg['n_solved']:2d}/{len(seeds)}  "
              f"min_err={s['tip_err_min_mm']['mean']:5.1f}±{s['tip_err_min_mm']['std']:4.1f}mm  "
              f"final_err={s['tip_err_final_mm']['mean']:5.1f}mm  "
              f"progress={s['progress_ratio']['mean']:+.3f}±{s['progress_ratio']['std']:.3f}")
        print(f"                   dir_err={s['direction_error_deg']['mean']:5.1f}°  "
              f"straight={s['straightness']['mean']:.3f}  "
              f"λ_cereb_max={s['lambda_cereb_norm_max']['mean']:.4f}  "
              f"jerk={s['jerk_rms']['mean']:.0f}  peak_v={s['peak_speed']['mean']:.2f}")
    env.close()

    print("\n=== Welch's t-test: 各 lambda K vs joint K=0 ===")
    base = results["joint K=0       "]["per_seed"]
    test_results = {}
    for name in [n for n, _ in conditions if n != "joint K=0       "]:
        per = results[name]["per_seed"]
        for key in ["direction_error_deg", "tip_err_min_mm", "progress_ratio", "tip_err_final_mm"]:
            t = welch([r[key] for r in per], [r[key] for r in base])
            test_results.setdefault(key, {})[name] = t
            if t:
                sig = "***" if t["p"]<0.001 else "**" if t["p"]<0.01 else "*" if t["p"]<0.05 else ""
                print(f"  {key:<22} {name} t={t['t']:+6.2f}  p={t['p']:.4g}{sig:<4}  d={t['d']:+.2f}")

    summary = {
        "phase": "1-6 F11",
        "purpose": "A2 λ-space cerebellar correction (Δλ via moment arm)",
        "n_seeds": len(seeds), "seeds": seeds, "elapsed_s": round(time.time()-t0, 1),
        "conditions": {n: {"n_solved": r["n_solved"], "stats": r["stats"]}
                       for n, r in results.items()},
        "stat_tests_vs_joint_K0": test_results,
        "raw_per_seed": {n: r["per_seed"] for n, r in results.items()},
    }
    out = RESULTS_DIR / "f11_cereb_lambda_target.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print(f"\n  elapsed: {time.time()-t0:.1f}s, saved → {out}")


if __name__ == "__main__":
    main()
