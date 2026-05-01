"""
experiment_myo_p15_f12_visuomotor.py — F12: B3 visuomotor feedback loop

F8-F11 で確立:
  - λ-EP virtual trajectory は biological plausibility (smooth, peak_v ~1.8m/s) を実現
  - direction_error 9.9° と final_err 150mm の精度ギャップは小脳補正 (joint or λ) では解けず
  - 真の機構は視覚運動フィードバック (Saunders & Knill 2003)

F12 で試す: tip_pos の周期的観測から IK を再実行し λ_target を更新する
  視覚運動フィードバック loop。ヒト視覚運動 latency ~100-200ms に相当。

  実装 (周期 N steps):
    if step_count % N == 0:
        λ_start_new = 現在の λ
        λ_target_new = IK(target_pos from current q)
        traj_t = 0.0
        traj_T = 残距離 × speed_gain / 0.5

  神経生理対応:
    - 視覚野 → PPC で tip と target を観測
    - 運動皮質が IK を再計算し λ 指令を更新
    - 100-200ms 遅延を period_steps × dt で表現

設計:
  env       : myoArmReachRandom-v0 + deterministic_reset
  test seeds: 0..49 reachable subset (n=20)
  base      : λ-traj c=20 sg=1.2 + new_CfC (cereb_target="lambda" K_cl=0)
              CfC は無効 (F11 で副作用大)
  conditions:
    1. neural endpoint_pd                 (参考)
    2. λ-traj baseline (no visuomotor)    (F9 と同じ)
    3. λ-traj visuo period=5  (100ms)
    4. λ-traj visuo period=10 (200ms, ヒト典型)
    5. λ-traj visuo period=25 (500ms, 遅い)
    6. λ-traj visuo period=50 (1s,    かなり遅い)
    7. pure λ-traj visuo period=10 (神経成分なし)

  指標 : F11 同様 + IK 呼び出し回数
  検定 : 各 visuomotor 条件 vs λ-traj baseline で Welch's t-test
  期待 : direction_error と final_err が大幅に改善、min_err も改善

出力:
  results/experiment_myo_p15/f12_visuomotor.json
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
CFC_OLD_PATH    = ROOT / "results" / "myo_cfc_data"        / "cfc_model.pt"

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
    ik_count = 0
    ik_residuals = []
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
        if info.get("ik_info") is not None:
            ik_count += 1
            ik_residuals.append(info["ik_info"]["residual_mm"])

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
        "ik_count": ik_count,
        "ik_residual_mm_max": float(np.max(ik_residuals)) if ik_residuals else 0.0,
        **km,
    }


def make_lambda_cfg(*, visuomotor: bool = False, period: int = 10,
                    neural: bool = True) -> MyoArmConfig:
    base = dict(
        Kp_ee=80.0, Kd_ee=15.0, Ki_ee=2.0, act_bias=0.15,
        traj_speed_gain=1.2, traj_dt=DT, vel_scale_min=0.10,
        Kd_traj=50.0, use_traj_plan=False,
        control_mode="lambda_ep", c_lambda=20.0, lambda_offset=0.005,
        lambda_trajectory=True, lambda_traj_speed_gain=1.2,
        visuomotor_feedback=visuomotor, visuomotor_period_steps=period,
    )
    if neural:
        base.update(K_cereb=0.0, K_ia=0.05, K_ib=0.03, K_ri=0.5,
                    io_mode="sparse", io_firing_rate_hz=1.0)
    else:
        base.update(K_cereb=0.0, K_ia=0.0, K_ib=0.0, K_ri=0.0,
                    io_mode="sparse", io_firing_rate_hz=0.0)
    return MyoArmConfig(**base)


def make_pd_cfg() -> MyoArmConfig:
    return MyoArmConfig(
        Kp_ee=80.0, Kd_ee=15.0, Ki_ee=2.0, act_bias=0.15,
        K_cereb=0.2, K_ia=0.05, K_ib=0.03, K_ri=0.5,
        io_mode="sparse", io_firing_rate_hz=1.0,
        traj_speed_gain=1.2, traj_dt=DT, vel_scale_min=0.10,
        Kd_traj=50.0, use_traj_plan=False,
    )


def run_condition(env, muscle_names, cfg, seeds, cfc_path):
    ctrl = MyoArmController(config=cfg, muscle_names=muscle_names)
    if cfc_path is not None and cfc_path.exists():
        ctrl.load_cfc(cfc_path)
    results = [run_episode(env, ctrl, seed=s) for s in seeds]
    keys = ["tip_err_min_mm", "tip_err_final_mm",
            "vel_peak_ratio", "straightness", "jerk_rms", "peak_speed",
            "direction_error_deg", "progress_ratio",
            "ik_count", "ik_residual_mm_max"]
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
    print("=== F12: B3 visuomotor feedback (周期的 re-IK) ===")
    env = gym.make("myoArmReachRandom-v0")
    deterministic_reset(env, 0)
    uw = env.unwrapped
    m = uw.mj_model
    muscle_names = [m.actuator(i).name for i in range(m.nu)]
    seeds = find_reachable_seeds(env)
    print(f"  test seeds: {seeds}")

    conditions = [
        ("neural endpoint_pd",                  make_pd_cfg(),                              CFC_OLD_PATH),
        ("λ-traj baseline",                     make_lambda_cfg(visuomotor=False),          CFC_LAMBDA_PATH),
        ("λ-traj visuo P=5  (100ms)",           make_lambda_cfg(visuomotor=True, period=5), CFC_LAMBDA_PATH),
        ("λ-traj visuo P=10 (200ms)",           make_lambda_cfg(visuomotor=True, period=10),CFC_LAMBDA_PATH),
        ("λ-traj visuo P=25 (500ms)",           make_lambda_cfg(visuomotor=True, period=25),CFC_LAMBDA_PATH),
        ("λ-traj visuo P=50 (1s)",              make_lambda_cfg(visuomotor=True, period=50),CFC_LAMBDA_PATH),
        ("pure λ-traj visuo P=10",
         make_lambda_cfg(visuomotor=True, period=10, neural=False), None),
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
              f"final_err={s['tip_err_final_mm']['mean']:5.1f}mm  "
              f"progress={s['progress_ratio']['mean']:+.3f}±{s['progress_ratio']['std']:.3f}")
        print(f"    dir_err={s['direction_error_deg']['mean']:5.1f}°  "
              f"straight={s['straightness']['mean']:.3f}  "
              f"jerk={s['jerk_rms']['mean']:.0f}  peak_v={s['peak_speed']['mean']:.2f}m/s")
        print(f"    IK calls/ep={s['ik_count']['mean']:.1f}  "
              f"IK residual_max={s['ik_residual_mm_max']['mean']:.2f}mm")
    env.close()

    # 検定
    print("\n=== Welch's t-test: vs λ-traj baseline (no visuomotor) ===")
    base = results["λ-traj baseline"]["per_seed"]
    test_results = {}
    for name in [n for n, _, _ in conditions if n not in ("λ-traj baseline", "neural endpoint_pd")]:
        per = results[name]["per_seed"]
        for key in ["direction_error_deg", "tip_err_min_mm", "progress_ratio", "tip_err_final_mm"]:
            t = welch([r[key] for r in per], [r[key] for r in base])
            test_results.setdefault(key, {})[name] = t
            if t:
                sig = "***" if t["p"]<0.001 else "**" if t["p"]<0.01 else "*" if t["p"]<0.05 else ""
                print(f"  {key:<22} {name:<26} t={t['t']:+6.2f}  p={t['p']:.4g}{sig:<4}  d={t['d']:+.2f}")

    # 検定: vs endpoint_pd (どこまで近づけるか)
    print("\n=== Welch's t-test: 各 visuomotor 条件 vs endpoint_pd ===")
    pd_per = results["neural endpoint_pd"]["per_seed"]
    for name in [n for n, _, _ in conditions if "visuo" in n or "pure" in n]:
        per = results[name]["per_seed"]
        for key in ["tip_err_min_mm", "direction_error_deg", "tip_err_final_mm"]:
            t = welch([r[key] for r in per], [r[key] for r in pd_per])
            if t:
                sig = "***" if t["p"]<0.001 else "**" if t["p"]<0.01 else "*" if t["p"]<0.05 else ""
                print(f"  {key:<22} {name:<26} t={t['t']:+6.2f}  p={t['p']:.4g}{sig:<4}  d={t['d']:+.2f}")

    summary = {
        "phase": "1-6 F12",
        "purpose": "B3 visuomotor feedback loop (periodic re-IK)",
        "n_seeds": len(seeds), "seeds": seeds, "elapsed_s": round(time.time()-t0, 1),
        "conditions": {n: {"n_solved": r["n_solved"], "stats": r["stats"]}
                       for n, r in results.items()},
        "stat_tests_vs_lambda_baseline": test_results,
        "raw_per_seed": {n: r["per_seed"] for n, r in results.items()},
    }
    out = RESULTS_DIR / "f12_visuomotor.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print(f"\n  elapsed: {time.time()-t0:.1f}s, saved → {out}")


if __name__ == "__main__":
    main()
