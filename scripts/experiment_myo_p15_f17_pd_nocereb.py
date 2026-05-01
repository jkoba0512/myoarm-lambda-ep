"""
experiment_myo_p15_f17_pd_nocereb.py — F17: PD baseline without cerebellar correction

Codex 第2回査読 Major #1 への対応:
F16 の endpoint_pd baseline は reflexes=True, K_cereb=0.2 (joint cereb 有効) になっており、
full controller は K_cereb=0 なので、headline comparison が "M1-level isolation, spinal
common" にならない (cerebellum が片側だけ)。

このスクリプトは endpoint_pd_nocereb (reflexes=True, K_cereb=0.0) を n=50 で実行し、
F16 の endpoint_pd と比較して「cerebellar branch has negligible effects on PD baseline」
を実証する。

出力: results/experiment_myo_p15/f17_pd_nocereb.json
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

RESULTS_DIR = ROOT / "results" / "experiment_myo_p15"
DT = DEFAULT_DT
N_REACHABLE = 50


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


def main():
    print("=== F17: PD baseline w/o cerebellar correction (n=50) ===")
    env = gym.make("myoArmReachRandom-v0")
    deterministic_reset(env, 0)
    uw = env.unwrapped
    m = uw.mj_model
    muscle_names = [m.actuator(i).name for i in range(m.nu)]
    seeds = find_reachable_seeds(env, pool=range(150), n=N_REACHABLE)
    print(f"  test seeds (n={len(seeds)}): {seeds[:10]}...{seeds[-5:]}")

    cfg = MyoArmConfig(
        Kp_ee=80.0, Kd_ee=15.0, Ki_ee=2.0, act_bias=0.15,
        K_cereb=0.0,                      # ← cerebellar correction OFF
        K_ia=0.05, K_ib=0.03, K_ri=0.5,   # reflexes ON (full common spinal layer)
        io_mode="sparse", io_firing_rate_hz=1.0,
        control_mode="endpoint_pd",
        lambda_trajectory=False,
        visuomotor_feedback=False,
        cereb_correction_target="joint",
        traj_dt=DT, use_traj_plan=False,
    )
    ctrl = MyoArmController(config=cfg, muscle_names=muscle_names)
    # CfC を load しないことで cerebellar correction は無効

    t0 = time.time()
    results = [run_episode(env, ctrl, seed=s) for s in seeds]
    keys = ["tip_err_min_mm", "tip_err_final_mm",
            "vel_peak_ratio", "straightness", "jerk_rms", "peak_speed",
            "movement_time", "skewness", "direction_error_deg", "progress_ratio"]
    s = stats_for_results(results, keys)
    print(f"\n  [endpoint_pd_nocereb]")
    print(f"    solve={sum(r['solved'] for r in results):2d}/{len(seeds)}  "
          f"min_err={s['tip_err_min_mm']['mean']:5.1f}±{s['tip_err_min_mm']['std']:5.1f}mm  "
          f"final_err={s['tip_err_final_mm']['mean']:5.1f}mm")
    print(f"    progress={s['progress_ratio']['mean']:+.3f}  "
          f"dir_err={s['direction_error_deg']['mean']:5.1f}°  "
          f"straight={s['straightness']['mean']:.3f}±{s['straightness']['std']:.3f}  "
          f"vpr={s['vel_peak_ratio']['mean']:.3f}±{s['vel_peak_ratio']['std']:.3f}  "
          f"peak_v={s['peak_speed']['mean']:.2f}±{s['peak_speed']['std']:.2f}  "
          f"jerk={s['jerk_rms']['mean']:.0f}±{s['jerk_rms']['std']:.0f}")
    env.close()

    # F16 の endpoint_pd (cereb 有効版) と比較
    print("\n=== Compare to F16 endpoint_pd (cereb K=0.2 on) ===")
    with open(RESULTS_DIR / "f16_n50.json") as f:
        f16 = json.load(f)
    pd_with_cereb = f16["raw_per_seed"]["endpoint_pd"]
    metric_keys = ["tip_err_min_mm", "tip_err_final_mm", "direction_error_deg",
                   "vel_peak_ratio", "straightness", "peak_speed", "jerk_rms"]
    test_results = {}
    for key in metric_keys:
        a = [r[key] for r in results]
        b = [r[key] for r in pd_with_cereb]
        # Welch
        t = welch_test(a, b)
        # paired Wilcoxon
        pairs = [(x, y) for x, y in zip(a, b)
                 if x is not None and y is not None
                 and not (isinstance(x, float) and np.isnan(x))
                 and not (isinstance(y, float) and np.isnan(y))]
        A = np.array([p[0] for p in pairs])
        B = np.array([p[1] for p in pairs])
        diffs = A - B
        if (diffs == 0).all():
            wp = 1.0
            w_stat = 0.0
        else:
            try:
                w_stat, wp = sp_stats.wilcoxon(A, B, alternative='two-sided')
                w_stat = float(w_stat); wp = float(wp)
            except Exception:
                w_stat = float('nan'); wp = float('nan')
        test_results[key] = {
            "welch": t,
            "wilcoxon": {"W": w_stat, "p": wp, "median_diff": float(np.median(diffs))},
        }
        d_str = f"{t['cohens_d']:+.3f}" if t.get('cohens_d') is not None else "n/a"
        print(f"  {key:24}  Welch p={t['p_value']:.3g} {sig_marker(t['p_value']):3} "
              f"d={d_str}  |  Wilcoxon p={wp:.3g} {sig_marker(wp):3} "
              f"med(Δ)={float(np.median(diffs)):+.3f}")

    summary = {
        "phase": "1-6 F17 PD nocereb baseline",
        "purpose": "Codex review v2 Major #1: show cerebellar branch has negligible effect on PD baseline",
        "n_seeds": len(seeds), "seeds": seeds,
        "elapsed_s": round(time.time()-t0, 1),
        "stats": s,
        "compare_to_f16_endpoint_pd": test_results,
        "raw_per_seed": results,
    }
    out = RESULTS_DIR / "f17_pd_nocereb.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print(f"\n  elapsed: {time.time()-t0:.1f}s, saved → {out}")


if __name__ == "__main__":
    main()
