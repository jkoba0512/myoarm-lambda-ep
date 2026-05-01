"""
experiment_myo_p15_random_validation.py — Phase 1-5 D1: Random env で統計的検証。

Fixed env では全シード同一結果 (std=0) で Welch's t-test が無効化される問題を
解決するため、myoArmReachRandom-v0 の reachable subset で検証する。

設計（事前宣言）:
  env       : myoArmReachRandom-v0
  seed pool : 0..49 から reach_dist < 0.85m のシードを取得 (reachable subset)
  上限      : 最初に得られた 20 シードを使用
  条件      : 5 つのコントローラ設定で同一シードを評価
              1. P1-1 baseline (use_traj_plan=False)
              2. P1-4 vel_scale
              3. P1-5-A K_ff=3, Kp_traj=20  (中庸)
              4. P1-5-B K_ff=4, Kp_traj=8   (straightness 重視)
              5. P1-5-C K_ff=4, Kp_traj=20  (vel_peak 重視, Fixed env best)
  主要指標  : vel_peak_ratio (mean ± std, n=20)
  検定      : 各 P1-5 条件 vs P1-4 で Welch's t-test (vel_peak_ratio)
  α         : 0.05

出力:
  results/experiment_myo_p15/random_validation.json
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

RESULTS_DIR    = ROOT / "results" / "experiment_myo_p15"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CFC_MODEL_PATH = ROOT / "results" / "myo_cfc_data" / "cfc_model.pt"

DT = 0.020
MAX_REACH_M    = 0.85
N_REACHABLE    = 20
SEED_POOL      = list(range(50))


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


def find_reachable_seeds(env: gym.Env) -> list[int]:
    """SEED_POOL から reach_dist < MAX_REACH_M を満たすシードを N_REACHABLE 個収集する。"""
    out = []
    for s in SEED_POOL:
        env.reset(seed=s)
        od = env.unwrapped.obs_dict
        d = float(np.linalg.norm(np.array(od["reach_err"])))
        if d < MAX_REACH_M:
            out.append((s, d))
        if len(out) >= N_REACHABLE:
            break
    return out


def run_episode(env, ctrl, seed, max_steps=600):
    obs, _ = env.reset(seed=seed)
    uw = env.unwrapped
    m, d = uw.mj_model, uw.mj_data
    ctrl.reset(); ctrl.initialize(m, d)
    positions = []; errs = []; solved = False
    for step in range(max_steps):
        od = uw.obs_dict
        q, dq      = np.array(od["qpos"]), np.array(od["qvel"])
        reach_err  = np.array(od["reach_err"])
        tip_pos    = np.array(od["tip_pos"])
        muscle_vel   = d.actuator_velocity.copy()
        muscle_force = d.actuator_force.copy()
        positions.append(tip_pos.copy()); errs.append(float(np.linalg.norm(reach_err)))
        a_total, _ = ctrl.step(q=q, dq=dq, reach_err=reach_err, tip_pos=tip_pos,
                               muscle_vel=muscle_vel, muscle_force=muscle_force, m=m, d=d)
        obs, _, term, trunc, info = env.step(a_total)
        ctrl.update_cerebellum(np.array(uw.obs_dict["qpos"]), m, d)
        if info.get("solved", False): solved = True
        if term or trunc: break
    km = compute_kinematics(np.array(positions), dt=DT)
    return {
        "seed": seed, "solved": solved,
        "tip_err_min_mm": min(errs)*1000 if errs else float("nan"),
        "tip_err_final_mm": errs[-1]*1000 if errs else float("nan"),
        **km,
    }


def run_condition(env, muscle_names, cfg: MyoArmConfig, seeds: list[int]) -> dict:
    ctrl = MyoArmController(config=cfg, muscle_names=muscle_names)
    if CFC_MODEL_PATH.exists():
        ctrl.load_cfc(CFC_MODEL_PATH)
    results = [run_episode(env, ctrl, seed=s) for s in seeds]

    def stats_for(key):
        vals = [r[key] for r in results
                if r.get(key) is not None and not np.isnan(r.get(key, float("nan")))]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {"mean": float(np.mean(vals)),
                "std":  float(np.std(vals, ddof=1)),  # 標本標準偏差
                "n":    len(vals)}

    n_solved = sum(r["solved"] for r in results)
    return {
        "n_solved": n_solved,
        "solve_rate": n_solved / len(seeds),
        "stats": {k: stats_for(k) for k in
                  ["tip_err_min_mm", "tip_err_final_mm",
                   "vel_peak_ratio", "straightness", "jerk_rms", "peak_speed", "movement_time"]},
        "per_seed": results,
    }


def main() -> None:
    print("=== Phase 1-5 D1: Random env での統計的検証 ===")
    print(f"env: myoArmReachRandom-v0  reachable_threshold: {MAX_REACH_M}m  target_n: {N_REACHABLE}")

    env = gym.make("myoArmReachRandom-v0")
    uw  = env.unwrapped
    env.reset(seed=0)
    m = uw.mj_model
    muscle_names = [m.actuator(i).name for i in range(m.nu)]

    # ── reachable seeds の抽出 ──
    reachable = find_reachable_seeds(env)
    seeds = [s for s, _ in reachable]
    print(f"  reachable seeds: {len(seeds)} 個   dist range: "
          f"{min(d for _,d in reachable)*100:.0f}-{max(d for _,d in reachable)*100:.0f}cm")
    print(f"  used seeds: {seeds}")

    # ── 5 つの条件 ──
    base_kwargs = dict(
        Kp_ee=80.0, Kd_ee=15.0, Ki_ee=2.0, act_bias=0.15, K_cereb=0.2,
        io_mode="sparse", io_firing_rate_hz=1.0,
        traj_speed_gain=1.2, traj_dt=DT, vel_scale_min=0.10,
        Kd_traj=50.0,
    )
    conditions = {
        "P1-1 baseline":          MyoArmConfig(**base_kwargs, use_traj_plan=False),
        "P1-4 vel_scale":         MyoArmConfig(**base_kwargs, use_traj_plan=True,
                                               traj_mode="vel_scale"),
        "P1-5-A K_ff=3 Kp_t=20":  MyoArmConfig(**base_kwargs, use_traj_plan=True,
                                               traj_mode="feedforward", K_ff=3.0, Kp_traj=20.0),
        "P1-5-B K_ff=4 Kp_t=8":   MyoArmConfig(**base_kwargs, use_traj_plan=True,
                                               traj_mode="feedforward", K_ff=4.0, Kp_traj=8.0),
        "P1-5-C K_ff=4 Kp_t=20":  MyoArmConfig(**base_kwargs, use_traj_plan=True,
                                               traj_mode="feedforward", K_ff=4.0, Kp_traj=20.0),
    }

    results = {}
    t0 = time.time()
    for name, cfg in conditions.items():
        agg = run_condition(env, muscle_names, cfg, seeds=seeds)
        results[name] = agg
        s = agg["stats"]
        print(f"\n  [{name}]")
        print(f"    solve={agg['n_solved']}/{len(seeds)}  "
              f"min_err={s['tip_err_min_mm']['mean']:5.1f}±{s['tip_err_min_mm']['std']:4.1f}mm  "
              f"vpr={s['vel_peak_ratio']['mean']:.3f}±{s['vel_peak_ratio']['std']:.3f}")
        print(f"    straight={s['straightness']['mean']:.3f}±{s['straightness']['std']:.3f}  "
              f"jerk={s['jerk_rms']['mean']:.0f}±{s['jerk_rms']['std']:.0f}  "
              f"peak_v={s['peak_speed']['mean']:.2f}m/s")

    env.close()
    elapsed = time.time() - t0

    # ── 統計検定: 各 P1-5 条件 vs P1-4 baseline ──
    print("\n=== 統計検定: 各 P1-5 条件 vs P1-4 vel_scale (vel_peak_ratio) ===")
    p14_vpr = [r["vel_peak_ratio"] for r in results["P1-4 vel_scale"]["per_seed"]
               if not np.isnan(r.get("vel_peak_ratio", float("nan")))]

    test_results = {}
    for name in ["P1-5-A K_ff=3 Kp_t=20", "P1-5-B K_ff=4 Kp_t=8", "P1-5-C K_ff=4 Kp_t=20"]:
        cond_vpr = [r["vel_peak_ratio"] for r in results[name]["per_seed"]
                    if not np.isnan(r.get("vel_peak_ratio", float("nan")))]
        if len(cond_vpr) >= 2 and len(p14_vpr) >= 2:
            t_stat, p_value = stats.ttest_ind(cond_vpr, p14_vpr, equal_var=False)
            sig = bool(p_value < 0.05)
            # 効果量 Cohen's d
            pooled_std = np.sqrt((np.var(cond_vpr, ddof=1) + np.var(p14_vpr, ddof=1)) / 2)
            cohens_d = (np.mean(cond_vpr) - np.mean(p14_vpr)) / pooled_std if pooled_std > 0 else float("nan")
        else:
            t_stat, p_value, sig, cohens_d = float("nan"), float("nan"), False, float("nan")

        test_results[name] = {
            "t_stat": float(t_stat) if not np.isnan(t_stat) else None,
            "p_value": float(p_value) if not np.isnan(p_value) else None,
            "cohens_d": float(cohens_d) if not np.isnan(cohens_d) else None,
            "significant": sig,
            "n_cond": len(cond_vpr),
            "n_p14": len(p14_vpr),
        }
        sig_mark = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else ""))
        d_str = f"{cohens_d:.2f}" if not np.isnan(cohens_d) else "n/a"
        print(f"  {name:<28} vs P1-4: t={t_stat:6.3f}  p={p_value:.4g}{sig_mark:<4}  d={d_str}")

    summary = {
        "phase": "1-5 D1",
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
        "stat_tests": test_results,
        "raw_per_seed": {name: r["per_seed"] for name, r in results.items()},
    }

    out = RESULTS_DIR / "random_validation.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)

    print(f"\n  elapsed: {elapsed:.1f}s, saved → {out}")


if __name__ == "__main__":
    main()
