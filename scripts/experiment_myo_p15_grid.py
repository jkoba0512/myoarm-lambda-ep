"""
experiment_myo_p15_grid.py — Phase 1-5: 強feedforward の厳密グリッド探索。

設計（事前宣言）:
  グリッド  : K_ff ∈ {1, 2, 3, 4} × Kp_traj ∈ {8, 20, 40}
  シード数  : 20
  主要指標  : solve_rate ≥ 0.9 かつ min_err ≤ 8mm を満たす条件のうち
             vel_peak_ratio が最大のもの
  副次指標  : straightness, jerk_rms, peak_speed
  統計検定  : 最良条件 vs P1-4 (vel_scale) を per-seed の vel_peak_ratio で
             Welch's t-test, α=0.05
  全ての条件で mean ± std と per-seed 値を保存。

出力:
  results/experiment_myo_p15/grid_summary.json
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


# 事前宣言：主要指標と判定基準
PRIMARY_CRITERIA = {
    "solve_rate_min": 0.9,
    "min_err_max_mm": 8.0,
    "primary_metric": "vel_peak_ratio",  # これを最大化
    "test": "welch_t",
    "alpha": 0.05,
}

# グリッド
K_FF_VALUES    = [1.0, 2.0, 3.0, 4.0]
KP_TRAJ_VALUES = [8.0, 20.0, 40.0]
N_SEEDS        = 20


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


def run_condition(env, muscle_names, cfg: MyoArmConfig, n_seeds: int) -> dict:
    ctrl = MyoArmController(config=cfg, muscle_names=muscle_names)
    if CFC_MODEL_PATH.exists():
        ctrl.load_cfc(CFC_MODEL_PATH)
    results = [run_episode(env, ctrl, seed=s) for s in range(n_seeds)]

    def stats_for(key):
        vals = [r[key] for r in results
                if r.get(key) is not None and not np.isnan(r.get(key, float("nan")))]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    n_solved = sum(r["solved"] for r in results)
    return {
        "n_solved": n_solved,
        "solve_rate": n_solved / n_seeds,
        "stats": {k: stats_for(k) for k in
                  ["tip_err_min_mm", "tip_err_final_mm",
                   "vel_peak_ratio", "straightness", "jerk_rms", "peak_speed", "movement_time"]},
        "per_seed": results,
    }


def main() -> None:
    print("=== Phase 1-5: 強feedforward グリッド探索 ===")
    print(f"env: myoArmReachFixed-v0  seeds: {N_SEEDS}  dt: {DT}s")
    print(f"grid: K_ff × Kp_traj = {len(K_FF_VALUES)} × {len(KP_TRAJ_VALUES)} = {len(K_FF_VALUES)*len(KP_TRAJ_VALUES)} 条件")
    print(f"事前宣言: {PRIMARY_CRITERIA}")

    env = gym.make("myoArmReachFixed-v0")
    uw  = env.unwrapped
    env.reset(seed=0)
    m = uw.mj_model
    muscle_names = [m.actuator(i).name for i in range(m.nu)]

    grid_results = {}
    t0 = time.time()
    for K_ff in K_FF_VALUES:
        for Kp_traj in KP_TRAJ_VALUES:
            cfg = MyoArmConfig(
                Kp_ee=80.0, Kd_ee=15.0, Ki_ee=2.0, act_bias=0.15, K_cereb=0.2,
                io_mode="sparse", io_firing_rate_hz=1.0,
                use_traj_plan=True, traj_mode="feedforward",
                traj_speed_gain=1.2, traj_dt=DT,
                K_ff=K_ff, Kp_traj=Kp_traj, Kd_traj=50.0,
            )
            agg = run_condition(env, muscle_names, cfg, n_seeds=N_SEEDS)
            key = f"K_ff={K_ff}_Kp_traj={Kp_traj}"
            grid_results[key] = {"K_ff": K_ff, "Kp_traj": Kp_traj, **agg}
            s = agg["stats"]
            print(
                f"  K_ff={K_ff:.0f}  Kp_traj={Kp_traj:5.1f}: "
                f"solve={agg['n_solved']:2d}/{N_SEEDS}  "
                f"min_err={s['tip_err_min_mm']['mean']:5.1f}±{s['tip_err_min_mm']['std']:.1f}mm  "
                f"vpr={s['vel_peak_ratio']['mean']:.3f}±{s['vel_peak_ratio']['std']:.3f}  "
                f"straight={s['straightness']['mean']:.3f}  "
                f"jerk={s['jerk_rms']['mean']:.0f}"
            )

    # P1-4 ベースライン (vel_scale モード) を 20 seeds で再実行（統計検定用）
    print("\n[Reference] P1-4 vel_scale baseline (20 seeds)")
    cfg_p14 = MyoArmConfig(
        Kp_ee=80.0, Kd_ee=15.0, Ki_ee=2.0, act_bias=0.15, K_cereb=0.2,
        io_mode="sparse", io_firing_rate_hz=1.0,
        use_traj_plan=True, traj_mode="vel_scale",
        traj_speed_gain=1.2, traj_dt=DT, vel_scale_min=0.10,
    )
    p14 = run_condition(env, muscle_names, cfg_p14, n_seeds=N_SEEDS)
    print(f"  vel_scale: solve={p14['n_solved']}/{N_SEEDS}  "
          f"vpr={p14['stats']['vel_peak_ratio']['mean']:.3f}±{p14['stats']['vel_peak_ratio']['std']:.3f}  "
          f"min_err={p14['stats']['tip_err_min_mm']['mean']:.1f}mm")
    env.close()
    elapsed = time.time() - t0

    # ── 事前宣言した主要指標で best を選択 ──
    eligible = [
        (k, v) for k, v in grid_results.items()
        if v["solve_rate"] >= PRIMARY_CRITERIA["solve_rate_min"]
        and v["stats"]["tip_err_min_mm"]["mean"] <= PRIMARY_CRITERIA["min_err_max_mm"]
    ]
    print(f"\n判定基準を満たす条件: {len(eligible)}/{len(grid_results)}")

    if eligible:
        best_key, best = max(
            eligible,
            key=lambda kv: kv[1]["stats"]["vel_peak_ratio"]["mean"]
            if not np.isnan(kv[1]["stats"]["vel_peak_ratio"]["mean"]) else -1.0
        )
        # ── 統計検定: best vs P1-4 (vel_peak_ratio) ──
        best_vpr_seeds = [r["vel_peak_ratio"] for r in best["per_seed"]
                          if not np.isnan(r.get("vel_peak_ratio", float("nan")))]
        p14_vpr_seeds  = [r["vel_peak_ratio"] for r in p14["per_seed"]
                          if not np.isnan(r.get("vel_peak_ratio", float("nan")))]
        t_stat, p_value = stats.ttest_ind(best_vpr_seeds, p14_vpr_seeds, equal_var=False)
        significant = bool(p_value < PRIMARY_CRITERIA["alpha"])
    else:
        best_key, best = None, None
        t_stat, p_value, significant = float("nan"), float("nan"), False

    summary = {
        "phase": "1-5",
        "env": "myoArmReachFixed-v0",
        "dt": DT,
        "n_seeds": N_SEEDS,
        "elapsed_s": round(elapsed, 1),
        "criteria": PRIMARY_CRITERIA,
        "grid_size": {"K_ff": K_FF_VALUES, "Kp_traj": KP_TRAJ_VALUES},
        "best": {
            "key":   best_key,
            "K_ff":  best["K_ff"]    if best else None,
            "Kp_traj": best["Kp_traj"] if best else None,
            "stats": best["stats"]   if best else None,
            "n_solved": best["n_solved"] if best else None,
        },
        "p14_reference": {
            "n_solved": p14["n_solved"],
            "stats": p14["stats"],
        },
        "stat_test": {
            "name": "Welch's t-test (best vs P1-4 vel_peak_ratio)",
            "t_stat": float(t_stat) if not np.isnan(t_stat) else None,
            "p_value": float(p_value) if not np.isnan(p_value) else None,
            "alpha": PRIMARY_CRITERIA["alpha"],
            "significant": significant,
        },
        "grid_results": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_seed"}
            for k, v in grid_results.items()
        },
        "raw_per_seed": {
            **{k: v["per_seed"] for k, v in grid_results.items()},
            "p14_vel_scale": p14["per_seed"],
        },
    }

    out = RESULTS_DIR / "grid_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)

    print()
    print("=== 主要指標で選ばれた最良条件 ===")
    if best:
        s = best["stats"]
        print(f"  {best_key}")
        print(f"  vel_peak_ratio: {s['vel_peak_ratio']['mean']:.3f} ± {s['vel_peak_ratio']['std']:.3f}")
        print(f"  min_err       : {s['tip_err_min_mm']['mean']:.1f} ± {s['tip_err_min_mm']['std']:.1f} mm")
        print(f"  straightness  : {s['straightness']['mean']:.3f} ± {s['straightness']['std']:.3f}")
        print(f"  jerk_rms      : {s['jerk_rms']['mean']:.0f} ± {s['jerk_rms']['std']:.0f} m/s³")
        print(f"  peak_speed    : {s['peak_speed']['mean']:.2f} ± {s['peak_speed']['std']:.2f} m/s")
        print(f"  solve         : {best['n_solved']}/{N_SEEDS}")
        print()
        print("=== 統計検定 (best vs P1-4 vel_peak_ratio) ===")
        print(f"  Welch's t-test: t={t_stat:.3f}, p={p_value:.4g}")
        print(f"  α={PRIMARY_CRITERIA['alpha']} で有意: {significant}")
    else:
        print("  判定基準を満たす条件なし")
    print(f"\n  elapsed: {elapsed:.1f}s, saved → {out}")


if __name__ == "__main__":
    main()
