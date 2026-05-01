"""
experiment_myo_p15_feedforward.py — Phase 1-5: 強 feedforward 制御の評価。

P1-4 (vel_scale) では vel_peak_ratio が 0.169 までしか改善せず、
ヒトの 0.50 に届かなかった。本実験では最小ジャーク加速度プロファイル
a_ref(τ) = (pf-p0)/T² × (60τ - 180τ² + 120τ³) を直接トルクに変換する
強 feedforward を導入する。

制御則:
  F_ff   = K_ff × a_ref
  F_fb   = Kp_traj × (p_ref - tip_pos) + Kd_traj × (v_ref - tip_vel)
  F_total = F_ff + F_fb + Ki × ∫err

K_ff スイープで vel_peak_ratio が最大になる値を探索する。

出力:
  results/experiment_myo_p15/feedforward_summary.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import myosuite  # noqa
import gymnasium as gym
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myoarm.myo_controller import MyoArmController, MyoArmConfig

RESULTS_DIR    = ROOT / "results" / "experiment_myo_p15"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CFC_MODEL_PATH = ROOT / "results" / "myo_cfc_data" / "cfc_model.pt"

DT = 0.020  # 実制御周期


def compute_kinematics(positions: np.ndarray, dt: float = DT) -> dict:
    """tip_pos 軌跡から運動学指標を計算する (P1-4 と同じ)。"""
    if len(positions) < 5:
        return {k: float("nan") for k in
                ["jerk_rms", "vel_peak_ratio", "straightness", "movement_time", "peak_speed"]}

    vel   = np.diff(positions, axis=0) / dt
    speed = np.linalg.norm(vel, axis=1)
    acc   = np.diff(vel, axis=0) / dt
    jerk  = np.diff(acc, axis=0) / dt

    thresh = 0.02
    onset  = next((i for i, s in enumerate(speed) if s > thresh), None)
    if onset is None:
        return {"jerk_rms": float("nan"), "vel_peak_ratio": float("nan"),
                "straightness": float("nan"), "movement_time": float("nan"),
                "peak_speed": float(np.max(speed)) if len(speed) > 0 else float("nan")}

    offset = next((i for i in range(onset+5, len(speed)) if speed[i] < thresh), len(speed)-1)

    movement_speed = speed[onset:offset+1]
    T_actual       = (offset - onset) * dt
    peak_idx       = int(np.argmax(movement_speed))
    vpr            = peak_idx / max(len(movement_speed) - 1, 1)

    jerk_seg = jerk[onset:offset-1] if offset-1 < len(jerk) else jerk[onset:]
    jerk_rms = float(np.sqrt(np.mean(np.sum(jerk_seg**2, axis=1)))) if len(jerk_seg) > 0 else float("nan")

    seg = positions[onset:offset+2]
    L_path = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))
    D      = float(np.linalg.norm(seg[-1] - seg[0]))
    straightness = D / max(L_path, 1e-6)

    return {
        "jerk_rms": jerk_rms,
        "vel_peak_ratio": float(vpr),
        "straightness": float(straightness),
        "movement_time": float(T_actual),
        "peak_speed": float(np.max(movement_speed)),
    }


def run_episode(env: gym.Env, ctrl: MyoArmController, seed: int, max_steps: int = 600) -> dict:
    obs, _ = env.reset(seed=seed)
    uw = env.unwrapped
    m, d = uw.mj_model, uw.mj_data
    ctrl.reset()
    ctrl.initialize(m, d)

    positions: list[np.ndarray] = []
    errs: list[float] = []
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
    valid = [r for r in results if not np.isnan(r.get("vel_peak_ratio", float("nan")))]

    def smean(key):
        v = [r[key] for r in valid if not np.isnan(r.get(key, float("nan")))]
        return float(np.mean(v)) if v else float("nan")

    return {
        "n_solved": sum(r["solved"] for r in results),
        "min_err_mm": smean("tip_err_min_mm"),
        "vel_peak_ratio": smean("vel_peak_ratio"),
        "straightness": smean("straightness"),
        "jerk_rms": smean("jerk_rms"),
        "peak_speed": smean("peak_speed"),
        "movement_time": smean("movement_time"),
        "per_seed": results,
    }


def main() -> None:
    n_seeds = 10
    print("=== Phase 1-5: 強 feedforward 制御 (K_ff スイープ) ===")
    print(f"env: myoArmReachFixed-v0  seeds: {n_seeds}  dt: {DT}s")

    env = gym.make("myoArmReachFixed-v0")
    uw  = env.unwrapped
    env.reset(seed=0)
    m = uw.mj_model
    muscle_names = [m.actuator(i).name for i in range(m.nu)]

    K_ff_values = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    sweep = {}
    t0 = time.time()
    for K_ff in K_ff_values:
        cfg = MyoArmConfig(
            Kp_ee=80.0, Kd_ee=15.0, Ki_ee=2.0, act_bias=0.15, K_cereb=0.2,
            io_mode="sparse", io_firing_rate_hz=1.0,
            use_traj_plan=True, traj_mode="feedforward",
            traj_speed_gain=1.2, traj_dt=DT,
            K_ff=K_ff, Kp_traj=8.0, Kd_traj=50.0,
        )
        agg = run_condition(env, muscle_names, cfg, n_seeds=n_seeds)
        sweep[f"K_ff={K_ff}"] = agg
        print(f"  K_ff={K_ff:6.1f}: solve={agg['n_solved']}/{n_seeds}  "
              f"min_err={agg['min_err_mm']:.1f}mm  "
              f"vel_peak={agg['vel_peak_ratio']:.3f}  "
              f"straight={agg['straightness']:.3f}  "
              f"jerk={agg['jerk_rms']:.0f}  "
              f"peak_v={agg['peak_speed']:.2f}m/s")

    env.close()
    elapsed = time.time() - t0

    # Find best K_ff (max vel_peak_ratio with solve >= 80%)
    best = max(
        ((k, v) for k, v in sweep.items() if v["n_solved"] >= int(0.8*n_seeds)
         and not np.isnan(v["vel_peak_ratio"])),
        key=lambda kv: kv[1]["vel_peak_ratio"],
        default=(None, None),
    )

    summary = {
        "phase": "1-5",
        "env": "myoArmReachFixed-v0",
        "dt": DT,
        "elapsed_s": round(elapsed, 1),
        "best": {"K_ff": best[0], **(best[1] or {})},
        "sweep": {k: {kk: vv for kk, vv in v.items() if kk != "per_seed"}
                  for k, v in sweep.items()},
        "comparison": {
            "P1-1 (no traj_plan)": {"vel_peak_ratio": 0.047, "min_err_mm": 5.4, "jerk_rms": 414.2},
            "P1-4 (vel_scale)":    {"vel_peak_ratio": 0.169, "min_err_mm": 4.5, "jerk_rms": 338.3},
            "human":               {"vel_peak_ratio": 0.50,  "straightness": 0.85},
        },
        "full_sweep": sweep,
    }

    out = RESULTS_DIR / "feedforward_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)

    print()
    print(f"=== Best: {best[0]} ===")
    if best[1]:
        print(f"  vel_peak_ratio: {best[1]['vel_peak_ratio']:.3f}  (P1-4: 0.169, human: 0.50)")
        print(f"  straightness  : {best[1]['straightness']:.3f}")
        print(f"  jerk_rms      : {best[1]['jerk_rms']:.0f} m/s³")
        print(f"  min_err       : {best[1]['min_err_mm']:.1f} mm")
        print(f"  peak_speed    : {best[1]['peak_speed']:.2f} m/s")
    print(f"  elapsed: {elapsed:.1f}s, saved → {out}")


if __name__ == "__main__":
    main()
