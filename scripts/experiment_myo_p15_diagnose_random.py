"""
experiment_myo_p15_diagnose_random.py — Phase 1-5 E1: Random env 失敗原因の系統診断。

5 つの仮説を順に検証する:
  H1. Jacobian 更新頻度が低い (50 ステップ → 1/5/25 で比較)
  H2. act_bias が Fixed env に過適合 (0.0/0.05/0.15/0.30 で比較)
  H3. target 方向依存性 (各シードの target 方向と final tip の方向を記録)
  H4. CfC 効果の有無 (load_cfc あり/なし で比較)
  H5. 純 PD (Phase 1-0 相当) が同じく失敗するか (use_traj_plan=False, K_cereb=0)

Random env の reachable subset (n=20) で評価し、
solve_rate と min_err、direction_error_deg を記録する。

出力:
  results/experiment_myo_p15/diagnose_random.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import myosuite  # noqa
import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myoarm.myo_controller import MyoArmController, MyoArmConfig
from myoarm.env_utils import deterministic_reset
import myoarm.myo_controller as mc  # _JACOBIAN_RECOMPUTE_INTERVAL に触るため

RESULTS_DIR    = ROOT / "results" / "experiment_myo_p15"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CFC_MODEL_PATH = ROOT / "results" / "myo_cfc_data" / "cfc_model.pt"

DT = 0.020
MAX_REACH_M = 0.85
N_REACHABLE = 20


def find_reachable_seeds(env: gym.Env) -> list[int]:
    out = []
    for s in range(50):
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
    target_dir = (target - tip0)
    target_dir_norm = target_dir / max(np.linalg.norm(target_dir), 1e-9)

    errs = []; tips = [tip0.copy()]; solved = False
    for step in range(max_steps):
        od = uw.obs_dict
        q, dq      = np.array(od["qpos"]), np.array(od["qvel"])
        reach_err  = np.array(od["reach_err"])
        tip_pos    = np.array(od["tip_pos"])
        muscle_vel   = d.actuator_velocity.copy()
        muscle_force = d.actuator_force.copy()
        errs.append(float(np.linalg.norm(reach_err)))
        tips.append(tip_pos.copy())

        a_total, _ = ctrl.step(q=q, dq=dq, reach_err=reach_err, tip_pos=tip_pos,
                               muscle_vel=muscle_vel, muscle_force=muscle_force, m=m, d=d)
        obs, _, term, trunc, info = env.step(a_total)
        ctrl.update_cerebellum(np.array(uw.obs_dict["qpos"]), m, d)
        if info.get("solved", False): solved = True
        if term or trunc: break

    tips = np.array(tips)
    final_tip = tips[-1]
    travel = final_tip - tip0
    travel_norm = travel / max(np.linalg.norm(travel), 1e-9)
    # 進行方向と target 方向のなす角
    cos_angle = float(np.clip(np.dot(travel_norm, target_dir_norm), -1.0, 1.0))
    direction_error_deg = float(np.degrees(np.arccos(cos_angle)))
    travel_dist = float(np.linalg.norm(travel))
    target_dist = float(np.linalg.norm(target_dir))
    progress = float(np.dot(travel, target_dir_norm))  # target 方向の進捗 [m]

    return {
        "seed": seed, "solved": solved,
        "target_dist_m": target_dist,
        "travel_dist_m": travel_dist,
        "progress_m": progress,
        "direction_error_deg": direction_error_deg,
        "tip_err_min_mm": min(errs)*1000 if errs else float("nan"),
        "tip_err_final_mm": errs[-1]*1000 if errs else float("nan"),
        "target_dir": target_dir_norm.tolist(),
    }


def make_cfg(**overrides) -> MyoArmConfig:
    base = dict(
        Kp_ee=80.0, Kd_ee=15.0, Ki_ee=2.0, act_bias=0.15, K_cereb=0.2,
        io_mode="sparse", io_firing_rate_hz=1.0,
        traj_speed_gain=1.2, traj_dt=DT, vel_scale_min=0.10,
        Kd_traj=50.0, use_traj_plan=False, traj_mode="vel_scale",
    )
    base.update(overrides)
    return MyoArmConfig(**base)


def run_all(env, muscle_names, cfg, seeds, load_cfc=True) -> dict:
    ctrl = MyoArmController(config=cfg, muscle_names=muscle_names)
    if load_cfc and CFC_MODEL_PATH.exists():
        ctrl.load_cfc(CFC_MODEL_PATH)
    rs = [run_episode(env, ctrl, seed=s) for s in seeds]
    n_solved = sum(r["solved"] for r in rs)
    progress_arr = np.array([r["progress_m"] / r["target_dist_m"] for r in rs])
    dir_err_arr  = np.array([r["direction_error_deg"] for r in rs])
    return {
        "n_solved": n_solved,
        "solve_rate": n_solved / len(seeds),
        "min_err_mm_mean":  float(np.mean([r["tip_err_min_mm"]   for r in rs])),
        "min_err_mm_std":   float(np.std ([r["tip_err_min_mm"]   for r in rs], ddof=1)),
        "progress_ratio_mean": float(np.mean(progress_arr)),
        "progress_ratio_std":  float(np.std (progress_arr, ddof=1)),
        "direction_error_deg_mean": float(np.mean(dir_err_arr)),
        "direction_error_deg_std":  float(np.std (dir_err_arr, ddof=1)),
        "per_seed": rs,
    }


def main() -> None:
    print("=== Phase 1-5 E1: Random env 失敗原因の系統診断 ===")
    env = gym.make("myoArmReachRandom-v0")
    uw  = env.unwrapped
    env.reset(seed=0)
    m = uw.mj_model
    muscle_names = [m.actuator(i).name for i in range(m.nu)]
    seeds = find_reachable_seeds(env)
    print(f"  reachable seeds: {seeds}")

    diagnostics = {}
    t0 = time.time()

    # ── H1: Jacobian 更新頻度 ──
    print("\n[H1] Jacobian 更新頻度 (1, 5, 25, 50)")
    h1 = {}
    orig_interval = mc._JACOBIAN_RECOMPUTE_INTERVAL
    for interval in [1, 5, 25, 50]:
        mc._JACOBIAN_RECOMPUTE_INTERVAL = interval
        agg = run_all(env, muscle_names, make_cfg(), seeds)
        h1[f"interval={interval}"] = agg
        print(f"  interval={interval:2d}: solve={agg['n_solved']:2d}/{N_REACHABLE}  "
              f"min_err={agg['min_err_mm_mean']:5.1f}±{agg['min_err_mm_std']:.0f}mm  "
              f"progress={agg['progress_ratio_mean']:.2f}  "
              f"dir_err={agg['direction_error_deg_mean']:5.1f}°")
    mc._JACOBIAN_RECOMPUTE_INTERVAL = orig_interval
    diagnostics["H1_jacobian_interval"] = h1

    # ── H2: act_bias ──
    print("\n[H2] act_bias (0.0, 0.05, 0.15, 0.30)")
    h2 = {}
    for bias in [0.0, 0.05, 0.15, 0.30]:
        agg = run_all(env, muscle_names, make_cfg(act_bias=bias), seeds)
        h2[f"act_bias={bias}"] = agg
        print(f"  bias={bias:.2f}: solve={agg['n_solved']:2d}/{N_REACHABLE}  "
              f"min_err={agg['min_err_mm_mean']:5.1f}mm  "
              f"progress={agg['progress_ratio_mean']:.2f}  "
              f"dir_err={agg['direction_error_deg_mean']:5.1f}°")
    diagnostics["H2_act_bias"] = h2

    # ── H4: CfC ありなし ──
    print("\n[H4] CfC 学習済みモデルの有無")
    h4 = {}
    for label, lc in [("CfC=loaded", True), ("CfC=random_init", False)]:
        agg = run_all(env, muscle_names, make_cfg(), seeds, load_cfc=lc)
        h4[label] = agg
        print(f"  {label:18s}: solve={agg['n_solved']:2d}/{N_REACHABLE}  "
              f"min_err={agg['min_err_mm_mean']:5.1f}mm  "
              f"progress={agg['progress_ratio_mean']:.2f}")
    diagnostics["H4_cfc"] = h4

    # ── H5: 純 PD (no neural) ──
    print("\n[H5] 純 PD (K_cereb=0, reflex/RI=0, IO firing_rate=0)")
    cfg_pure_pd = make_cfg(K_cereb=0.0, K_ia=0.0, K_ib=0.0, K_ri=0.0,
                           io_firing_rate_hz=0.0)
    agg = run_all(env, muscle_names, cfg_pure_pd, seeds, load_cfc=False)
    diagnostics["H5_pure_pd"] = agg
    print(f"  pure PD: solve={agg['n_solved']:2d}/{N_REACHABLE}  "
          f"min_err={agg['min_err_mm_mean']:5.1f}mm  "
          f"progress={agg['progress_ratio_mean']:.2f}  "
          f"dir_err={agg['direction_error_deg_mean']:5.1f}°")

    # ── H3: target 方向依存性 ──
    print("\n[H3] target 方向と direction_error の相関 (P1-1 baseline 結果から)")
    p11 = h1["interval=50"]  # default の P1-1 結果を使う
    print(f"  per-seed direction_error_deg と target_dir の相関:")
    print(f"  {'seed':>4} {'tgt_dir (x,y,z)':>22} {'dist_m':>7} {'dir_err°':>9} "
          f"{'progress':>9} {'min_err_mm':>11} {'solved':>7}")
    h3 = []
    for r in p11["per_seed"]:
        h3.append({
            "seed": r["seed"], "target_dir": r["target_dir"],
            "target_dist_m": r["target_dist_m"], "dir_err_deg": r["direction_error_deg"],
            "progress_ratio": r["progress_m"] / r["target_dist_m"],
            "min_err_mm": r["tip_err_min_mm"], "solved": r["solved"],
        })
        print(f"  {r['seed']:>4} ({r['target_dir'][0]:+.2f},{r['target_dir'][1]:+.2f},"
              f"{r['target_dir'][2]:+.2f})  {r['target_dist_m']:.2f}m "
              f"{r['direction_error_deg']:6.1f}°  "
              f"{r['progress_m']/r['target_dist_m']:+.2f}  "
              f"{r['tip_err_min_mm']:7.1f}    {'✓' if r['solved'] else ' '}")
    diagnostics["H3_per_seed_directions"] = h3

    env.close()
    elapsed = time.time() - t0

    summary = {
        "phase": "1-5 E1 diagnose",
        "env": "myoArmReachRandom-v0",
        "n_seeds": len(seeds),
        "seeds": seeds,
        "elapsed_s": round(elapsed, 1),
        "diagnostics": diagnostics,
    }
    out = RESULTS_DIR / "diagnose_random.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)

    # ── 結論 ──
    print("\n=== 仮説別の所見 ===")
    h1_solves = [v["n_solved"] for v in h1.values()]
    h2_solves = [v["n_solved"] for v in h2.values()]
    h4_solves = [v["n_solved"] for v in h4.values()]
    print(f"  H1 jacobian: solve range {min(h1_solves)}-{max(h1_solves)}/20 → "
          f"{'effect' if max(h1_solves) - min(h1_solves) >= 3 else 'no effect'}")
    print(f"  H2 act_bias: solve range {min(h2_solves)}-{max(h2_solves)}/20 → "
          f"{'effect' if max(h2_solves) - min(h2_solves) >= 3 else 'no effect'}")
    print(f"  H4 CfC:      solve {h4_solves} → "
          f"{'effect' if abs(h4_solves[0]-h4_solves[1]) >= 3 else 'no effect'}")
    print(f"  H5 pure PD:  solve {diagnostics['H5_pure_pd']['n_solved']}/20 → "
          f"{'神経成分が問題' if diagnostics['H5_pure_pd']['n_solved'] > p11['n_solved']+2 else '構造的問題'}")
    dir_errs = [r['dir_err_deg'] for r in h3]
    print(f"  H3 dir_err:  mean={np.mean(dir_errs):.1f}°, "
          f"max={max(dir_errs):.1f}°, "
          f"max_seed={h3[int(np.argmax(dir_errs))]['seed']}")

    print(f"\n  elapsed: {elapsed:.1f}s, saved → {out}")


if __name__ == "__main__":
    main()
