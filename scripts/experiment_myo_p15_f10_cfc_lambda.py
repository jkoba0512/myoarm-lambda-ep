"""
experiment_myo_p15_f10_cfc_lambda.py — F10: λ-EP virtual trajectory に
                                       λ-mode 訓練済 CfC で online 補正を加える
                                       (Kawato/Wolpert/Ito の小脳逆モデル理論)

F9 で λ-EP virtual trajectory は biological plausibility を実現したが、
direction_error が 9.9° で改善せず、min_err は 138mm で endpoint_pd より悪い。
F4/F8 で既存 CfC が「実質効いていない」と判明済 (Fixed env + ランダム活性化で訓練)。

F10 で試す: λ-EP rollouts で再訓練した CfC を online correction として使う。
  神経生理: 小脳が運動指令の予測誤差から逆モデルを学習し、descending
            command に補正信号を加える (Kawato 1999, Ito 2008)。
  実装    : 既存の CfC 補正パイプラインそのまま (delta_q_err → J_act^+ → Δa)、
            ただしモデルを λ-EP データで訓練したものに差し替える。
            K_cereb スイープで補正の強さを探る。

設計:
  env       : myoArmReachRandom-v0 + deterministic_reset
  test seeds: 0..49 reachable subset (n=20, 訓練 100..299 と分離)
  λ params  : c_λ=20, sg=1.2, lambda_offset=0.005 (F9 best biological)

  conditions:
    1. neural endpoint_pd (参考)
    2. neural λ-traj K_cereb=0       (CfC なし、F9 と同じ)
    3. neural λ-traj K_cereb=0.2 + old_CfC (Fixed env / random act 訓練)
    4. neural λ-traj K_cereb=0.2 + new_CfC (λ-EP 再訓練)
    5. neural λ-traj K_cereb=0.5 + new_CfC
    6. neural λ-traj K_cereb=1.0 + new_CfC

  指標     : 同 F9
  検定     : 各 CfC 条件 vs K_cereb=0 baseline で Welch's t-test
             (主要指標 = direction_error_deg、副次 = tip_err_min_mm, progress)
  期待     : new_CfC で direction_error が低下、min_err 改善

出力:
  results/experiment_myo_p15/f10_cfc_lambda.json
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
CFC_OLD_PATH    = ROOT / "results" / "myo_cfc_data"        / "cfc_model.pt"
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
        "pred_err_mean": float(np.mean(pred_err_norms)) if pred_err_norms else 0.0,
        "pred_err_max":  float(np.max(pred_err_norms))  if pred_err_norms else 0.0,
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


def make_pd_cfg() -> MyoArmConfig:
    return MyoArmConfig(
        Kp_ee=80.0, Kd_ee=15.0, Ki_ee=2.0, act_bias=0.15,
        K_cereb=0.2, K_ia=0.05, K_ib=0.03, K_ri=0.5,
        io_mode="sparse", io_firing_rate_hz=1.0,
        traj_speed_gain=1.2, traj_dt=DT, vel_scale_min=0.10,
        Kd_traj=50.0, use_traj_plan=False,
    )


def run_condition(env, muscle_names, cfg: MyoArmConfig, seeds: list[int],
                  cfc_path: Path | None) -> dict:
    ctrl = MyoArmController(config=cfg, muscle_names=muscle_names)
    if cfc_path is not None and cfc_path.exists():
        ctrl.load_cfc(cfc_path)
    results = [run_episode(env, ctrl, seed=s) for s in seeds]

    metric_keys = ["tip_err_min_mm", "tip_err_final_mm",
                   "vel_peak_ratio", "straightness", "jerk_rms",
                   "peak_speed", "movement_time",
                   "direction_error_deg", "progress_ratio",
                   "pred_err_mean", "pred_err_max"]

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
    print("=== Phase 1-6 F10: λ-traj + 再訓練 CfC で online 補正 ===")
    print(f"old CfC: {CFC_OLD_PATH}")
    print(f"new CfC: {CFC_LAMBDA_PATH}")

    env = gym.make("myoArmReachRandom-v0")
    uw  = env.unwrapped
    deterministic_reset(env, 0)
    m = uw.mj_model
    muscle_names = [m.actuator(i).name for i in range(m.nu)]

    seeds = find_reachable_seeds(env)
    print(f"  reachable test seeds: {seeds}")

    conditions = [
        ("neural endpoint_pd",
         make_pd_cfg(), CFC_OLD_PATH),  # 参考: PD + 既存 CfC
        ("λ-traj K=0 (no CfC)",
         make_lambda_cfg(K_cereb=0.0), None),
        ("λ-traj K=0.2 + old_CfC",
         make_lambda_cfg(K_cereb=0.2), CFC_OLD_PATH),
        ("λ-traj K=0.2 + new_CfC",
         make_lambda_cfg(K_cereb=0.2), CFC_LAMBDA_PATH),
        ("λ-traj K=0.5 + new_CfC",
         make_lambda_cfg(K_cereb=0.5), CFC_LAMBDA_PATH),
        ("λ-traj K=1.0 + new_CfC",
         make_lambda_cfg(K_cereb=1.0), CFC_LAMBDA_PATH),
    ]

    t0 = time.time()
    results = {}
    for name, cfg, cfc_path in conditions:
        agg = run_condition(env, muscle_names, cfg, seeds=seeds, cfc_path=cfc_path)
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
        print(f"    pred_err: mean={s['pred_err_mean']['mean']:.4f}  "
              f"max={s['pred_err_max']['mean']:.4f}")

    env.close()
    elapsed = time.time() - t0

    # ── 統計検定 ──
    print("\n=== Welch's t-test: vs λ-traj K=0 (no CfC) ===")
    test_results = {}
    base_per_seed = results["λ-traj K=0 (no CfC)"]["per_seed"]
    for key in ["direction_error_deg", "tip_err_min_mm", "progress_ratio", "tip_err_final_mm"]:
        base_vals = [r[key] for r in base_per_seed]
        for name in [n for n, _, _ in conditions if n not in
                     ("λ-traj K=0 (no CfC)", "neural endpoint_pd")]:
            cond_vals = [r[key] for r in results[name]["per_seed"]]
            t = welch_test(cond_vals, base_vals)
            test_results.setdefault(key, {})[name] = t
            if t["p_value"] is not None:
                sig = ("***" if t["p_value"] < 0.001 else
                       "**"  if t["p_value"] < 0.01  else
                       "*"   if t["p_value"] < 0.05  else "")
                d = f"{t['cohens_d']:+.2f}" if t['cohens_d'] is not None else "n/a"
                print(f"  {key:<22} {name:<28} t={t['t_stat']:+6.2f}  p={t['p_value']:.4g}{sig:<4} d={d}")

    # old vs new CfC (両方 K=0.2)
    print("\n=== old_CfC vs new_CfC (両方 K=0.2) ===")
    old_per = results["λ-traj K=0.2 + old_CfC"]["per_seed"]
    new_per = results["λ-traj K=0.2 + new_CfC"]["per_seed"]
    for key in ["direction_error_deg", "tip_err_min_mm", "progress_ratio", "pred_err_mean"]:
        a = [r[key] for r in new_per]
        b = [r[key] for r in old_per]
        t = welch_test(a, b)
        if t["p_value"] is not None:
            sig = ("***" if t["p_value"] < 0.001 else
                   "**"  if t["p_value"] < 0.01  else
                   "*"   if t["p_value"] < 0.05  else "")
            d = f"{t['cohens_d']:+.2f}" if t['cohens_d'] is not None else "n/a"
            print(f"  {key:<22} new vs old      t={t['t_stat']:+6.2f}  p={t['p_value']:.4g}{sig:<4} d={d}")

    summary = {
        "phase": "1-6 F10",
        "purpose": "λ-EP 再訓練 CfC で cerebellar online correction",
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
        "stat_tests_vs_no_cfc": test_results,
        "raw_per_seed": {name: r["per_seed"] for name, r in results.items()},
    }

    out = RESULTS_DIR / "f10_cfc_lambda.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)

    print(f"\n  elapsed: {elapsed:.1f}s, saved → {out}")


if __name__ == "__main__":
    main()
