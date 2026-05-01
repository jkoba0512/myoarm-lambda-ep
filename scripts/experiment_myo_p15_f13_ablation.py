"""
experiment_myo_p15_f13_ablation.py — F13: F12 best 構成の成分 ablation

F12 で得た最良条件 `pure λ visuo P=10` (神経成分すべて off) から、
各成分を 1 つずつ on/off して必須要素を特定する。

ベースライン (F12 best):
  control_mode    = "lambda_ep"
  lambda_trajectory   = True (virtual trajectory)
  visuomotor_feedback = True, period_steps = 10 (200ms)
  c_lambda=20, lambda_offset=0.005
  K_cereb=0, K_ia=0, K_ib=0, K_ri=0 (神経成分すべて off)

ablation 条件 (8 conditions):
  1. F12 best baseline                       (基準)
  2. − virtual_trajectory    (静的 λ)
  3. − visuomotor                            (F8/F9 baseline)
  4. − virtual_trajectory − visuomotor       (静的 λ のみ)
  5. + reflexes (Ia/Ib/RI)                   (反射追加)
  6. + reflexes + K_cereb=0.2 (joint cereb)
  7. + reflexes + K_cereb_lambda=0.5 (λ cereb)
  8. endpoint_pd reference                   (PD 制御参照)

主要指標 : tip_err_min_mm, direction_error_deg, peak_speed, jerk_rms
検定     : 各条件 vs F12 best baseline で Welch's t-test (Cohen's d)
n        : 20 reachable seeds (test pool 0..49)

期待される所見:
  - virtual_trajectory off で peak_v 上昇 (smoothness 寄与)
  - visuomotor off で min_err 大悪化 (精度の主源)
  - reflexes/CfC は寄与小 ~ 中

出力: results/experiment_myo_p15/f13_ablation.json
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
from myoarm.exp_utils import (
    compute_kinematics, find_reachable_seeds, welch_test,
    stats_for_results, sig_marker, DEFAULT_DT,
)

RESULTS_DIR     = ROOT / "results" / "experiment_myo_p15"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CFC_LAMBDA_PATH = ROOT / "results" / "myo_cfc_data_lambda" / "cfc_model.pt"
CFC_OLD_PATH    = ROOT / "results" / "myo_cfc_data"        / "cfc_model.pt"

DT          = DEFAULT_DT
N_REACHABLE = 20


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
        if info.get("solved", False):
            solved = True
        if term or trunc:
            break

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
        "tip_err_min_mm":      min(errs)*1000 if errs else float("nan"),
        "tip_err_final_mm":    errs[-1]*1000 if errs else float("nan"),
        "direction_error_deg": direction_error_deg,
        "progress_ratio":      progress_ratio,
        "target_dist_m":       target_norm,
        **km,
    }


def make_cfg(*,
             control_mode: str = "lambda_ep",
             lambda_trajectory: bool = True,
             visuomotor: bool = True,
             reflexes: bool = False,
             K_cereb: float = 0.0,
             cereb_target: str = "joint",
             K_cereb_lambda: float = 0.0) -> MyoArmConfig:
    """ablation 用の設定生成。F12 best 構成を base にフラグで切替。"""
    return MyoArmConfig(
        Kp_ee=80.0, Kd_ee=15.0, Ki_ee=2.0, act_bias=0.15,
        # 神経成分
        K_cereb=K_cereb,
        K_ia=0.05 if reflexes else 0.0,
        K_ib=0.03 if reflexes else 0.0,
        K_ri=0.5  if reflexes else 0.0,
        io_mode="sparse",
        io_firing_rate_hz=1.0 if reflexes else 0.0,
        # 制御モード
        control_mode=control_mode,
        c_lambda=20.0, lambda_offset=0.005,
        lambda_trajectory=lambda_trajectory,
        lambda_traj_speed_gain=1.2,
        # 視覚運動
        visuomotor_feedback=visuomotor,
        visuomotor_period_steps=10,
        # CfC 補正
        cereb_correction_target=cereb_target,
        K_cereb_lambda=K_cereb_lambda,
        # 旧 PD 用 (lambda_ep では未使用)
        traj_dt=DT, use_traj_plan=False,
    )


def run_condition(env, muscle_names, cfg: MyoArmConfig, seeds: list[int],
                  cfc_path: Path | None) -> dict:
    ctrl = MyoArmController(config=cfg, muscle_names=muscle_names)
    if cfc_path is not None and cfc_path.exists():
        ctrl.load_cfc(cfc_path)
    results = [run_episode(env, ctrl, seed=s) for s in seeds]
    keys = ["tip_err_min_mm", "tip_err_final_mm",
            "vel_peak_ratio", "straightness", "jerk_rms", "peak_speed",
            "direction_error_deg", "progress_ratio"]
    return {
        "n_solved": sum(r["solved"] for r in results),
        "stats": stats_for_results(results, keys),
        "per_seed": results,
    }


def main():
    print("=== F13 Ablation: F12 best の成分 on/off で必須要素を特定 ===")
    env = gym.make("myoArmReachRandom-v0")
    deterministic_reset(env, 0)
    uw = env.unwrapped
    m = uw.mj_model
    muscle_names = [m.actuator(i).name for i in range(m.nu)]
    seeds = find_reachable_seeds(env, pool=range(50), n=N_REACHABLE)
    print(f"  test seeds (n={len(seeds)}): {seeds}")

    # 8 ablation conditions
    conditions = [
        # (name, cfg, cfc_path)
        ("F12 best (pure λ visuo)",
         make_cfg(),  # default = baseline
         None),
        ("− virtual_trajectory",
         make_cfg(lambda_trajectory=False),
         None),
        ("− visuomotor",
         make_cfg(visuomotor=False),
         None),
        ("− vt − visuo (static λ)",
         make_cfg(lambda_trajectory=False, visuomotor=False),
         None),
        ("+ reflexes",
         make_cfg(reflexes=True),
         None),
        ("+ reflexes + K_cereb=0.2 joint",
         make_cfg(reflexes=True, K_cereb=0.2, cereb_target="joint"),
         CFC_LAMBDA_PATH),
        ("+ reflexes + K_cereb_λ=0.5",
         make_cfg(reflexes=True, cereb_target="lambda", K_cereb_lambda=0.5),
         CFC_LAMBDA_PATH),
        ("endpoint_pd (reference)",
         make_cfg(control_mode="endpoint_pd", lambda_trajectory=False,
                  visuomotor=False, reflexes=True, K_cereb=0.2),
         CFC_OLD_PATH),
    ]

    t0 = time.time()
    results = {}
    for name, cfg, cfc_path in conditions:
        agg = run_condition(env, muscle_names, cfg, seeds, cfc_path)
        results[name] = agg
        s = agg["stats"]
        print(f"\n  [{name}]")
        print(f"    solve={agg['n_solved']:2d}/{len(seeds)}  "
              f"min_err={s['tip_err_min_mm']['mean']:5.1f}±{s['tip_err_min_mm']['std']:5.1f}mm  "
              f"final_err={s['tip_err_final_mm']['mean']:5.1f}mm")
        print(f"    progress={s['progress_ratio']['mean']:+.3f}  "
              f"dir_err={s['direction_error_deg']['mean']:5.1f}°  "
              f"straight={s['straightness']['mean']:.3f}  "
              f"peak_v={s['peak_speed']['mean']:.2f}  jerk={s['jerk_rms']['mean']:.0f}")
    env.close()

    # Welch's t-test: 各 ablation vs F12 best baseline
    print("\n=== Welch's t-test: 各条件 vs F12 best baseline ===")
    base_per = results["F12 best (pure λ visuo)"]["per_seed"]
    test_results = {}
    metric_keys = ["tip_err_min_mm", "tip_err_final_mm",
                   "direction_error_deg", "peak_speed", "jerk_rms",
                   "progress_ratio"]
    for name in [n for n, _, _ in conditions[1:]]:
        cond_per = results[name]["per_seed"]
        for key in metric_keys:
            t = welch_test([r[key] for r in cond_per], [r[key] for r in base_per])
            test_results.setdefault(key, {})[name] = t
            if t["p_value"] is not None:
                d_str = f"{t['cohens_d']:+.2f}" if t['cohens_d'] is not None else "n/a"
                print(f"  {key:<22} {name:<32} t={t['t_stat']:+6.2f}  "
                      f"p={t['p_value']:.4g}{sig_marker(t['p_value']):<4} d={d_str}")

    summary = {
        "phase": "1-6 F13 ablation",
        "purpose": "F12 best 構成の成分 ablation",
        "n_seeds": len(seeds), "seeds": seeds,
        "elapsed_s": round(time.time()-t0, 1),
        "conditions": {n: {"n_solved": r["n_solved"], "stats": r["stats"]}
                       for n, r in results.items()},
        "stat_tests_vs_F12_best": test_results,
        "raw_per_seed": {n: r["per_seed"] for n, r in results.items()},
    }
    out = RESULTS_DIR / "f13_ablation.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print(f"\n  elapsed: {time.time()-t0:.1f}s, saved → {out}")


if __name__ == "__main__":
    main()
