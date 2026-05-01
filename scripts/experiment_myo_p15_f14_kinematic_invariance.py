"""
experiment_myo_p15_f14_kinematic_invariance.py — F14: ヒト reach kinematic
invariance との照合

F13 で vel_peak_ratio=0.067 (peak が movement の開始 6.7% で起きる) が判明、
これはヒト reach の 0.40-0.50 (Flash & Hogan 1985) と乖離。
F12 best が「biological motion」と主張するなら以下を満たすべき:

  metric                  | human reference         | ours (must measure)
  ------------------------|------------------------|---------------------
  vel_peak_ratio          | 0.40-0.50 (F&H 1985)   | F13 で 0.067 (異常?)
  velocity profile shape  | bell-shaped, symmetric | skewness 計測
  Morasso D/L straight.   | > 0.85 (Morasso 1981)  | F12 で 0.671
  normalized jerk (J*T⁵/D²) | min 360 (Hogan 1984) | 計測
  peak speed              | 0.5-1.5 m/s (depends)  | F12 で 1.47 ✓

 F14 で測定する追加指標:
  - skewness of speed profile (negative = peak rightward of midpoint)
  - kurtosis (peakedness)
  - symmetry_ratio (time before peak / total time, 0.5 = symmetric)
  - Morasso D/L 確認
  - normalized jerk = jerk_rms × T^5 / D^2

設計:
  test seeds: 0..49 reachable subset (n=20)
  conditions:
    1. F12 best (pure λ visuo P=10)        ← F12 結果
    2. + reflexes (full neural)            ← 反射 ON で human-like か
    3. − virtual_trajectory                ← smooth λ の効果
    4. − visuomotor                        ← visuomotor の効果
    5. endpoint_pd (reference)             ← PD 制御の比較
  出力: 各条件の per-seed kinematic 詳細 + trajectory 保存

期待される所見:
  - F12 best のヒト適合度を確認
  - どの成分が bell-shape / 直線性 に寄与しているか分解

出力:
  results/experiment_myo_p15/f14_kinematic_invariance.json
  results/experiment_myo_p15/f14_trajectories.npz  (per-seed positions)
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

RESULTS_DIR     = ROOT / "results" / "experiment_myo_p15"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CFC_LAMBDA_PATH = ROOT / "results" / "myo_cfc_data_lambda" / "cfc_model.pt"
CFC_OLD_PATH    = ROOT / "results" / "myo_cfc_data"        / "cfc_model.pt"

DT          = DEFAULT_DT
N_REACHABLE = 20

# ── ヒト reference 値 (literature) ──────────────────────────────────
HUMAN_REF = {
    "vel_peak_ratio": (0.40, 0.50),    # Flash & Hogan 1985
    "skewness":       (-0.5, 0.5),     # roughly symmetric (~ 0)
    "symmetry_ratio": (0.40, 0.55),    # peak near midpoint
    "straightness":   (0.85, 1.00),    # Morasso 1981
    "norm_jerk_min":  360,             # Hogan 1984 (theoretical minimum-jerk)
    "peak_speed":     (0.30, 2.00),    # depends on amplitude
}


def compute_full_kinematics(positions: np.ndarray, target_dist: float,
                            dt: float = DT) -> dict:
    """通常 compute_kinematics に追加して bell-shape, normalized jerk 等を返す。"""
    if len(positions) < 5:
        return {k: float("nan") for k in
                ["jerk_rms", "vel_peak_ratio", "straightness", "movement_time",
                 "peak_speed", "skewness", "kurtosis", "symmetry_ratio",
                 "norm_jerk", "n_speed_samples"]}
    vel = np.diff(positions, axis=0) / dt
    speed = np.linalg.norm(vel, axis=1)
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt

    thresh = 0.02
    onset = next((i for i, s in enumerate(speed) if s > thresh), None)
    if onset is None:
        return {k: float("nan") for k in
                ["jerk_rms", "vel_peak_ratio", "straightness", "movement_time",
                 "peak_speed", "skewness", "kurtosis", "symmetry_ratio",
                 "norm_jerk", "n_speed_samples"]}
    offset = next((i for i in range(onset+5, len(speed)) if speed[i] < thresh),
                  len(speed)-1)
    movement_speed = speed[onset:offset+1]
    n_samples = len(movement_speed)
    T_actual = (offset - onset) * dt
    peak_idx = int(np.argmax(movement_speed))
    vpr = peak_idx / max(n_samples - 1, 1)

    # bell-shape statistics
    if n_samples >= 5:
        skew = float(sp_stats.skew(movement_speed))
        kurt = float(sp_stats.kurtosis(movement_speed))
    else:
        skew = float("nan"); kurt = float("nan")
    symmetry_ratio = peak_idx / max(n_samples - 1, 1)  # = vpr

    # jerk
    jerk_seg = jerk[onset:offset-1] if offset-1 < len(jerk) else jerk[onset:]
    jerk_rms = (float(np.sqrt(np.mean(np.sum(jerk_seg**2, axis=1))))
                if len(jerk_seg) > 0 else float("nan"))

    # path metrics
    seg = positions[onset:offset+2]
    L_path = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))
    D = float(np.linalg.norm(seg[-1] - seg[0]))
    straightness = D / max(L_path, 1e-6)

    # normalized jerk: J_rms × T^5 / D^2  (Hogan 1984)
    # min-jerk theoretical: 360 / T^5 × something — actually formula is
    # J_norm = sqrt(0.5 * integral(jerk^2) * T^5 / D^2)
    # 実用簡易: jerk_rms * T^(2.5) / D は dimensionless 近似
    if T_actual > 0 and D > 0.01:
        norm_jerk = jerk_rms * (T_actual**2.5) / D
    else:
        norm_jerk = float("nan")

    return {
        "jerk_rms":        jerk_rms,
        "vel_peak_ratio":  float(vpr),
        "straightness":    float(straightness),
        "movement_time":   float(T_actual),
        "peak_speed":      float(np.max(movement_speed)),
        "skewness":        skew,
        "kurtosis":        kurt,
        "symmetry_ratio":  float(symmetry_ratio),
        "norm_jerk":       float(norm_jerk),
        "n_speed_samples": int(n_samples),
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

    km = compute_full_kinematics(positions, target_dist=target_norm, dt=DT)
    return {
        "seed": seed, "solved": solved,
        "tip_err_min_mm":      min(errs)*1000 if errs else float("nan"),
        "tip_err_final_mm":    errs[-1]*1000 if errs else float("nan"),
        "direction_error_deg": direction_error_deg,
        "progress_ratio":      progress_ratio,
        "target_dist_m":       target_norm,
        "_positions":          positions,  # 軌跡保存用
        **km,
    }


def make_cfg(*,
             control_mode: str = "lambda_ep",
             lambda_trajectory: bool = True,
             visuomotor: bool = True,
             reflexes: bool = False,
             K_cereb: float = 0.0) -> MyoArmConfig:
    return MyoArmConfig(
        Kp_ee=80.0, Kd_ee=15.0, Ki_ee=2.0, act_bias=0.15,
        K_cereb=K_cereb,
        K_ia=0.05 if reflexes else 0.0,
        K_ib=0.03 if reflexes else 0.0,
        K_ri=0.5  if reflexes else 0.0,
        io_mode="sparse",
        io_firing_rate_hz=1.0 if reflexes else 0.0,
        control_mode=control_mode,
        c_lambda=20.0, lambda_offset=0.005,
        lambda_trajectory=lambda_trajectory,
        lambda_traj_speed_gain=1.2,
        visuomotor_feedback=visuomotor,
        visuomotor_period_steps=10,
        traj_dt=DT, use_traj_plan=False,
    )


def run_condition(env, muscle_names, cfg, seeds, cfc_path):
    ctrl = MyoArmController(config=cfg, muscle_names=muscle_names)
    if cfc_path is not None and cfc_path.exists():
        ctrl.load_cfc(cfc_path)
    results = [run_episode(env, ctrl, seed=s) for s in seeds]
    keys = ["tip_err_min_mm", "tip_err_final_mm",
            "vel_peak_ratio", "straightness", "jerk_rms", "peak_speed",
            "movement_time", "skewness", "kurtosis", "symmetry_ratio",
            "norm_jerk", "direction_error_deg", "progress_ratio"]
    return {
        "n_solved": sum(r["solved"] for r in results),
        "stats":    stats_for_results(results, keys),
        "per_seed": results,
    }


def fmt_human_check(value: float, key: str) -> str:
    """ヒトリファレンス範囲との比較マーカー。"""
    if np.isnan(value):
        return "n/a"
    ref = HUMAN_REF.get(key)
    if ref is None:
        return ""
    if isinstance(ref, tuple):
        lo, hi = ref
        if lo <= value <= hi:  return "✓ in range"
        if value < lo:         return f"⚠ below {lo}"
        return f"⚠ above {hi}"
    return ""


def main():
    print("=== F14: kinematic invariance vs ヒト reach reference ===")
    env = gym.make("myoArmReachRandom-v0")
    deterministic_reset(env, 0)
    uw = env.unwrapped
    m = uw.mj_model
    muscle_names = [m.actuator(i).name for i in range(m.nu)]
    seeds = find_reachable_seeds(env, pool=range(50), n=N_REACHABLE)
    print(f"  test seeds: {seeds}")

    conditions = [
        ("F12 best (pure λ visuo)",
         make_cfg(), None),
        ("+ reflexes",
         make_cfg(reflexes=True), None),
        ("− virtual_trajectory",
         make_cfg(lambda_trajectory=False), None),
        ("− visuomotor",
         make_cfg(visuomotor=False), None),
        ("endpoint_pd reference",
         make_cfg(control_mode="endpoint_pd", lambda_trajectory=False,
                  visuomotor=False, reflexes=True, K_cereb=0.2),
         CFC_OLD_PATH),
    ]

    t0 = time.time()
    results = {}
    saved_trajs = {}
    for name, cfg, cfc in conditions:
        agg = run_condition(env, muscle_names, cfg, seeds, cfc)
        results[name] = agg

        # 軌跡保存 (per-seed)
        saved_trajs[name] = [r["_positions"] for r in agg["per_seed"]]

        s = agg["stats"]
        print(f"\n  [{name}]")
        print(f"    min_err={s['tip_err_min_mm']['mean']:5.1f}mm  "
              f"vpr={s['vel_peak_ratio']['mean']:.3f} {fmt_human_check(s['vel_peak_ratio']['mean'], 'vel_peak_ratio')}")
        print(f"    skewness={s['skewness']['mean']:+.2f} {fmt_human_check(s['skewness']['mean'], 'skewness')}  "
              f"kurtosis={s['kurtosis']['mean']:+.2f}  "
              f"symmetry={s['symmetry_ratio']['mean']:.3f} {fmt_human_check(s['symmetry_ratio']['mean'], 'symmetry_ratio')}")
        print(f"    straight={s['straightness']['mean']:.3f} {fmt_human_check(s['straightness']['mean'], 'straightness')}  "
              f"peak_v={s['peak_speed']['mean']:.2f} {fmt_human_check(s['peak_speed']['mean'], 'peak_speed')}  "
              f"jerk_rms={s['jerk_rms']['mean']:.0f}")
        print(f"    norm_jerk={s['norm_jerk']['mean']:.2f}  "
              f"movement_time={s['movement_time']['mean']:.2f}s")
    env.close()

    # F12 best vs その他で検定
    print("\n=== Welch's t-test: vs F12 best ===")
    base_per = results["F12 best (pure λ visuo)"]["per_seed"]
    test_results = {}
    for name in [n for n, _, _ in conditions[1:]]:
        cond_per = results[name]["per_seed"]
        for key in ["vel_peak_ratio", "skewness", "straightness", "peak_speed",
                    "norm_jerk", "tip_err_min_mm", "direction_error_deg"]:
            t = welch_test([r[key] for r in cond_per], [r[key] for r in base_per])
            test_results.setdefault(key, {})[name] = t
            if t["p_value"] is not None:
                d_str = f"{t['cohens_d']:+.2f}" if t['cohens_d'] is not None else "n/a"
                print(f"  {key:<22} {name:<28} t={t['t_stat']:+6.2f}  "
                      f"p={t['p_value']:.4g}{sig_marker(t['p_value']):<4} d={d_str}")

    # 軌跡保存
    np.savez(
        RESULTS_DIR / "f14_trajectories.npz",
        seeds=np.array(seeds),
        **{f"traj_{name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('−', 'minus')}":
           np.array([t for t in saved_trajs[name]], dtype=object)
           for name in saved_trajs}
    )

    # サマリ JSON
    summary = {
        "phase": "1-6 F14",
        "purpose": "kinematic invariance verification vs human reach",
        "human_reference": HUMAN_REF,
        "n_seeds": len(seeds), "seeds": seeds,
        "elapsed_s": round(time.time()-t0, 1),
        "conditions": {n: {"n_solved": r["n_solved"], "stats": r["stats"]}
                       for n, r in results.items()},
        "stat_tests_vs_F12_best": test_results,
        "raw_per_seed": {n: [{k: v for k, v in r.items() if k != "_positions"}
                             for r in res["per_seed"]]
                         for n, res in results.items()},
    }
    out = RESULTS_DIR / "f14_kinematic_invariance.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print(f"\n  elapsed: {time.time()-t0:.1f}s")
    print(f"  saved → {out}")
    print(f"  saved → {RESULTS_DIR / 'f14_trajectories.npz'}")


if __name__ == "__main__":
    main()
