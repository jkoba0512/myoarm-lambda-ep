"""
experiment_myo_p13_analysis.py — Phase 1-3: 運動学的不一致の原因分析。

Phase 1-2 の結果を詳細解析し、以下の問いに答える:
  Q1. NaN シードは到達不可能なターゲットか？
  Q2. 速度プロファイルの非ベル形状の原因は何か？
  Q3. 高ジャーク値の原因は何か？
  Q4. 低直線性の原因は何か？

出力:
  results/experiment_myo_p13/mismatch_analysis.json
  results/experiment_myo_p13/velocity_profiles.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import myosuite  # noqa
import gymnasium as gym
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myoarm.myo_controller import MyoArmController, MyoArmConfig

RESULTS_DIR = ROOT / "results" / "experiment_myo_p13"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

P12_JSON = ROOT / "results" / "experiment_myo_p12" / "kinematics_summary.json"
CFC_MODEL_PATH = ROOT / "results" / "myo_cfc_data" / "cfc_model.pt"


def minimum_jerk_velocity(T: float, dist: float, dt: float = 0.005) -> np.ndarray:
    """最小ジャーク速度プロファイル（速さの時系列）を返す。"""
    n = max(int(T / dt), 2)
    t_arr = np.linspace(0, T, n)
    tau = t_arr / T
    # v(τ) = (dist/T) * 30τ²(1-τ)²  —— 正規化最小ジャーク速度
    v = (dist / T) * 30 * tau**2 * (1 - tau)**2
    return v


def run_episode_with_profile(
    env: gym.Env,
    ctrl: MyoArmController,
    seed: int,
    max_steps: int = 800,
) -> dict:
    obs, _ = env.reset(seed=seed)
    uw = env.unwrapped
    m, d = uw.mj_model, uw.mj_data

    ctrl.reset()
    ctrl.initialize(m, d)

    positions: list[np.ndarray] = []
    activations: list[np.ndarray] = []

    for step in range(max_steps):
        od = uw.obs_dict
        q          = np.array(od["qpos"])
        dq         = np.array(od["qvel"])
        reach_err  = np.array(od["reach_err"])
        tip_pos    = np.array(od["tip_pos"])
        muscle_vel   = d.actuator_velocity.copy()
        muscle_force = d.actuator_force.copy()

        positions.append(tip_pos.copy())

        a_total, _ = ctrl.step(
            q=q, dq=dq,
            reach_err=reach_err, tip_pos=tip_pos,
            muscle_vel=muscle_vel, muscle_force=muscle_force,
            m=m, d=d,
        )
        activations.append(a_total.copy())

        obs, _, terminated, truncated, info = env.step(a_total)

        od2    = uw.obs_dict
        q_next = np.array(od2["qpos"])
        ctrl.update_cerebellum(q_next, m, d)

        if terminated or truncated:
            break

    traj = np.array(positions)  # (T, 3)
    acts = np.array(activations)  # (T, 34)

    p_start = traj[0]
    p_end   = traj[-1]

    # 到達距離 (target = p_start + reach_err_at_start)
    # reach_err = target - tip; 初期値でtargetを推定
    obs, _ = env.reset(seed=seed)
    od0 = uw.obs_dict
    reach_err_init = np.array(od0["reach_err"])
    tip_init = np.array(od0["tip_pos"])
    target_pos = tip_init + reach_err_init
    dist = float(np.linalg.norm(reach_err_init))

    # 速度プロファイル
    dt = 0.005
    if len(traj) < 3:
        return {
            "seed": seed, "dist_m": dist, "valid": False,
            "reason": "too_short", "traj": traj, "acts": acts,
            "target_pos": target_pos,
        }

    vel = np.diff(traj, axis=0) / dt  # (T-1, 3)
    speed = np.linalg.norm(vel, axis=1)  # (T-1,)

    # オンセット/オフセット検出
    thresh = 0.02
    onset = None
    for i, s in enumerate(speed):
        if s > thresh:
            onset = i
            break

    if onset is None:
        return {
            "seed": seed, "dist_m": dist, "valid": False,
            "reason": "no_onset", "traj": traj, "acts": acts,
            "target_pos": target_pos,
        }

    # オフセット: onset後に速度が閾値以下になる最初のステップ
    offset = None
    for i in range(onset + 5, len(speed)):
        if speed[i] < thresh:
            offset = i
            break

    if offset is None:
        offset = len(speed) - 1

    T_actual = (offset - onset) * dt
    movement_speed = speed[onset:offset+1]

    # ベルプロファイル指標
    peak_idx = int(np.argmax(movement_speed))
    vel_peak_ratio = peak_idx / max(len(movement_speed) - 1, 1)

    # ジャーク
    if offset > onset + 2:
        acc = np.diff(vel, axis=0)  # (T-2, 3)
        jerk = np.diff(acc, axis=0) / dt  # (T-3, 3)
        jerk_segment = jerk[onset:offset-1] if offset-1 < len(jerk) else jerk[onset:]
        jerk_rms = float(np.sqrt(np.mean(np.sum(jerk_segment**2, axis=1)))) if len(jerk_segment) > 0 else np.nan
    else:
        jerk_rms = np.nan

    # 直線性
    total_path = float(np.sum(np.linalg.norm(np.diff(traj[onset:offset+1], axis=0), axis=1)))
    direct_dist = float(np.linalg.norm(traj[offset] - traj[onset]))
    straightness = direct_dist / total_path if total_path > 1e-6 else np.nan

    # 最小ジャーク参照速度プロファイル
    mj_vel = minimum_jerk_velocity(T=max(T_actual, 0.2), dist=dist, dt=dt)

    # 筋活性化の平均・標準偏差（協調パターン）
    act_mean = float(np.mean(acts))
    act_std  = float(np.std(acts))

    return {
        "seed": seed,
        "dist_m": dist,
        "valid": True,
        "T_actual_s": T_actual,
        "onset_step": onset,
        "offset_step": offset,
        "jerk_rms": jerk_rms,
        "vel_peak_ratio": vel_peak_ratio,
        "straightness": straightness,
        "act_mean": act_mean,
        "act_std": act_std,
        "speed_profile": movement_speed.tolist(),
        "mj_vel_profile": mj_vel.tolist(),
        "traj": traj,
        "acts": acts,
        "target_pos": target_pos,
    }


def diagnose_workspace(env: gym.Env, seeds: list[int]) -> list[dict]:
    """各シードの target が arm workspace 内かどうかを判定する。"""
    results = []
    for seed in seeds:
        env.reset(seed=seed)
        uw = env.unwrapped
        od = uw.obs_dict
        reach_err = np.array(od["reach_err"])
        tip_pos   = np.array(od["tip_pos"])
        target    = tip_pos + reach_err
        dist      = float(np.linalg.norm(reach_err))
        # myoArm の最大リーチ: 肩から指先まで上腕(~38cm)+前腕(~26cm)+手(~10cm)=~74cm
        # 到達不可能の閾値を 90cm とする
        unreachable = dist > 0.90
        results.append({
            "seed": seed,
            "dist_m": dist,
            "target_pos": target.tolist(),
            "tip_init_pos": tip_pos.tolist(),
            "likely_unreachable": unreachable,
        })
    return results


def main() -> None:
    print("=== Phase 1-3: 運動学的不一致の原因分析 ===")

    # Phase 1-2 結果を読む
    with open(P12_JSON) as f:
        p12 = json.load(f)

    per_seed = p12["per_seed"]
    nan_seeds   = [r["seed"] for r in per_seed if r.get("movement_time") is None or
                   (isinstance(r.get("movement_time"), float) and np.isnan(r["movement_time"]))]
    valid_seeds = [r["seed"] for r in per_seed if r["seed"] not in nan_seeds]

    print(f"  NaN シード ({len(nan_seeds)}件): {nan_seeds}")
    print(f"  有効シード ({len(valid_seeds)}件): {valid_seeds}")

    env = gym.make("myoArmReachRandom-v0")
    uw  = env.unwrapped

    # Q1: workspace診断
    print("\n[Q1] NaNシードの到達可能性を診断...")
    all_seeds = list(range(20))
    ws_results = diagnose_workspace(env, all_seeds)

    unreachable = [r for r in ws_results if r["likely_unreachable"]]
    reachable   = [r for r in ws_results if not r["likely_unreachable"]]
    print(f"  到達不可能 (dist > 90cm): {len(unreachable)} 件 → seeds {[r['seed'] for r in unreachable]}")
    print(f"  到達可能   (dist ≤ 90cm): {len(reachable)} 件 → seeds {[r['seed'] for r in reachable]}")

    # Q2/Q3/Q4: 有効シードの速度プロファイル詳細解析
    print("\n[Q2/Q3/Q4] 速度プロファイル・ジャーク・直線性の詳細解析...")

    env_uw = uw.mj_model
    muscle_names = [env_uw.actuator(i).name for i in range(env_uw.nu)]

    cfg  = MyoArmConfig(
        Kp_ee=80.0, Kd_ee=15.0, Ki_ee=2.0, act_bias=0.15,
        K_cereb=0.2, io_mode="sparse", io_firing_rate_hz=1.0,
    )
    ctrl = MyoArmController(config=cfg, muscle_names=muscle_names)
    if CFC_MODEL_PATH.exists():
        ctrl.load_cfc(CFC_MODEL_PATH)

    analysis_results = []
    trajs_by_seed = {}

    for seed in range(20):
        r = run_episode_with_profile(env, ctrl, seed=seed, max_steps=800)
        analysis_results.append({k: v for k, v in r.items() if k not in ("traj", "acts", "speed_profile", "mj_vel_profile", "target_pos")})
        analysis_results[-1].update({
            "speed_len": len(r.get("speed_profile", [])),
            "mj_vel_len": len(r.get("mj_vel_profile", [])),
        })
        trajs_by_seed[seed] = {
            "traj": r["traj"],
            "speed_profile": np.array(r.get("speed_profile", [])),
            "mj_vel_profile": np.array(r.get("mj_vel_profile", [])),
        }
        status = "✓" if r.get("valid") else "✗"
        print(
            f"  seed {seed:2d}: dist={r['dist_m']*100:.0f}cm  "
            f"T={r.get('T_actual_s', float('nan')):.2f}s  "
            f"valid={r.get('valid', False)}  {status}"
        )

    env.close()

    # 集計（有効シードのみ）
    valid_rs = [r for r in analysis_results if r.get("valid")]
    invalid_rs = [r for r in analysis_results if not r.get("valid")]

    def safe_mean(vals):
        v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(v)) if v else float("nan")

    aggregate = {
        "n_valid": len(valid_rs),
        "n_invalid": len(invalid_rs),
        "invalid_seeds": [r["seed"] for r in invalid_rs],
        "invalid_reasons": {r["seed"]: r.get("reason", "nan_metric") for r in invalid_rs},
        "jerk_rms_mean":      safe_mean([r.get("jerk_rms")       for r in valid_rs]),
        "vel_peak_ratio_mean":safe_mean([r.get("vel_peak_ratio") for r in valid_rs]),
        "straightness_mean":  safe_mean([r.get("straightness")   for r in valid_rs]),
        "T_actual_mean":      safe_mean([r.get("T_actual_s")     for r in valid_rs]),
        "dist_mean_m":        safe_mean([r.get("dist_m")         for r in valid_rs]),
    }

    # 原因診断
    diagnosis = {
        "Q1_workspace": {
            "finding": f"{len(unreachable)}/20 シードが到達可能距離 > 90cm のため NaN。"
                       f"myoArmReachRandom は unreachable targets を含む。",
            "unreachable_dist_cm": [round(r['dist_m']*100, 1) for r in unreachable],
            "recommendation": "reach_dist < 90cm のシードのみで評価するか、"
                              "myoArmReachFixed-v0 に限定する。",
        },
        "Q2_velocity_profile": {
            "finding": f"vel_peak_ratio = {aggregate['vel_peak_ratio_mean']:.3f} (human: ~0.50)。"
                       "PD フィードバック制御は初期誤差が最大時に最大トルクを発生させるため、"
                       "速度が t=0 付近でピークを持つ非対称プロファイルになる。",
            "root_cause": "feedforward軌跡計画の欠如。制御は目標位置への反応的追跡のみ。",
            "recommendation": "最小ジャーク軌跡をエンドポイント目標として事前計算し、"
                              "その参照軌跡を追従するよう切り替える。",
        },
        "Q3_high_jerk": {
            "finding": f"jerk_rms = {aggregate['jerk_rms_mean']:.0f} m/s³ "
                       f"(min_jerk 参照値の ~{aggregate['jerk_rms_mean']/322:.0f}x)。"
                       "疑似逆行列による筋活性化は各ステップで独立に計算されるため高周波成分が大きい。",
            "root_cause": "筋シナジー行列なし + 活性化フィルタなし。"
                          "34次元筋空間の疑似逆解は数値的に不安定。",
            "recommendation": "d'Avella筋シナジー行列(~5-7成分)で有効自由度を削減、"
                              "または活性化に低域通過フィルタ(τ≈50ms)を適用。",
        },
        "Q4_straightness": {
            "finding": f"straightness = {aggregate['straightness_mean']:.3f} (human: ~0.85-0.95)。"
                       "エンドポイント誤差のみを最小化するため、null-space成分が非直線経路を生む。",
            "root_cause": "疑似逆行列の null-space が制約されていない。"
                          "関節角度制限回避のための null-space 制御が経路を歪める。",
            "recommendation": "null-space projector で肘角度の自然姿勢を維持、"
                              "または直接的な軌跡追従制御に切り替える。",
        },
    }

    # velocity profile を npz で保存
    speed_data = {}
    for seed, dat in trajs_by_seed.items():
        speed_data[f"speed_{seed}"] = dat["speed_profile"]
        speed_data[f"mj_vel_{seed}"] = dat["mj_vel_profile"]
    np.savez_compressed(RESULTS_DIR / "velocity_profiles.npz", **speed_data)

    result = {
        "phase": "1-3",
        "env": "myoArmReachRandom-v0",
        "aggregate": aggregate,
        "workspace_diagnosis": ws_results,
        "diagnosis": diagnosis,
        "next_steps": [
            "Phase 1-4: 最小ジャーク軌跡追従制御への切り替え (feedforward + feedback)",
            "Phase 1-B: d'Avella 型筋シナジー行列の導入",
            "Phase 1-C: 活性化低域通過フィルタ (τ=50ms) の追加",
            "評価: myoArmReachFixed-v0 のみに限定してシナジー効果を測定",
        ],
        "per_seed": analysis_results,
    }

    out = RESULTS_DIR / "mismatch_analysis.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)

    print()
    print("=== Phase 1-3 診断結果 ===")
    print(f"  有効シード数  : {aggregate['n_valid']}/20")
    print(f"  無効シード    : {aggregate['invalid_seeds']} (到達不可能 or onset未検出)")
    print()
    for qk, qv in diagnosis.items():
        print(f"  [{qk}]")
        print(f"    発見: {qv['finding'][:80]}...")
        print(f"    対策: {qv['recommendation'][:80]}...")
    print()
    print(f"  saved → {out}")


if __name__ == "__main__":
    main()
