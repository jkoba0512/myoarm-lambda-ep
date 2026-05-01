"""
experiment_myo_p15_f3_synergy.py — F3: 筋シナジー(NMF)で純PDの一般化を改善する。

F4 で純PDが Random env で min_err=124mm に留まることを確認した。仮説:
  3D の F_ee を J_act^+ で 34 筋に分配する際、ヤコビアン疑似逆行列の
  条件数が大きく、関節→筋へのノイズ増幅が悪影響を与えている。
  筋活性化を低次元シナジー基底 W (34×k) の張る部分空間に拘束すれば、
  control authority が機能的部分集合に絞られて一般化が改善するはず。

訓練・テスト分離（汎化を正しく測るため必須）:
  train seeds : reach_dist<0.85m を満たすシード (50..149 から先頭 20)
  test  seeds : reach_dist<0.85m を満たすシード (0..49 から先頭 20)  ← F4 と共通

手順:
  (A) 純PD で train seeds を走らせ a_total を毎ステップ収集 → A (T×34)
  (B) 自前 NMF(k∈{4,6,8}) で W_k (34×k), H_k (k×T) に分解
  (C) test seeds で純PD + 射影 a' = clip(W_k @ pinv(W_k) @ a_total, 0, 1) を評価
       比較対象: pure_PD (k=∞ 相当)

指標     : F4 と同一 (tip_err, vel_peak_ratio, straightness, jerk, dir_err, progress)
検定     : 各 k vs pure_PD で Welch's t-test (主要指標 = tip_err_min_mm)
α        : 0.05

NMF 実装 (sklearn なしで自己完結):
  Lee & Seung 2001 の multiplicative update を numpy で実装。
  ・ 正値性は初期化と更新式で自動的に保たれる
  ・ 終了基準: max_iter または ||A - WH||_F の相対変化 < tol
  ・ シード固定で再現性を担保

出力:
  results/experiment_myo_p15/f3_synergy.json
  results/experiment_myo_p15/f3_synergy_basis.npz   (W_k4/W_k6/W_k8, A_train)
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

RESULTS_DIR    = ROOT / "results" / "experiment_myo_p15"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DT          = 0.020
MAX_REACH_M = 0.85
N_REACHABLE = 20
TRAIN_POOL  = list(range(50, 150))
TEST_POOL   = list(range(50))
SYNERGY_KS  = [4, 6, 8]
NMF_SEED    = 0


# ──────────────────────────────────────────────────────────────────────
# 自前 NMF (Lee & Seung 2001, multiplicative update)
# ──────────────────────────────────────────────────────────────────────

def nmf_multiplicative(A: np.ndarray, k: int, *,
                       max_iter: int = 500,
                       tol: float = 1e-5,
                       seed: int = 0,
                       eps: float = 1e-9) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """A ≈ W H, A ∈ R+^(m×n), W ∈ R+^(m×k), H ∈ R+^(k×n).

    multiplicative update:
        W ← W ⊙ (A H^T) / (W H H^T + eps)
        H ← H ⊙ (W^T A) / (W^T W H + eps)
    終了: 相対 Frobenius 残差変化 < tol、または max_iter 到達。
    """
    rng = np.random.default_rng(seed)
    m, n = A.shape
    A_safe = np.maximum(A, 0.0)
    scale = float(np.sqrt(A_safe.mean() / k)) if A_safe.mean() > 0 else 1.0
    W = rng.uniform(eps, scale, size=(m, k))
    H = rng.uniform(eps, scale, size=(k, n))

    losses = []
    prev = None
    for it in range(max_iter):
        WH = W @ H
        loss = float(np.linalg.norm(A_safe - WH, ord="fro"))
        losses.append(loss)
        if prev is not None and abs(prev - loss) / max(prev, eps) < tol:
            break
        prev = loss

        # H update
        num_H = W.T @ A_safe
        den_H = W.T @ W @ H + eps
        H *= num_H / den_H

        # W update
        num_W = A_safe @ H.T
        den_W = W @ (H @ H.T) + eps
        W *= num_W / den_W

    return W, H, losses


# ──────────────────────────────────────────────────────────────────────
# 運動学 (F4 と同一)
# ──────────────────────────────────────────────────────────────────────

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


def find_reachable_seeds(env: gym.Env, pool: list[int], n: int) -> list[int]:
    out = []
    for s in pool:
        deterministic_reset(env, s)
        od = env.unwrapped.obs_dict
        d = float(np.linalg.norm(np.array(od["reach_err"])))
        if d < MAX_REACH_M:
            out.append(s)
        if len(out) >= n:
            break
    return out


# ──────────────────────────────────────────────────────────────────────
# 1 エピソード (synergy_proj=W@pinv(W) を a_total にかける)
# ──────────────────────────────────────────────────────────────────────

def run_episode(env, ctrl, seed, *, synergy_proj: np.ndarray | None = None,
                max_steps: int = 600, log_a_total: bool = False) -> dict:
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
    a_log = [] if log_a_total else None
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
        if synergy_proj is not None:
            a_total = np.clip(synergy_proj @ a_total, 0.0, 1.0).astype(np.float32)
        if log_a_total:
            a_log.append(np.array(a_total, copy=True))

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
    cos_angle = float(np.clip(np.dot(travel_unit, target_dir_unit), -1.0, 1.0))
    direction_error_deg = float(np.degrees(np.arccos(cos_angle)))
    progress_m = float(np.dot(travel, target_dir_unit))
    progress_ratio = progress_m / max(target_norm, 1e-9)

    km = compute_kinematics(positions, dt=DT)
    out = {
        "seed": seed, "solved": solved,
        "tip_err_min_mm":   min(errs)*1000 if errs else float("nan"),
        "tip_err_final_mm": errs[-1]*1000 if errs else float("nan"),
        "direction_error_deg": direction_error_deg,
        "progress_ratio": progress_ratio,
        "target_dist_m": target_norm,
        **km,
    }
    if log_a_total:
        out["_a_total"] = np.array(a_log)  # (T, 34)
    return out


def make_pd_cfg() -> MyoArmConfig:
    return MyoArmConfig(
        Kp_ee=80.0, Kd_ee=15.0, Ki_ee=2.0, act_bias=0.15,
        K_cereb=0.0, K_ia=0.0, K_ib=0.0, K_ri=0.0,
        io_mode="sparse", io_firing_rate_hz=0.0,
        traj_speed_gain=1.2, traj_dt=DT, vel_scale_min=0.10,
        Kd_traj=50.0, use_traj_plan=False,
    )


def run_condition(env, muscle_names, seeds: list[int],
                  synergy_proj: np.ndarray | None = None) -> dict:
    cfg = make_pd_cfg()
    ctrl = MyoArmController(config=cfg, muscle_names=muscle_names)
    results = [run_episode(env, ctrl, seed=s, synergy_proj=synergy_proj) for s in seeds]

    metric_keys = ["tip_err_min_mm", "tip_err_final_mm",
                   "vel_peak_ratio", "straightness", "jerk_rms",
                   "peak_speed", "movement_time",
                   "direction_error_deg", "progress_ratio"]

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


def collect_train_activations(env, muscle_names, seeds: list[int]) -> np.ndarray:
    """train seeds で純PDを走らせ a_total を集める → (T_total, 34)。"""
    cfg = make_pd_cfg()
    ctrl = MyoArmController(config=cfg, muscle_names=muscle_names)
    A_list = []
    for s in seeds:
        r = run_episode(env, ctrl, seed=s, log_a_total=True)
        A_list.append(r["_a_total"])
    A = np.concatenate(A_list, axis=0)  # (T_total, 34)
    return A


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


# ──────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Phase 1-5 F3: 筋シナジー(NMF)で純PDの一般化を改善 ===")

    env = gym.make("myoArmReachRandom-v0")
    uw  = env.unwrapped
    env.reset(seed=0)
    m = uw.mj_model
    muscle_names = [m.actuator(i).name for i in range(m.nu)]

    # ── train/test 分離 ──
    test_seeds  = find_reachable_seeds(env, TEST_POOL,  N_REACHABLE)
    train_seeds = find_reachable_seeds(env, TRAIN_POOL, N_REACHABLE)
    print(f"  test  seeds (n={len(test_seeds)}): {test_seeds}")
    print(f"  train seeds (n={len(train_seeds)}): {train_seeds}")
    overlap = set(test_seeds) & set(train_seeds)
    assert not overlap, f"train/test seeds overlap: {overlap}"

    t0 = time.time()

    # ── (A) 訓練フェーズ: 純PD で a_total 収集 ──
    print("\n[A] 訓練フェーズ: 純PD で train seeds の a_total 収集")
    A_train = collect_train_activations(env, muscle_names, train_seeds)
    print(f"  collected A: shape={A_train.shape}  range=[{A_train.min():.3f}, {A_train.max():.3f}]  "
          f"mean={A_train.mean():.3f}")

    # ── (B) NMF 分解 ──
    print("\n[B] NMF 分解")
    A_T = A_train.T  # NMF expects (m=34, n=T)
    bases = {}
    nmf_loss = {}
    for k in SYNERGY_KS:
        W, H, losses = nmf_multiplicative(A_T, k=k, seed=NMF_SEED)
        recon = W @ H
        relerr = float(np.linalg.norm(A_T - recon) / max(np.linalg.norm(A_T), 1e-9))
        bases[k] = W
        nmf_loss[k] = {"final_fro": losses[-1], "iters": len(losses), "rel_err": relerr}
        print(f"  k={k}: iters={len(losses):3d}  fro_loss={losses[-1]:.3f}  "
              f"rel_err={relerr:.3f}")

    # ── (C) 評価フェーズ: test seeds で純PD vs 純PD+synergy(k) ──
    print("\n[C] 評価フェーズ: test seeds で純PD と各 synergy(k) を比較")
    results = {}

    # baseline: pure_PD (no projection)
    agg = run_condition(env, muscle_names, test_seeds, synergy_proj=None)
    results["pure_PD"] = agg
    s = agg["stats"]
    print(f"\n  [pure_PD]               solve={agg['n_solved']:2d}/{len(test_seeds)}  "
          f"min_err={s['tip_err_min_mm']['mean']:5.1f}±{s['tip_err_min_mm']['std']:5.1f}mm  "
          f"progress={s['progress_ratio']['mean']:+.2f}  "
          f"dir_err={s['direction_error_deg']['mean']:5.1f}°")

    for k in SYNERGY_KS:
        W = bases[k]                                # (34, k)
        proj = W @ np.linalg.pinv(W)                # (34, 34)
        agg = run_condition(env, muscle_names, test_seeds, synergy_proj=proj)
        results[f"synergy_k{k}"] = agg
        s = agg["stats"]
        print(f"  [synergy_k{k}]            solve={agg['n_solved']:2d}/{len(test_seeds)}  "
              f"min_err={s['tip_err_min_mm']['mean']:5.1f}±{s['tip_err_min_mm']['std']:5.1f}mm  "
              f"progress={s['progress_ratio']['mean']:+.2f}  "
              f"dir_err={s['direction_error_deg']['mean']:5.1f}°")
        print(f"                          vpr={s['vel_peak_ratio']['mean']:.3f}  "
              f"straight={s['straightness']['mean']:.3f}  "
              f"jerk={s['jerk_rms']['mean']:.0f}")

    env.close()
    elapsed = time.time() - t0

    # ── 統計検定: 各 synergy_k vs pure_PD ──
    print("\n=== Welch's t-test: 各 synergy_k vs pure_PD ===")
    test_results = {}
    pd_per_seed = results["pure_PD"]["per_seed"]
    for key in ["tip_err_min_mm", "progress_ratio", "direction_error_deg"]:
        pd_vals = [r[key] for r in pd_per_seed]
        for k in SYNERGY_KS:
            cond_vals = [r[key] for r in results[f"synergy_k{k}"]["per_seed"]]
            t = welch_test(cond_vals, pd_vals)
            test_results.setdefault(key, {})[f"synergy_k{k}"] = t
            if t["p_value"] is not None:
                sig = ("***" if t["p_value"] < 0.001 else
                       "**"  if t["p_value"] < 0.01  else
                       "*"   if t["p_value"] < 0.05  else "")
                d = f"{t['cohens_d']:+.2f}" if t['cohens_d'] is not None else "n/a"
                print(f"  {key:<22} synergy_k{k} vs pure_PD  t={t['t_stat']:+6.2f}  "
                      f"p={t['p_value']:.4g}{sig:<4}  d={d}")
            else:
                print(f"  {key:<22} synergy_k{k} (insufficient data)")

    summary = {
        "phase": "1-5 F3",
        "purpose": "筋シナジー射影で純PDのRandom env一般化を改善",
        "env": "myoArmReachRandom-v0",
        "dt": DT,
        "reachable_threshold_m": MAX_REACH_M,
        "train_seeds": train_seeds,
        "test_seeds":  test_seeds,
        "synergy_ks":  SYNERGY_KS,
        "nmf": {
            "method": "multiplicative_update_lee_seung_2001",
            "max_iter": 500, "tol": 1e-5, "seed": NMF_SEED,
            "A_train_shape": list(A_train.shape),
            "loss": nmf_loss,
        },
        "elapsed_s": round(elapsed, 1),
        "conditions": {
            name: {
                "n_solved":   r["n_solved"],
                "solve_rate": r["solve_rate"],
                "stats":      r["stats"],
            }
            for name, r in results.items()
        },
        "stat_tests_vs_pure_PD": test_results,
        "raw_per_seed": {
            name: [{kk: vv for kk, vv in r.items() if kk != "_a_total"}
                   for r in res["per_seed"]]
            for name, res in results.items()
        },
    }

    out_json = RESULTS_DIR / "f3_synergy.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)

    out_npz = RESULTS_DIR / "f3_synergy_basis.npz"
    np.savez(out_npz,
             A_train=A_train,
             **{f"W_k{k}": bases[k] for k in SYNERGY_KS})

    print(f"\n  elapsed: {elapsed:.1f}s")
    print(f"  saved → {out_json}")
    print(f"  saved → {out_npz}")


if __name__ == "__main__":
    main()
