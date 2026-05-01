"""
exp_utils.py — myoArm 実験スクリプト共通ヘルパ。

F4-F12 で重複していた以下を集約 (各スクリプトで ~50 行重複していた):
  - compute_kinematics : tip 軌跡 (T,3) から運動学指標
  - find_reachable_seeds: reach_dist<MAX で reachable subset を抽出
  - welch_test         : Welch's t-test + Cohen's d
  - stats_for_results  : per_seed 結果リストから mean/std/n を計算

新規スクリプト (F13 以降) は `from myoarm.exp_utils import ...` で利用する。
既存 F1-F12 はそのまま動作 (互換性維持のため触らない)。

定数 DEFAULT_DT, DEFAULT_MAX_REACH_M はメモリ feedback_dt_unit に基づく:
  myoArm 実制御周期 dt=0.020s (frame_skip=10 × mj_dt=0.002s)
  reachable threshold ≈ workspace 上限 0.85m
"""

from __future__ import annotations

from typing import Iterable

import gymnasium as gym
import numpy as np
from scipy import stats


DEFAULT_DT          = 0.020
DEFAULT_MAX_REACH_M = 0.85


def compute_kinematics(positions: np.ndarray, dt: float = DEFAULT_DT) -> dict:
    """tip 軌跡 (T, 3) から運動学指標を計算する。

    Returns
    -------
    dict with keys:
      jerk_rms       : 速度の三階微分 RMS norm [m/s³]
      vel_peak_ratio : 速度ピークが movement の何 % で起きるか (人間 ~0.4-0.5)
      straightness   : 直線距離 / 経路長 (1 = 完全直線)
      movement_time  : speed > 0.02 m/s の継続時間 [s]
      peak_speed     : 最大速度 [m/s]

    境界条件: 軌跡が短すぎる (<5 サンプル) or 動きがない場合は NaN 多数返却。
    """
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
        "jerk_rms":       jerk_rms,
        "vel_peak_ratio": float(vpr),
        "straightness":   float(straightness),
        "movement_time":  float(T_actual),
        "peak_speed":     float(np.max(movement_speed)),
    }


def find_reachable_seeds(
    env: gym.Env,
    pool: Iterable[int] = range(50),
    n: int = 20,
    max_reach_m: float = DEFAULT_MAX_REACH_M,
) -> list[int]:
    """reach_dist < max_reach_m を満たす seed を pool から最大 n 個取得する。

    内部で deterministic_reset を使うので env_utils が必須。
    `pool=range(50)` と `n=20` の組み合わせは F4/F8/F12 と一致する標準テストセット。
    """
    # 循環 import 回避のためここで import
    from myoarm.env_utils import deterministic_reset

    out: list[int] = []
    for s in pool:
        deterministic_reset(env, s)
        od = env.unwrapped.obs_dict
        d = float(np.linalg.norm(np.array(od["reach_err"])))
        if d < max_reach_m:
            out.append(int(s))
        if len(out) >= n:
            break
    return out


def welch_test(a, b) -> dict:
    """Welch's t-test (unequal variances) + Cohen's d を計算する。

    Returns
    -------
    dict with keys:
      t_stat, p_value, cohens_d : 検定統計量
      significant               : bool, p < 0.05
      n_a, n_b                  : 有効サンプル数 (NaN/None 除去後)

    NaN/None は両配列から除去される。各配列が 2 未満のとき t/p/d は None。
    """
    a = [x for x in a if x is not None and not (isinstance(x, float) and np.isnan(x))]
    b = [x for x in b if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if len(a) < 2 or len(b) < 2:
        return {"t_stat": None, "p_value": None, "cohens_d": None,
                "significant": False, "n_a": len(a), "n_b": len(b)}
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    cohens_d = (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else float("nan")
    return {
        "t_stat":      float(t_stat),
        "p_value":     float(p_value),
        "cohens_d":    float(cohens_d) if not np.isnan(cohens_d) else None,
        "significant": bool(p_value < 0.05),
        "n_a":         len(a),
        "n_b":         len(b),
    }


def stats_for_results(results: list[dict], keys: list[str]) -> dict[str, dict]:
    """run_episode の per_seed 結果リストから各 key の mean/std/n を計算する。

    使い方:
      keys = ["tip_err_min_mm", "direction_error_deg", "progress_ratio"]
      stats = stats_for_results(per_seed_list, keys)
      # stats["tip_err_min_mm"] = {"mean": ..., "std": ..., "n": ...}
    """
    out: dict[str, dict] = {}
    for k in keys:
        vals = [r[k] for r in results
                if r.get(k) is not None
                and not (isinstance(r.get(k), float) and np.isnan(r.get(k)))]
        if not vals:
            out[k] = {"mean": float("nan"), "std": float("nan"), "n": 0}
        else:
            out[k] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0,
                "n":    len(vals),
            }
    return out


def sig_marker(p: float | None) -> str:
    """p-value から有意水準マーカーを返す (*, **, ***, '')。"""
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""
