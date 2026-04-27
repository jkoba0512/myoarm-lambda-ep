"""
CPG/LIF 位相整合解析

phase_log.json を読んで以下の指標を算出する:
  - cpg_lif_phase_diff : 外乱後最初の LIF 発火時刻における CPG 位相
  - sign_agreement_rate: q_error と同ステップの cpg_output が逆符号（補正方向）の割合
  - lif_density_at_peak: 外乱ピーク誤差周辺 ±50ms の LIF 発火密度
  - improvement_rate   : CPG+CfC vs CPG+CfC+LIF_FB の mae 改善率（metrics.json から）

入力:
  results/experiment_franka_2c/condition_sweep_2c/*/seed*/phase_log.json
  results/experiment_franka_2c/condition_sweep_2c/*/seed*/metrics.json

出力:
  results/franka_phase_alignment_summary.json

使い方:
  .venv/bin/python scripts/analyze_phase_alignment.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parents[1]
RESULTS_DIR = ROOT / "results"
OUTPUT = RESULTS_DIR / "franka_phase_alignment_summary.json"
COND_DIR = RESULTS_DIR / "experiment_franka_2c" / "condition_sweep_2c"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def sign_agreement_rate(cpg_output: np.ndarray, q_error: np.ndarray) -> float:
    """cpg_output と q_error が逆符号（補正方向）の割合を返す。"""
    cpg_flat = cpg_output.flatten()
    err_flat = q_error.flatten()
    nonzero = np.abs(err_flat) > 1e-6
    if not nonzero.any():
        return float("nan")
    return float(np.mean(cpg_flat[nonzero] * err_flat[nonzero] < 0))


def lif_density_at_peak(
    lif_fired: np.ndarray,  # (T, n_joints) bool
    q_error: np.ndarray,    # (T, n_joints)
    t: np.ndarray,          # (T,)
    load_time_s: float,
    window_s: float = 0.05,
) -> float:
    """外乱後の誤差ピーク周辺 ±window_s の LIF 発火密度を返す。"""
    post_mask = t >= load_time_s
    if not post_mask.any():
        return float("nan")
    post_err = np.abs(q_error[post_mask]).mean(axis=1)
    peak_idx_rel = int(np.argmax(post_err))
    peak_t = t[post_mask][peak_idx_rel]

    window_mask = np.abs(t - peak_t) <= window_s
    if not window_mask.any():
        return float("nan")
    return float(lif_fired[window_mask].mean())


def first_lif_phase_after_load(
    lif_fired: np.ndarray,  # (T, n_joints) bool
    cpg_phase: np.ndarray,  # (T, n_joints)
    t: np.ndarray,          # (T,)
    load_time_s: float,
) -> float | None:
    """外乱後に初めて LIF が発火したステップの CPG 位相平均を返す。"""
    post_indices = np.where(t >= load_time_s)[0]
    for idx in post_indices:
        if lif_fired[idx].any():
            return float(cpg_phase[idx, lif_fired[idx]].mean())
    return None


def analyze_condition(cond_path: Path) -> dict[str, Any] | None:
    """1 条件のすべての seed を集約して指標を返す。"""
    seed_dirs = sorted(cond_path.glob("seed*"))
    if not seed_dirs:
        return None

    per_seed: list[dict[str, Any]] = []
    for seed_dir in seed_dirs:
        plog = load_json(seed_dir / "phase_log.json")
        mlog = load_json(seed_dir / "metrics.json")
        if plog is None or mlog is None:
            continue

        load_t = plog.get("meta", {}).get("load_time_s", 3.0)

        cond_results: dict[str, Any] = {}
        for label in ("CPG+CfC", "CPG+CfC+LIF_FB"):
            if label not in plog:
                continue
            pl = plog[label]
            t_arr   = np.array(pl["t"])
            cpg_out = np.array(pl["cpg_output"])  # (T, n_joints)
            cpg_ph  = np.array(pl["cpg_phase"])   # (T, n_joints)
            lif_f   = np.array(pl["lif_fired"], dtype=bool)  # (T, n_joints)
            q_err   = np.array(pl["q_error"])     # (T, n_joints)

            sar = sign_agreement_rate(cpg_out, q_err)
            ld  = lif_density_at_peak(lif_f, q_err, t_arr, load_t)
            phase0 = first_lif_phase_after_load(lif_f, cpg_ph, t_arr, load_t)
            cond_results[label] = {
                "sign_agreement_rate": sar,
                "lif_density_at_peak": ld,
                "first_lif_phase": phase0,
            }

        # mae 改善率
        imp_rate = float("nan")
        if mlog and "conditions" in mlog:
            mae_no = mlog["conditions"].get("CPG+CfC", {}).get("mae_post_mrad")
            mae_fb = mlog["conditions"].get("CPG+CfC+LIF_FB", {}).get("mae_post_mrad")
            if mae_no and mae_fb and mae_no > 0:
                imp_rate = (mae_no - mae_fb) / mae_no * 100.0

        per_seed.append({
            "seed": int(seed_dir.name.replace("seed", "")),
            "improvement_rate_pct": imp_rate,
            "conditions": cond_results,
        })

    if not per_seed:
        return None

    # seed 間平均
    imp_vals = [s["improvement_rate_pct"] for s in per_seed
                if not math.isnan(s["improvement_rate_pct"])]
    sar_fb_vals = [
        s["conditions"].get("CPG+CfC+LIF_FB", {}).get("sign_agreement_rate", float("nan"))
        for s in per_seed
    ]
    sar_fb_vals = [v for v in sar_fb_vals if not math.isnan(v)]
    ld_fb_vals = [
        s["conditions"].get("CPG+CfC+LIF_FB", {}).get("lif_density_at_peak", float("nan"))
        for s in per_seed
    ]
    ld_fb_vals = [v for v in ld_fb_vals if not math.isnan(v)]

    meta_path = next(iter(seed_dirs)) / "metrics.json"
    meta = load_json(meta_path) or {}

    return {
        "condition": cond_path.name,
        "cpg_tau": meta.get("cpg_tau"),
        "cpg_tau_r": meta.get("cpg_tau_r"),
        "n_seeds": len(per_seed),
        "improvement_rate_pct_mean": float(np.mean(imp_vals)) if imp_vals else None,
        "improvement_rate_pct_std":  float(np.std(imp_vals, ddof=1)) if len(imp_vals) > 1 else None,
        "sign_agreement_rate_mean":  float(np.mean(sar_fb_vals)) if sar_fb_vals else None,
        "lif_density_at_peak_mean":  float(np.mean(ld_fb_vals)) if ld_fb_vals else None,
        "per_seed": per_seed,
    }


def main() -> None:
    if not COND_DIR.exists():
        print(f"ERROR: {COND_DIR} が存在しません。先に実験を実行してください。")
        return

    results = []
    for cond_path in sorted(COND_DIR.iterdir()):
        if not cond_path.is_dir():
            continue
        # phase_log.json があるものだけ解析
        has_plog = any(cond_path.glob("seed*/phase_log.json"))
        if not has_plog:
            print(f"  skip (no phase_log): {cond_path.name}")
            continue
        res = analyze_condition(cond_path)
        if res:
            results.append(res)
            print(f"  {cond_path.name:30s}  imp={res['improvement_rate_pct_mean']:.1f}%"
                  f"  SAR={res['sign_agreement_rate_mean']:.3f}"
                  f"  tau={res['cpg_tau']}  tau_r={res['cpg_tau_r']}")

    if not results:
        print("解析対象の phase_log.json が見つかりませんでした。")
        print("--save-phase-log フラグ付きで 2C 実験を先に実行してください。")
        return

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {OUTPUT}")


if __name__ == "__main__":
    main()
