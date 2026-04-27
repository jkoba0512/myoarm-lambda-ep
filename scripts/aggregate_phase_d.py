"""
Phase D 結果集計スクリプト

D-main:  PD+CfC vs PD+CfC+Reflex (Full) — 反射弓の追加効果
D-secondary: PD vs PD+CfC vs Full — Sim-to-Real 前段の全体ロバスト性

入力:
  results/experiment_franka_2b/condition_sweep_phase_d/*/seed*/metrics.json

出力:
  results/franka_phase_d_summary.json

使い方:
  .venv/bin/python scripts/aggregate_phase_d.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parents[1]
RESULTS_DIR = ROOT / "results"
PHASE_D_DIR = RESULTS_DIR / "experiment_franka_2b" / "condition_sweep_phase_d"
OUTPUT = RESULTS_DIR / "franka_phase_d_summary.json"


def stats(values: list[float | None]) -> dict[str, Any]:
    valid = [v for v in values if v is not None and not math.isnan(v)]
    n_total = len(values)
    n_valid = len(valid)
    fail_rate = (n_total - n_valid) / n_total if n_total > 0 else None
    if n_valid == 0:
        return {"mean": None, "std": None, "median": None,
                "n": n_total, "n_valid": 0, "fail_rate": fail_rate}
    arr = np.array(valid, dtype=float)
    return {
        "mean":      float(np.mean(arr)),
        "std":       float(np.std(arr, ddof=1)) if n_valid > 1 else 0.0,
        "median":    float(np.median(arr)),
        "min":       float(np.min(arr)),
        "max":       float(np.max(arr)),
        "n":         n_total,
        "n_valid":   n_valid,
        "fail_rate": fail_rate,
    }


def cohens_d(a_vals: list[float | None], b_vals: list[float | None]) -> float | None:
    a = [v for v in a_vals if v is not None and not math.isnan(v)]
    b = [v for v in b_vals if v is not None and not math.isnan(v)]
    if len(a) < 2 or len(b) < 2:
        return None
    pooled_std = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled_std == 0:
        return None
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def load_condition(cond_dir: Path) -> dict[str, Any] | None:
    seed_dirs = sorted(cond_dir.glob("seed*"))
    if not seed_dirs:
        return None

    # controller labels to collect
    ctrl_labels = ["PD+CfC", "Full"]

    per_label: dict[str, dict[str, list]] = {
        lbl: {"peak_err_rad": [], "mae_post_mrad": [], "recovery_time_s": []}
        for lbl in ctrl_labels
    }
    meta: dict[str, Any] = {}
    n_seeds = 0

    for sd in seed_dirs:
        mf = sd / "metrics.json"
        if not mf.exists():
            continue
        with open(mf) as f:
            rec = json.load(f)

        if not meta:
            meta = {k: rec.get(k) for k in (
                "pd_gain_scale", "obs_noise_std", "obs_delay_steps",
                "torque_saturation", "model_mass_scale", "model_friction_scale",
            )}

        for dl_name, dl_data in rec.get("disturbance_levels", {}).items():
            for lbl in ctrl_labels:
                if lbl not in dl_data:
                    continue
                d = dl_data[lbl]
                per_label[lbl]["peak_err_rad"].append(d.get("peak_err_rad"))
                per_label[lbl]["mae_post_mrad"].append(d.get("mae_post_mrad"))
                per_label[lbl]["recovery_time_s"].append(d.get("recovery_time_s"))
        n_seeds += 1

    if n_seeds == 0:
        return None

    cfc_peak = per_label["PD+CfC"]["peak_err_rad"]
    full_peak = per_label["Full"]["peak_err_rad"]
    cfc_mae  = per_label["PD+CfC"]["mae_post_mrad"]
    full_mae  = per_label["Full"]["mae_post_mrad"]
    cfc_rt   = per_label["PD+CfC"]["recovery_time_s"]
    full_rt   = per_label["Full"]["recovery_time_s"]

    return {
        "condition": cond_dir.name,
        "meta": meta,
        "n_seeds": n_seeds,
        "PD+CfC": {
            "peak_err_rad":    stats(cfc_peak),
            "mae_post_mrad":   stats(cfc_mae),
            "recovery_time_s": stats(cfc_rt),
        },
        "Full": {
            "peak_err_rad":    stats(full_peak),
            "mae_post_mrad":   stats(full_mae),
            "recovery_time_s": stats(full_rt),
        },
        "D_main_effect_size": {
            "cohens_d_peak_err": cohens_d(cfc_peak, full_peak),
            "cohens_d_mae":      cohens_d(cfc_mae,  full_mae),
            "cohens_d_rt":       cohens_d(cfc_rt,   full_rt),
        },
    }


def main() -> None:
    if not PHASE_D_DIR.exists():
        print(f"ERROR: {PHASE_D_DIR} が存在しません。Phase D 実験を先に実行してください。")
        return

    results = []
    for cond_dir in sorted(PHASE_D_DIR.iterdir()):
        if not cond_dir.is_dir():
            continue
        res = load_condition(cond_dir)
        if res is None:
            print(f"  skip (no data): {cond_dir.name}")
            continue
        results.append(res)
        cfc_mean = res["PD+CfC"]["peak_err_rad"].get("mean")
        full_mean = res["Full"]["peak_err_rad"].get("mean")
        cd = res["D_main_effect_size"]["cohens_d_peak_err"]
        cfc_str  = f"{cfc_mean:.4f}" if cfc_mean is not None else "N/A"
        full_str = f"{full_mean:.4f}" if full_mean is not None else "N/A"
        cd_str   = f"{cd:.3f}"        if cd is not None         else "N/A"
        print(f"  {cond_dir.name:25s}  PD+CfC peak={cfc_str}  Full peak={full_str}  d={cd_str}")

    payload = {"phase_d": results}
    with open(OUTPUT, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {OUTPUT}")


if __name__ == "__main__":
    main()
