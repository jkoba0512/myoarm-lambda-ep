"""
条件スイープ結果を集計する。

入力:
  results/experiment_franka_2b/condition_sweep_2b/*/seed*/metrics.json
  results/experiment_franka_2c/condition_sweep_2c/*/seed*/metrics.json

出力:
  results/franka_condition_sweep_summary.json

使い方:
  .venv/bin/python scripts/aggregate_condition_sweeps.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).parents[1]
RESULTS_DIR = ROOT / "results"
OUTPUT = RESULTS_DIR / "franka_condition_sweep_summary.json"


def stats(values: list[float | None]) -> dict[str, Any]:
    valid = [v for v in values if v is not None and not math.isnan(v)]
    n_total = len(values)
    n_valid = len(valid)
    fail_rate = (n_total - n_valid) / n_total if n_total > 0 else None
    if n_valid == 0:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "n": n_total,
            "n_valid": 0,
            "fail_rate": fail_rate,
        }
    arr = np.array(valid, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if n_valid > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": n_total,
        "n_valid": n_valid,
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


def load_jsons(pattern: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(ROOT.glob(pattern)):
        with open(path) as f:
            records.append(json.load(f))
    return records


def aggregate_2b(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        sweep_name = rec.get("sweep_name", "")
        condition_name = Path(sweep_name).name
        by_condition.setdefault(condition_name, []).append(rec)

    out: dict[str, Any] = {}
    for condition_name, recs in sorted(by_condition.items()):
        pdcfc_peak = []
        full_peak = []
        pdcfc_mae = []
        full_mae = []
        pdcfc_rt = []
        full_rt = []
        for rec in recs:
            level_name, level = next(iter(rec["disturbance_levels"].items()))
            _ = level_name
            pdcfc_peak.append(level["PD+CfC"]["peak_err_rad"])
            full_peak.append(level["Full"]["peak_err_rad"])
            pdcfc_mae.append(level["PD+CfC"]["mae_post_mrad"])
            full_mae.append(level["Full"]["mae_post_mrad"])
            pdcfc_rt.append(level["PD+CfC"]["recovery_time_s"])
            full_rt.append(level["Full"]["recovery_time_s"])
        out[condition_name] = {
            "PD+CfC": {
                "peak_err_rad": stats(pdcfc_peak),
                "mae_post_mrad": stats(pdcfc_mae),
                "recovery_time_s": stats(pdcfc_rt),
            },
            "Full": {
                "peak_err_rad": stats(full_peak),
                "mae_post_mrad": stats(full_mae),
                "recovery_time_s": stats(full_rt),
            },
            "effect_size": {
                "cohens_d_peak_err_PDCfC_vs_Full": cohens_d(pdcfc_peak, full_peak),
                "cohens_d_mae_post_PDCfC_vs_Full": cohens_d(pdcfc_mae, full_mae),
                "cohens_d_recovery_time_PDCfC_vs_Full": cohens_d(pdcfc_rt, full_rt),
            },
            "n_seeds": len(recs),
        }
    return out


def aggregate_2c(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        sweep_name = rec.get("sweep_name", "")
        condition_name = Path(sweep_name).name
        by_condition.setdefault(condition_name, []).append(rec)

    out: dict[str, Any] = {}
    for condition_name, recs in sorted(by_condition.items()):
        nofb_pre = []
        nofb_post = []
        nofb_mae = []
        nofb_rt = []
        fb_pre = []
        fb_post = []
        fb_mae = []
        fb_rt = []
        _cpg_tau = recs[0].get("cpg_tau", 0.3)
        meta = {
            "load_torque_nm": recs[0]["load_torque_nm"],
            "cpg_amplitude": recs[0]["cpg_amplitude"],
            "cpg_tau": _cpg_tau,
            "cpg_tau_r": recs[0].get("cpg_tau_r", _cpg_tau * 2.0),
            "load_time_s": recs[0]["load_time_s"],
            "load_joint": recs[0]["load_joint"],
            "endpoint_joint": recs[0]["endpoint_joint"],
        }
        for rec in recs:
            c0 = rec["conditions"]["CPG+CfC"]
            c1 = rec["conditions"]["CPG+CfC+LIF_FB"]
            nofb_pre.append(c0["ep_err_pre_mrad"])
            nofb_post.append(c0["ep_err_post_mrad"])
            nofb_mae.append(c0["mae_post_mrad"])
            nofb_rt.append(c0["recovery_time_s"])
            fb_pre.append(c1["ep_err_pre_mrad"])
            fb_post.append(c1["ep_err_post_mrad"])
            fb_mae.append(c1["mae_post_mrad"])
            fb_rt.append(c1["recovery_time_s"])
        out[condition_name] = {
            "meta": meta,
            "CPG+CfC": {
                "ep_err_pre_mrad": stats(nofb_pre),
                "ep_err_post_mrad": stats(nofb_post),
                "mae_post_mrad": stats(nofb_mae),
                "recovery_time_s": stats(nofb_rt),
            },
            "CPG+CfC+LIF_FB": {
                "ep_err_pre_mrad": stats(fb_pre),
                "ep_err_post_mrad": stats(fb_post),
                "mae_post_mrad": stats(fb_mae),
                "recovery_time_s": stats(fb_rt),
            },
            "effect_size": {
                "cohens_d_ep_err_post_nofb_vs_fb": cohens_d(nofb_post, fb_post),
                "cohens_d_mae_post_nofb_vs_fb": cohens_d(nofb_mae, fb_mae),
                "cohens_d_recovery_time_nofb_vs_fb": cohens_d(nofb_rt, fb_rt),
            },
            "n_seeds": len(recs),
        }
    return out


def main() -> None:
    rec2b = load_jsons("results/experiment_franka_2b/condition_sweep_2b/*/seed*/metrics.json")
    rec2c = load_jsons("results/experiment_franka_2c/condition_sweep_2c/*/seed*/metrics.json")
    payload = {
        "2b_condition_sweep": aggregate_2b(rec2b),
        "2c_condition_sweep": aggregate_2c(rec2c),
    }
    with open(OUTPUT, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"保存: {OUTPUT}")


if __name__ == "__main__":
    main()
