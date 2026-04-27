"""
集計スクリプト: 全 seed の metrics.json を読み込み、
平均・標準偏差・中央値・効果量・失敗率を計算して
results/franka_master_summary.json を出力する。

使い方:
  uv run python scripts/aggregate_results.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT        = Path(__file__).parents[1]
RESULTS_DIR = ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "franka_master_summary.json"


# ──────────────────────────────────────────────────────────────────────
def load_seed_jsons(exp: str) -> list[dict]:
    """results/experiment_franka_{exp}/seed*/metrics.json を全部読む。"""
    pattern = RESULTS_DIR / f"experiment_franka_{exp}" / "seed*" / "metrics.json"
    files   = sorted(Path(RESULTS_DIR / f"experiment_franka_{exp}").glob("seed*/metrics.json"))
    data    = []
    for f in files:
        with open(f) as fp:
            data.append(json.load(fp))
    return data


def stats(values: list[float | None]) -> dict:
    """None を失敗として扱い、有効値で統計を計算する。"""
    valid = [v for v in values if v is not None and not math.isnan(v)]
    n_total   = len(values)
    n_valid   = len(valid)
    fail_rate = (n_total - n_valid) / n_total if n_total > 0 else None

    if n_valid == 0:
        return {"mean": None, "std": None, "median": None,
                "min": None, "max": None,
                "n": n_total, "n_valid": 0, "fail_rate": fail_rate}

    arr = np.array(valid)
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
    """Cohen's d（効果量）: (mean_a - mean_b) / pooled_std"""
    a = [v for v in a_vals if v is not None and not math.isnan(v)]
    b = [v for v in b_vals if v is not None and not math.isnan(v)]
    if len(a) < 2 or len(b) < 2:
        return None
    mean_diff  = np.mean(a) - np.mean(b)
    pooled_std = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float(mean_diff / pooled_std) if pooled_std > 0 else None


# ──────────────────────────────────────────────────────────────────────
def aggregate_2a(records: list[dict]) -> dict:
    """2A: 各条件の static_mae_mrad を集計。"""
    conditions = ["PD", "PD+CfC", "Full"]
    result: dict[str, Any] = {}

    for cond in conditions:
        vals = []
        for r in records:
            v = r.get("conditions", {}).get(cond, {}).get("static_mae_mrad")
            vals.append(v)
        result[cond] = {"static_mae_mrad": stats(vals)}

    # 効果量: PD vs PD+CfC, PD vs Full
    pd_vals  = [r.get("conditions", {}).get("PD",     {}).get("static_mae_mrad") for r in records]
    cfc_vals = [r.get("conditions", {}).get("PD+CfC", {}).get("static_mae_mrad") for r in records]
    full_vals = [r.get("conditions", {}).get("Full",  {}).get("static_mae_mrad") for r in records]
    result["effect_size"] = {
        "cohens_d_PD_vs_PDCfC": cohens_d(pd_vals, cfc_vals),
        "cohens_d_PD_vs_Full":  cohens_d(pd_vals, full_vals),
    }
    return result


def aggregate_2b(records: list[dict]) -> dict:
    """2B: 外乱強度ごとに peak_err_rad, recovery_time_s を集計。"""
    dist_levels = ["light_30Nm", "medium_60Nm", "heavy_87Nm"]
    conditions  = ["PD+CfC", "Full"]
    result: dict[str, Any] = {}

    for dl in dist_levels:
        result[dl] = {}
        for cond in conditions:
            peak_vals = []
            rt_vals   = []
            for r in records:
                entry = r.get("disturbance_levels", {}).get(dl, {}).get(cond, {})
                peak_vals.append(entry.get("peak_err_rad"))
                rt_vals.append(entry.get("recovery_time_s"))
            result[dl][cond] = {
                "peak_err_rad":    stats(peak_vals),
                "recovery_time_s": stats(rt_vals),
            }
        # 効果量
        cfc_peak  = [r.get("disturbance_levels",{}).get(dl,{}).get("PD+CfC",{}).get("peak_err_rad") for r in records]
        full_peak = [r.get("disturbance_levels",{}).get(dl,{}).get("Full",{}).get("peak_err_rad")   for r in records]
        result[dl]["effect_size"] = {
            "cohens_d_peak_err_PDCfC_vs_Full": cohens_d(cfc_peak, full_peak),
        }
    return result


def aggregate_2c(records: list[dict]) -> dict:
    """2C: 各条件の ep_err_post_mrad を集計。"""
    conditions = ["CPG+CfC", "CPG+CfC+LIF_FB"]
    result: dict[str, Any] = {}

    for cond in conditions:
        ep_pre  = [r.get("conditions", {}).get(cond, {}).get("ep_err_pre_mrad")  for r in records]
        ep_post = [r.get("conditions", {}).get(cond, {}).get("ep_err_post_mrad") for r in records]
        mae_post = [r.get("conditions", {}).get(cond, {}).get("mae_post_mrad")    for r in records]
        result[cond] = {
            "ep_err_pre_mrad":  stats(ep_pre),
            "ep_err_post_mrad": stats(ep_post),
            "mae_post_mrad":    stats(mae_post),
        }

    no_vals = [r.get("conditions", {}).get("CPG+CfC",        {}).get("ep_err_post_mrad") for r in records]
    fb_vals = [r.get("conditions", {}).get("CPG+CfC+LIF_FB", {}).get("ep_err_post_mrad") for r in records]
    result["effect_size"] = {
        "cohens_d_ep_post_nofb_vs_fb": cohens_d(no_vals, fb_vals),
    }
    return result


def aggregate_2d(records: list[dict]) -> dict:
    """2D: 各条件の static_mae_mrad, peak_err_rad, recovery_time_s を集計。"""
    conditions = ["PD", "PD+CfC", "Full"]
    result: dict[str, Any] = {}

    for cond in conditions:
        mae_vals  = [r.get("conditions", {}).get(cond, {}).get("static_mae_mrad") for r in records]
        peak_vals = [r.get("conditions", {}).get(cond, {}).get("peak_err_rad")    for r in records]
        rt_vals   = [r.get("conditions", {}).get(cond, {}).get("recovery_time_s") for r in records]
        result[cond] = {
            "static_mae_mrad": stats(mae_vals),
            "peak_err_rad":    stats(peak_vals),
            "recovery_time_s": stats(rt_vals),
        }

    pd_mae  = [r.get("conditions",{}).get("PD",   {}).get("static_mae_mrad") for r in records]
    full_mae = [r.get("conditions",{}).get("Full", {}).get("static_mae_mrad") for r in records]
    result["effect_size"] = {
        "cohens_d_mae_PD_vs_Full": cohens_d(pd_mae, full_mae),
    }
    return result


# ──────────────────────────────────────────────────────────────────────
def print_summary(master: dict) -> None:
    seeds = master.get("seeds_used", [])
    print(f"\n{'='*60}")
    print(f"  franka_master_summary  ({len(seeds)} seeds: {seeds})")
    print(f"{'='*60}")

    # 2A
    print("\n[2A] Ablation (static_mae_mrad)")
    for cond in ["PD", "PD+CfC", "Full"]:
        s = master["2a"][cond]["static_mae_mrad"]
        if s["mean"] is not None:
            print(f"  {cond:12s}: {s['mean']:.1f} ± {s['std']:.1f}  "
                  f"[{s['min']:.1f}, {s['max']:.1f}]  fail={s['fail_rate']:.0%}")

    # 2B heavy
    print("\n[2B] Disturbance rejection (heavy_87Nm, peak_err_rad)")
    for cond in ["PD+CfC", "Full"]:
        s = master["2b"]["heavy_87Nm"][cond]["peak_err_rad"]
        if s["mean"] is not None:
            print(f"  {cond:8s}: {s['mean']:.3f} ± {s['std']:.3f}  fail={s['fail_rate']:.0%}")

    # 2C
    print("\n[2C] Cyclic motion (ep_err_post_mrad)")
    for cond in ["CPG+CfC", "CPG+CfC+LIF_FB"]:
        s = master["2c"][cond]["ep_err_post_mrad"]
        if s["mean"] is not None:
            print(f"  {cond:20s}: {s['mean']:.1f} ± {s['std']:.1f}  fail={s['fail_rate']:.0%}")

    # 2D
    print("\n[2D] Integrated (static_mae_mrad)")
    for cond in ["PD", "PD+CfC", "Full"]:
        s = master["2d"][cond]["static_mae_mrad"]
        if s["mean"] is not None:
            print(f"  {cond:8s}: {s['mean']:.1f} ± {s['std']:.1f}  fail={s['fail_rate']:.0%}")

    print(f"\n  → 保存先: {OUTPUT_PATH}")


# ──────────────────────────────────────────────────────────────────────
def main():
    master: dict[str, Any] = {}

    for exp, agg_fn in [("2a", aggregate_2a), ("2b", aggregate_2b),
                        ("2c", aggregate_2c), ("2d", aggregate_2d)]:
        records = load_seed_jsons(exp)
        if not records:
            print(f"  [WARN] {exp}: seed データなし（先に run_seed_sweep.py を実行してください）")
            master[exp] = {}
            continue
        seeds_in = sorted({r["seed"] for r in records})
        print(f"  {exp}: {len(records)} seeds 読み込み ({seeds_in})")
        master[exp] = agg_fn(records)

    # 使用 seed 一覧（全実験の和集合）
    all_seeds: set[int] = set()
    for exp in ["2a", "2b", "2c", "2d"]:
        recs = load_seed_jsons(exp)
        all_seeds.update(r["seed"] for r in recs)
    master["seeds_used"] = sorted(all_seeds)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(master, f, indent=2, ensure_ascii=False)

    print_summary(master)


if __name__ == "__main__":
    main()
