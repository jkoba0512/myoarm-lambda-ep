"""EF（完全階層）結果の集計スクリプト。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
RESULTS_DIR = ROOT / "results" / "experiment_franka_ef"

SEEDS      = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42]
D_CONDS    = ["D0", "D1", "D5"]
CTRL_CONDS = ["EF-PD", "EF-E4", "EF-cereb", "EF-full"]
METRICS    = [
    "mae_post_mrad", "energy_post_J", "jerk_post", "cc_ratio",
    "recovery_time_s", "peak_err_rad", "pred_err_mrad",
]


def load_seeds(sweep_name: str) -> dict[int, dict]:
    data: dict[int, dict] = {}
    base = RESULTS_DIR / sweep_name
    for seed in SEEDS:
        p = base / f"seed{seed}" / "metrics.json"
        if p.exists():
            with open(p) as f:
                data[seed] = json.load(f)
        else:
            print(f"  WARNING: missing {p}")
    return data


def aggregate(sweep_name: str) -> dict:
    all_seeds = load_seeds(sweep_name)
    loaded    = sorted(all_seeds.keys())
    print(f"Loaded {len(loaded)} seeds: {loaded}")

    collected = {d: {c: {m: [] for m in METRICS} for c in CTRL_CONDS} for d in D_CONDS}

    for seed, data in all_seeds.items():
        for d in D_CONDS:
            for c in CTRL_CONDS:
                row = data.get("results", {}).get(d, {}).get(c, {})
                for m in METRICS:
                    v = row.get(m)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        collected[d][c][m].append(float(v))

    summary_metrics: dict = {}
    for d in D_CONDS:
        summary_metrics[d] = {}
        for c in CTRL_CONDS:
            summary_metrics[d][c] = {}
            for m in METRICS:
                vals = collected[d][c][m]
                summary_metrics[d][c][m] = {
                    "mean": float(np.mean(vals)) if vals else float("nan"),
                    "std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
                    "n":    len(vals),
                }

    def mean_of(d, c, m):
        return summary_metrics[d][c][m]["mean"]

    # 成功判定: EF-full < EF-E4 in mae_post (D0 or D1)
    pass_mae = (
        mean_of("D0", "EF-full", "mae_post_mrad") < mean_of("D0", "EF-E4", "mae_post_mrad") or
        mean_of("D1", "EF-full", "mae_post_mrad") < mean_of("D1", "EF-E4", "mae_post_mrad")
    )
    # EF-full エネルギーが EF-E4 以下（D0 or D1）
    ENERGY_TOL = 0.10
    pass_energy = (
        mean_of("D0", "EF-full", "energy_post_J") <= mean_of("D0", "EF-E4", "energy_post_J") * (1 + ENERGY_TOL) or
        mean_of("D1", "EF-full", "energy_post_J") <= mean_of("D1", "EF-E4", "energy_post_J") * (1 + ENERGY_TOL)
    )

    # 各条件の改善率（対 EF-PD）
    improvement: dict = {}
    for c in ["EF-E4", "EF-cereb", "EF-full"]:
        improvement[c] = {}
        for d in D_CONDS:
            base_v  = mean_of(d, "EF-PD", "mae_post_mrad")
            cond_v  = mean_of(d, c, "mae_post_mrad")
            improvement[c][d] = round(base_v - cond_v, 1)  # 正 = 改善 [mrad]

    highlights = {
        "mae_post_mrad": {
            d: {c: round(mean_of(d, c, "mae_post_mrad"), 1) for c in CTRL_CONDS}
            for d in D_CONDS
        },
        "energy_post_J": {
            d: {c: round(mean_of(d, c, "energy_post_J"), 1) for c in CTRL_CONDS}
            for d in D_CONDS
        },
        "cc_ratio": {
            d: {c: round(mean_of(d, c, "cc_ratio"), 4) for c in CTRL_CONDS}
            for d in D_CONDS
        },
        "pred_err_mrad": {
            d: {c: round(mean_of(d, c, "pred_err_mrad"), 2) for c in CTRL_CONDS}
            for d in D_CONDS
        },
    }

    return {
        "sweep_name":    sweep_name,
        "seeds":         loaded,
        "metrics":       summary_metrics,
        "pass_mae":      pass_mae,
        "pass_energy":   pass_energy,
        "pass":          pass_mae,
        "improvement_vs_PD_mrad": improvement,
        "highlights":    highlights,
    }


def print_table(summary: dict) -> None:
    print("\n" + "=" * 80)
    print(f"EF 集計結果 (sweep: {summary['sweep_name']})")
    print("=" * 80)
    for metric_key, metric_label in [
        ("mae_post_mrad", "MAE_post [mrad]"),
        ("energy_post_J", "Energy_post [Nm²·s]"),
        ("cc_ratio",      "CC ratio"),
        ("pred_err_mrad", "Pred error [mrad]"),
    ]:
        h = summary["highlights"].get(metric_key, {})
        print(f"\n  {metric_label}:")
        print(f"    {'':12s}" + "".join(f"  {d:>10s}" for d in D_CONDS))
        for c in CTRL_CONDS:
            row = f"    {c:12s}"
            for d in D_CONDS:
                v = h.get(d, {}).get(c, float("nan"))
                row += f"  {v:>10.1f}"
            print(row)

    print("\n  MAE_post 改善量（対 EF-PD） [mrad]:")
    print(f"    {'':12s}" + "".join(f"  {d:>10s}" for d in D_CONDS))
    for c in ["EF-E4", "EF-cereb", "EF-full"]:
        row = f"    {c:12s}"
        for d in D_CONDS:
            v = summary["improvement_vs_PD_mrad"].get(c, {}).get(d, float("nan"))
            row += f"  {v:>+10.1f}"
        print(row)

    print(f"\n  pass_mae    : {summary['pass_mae']}")
    print(f"  pass_energy : {summary['pass_energy']}")
    print(f"  overall PASS: {summary['pass']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-name", type=str, default="sweep11")
    return p.parse_args()


def main():
    args    = parse_args()
    summary = aggregate(args.sweep_name)
    print_table(summary)
    out = ROOT / "results" / "franka_ef_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
