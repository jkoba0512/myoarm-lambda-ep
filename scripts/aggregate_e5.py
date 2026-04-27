"""
E5 結果の集計スクリプト

11 seed の metrics.json を読み込み、各指標の平均・標準偏差を計算する。
出力: results/franka_e5_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
RESULTS_DIR = ROOT / "results" / "experiment_franka_e5"

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42]
D_CONDS   = ["D0", "D1", "D5"]
CTRL_CONDS = ["E5-PD", "E5-E3", "E5-MCA", "E5-full"]
METRICS   = ["mae_post_mrad", "energy_J", "energy_post_J", "jerk", "jerk_post", "cc_ratio", "recovery_time_s", "peak_err_rad"]


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
    loaded_seeds = sorted(all_seeds.keys())
    print(f"Loaded {len(loaded_seeds)} seeds: {loaded_seeds}")

    # metrics[d_cond][ctrl_cond][metric] = list of values over seeds
    collected: dict = {d: {c: {m: [] for m in METRICS} for c in CTRL_CONDS} for d in D_CONDS}

    for seed, data in all_seeds.items():
        for d in D_CONDS:
            for c in CTRL_CONDS:
                row = data.get("results", {}).get(d, {}).get(c, {})
                for m in METRICS:
                    v = row.get(m)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        collected[d][c][m].append(float(v))

    # 平均・std を計算
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

    # 成功判定: E5-full の energy_post_J ≤ E5-E3 の energy_post_J (D0 か D1)
    def mean_of(d, c, m):
        return summary_metrics[d][c][m]["mean"]

    pass_energy = (
        mean_of("D0", "E5-full", "energy_post_J") <= mean_of("D0", "E5-E3", "energy_post_J") or
        mean_of("D1", "E5-full", "energy_post_J") <= mean_of("D1", "E5-E3", "energy_post_J")
    )
    # jerk は物理現象支配のため全条件で実質同一。
    # 「悪化させていない（+5% 以内）」を成功とする。
    JERK_TOLERANCE = 0.05
    pass_jerk = (
        mean_of("D0", "E5-full", "jerk_post") <= mean_of("D0", "E5-E3", "jerk_post") * (1 + JERK_TOLERANCE) or
        mean_of("D1", "E5-full", "jerk_post") <= mean_of("D1", "E5-E3", "jerk_post") * (1 + JERK_TOLERANCE)
    )

    summary = {
        "sweep_name":    sweep_name,
        "seeds":         loaded_seeds,
        "metrics":       summary_metrics,
        "pass_energy":   pass_energy,
        "pass_jerk":     pass_jerk,
        "pass":          pass_energy and pass_jerk,
        "highlights": {
            "energy_post_J": {
                d: {c: round(mean_of(d, c, "energy_post_J"), 1) for c in CTRL_CONDS}
                for d in D_CONDS
            },
            "jerk_post": {
                d: {c: float(f"{mean_of(d, c, 'jerk_post'):.3e}") for c in CTRL_CONDS}
                for d in D_CONDS
            },
            "cc_ratio": {
                d: {c: round(mean_of(d, c, "cc_ratio"), 4) for c in CTRL_CONDS}
                for d in D_CONDS
            },
            "mae_post_mrad": {
                d: {c: round(mean_of(d, c, "mae_post_mrad"), 1) for c in CTRL_CONDS}
                for d in D_CONDS
            },
        },
    }
    return summary


def print_table(summary: dict) -> None:
    print("\n" + "=" * 80)
    print(f"E5 集計結果 (sweep: {summary['sweep_name']})")
    print("=" * 80)

    for metric_key, metric_label in [
        ("mae_post_mrad", "MAE_post [mrad]"),
        ("energy_post_J", "Energy_post [Nm²·s]"),
        ("jerk_post",     "Jerk_post [rad²/s⁵]"),
        ("cc_ratio",      "CC ratio"),
    ]:
        h = summary["highlights"].get(metric_key, {})
        print(f"\n  {metric_label}:")
        header = f"    {'':10s}" + "".join(f"  {d:>12s}" for d in ["D0", "D1", "D5"])
        print(header)
        for c in ["E5-PD", "E5-E3", "E5-MCA", "E5-full"]:
            row = f"    {c:10s}"
            for d in ["D0", "D1", "D5"]:
                v = h.get(d, {}).get(c, float("nan"))
                row += f"  {v:>12.3e}" if metric_key == "jerk_post" else f"  {v:>12.1f}"
            print(row)

    print(f"\n  pass_energy : {summary['pass_energy']}")
    print(f"  pass_jerk   : {summary['pass_jerk']}")
    print(f"  overall PASS: {summary['pass']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-name", type=str, default="sweep11")
    return p.parse_args()


def main():
    args    = parse_args()
    summary = aggregate(args.sweep_name)
    print_table(summary)

    out = ROOT / "results" / "franka_e5_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
