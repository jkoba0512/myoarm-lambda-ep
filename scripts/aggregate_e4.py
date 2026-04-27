"""
E4 sweep の集計と成功判定

E4-hold 成功基準:
  E4-full の mae_post_mrad が D0/D1/D5 の少なくとも1条件で E4-E3 以下

E4-switch 成功基準:
  E4-MCA-switch の hold_mae が E4-PD-switch の hold_mae 以下
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT    = Path(__file__).parents[1]
E4_DIR  = ROOT / "results" / "experiment_franka_e4"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

SEEDS        = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42]
D_CONDITIONS = ["D0", "D1", "D5"]
HOLD_CONDS   = ["E4-PD", "E4-E3", "E4-MCA", "E4-full"]
SWITCH_CONDS = ["E4-PD-switch", "E4-MCA-switch"]


def load_e4(sweep_name: str) -> tuple[dict, dict]:
    """
    E4 metrics をシードにわたって集計する。

    Returns
    -------
    hold_data   : [d_cond][cname] → list of mae_post_mrad
    switch_data : [cname] → {"hold_mae": [...], "oscillate_mae": [...], "transition_jerk": [...]}
    """
    hold_data: dict[str, dict[str, list[float]]] = {d: {} for d in D_CONDITIONS}
    switch_data: dict[str, dict[str, list[float]]] = {c: {} for c in SWITCH_CONDS}

    base = E4_DIR / sweep_name
    for seed in SEEDS:
        path = base / f"seed{seed}" / "metrics.json"
        if not path.exists():
            print(f"  WARN: missing {path}")
            continue
        with open(path) as f:
            data = json.load(f)

        # hold
        for d_name in D_CONDITIONS:
            d_res = data.get("hold_results", {}).get(d_name, {})
            for cname, cdata in d_res.items():
                mae = cdata.get("mae_post_mrad")
                if mae is None:
                    mae = cdata.get("mae_mrad")
                if mae is None or (isinstance(mae, float) and math.isnan(mae)):
                    continue
                hold_data[d_name].setdefault(cname, []).append(float(mae))

        # switch
        for cname in SWITCH_CONDS:
            sw = data.get("switch_results", {}).get(cname, {})
            if not sw:
                continue
            for key in ("hold_mae_mrad", "oscillate_mae_mrad", "transition_jerk"):
                v = sw.get(key)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                switch_data[cname].setdefault(key, []).append(float(v))

    return hold_data, switch_data


def safe_mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else math.nan


def safe_std(vals: list[float]) -> float:
    return float(np.std(vals, ddof=1)) if len(vals) >= 2 else math.nan


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-name", default="sweep11")
    args = p.parse_args()

    print(f"\n{'='*65}")
    print(f"  E4 Aggregation  sweep={args.sweep_name}")
    print(f"{'='*65}")

    hold_data, switch_data = load_e4(args.sweep_name)

    # ── E4-hold 集計 ──────────────────────────────────────────────
    means_mae_post: dict[str, dict[str, float]] = {}  # [cname][d_name]

    print(f"\n--- E4-hold: MAE_post 平均 ± std [mrad] ---")
    print(f"  {'cond':<12}  {'D0':>18}  {'D1':>18}  {'D5':>18}  n")

    for cname in HOLD_CONDS:
        row: dict[str, float] = {}
        cells = []
        for d_name in D_CONDITIONS:
            vals = hold_data[d_name].get(cname, [])
            m    = safe_mean(vals)
            s    = safe_std(vals)
            row[d_name] = m
            cells.append(f"{m:6.1f}±{s:4.1f}" if vals else "     nan")
        means_mae_post[cname] = row
        n = len(hold_data.get("D0", {}).get(cname, []))
        print(f"  {cname:<12}  {'  '.join(f'{c:>18}' for c in cells)}  {n}")

    # E4-hold 成功判定: E4-full <= E4-E3 in any D condition
    improvement_vs_e3: dict[str, float] = {}
    for d_name in D_CONDITIONS:
        full_m = means_mae_post.get("E4-full", {}).get(d_name, math.nan)
        e3_m   = means_mae_post.get("E4-E3",   {}).get(d_name, math.nan)
        improvement_vs_e3[d_name] = e3_m - full_m  # 正なら改善

    pass_hold = any(
        improvement_vs_e3.get(d, -math.inf) >= 0.0
        for d in D_CONDITIONS
        if not math.isnan(improvement_vs_e3.get(d, math.nan))
    )
    print(f"\n  E4-hold 判定: {'PASS ✓' if pass_hold else 'FAIL ✗'}")
    for d_name in D_CONDITIONS:
        delta = improvement_vs_e3.get(d_name, math.nan)
        print(f"    {d_name}: E4-full vs E4-E3 delta = {delta:+.1f} mrad  "
              f"({'改善' if delta >= 0 else '悪化'})")

    # ── E4-switch 集計 ─────────────────────────────────────────────
    print(f"\n--- E4-switch: hold/oscillate MAE [mrad] & jerk ---")
    print(f"  {'cond':<20}  {'hold_mae':>18}  {'osc_mae':>18}  {'jerk':>12}  n")

    means_hold_mae:     dict[str, float] = {}
    means_oscillate_mae: dict[str, float] = {}

    for cname in SWITCH_CONDS:
        sw = switch_data.get(cname, {})
        hm_vals  = sw.get("hold_mae_mrad", [])
        om_vals  = sw.get("oscillate_mae_mrad", [])
        jk_vals  = sw.get("transition_jerk", [])

        hm = safe_mean(hm_vals)
        om = safe_mean(om_vals)
        jk = safe_mean(jk_vals)
        hs = safe_std(hm_vals)
        os_ = safe_std(om_vals)
        jks = safe_std(jk_vals)

        means_hold_mae[cname]      = hm
        means_oscillate_mae[cname] = om

        hm_str = f"{hm:6.1f}±{hs:4.1f}" if hm_vals else "     nan"
        om_str = f"{om:6.1f}±{os_:4.1f}" if om_vals else "     nan"
        jk_str = f"{jk:.4f}±{jks:.4f}" if jk_vals else "     nan"
        n = len(hm_vals)
        print(f"  {cname:<20}  {hm_str:>18}  {om_str:>18}  {jk_str:>12}  {n}")

    # E4-switch 成功判定: MCA-switch hold_mae <= PD-switch hold_mae
    pd_hm  = means_hold_mae.get("E4-PD-switch",  math.nan)
    mca_hm = means_hold_mae.get("E4-MCA-switch", math.nan)
    pass_switch = (
        not math.isnan(pd_hm) and not math.isnan(mca_hm) and mca_hm <= pd_hm
    )
    print(f"\n  E4-switch 判定: {'PASS ✓' if pass_switch else 'FAIL ✗'}")
    print(f"    E4-MCA-switch hold_mae={mca_hm:.1f} <= E4-PD-switch hold_mae={pd_hm:.1f}? {pass_switch}")

    # ── JSON 保存 ─────────────────────────────────────────────────
    summary = {
        "sweep_name": args.sweep_name,
        "seeds":      SEEDS,
        "E4_hold": {
            "means_mae_post_mrad": means_mae_post,
            "pass": pass_hold,
            "improvement_vs_e3": {
                d: float(improvement_vs_e3[d]) for d in D_CONDITIONS
            },
        },
        "E4_switch": {
            "means_hold_mae_mrad":      {c: float(v) for c, v in means_hold_mae.items()},
            "means_oscillate_mae_mrad": {c: float(v) for c, v in means_oscillate_mae.items()},
            "pass_switch": pass_switch,
        },
    }
    out_path = OUT_DIR / "franka_e4_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n結果保存: {out_path}")

    # ── 詳細 std テーブル ─────────────────────────────────────────
    print("\n--- E4-hold MAE_post 平均 ± std [mrad] 詳細 ---")
    for cname in HOLD_CONDS:
        row = []
        for d_name in D_CONDITIONS:
            vals = hold_data[d_name].get(cname, [])
            if vals:
                row.append(f"{np.mean(vals):6.1f}±{np.std(vals, ddof=1):4.1f}")
            else:
                row.append("     nan")
        print(f"  {cname:<12}  {'  '.join(row)}")


if __name__ == "__main__":
    main()
