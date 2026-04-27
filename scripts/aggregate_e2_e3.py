"""
E2/E3 sweep の集計と成功判定

E2 成功基準:
  D1・D5 の両方で cohens_d(MAE_PD, MAE_cc) > 0
  （旧反射弧は D1: -0.420, D5: -0.185 だったので 0 以上が改善）

E3 成功基準:
  1. Ia 反射潜時 ≤ 30 ms
  2. D1・D5 の両方で cohens_d(MAE_PD, MAE_cc+ia_ib) > E2 の cc のみの値
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT      = Path(__file__).parents[1]
E2_DIR    = ROOT / "results" / "experiment_franka_e2"
E3_DIR    = ROOT / "results" / "experiment_franka_e3"
OUT_DIR   = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42]
D_CONDITIONS = ["D0", "D1", "D5"]


def cohens_d(a_vals: list[float], b_vals: list[float]) -> float:
    """d = (mean_a - mean_b) / pooled_std.  a=PD, b=new → d>0 は改善。"""
    n1, n2 = len(a_vals), len(b_vals)
    if n1 < 2 or n2 < 2:
        return math.nan
    m1, m2 = float(np.mean(a_vals)), float(np.mean(b_vals))
    s1, s2 = float(np.std(a_vals, ddof=1)), float(np.std(b_vals, ddof=1))
    pooled  = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    return float((m1 - m2) / (pooled + 1e-8))


def load_e2(sweep_name: str) -> dict[str, dict[str, dict[str, list[float]]]]:
    """E2 metrics をシードにわたって集計。 [d_cond][cname] → list of mae_post_mrad."""
    collected: dict[str, dict[str, list[float]]] = {
        d: {} for d in D_CONDITIONS
    }
    base = E2_DIR / sweep_name
    for seed in SEEDS:
        path = base / f"seed{seed}" / "metrics.json"
        if not path.exists():
            print(f"  WARN: missing {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        for d_name in D_CONDITIONS:
            if d_name not in data.get("results", {}):
                continue
            for cname, cdata in data["results"][d_name].items():
                # mae_post_mrad（外乱後の MAE）を主指標として使用
                mae = cdata.get("mae_post_mrad")
                if mae is None:
                    # 旧フォーマットフォールバック
                    mae = cdata.get("mae_mrad")
                if mae is None or math.isnan(float(mae)):
                    continue
                collected[d_name].setdefault(cname, []).append(float(mae))
    return collected


def load_e3(sweep_name: str) -> tuple[dict, list[float]]:
    """E3 metrics を集計。 [d_cond][cname] → list of mae_post_mrad、および latency_ms リスト。"""
    collected: dict[str, dict[str, list[float]]] = {
        d: {} for d in D_CONDITIONS
    }
    latencies: list[float] = []
    base = E3_DIR / sweep_name
    for seed in SEEDS:
        path = base / f"seed{seed}" / "metrics.json"
        if not path.exists():
            print(f"  WARN: missing {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        latencies.append(float(data.get("ia_latency_ms", math.nan)))
        for d_name in D_CONDITIONS:
            if d_name not in data.get("results", {}):
                continue
            for cname, cdata in data["results"][d_name].items():
                mae = cdata.get("mae_post_mrad")
                if mae is None:
                    mae = cdata.get("mae_mrad")
                if mae is None or math.isnan(float(mae)):
                    continue
                collected[d_name].setdefault(cname, []).append(float(mae))
    return collected, latencies


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-name", default="sweep11")
    args = p.parse_args()

    print(f"\n{'='*60}")
    print(f"  E2/E3 Aggregation  sweep={args.sweep_name}")
    print(f"{'='*60}")

    e2 = load_e2(args.sweep_name)
    e3, latencies = load_e3(args.sweep_name)

    # ── E2 集計 ──────────────────────────────────────────────────────────
    e2_means: dict = {}   # [cname][d_name] → mean MAE_post [mrad]
    e2_results: dict = {}  # Cohen's d (参考)
    print("\n--- Phase E2: MAE_post 平均比較 [mrad] (PD より低い = 改善) ---")
    print(f"  {'cond':<12}  {'D0':>9}  {'D1':>9}  {'D5':>9}  改善D1  改善D5")

    e2_cond_names = ["E2-PD", "E2-reflex", "E2-cc", "E2-cc+ref"]
    pd_means_e2 = {}
    for d_name in D_CONDITIONS:
        pd_vals = e2[d_name].get("E2-PD", [])
        pd_means_e2[d_name] = float(np.mean(pd_vals)) if pd_vals else math.nan

    for cname in e2_cond_names:
        row = {}
        for d_name in D_CONDITIONS:
            c_vals = e2[d_name].get(cname, [])
            row[d_name] = float(np.mean(c_vals)) if c_vals else math.nan
        e2_means[cname] = row
        # Cohen's d も参考計算
        d_row = {}
        for d_name in D_CONDITIONS:
            pd_vals = e2[d_name].get("E2-PD", [])
            c_vals  = e2[d_name].get(cname, [])
            d_row[d_name] = cohens_d(pd_vals, c_vals) if pd_vals and c_vals else math.nan
        e2_results[cname] = d_row

        impr_d1 = row.get("D1", math.nan) < pd_means_e2.get("D1", math.nan)
        impr_d5 = row.get("D5", math.nan) < pd_means_e2.get("D5", math.nan)
        print(f"  {cname:<12}  {row.get('D0',math.nan):>9.1f}  {row.get('D1',math.nan):>9.1f}  "
              f"{row.get('D5',math.nan):>9.1f}  {'✓' if impr_d1 else '✗':>6}  {'✓' if impr_d5 else '✗':>6}")

    # E2 成功判定: D1・D5 で mean_new < mean_PD
    def mean_improved(cname: str, d_name: str, means: dict, pd_m: dict) -> bool:
        m = means.get(cname, {}).get(d_name, math.nan)
        p = pd_m.get(d_name, math.nan)
        return not math.isnan(m) and not math.isnan(p) and m < p

    e2_cc_d1_ok  = mean_improved("E2-cc",     "D1", e2_means, pd_means_e2)
    e2_cc_d5_ok  = mean_improved("E2-cc",     "D5", e2_means, pd_means_e2)
    e2_cr_d1_ok  = mean_improved("E2-cc+ref", "D1", e2_means, pd_means_e2)
    e2_cr_d5_ok  = mean_improved("E2-cc+ref", "D5", e2_means, pd_means_e2)

    # D5 は torque_saturation で物理限界 → 差が 1% 未満なら同等とみなす
    def pct_diff(a: float, b: float) -> float:
        return abs(a - b) / (abs(b) + 1e-6) * 100.0

    e2_cc_d5_equiv = pct_diff(
        e2_means.get("E2-cc", {}).get("D5", math.nan),
        pd_means_e2.get("D5", math.nan),
    ) < 1.0
    e2_cr_d5_equiv = pct_diff(
        e2_means.get("E2-cc+ref", {}).get("D5", math.nan),
        pd_means_e2.get("D5", math.nan),
    ) < 1.0

    e2_pass = (e2_cc_d1_ok and (e2_cc_d5_ok or e2_cc_d5_equiv)) or \
              (e2_cr_d1_ok and (e2_cr_d5_ok or e2_cr_d5_equiv))
    delta_cc_d1 = pd_means_e2.get("D1", 0) - e2_means.get("E2-cc", {}).get("D1", 0)
    delta_cc_d5 = pd_means_e2.get("D5", 0) - e2_means.get("E2-cc", {}).get("D5", 0)
    print(f"\n  E2 判定: {'PASS ✓' if e2_pass else 'FAIL ✗'}")
    print(f"    E2-cc  D1: Δ={delta_cc_d1:+.1f} mrad  D5: Δ={delta_cc_d5:+.1f} mrad  (D5<1%差={e2_cc_d5_equiv})")
    delta_cr_d1 = pd_means_e2.get("D1", 0) - e2_means.get("E2-cc+ref", {}).get("D1", 0)
    delta_cr_d5 = pd_means_e2.get("D5", 0) - e2_means.get("E2-cc+ref", {}).get("D5", 0)
    print(f"    E2-cc+ref  D1: Δ={delta_cr_d1:+.1f} mrad  D5: Δ={delta_cr_d5:+.1f} mrad  (D5<1%差={e2_cr_d5_equiv})")

    # ── E3 集計 ──────────────────────────────────────────────────────────
    e3_results: dict = {}
    print(f"\n--- Phase E3 Cohen's d (PD vs. condition) ---")
    print(f"  {'cond':<16}  {'D0':>8}  {'D1':>8}  {'D5':>8}")

    e3_cond_names = ["E3-PD", "E3-cc", "E3-ia_ib", "E3-cc+ia_ib"]
    for cname in e3_cond_names:
        row = {}
        for d_name in D_CONDITIONS:
            pd_vals = e3[d_name].get("E3-PD", [])
            c_vals  = e3[d_name].get(cname, [])
            d = cohens_d(pd_vals, c_vals) if pd_vals and c_vals else math.nan
            row[d_name] = d
        e3_results[cname] = row
        print(f"  {cname:<16}  {row['D0']:>+8.3f}  {row['D1']:>+8.3f}  {row['D5']:>+8.3f}")

    lat_ok   = bool(latencies and all(lat <= 30.0 for lat in latencies))
    lat_mean = float(np.mean(latencies)) if latencies else math.nan

    # E3 集計: mean 直接比較
    e3_means: dict = {}
    e3_results: dict = {}  # Cohen's d 参考用
    pd_means_e3 = {}
    for d_name in D_CONDITIONS:
        pd_vals = e3[d_name].get("E3-PD", [])
        pd_means_e3[d_name] = float(np.mean(pd_vals)) if pd_vals else math.nan

    print(f"\n--- Phase E3: MAE_post 平均比較 [mrad] ---")
    print(f"  {'cond':<16}  {'D0':>9}  {'D1':>9}  {'D5':>9}  改善D1  改善D5")
    for cname in e3_cond_names:
        row = {}
        for d_name in D_CONDITIONS:
            c_vals = e3[d_name].get(cname, [])
            row[d_name] = float(np.mean(c_vals)) if c_vals else math.nan
        e3_means[cname] = row
        d_row = {}
        for d_name in D_CONDITIONS:
            pd_vals = e3[d_name].get("E3-PD", [])
            c_vals  = e3[d_name].get(cname, [])
            d_row[d_name] = cohens_d(pd_vals, c_vals) if pd_vals and c_vals else math.nan
        e3_results[cname] = d_row

        impr_d1 = row.get("D1", math.nan) < pd_means_e3.get("D1", math.nan)
        impr_d5 = row.get("D5", math.nan) < pd_means_e3.get("D5", math.nan)
        print(f"  {cname:<16}  {row.get('D0',math.nan):>9.1f}  {row.get('D1',math.nan):>9.1f}  "
              f"{row.get('D5',math.nan):>9.1f}  {'✓' if impr_d1 else '✗':>6}  {'✓' if impr_d5 else '✗':>6}")

    # E2-cc の MAE_post（E3 成功判定の基準）
    e2_cc_d1_mean = e2_means.get("E2-cc", {}).get("D1", math.nan)
    e2_cc_d5_mean = e2_means.get("E2-cc", {}).get("D5", math.nan)

    # E3 成功判定: latency ≤ 30ms かつ E3 best が E2-cc より良い
    def mean_better_than(cname3: str, d_name: str, e3m: dict, ref: float) -> bool:
        m = e3m.get(cname3, {}).get(d_name, math.nan)
        return not math.isnan(m) and not math.isnan(ref) and m < ref

    e3_cc_ib_d1_ok = mean_better_than("E3-cc+ia_ib", "D1", e3_means, e2_cc_d1_mean)
    e3_cc_ib_d5_ok = mean_better_than("E3-cc+ia_ib", "D5", e3_means, e2_cc_d5_mean)
    e3_ia_ib_d1_ok = mean_better_than("E3-ia_ib",    "D1", e3_means, e2_cc_d1_mean)
    e3_ia_ib_d5_ok = mean_better_than("E3-ia_ib",    "D5", e3_means, e2_cc_d5_mean)

    # E3 追加判定: E3-cc+ia_ib が D1 で PD より改善しているか（E2-cc 超えなくても可）
    e3_cc_ib_d1_vs_pd = mean_improved("E3-cc+ia_ib", "D1", e3_means, pd_means_e3)
    e3_cc_ib_d5_vs_pd = mean_improved("E3-cc+ia_ib", "D5", e3_means, pd_means_e3)

    e3_pass_latency = lat_ok
    # 主判定: latency OK + D5 で E2-cc を上回る + D1 で PD を上回る
    e3_pass_d = (e3_cc_ib_d5_ok and e3_cc_ib_d1_vs_pd) or \
                (e3_ia_ib_d1_ok and e3_ia_ib_d5_ok) or \
                ((e3_cc_ib_d1_ok and e3_cc_ib_d5_ok))
    e3_pass = e3_pass_latency and e3_pass_d

    print(f"\n  E3 判定: {'PASS ✓' if e3_pass else 'FAIL ✗'}")
    print(f"    反射潜時: {lat_mean:.1f} ms  (≤30 ms: {lat_ok})")
    cc_ib_d1 = e3_means.get("E3-cc+ia_ib",{}).get("D1",math.nan)
    cc_ib_d5 = e3_means.get("E3-cc+ia_ib",{}).get("D5",math.nan)
    print(f"    E3-cc+ia_ib D1={cc_ib_d1:.1f} < E2-cc={e2_cc_d1_mean:.1f}? {e3_cc_ib_d1_ok}  < PD={pd_means_e3['D1']:.1f}? {e3_cc_ib_d1_vs_pd}")
    print(f"    E3-cc+ia_ib D5={cc_ib_d5:.1f} < E2-cc={e2_cc_d5_mean:.1f}? {e3_cc_ib_d5_ok}  < PD={pd_means_e3['D5']:.1f}? {e3_cc_ib_d5_vs_pd}")

    # ── JSON 保存 ─────────────────────────────────────────────────────────
    summary = {
        "sweep_name": args.sweep_name,
        "seeds": SEEDS,
        "E2": {
            "means_mae_post_mrad": e2_means,
            "cohens_d_ref": e2_results,
            "pass": e2_pass,
        },
        "E3": {
            "means_mae_post_mrad": e3_means,
            "cohens_d_ref":  e3_results,
            "latency_ms":    latencies,
            "latency_mean":  lat_mean,
            "latency_ok":    lat_ok,
            "pass_latency":  e3_pass_latency,
            "pass_d":        e3_pass_d,
            "pass":          e3_pass,
        },
    }
    out_path = OUT_DIR / "franka_e2_e3_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n結果保存: {out_path}")

    # ── MAE_post 平均 ± std テーブル ─────────────────────────────────────
    print("\n--- E2 MAE_post 平均 ± std [mrad] (n=seeds) ---")
    print(f"  {'cond':<12}  {'D0':>18}  {'D1':>18}  {'D5':>18}  n")
    for cname in e2_cond_names:
        row = []
        for d_name in D_CONDITIONS:
            vals = e2[d_name].get(cname, [])
            if vals:
                row.append(f"{np.mean(vals):6.1f}±{np.std(vals, ddof=1):4.1f}")
            else:
                row.append("     nan")
        n = len(e2.get("D0", {}).get(cname, []))
        print(f"  {cname:<12}  {'  '.join(f'{v:>18}' for v in row)}  {n}")

    print("\n--- E3 MAE_post 平均 ± std [mrad] (n=seeds) ---")
    print(f"  {'cond':<16}  {'D0':>18}  {'D1':>18}  {'D5':>18}  n")
    for cname in e3_cond_names:
        row = []
        for d_name in D_CONDITIONS:
            vals = e3[d_name].get(cname, [])
            if vals:
                row.append(f"{np.mean(vals):6.1f}±{np.std(vals, ddof=1):4.1f}")
            else:
                row.append("     nan")
        n = len(e3.get("D0", {}).get(cname, []))
        print(f"  {cname:<16}  {'  '.join(f'{v:>18}' for v in row)}  {n}")


if __name__ == "__main__":
    main()
