"""
主結果 JSON から論文用の1枚表を Markdown で生成する。

入力:
  results/franka_main_results.json

出力:
  results/paper_main_table.md
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RESULTS_DIR = ROOT / "results"
MAIN_RESULTS = RESULTS_DIR / "franka_main_results.json"
OUTPUT = RESULTS_DIR / "paper_main_table.md"


def fmt_mean_std(entry: dict, digits: int = 1) -> str:
    mean = entry["mean"]
    std = entry["std"]
    if mean is None:
        return "NA"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def fmt_d(val: float | None) -> str:
    return "NA" if val is None else f"{val:.2f}"


def main() -> None:
    with open(MAIN_RESULTS) as f:
        data = json.load(f)

    d2a = data["main_metrics"]["2a_static_holding"]
    d2b = data["main_metrics"]["2b_disturbance"]["levels"]
    d2c = data["main_metrics"]["2c_cyclic_motion"]
    d2d = data["main_metrics"]["2d_integrated"]

    lines: list[str] = []
    lines.append("# neuro-arm-control Main Results Table")
    lines.append("")
    lines.append(f"Seeds: `{data['seeds_used']}`")
    lines.append("")
    lines.append("## 2A Static Holding")
    lines.append("")
    lines.append("| Condition | static_mae [mrad] | Cohen's d vs PD |")
    lines.append("|---|---:|---:|")
    lines.append(f"| PD | {fmt_mean_std(d2a['conditions']['PD'])} | — |")
    lines.append(f"| PD+CfC | {fmt_mean_std(d2a['conditions']['PD+CfC'])} | {fmt_d(d2a['effect_size']['cohens_d_PD_vs_PDCfC'])} |")
    lines.append(f"| Full | {fmt_mean_std(d2a['conditions']['Full'])} | {fmt_d(d2a['effect_size']['cohens_d_PD_vs_Full'])} |")
    lines.append("")
    lines.append("## 2B Disturbance Rejection")
    lines.append("")
    lines.append("| Disturbance | PD+CfC peak_err [rad] | Full peak_err [rad] | Cohen's d |")
    lines.append("|---|---:|---:|---:|")
    for level in ["light_30Nm", "medium_60Nm", "heavy_87Nm"]:
        pdcfc = fmt_mean_std(d2b[level]["PD+CfC"]["peak_err_rad"], digits=3)
        full = fmt_mean_std(d2b[level]["Full"]["peak_err_rad"], digits=3)
        dval = fmt_d(d2b[level]["effect_size"]["cohens_d_peak_err_PDCfC_vs_Full"])
        lines.append(f"| {level} | {pdcfc} | {full} | {dval} |")
    lines.append("")
    lines.append("## 2C Cyclic Motion")
    lines.append("")
    lines.append("| Condition | ep_err_post [mrad] | mae_post [mrad] | Cohen's d |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| CPG+CfC | {fmt_mean_std(d2c['conditions']['CPG+CfC']['ep_err_post_mrad'])} | {fmt_mean_std(d2c['conditions']['CPG+CfC']['mae_post_mrad'])} | — |")
    lines.append(f"| CPG+CfC+LIF_FB | {fmt_mean_std(d2c['conditions']['CPG+CfC+LIF_FB']['ep_err_post_mrad'])} | {fmt_mean_std(d2c['conditions']['CPG+CfC+LIF_FB']['mae_post_mrad'])} | {fmt_d(d2c['effect_size']['cohens_d_ep_post_nofb_vs_fb'])} |")
    lines.append("")
    lines.append("## 2D Integrated Evaluation")
    lines.append("")
    lines.append("| Condition | static_mae [mrad] | peak_err [rad] | recovery_time [s] |")
    lines.append("|---|---:|---:|---:|")
    for cond in ["PD", "PD+CfC", "Full"]:
        mae = fmt_mean_std(d2d["conditions"][cond]["static_mae_mrad"])
        peak = fmt_mean_std(d2d["conditions"][cond]["peak_err_rad"], digits=3)
        rt = fmt_mean_std(d2d["conditions"][cond]["recovery_time_s"], digits=3)
        lines.append(f"| {cond} | {mae} | {peak} | {rt} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This table contains only main metrics.")
    lines.append("- `tau_sys` and Cartesian error are treated as secondary analyses.")

    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"保存: {OUTPUT}")


if __name__ == "__main__":
    main()
