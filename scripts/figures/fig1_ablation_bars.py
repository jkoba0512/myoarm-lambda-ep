"""Fig 1: Ablation bar chart (n=50, F16 results).

4 panels: peak_speed / jerk_rms / vel_peak_ratio / tip_err_min_mm
Comparison: endpoint_pd vs lambda-traj vs F12 best vs F12+reflexes.

Output: figures/fig1_ablation.{pdf,png}
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "experiment_myo_p15" / "f16_n50.json"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


CONDITIONS = [
    ("endpoint_pd", "Endpoint PD\n+ spinal"),
    ("λ-traj baseline (no visuo)", "λ-traj\n(no visuo)"),
    ("F12 best (pure λ visuo)", "λ + visuo\n(pure)"),
    ("F12 best + reflexes", "λ + visuo\n+ reflex"),
]

PANELS = [
    ("peak_speed", "Peak speed (m/s)", None),
    ("jerk_rms", r"Jerk RMS (m/s$^3$)", None),
    ("vel_peak_ratio", "Velocity peak ratio", (0.40, 0.50)),
    ("tip_err_min_mm", "Min tip error (mm)", None),
]

PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]


def cohen_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    return (a.mean() - b.mean()) / pooled


def sig_marker(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def main():
    with DATA.open() as f:
        d = json.load(f)
    raw = d["raw_per_seed"]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    axes = axes.flatten()

    colors = ["#888888", "#9ecae1", "#4292c6", "#08519c"]

    for ax, (metric, ylabel, human_band), label in zip(axes, PANELS, PANEL_LABELS):
        means, sems, all_vals = [], [], []
        for cond_key, _ in CONDITIONS:
            vals = np.array([row[metric] for row in raw[cond_key]])
            all_vals.append(vals)
            means.append(vals.mean())
            sems.append(vals.std(ddof=1) / np.sqrt(vals.size))

        x = np.arange(len(CONDITIONS))
        bars = ax.bar(x, means, yerr=sems, color=colors, edgecolor="black",
                      linewidth=0.6, capsize=3, error_kw={"elinewidth": 0.8})

        if human_band is not None:
            ax.axhspan(human_band[0], human_band[1], color="#d4f1d4", alpha=0.6,
                       zorder=0, label="Human range")
            ax.legend(loc="upper left", frameon=False, fontsize=6.5)

        # Paired Wilcoxon signed-rank test (same seeds across conditions).
        ref = all_vals[0]
        y_max = max(m + s for m, s in zip(means, sems))
        for i in range(1, len(CONDITIONS)):
            diffs = all_vals[i] - ref
            if (diffs == 0).all():
                p = 1.0
            else:
                _, p = wilcoxon(all_vals[i], ref, alternative="two-sided")
            mark = sig_marker(p)
            y = means[i] + sems[i] + 0.04 * y_max
            ax.text(i, y, mark, ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([c[1] for c in CONDITIONS])
        ax.set_ylabel(ylabel)
        ax.text(-0.18, 1.02, label, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom")

        ax.margins(y=0.15)

    fig.suptitle("Ablation of biological reach control mechanisms (n = 50)",
                 fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    pdf_path = OUT / "fig1_ablation.pdf"
    png_path = OUT / "fig1_ablation.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

    print("\n--- Summary table (vs endpoint_pd) ---")
    ref = np.array([row[PANELS[0][0]] for row in raw["endpoint_pd"]])
    for cond_key, label in CONDITIONS:
        line = f"{label.replace(chr(10), ' '):28s}"
        for metric, _, _ in PANELS:
            vals = np.array([row[metric] for row in raw[cond_key]])
            ref_m = np.array([row[metric] for row in raw["endpoint_pd"]])
            if cond_key == "endpoint_pd":
                line += f"  {metric}: {vals.mean():7.2f}±{vals.std(ddof=1)/np.sqrt(50):4.2f}"
            else:
                diffs = vals - ref_m
                if (diffs == 0).all():
                    p = 1.0
                else:
                    _, p = wilcoxon(vals, ref_m, alternative="two-sided")
                d_val = cohen_d(vals, ref_m)
                line += f"  {metric}: {vals.mean():7.2f}±{vals.std(ddof=1)/np.sqrt(50):4.2f} (d={d_val:+.2f}{sig_marker(p)})"
        print(line)


if __name__ == "__main__":
    main()
