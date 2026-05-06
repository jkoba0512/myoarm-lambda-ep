"""Fig 4: Orthogonal contributions of three controller components.

Top (a):    Schematic — VT / VM / Reflex → Smoothness / Accuracy / Bell-shape.
Bottom (b): Cohen's d heatmap (signed: positive = component improves the axis).

Data: results/experiment_myo_p15/f13_ablation.json (n = 20 per condition).

Component effects defined relative to F12-best baseline:
    VT      : F12-best (with VT)          vs −virtual_trajectory
    VM      : F12-best (with VM)          vs −visuomotor
    Reflex  : +reflexes (with reflex)     vs F12-best (no reflex)

Sign convention: positive d  ⇒  component pushes the metric in the
desirable direction (lower for error / speed / jerk, higher for vpr).

Output: figures/fig4_orthogonal.{pdf,png}
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "experiment_myo_p15" / "f13_ablation.json"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


# (component_name, "with" condition, "without" condition, color)
COMPONENTS = [
    ("Virtual trajectory\n(λ̇, cerebellum)",      "F12 best (pure λ visuo)", "− virtual_trajectory", "#9ecae1"),
    ("Visuomotor\n(Δλ, parietal→motor)",          "F12 best (pure λ visuo)", "− visuomotor",         "#4292c6"),
    ("Stretch reflex\n(γ_d, K_cereb)",            "+ reflexes",              "F12 best (pure λ visuo)", "#08519c"),
]

# (metric_key, label, group, lower_is_better)
METRICS = [
    ("peak_speed",          "Peak speed\n(m/s)",         "Smoothness", True),
    ("jerk_rms",            "Jerk RMS\n(m/s³)",          "Smoothness", True),
    ("tip_err_min_mm",      "Min tip err\n(mm)",         "Accuracy",   True),
    ("direction_error_deg", "Direction err\n(°)",        "Accuracy",   True),
    ("vel_peak_ratio",      "Velocity peak\nratio",      "Bell-shape", False),
]

AXES = [
    ("Smoothness",  "peak v, jerk"),
    ("Accuracy",    "tip err, direction"),
    ("Bell-shape",  "vel peak ratio"),
]
AXIS_COLOR = "#d9d9d9"


def signed_cohen_d(a_with, a_without, lower_is_better):
    """Signed Cohen's d such that positive = improvement (component helps).

    a_with     : metric values when component is present
    a_without  : metric values when component is absent
    lower_is_better : if True, "improvement" = lower mean.
    """
    a, b = np.asarray(a_with), np.asarray(a_without)
    pooled = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    if lower_is_better:
        return (b.mean() - a.mean()) / pooled
    return (a.mean() - b.mean()) / pooled


def sig_marker(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def compute_effects(raw):
    """Return effects (dict): comp_idx -> metric_idx -> (d_signed, p, n).

    Significance is computed by paired Wilcoxon signed-rank test across
    seeds (the same seed list is used for "with" and "without" conditions
    via deterministic_reset). Cohen's d is reported as the effect size.
    """
    out = {}
    for ci, (_, with_key, without_key, _) in enumerate(COMPONENTS):
        out[ci] = {}
        for mi, (mkey, _, _, lower_better) in enumerate(METRICS):
            vals_with    = np.array([row[mkey] for row in raw[with_key]])
            vals_without = np.array([row[mkey] for row in raw[without_key]])
            d = signed_cohen_d(vals_with, vals_without, lower_better)
            diffs = vals_with - vals_without
            if (diffs == 0).all():
                p = 1.0
            else:
                _, p = wilcoxon(vals_with, vals_without, alternative="two-sided")
            out[ci][mi] = (d, p, vals_with.size)
    return out


def draw_schematic(ax, effects):
    """Top panel: 3 component boxes (left) → 3 axis boxes (right)."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Component boxes (left)
    comp_x_lo, comp_x_hi = 0.4, 3.6
    comp_centers_y = [7.8, 5.0, 2.2]
    comp_box_h = 1.6
    for (label, _, _, color), cy in zip(COMPONENTS, comp_centers_y):
        box = FancyBboxPatch(
            (comp_x_lo, cy - comp_box_h / 2), comp_x_hi - comp_x_lo, comp_box_h,
            boxstyle="round,pad=0.05,rounding_size=0.18",
            linewidth=1.0, edgecolor="black", facecolor=color,
        )
        ax.add_patch(box)
        text_color = "white" if color == "#08519c" else "black"
        ax.text((comp_x_lo + comp_x_hi) / 2, cy, label,
                ha="center", va="center", fontsize=8, color=text_color)

    # Axis boxes (right)
    axis_x_lo, axis_x_hi = 6.4, 9.6
    axis_centers_y = [7.8, 5.0, 2.2]
    axis_box_h = 1.6
    for (label, sub), cy in zip(AXES, axis_centers_y):
        box = FancyBboxPatch(
            (axis_x_lo, cy - axis_box_h / 2), axis_x_hi - axis_x_lo, axis_box_h,
            boxstyle="round,pad=0.05,rounding_size=0.18",
            linewidth=1.0, edgecolor="black", facecolor=AXIS_COLOR,
        )
        ax.add_patch(box)
        ax.text((axis_x_lo + axis_x_hi) / 2, cy + 0.25, label,
                ha="center", va="center", fontsize=8, fontweight="bold", color="black")
        ax.text((axis_x_lo + axis_x_hi) / 2, cy - 0.30, sub,
                ha="center", va="center", fontsize=7, color="#444444", style="italic")

    # Arrows: each component → each axis. Strong if |d| > 0.5 on the primary metric.
    # Primary metric per axis: peak_speed (smoothness), tip_err_min (accuracy), vpr (bell).
    primary_metric_idx = [0, 2, 4]  # peak_speed, tip_err_min, vpr
    for ci, cy_c in enumerate(comp_centers_y):
        for ai, cy_a in enumerate(axis_centers_y):
            mi = primary_metric_idx[ai]
            d, p, _ = effects[ci][mi]
            is_diag = (ci == ai)
            strong = abs(d) > 0.5
            if strong and d > 0:
                lw, color, ls, alpha = 2.4, "black", "-", 1.0
            elif strong and d < 0:
                lw, color, ls, alpha = 1.6, "#cb181d", "-", 0.85  # significant worsening
            else:
                lw, color, ls, alpha = 0.7, "#999999", (0, (3, 3)), 0.7
            arrow = FancyArrowPatch(
                (comp_x_hi + 0.05, cy_c),
                (axis_x_lo - 0.05, cy_a),
                arrowstyle="-|>", mutation_scale=12,
                linewidth=lw, color=color, linestyle=ls, alpha=alpha,
                shrinkA=0, shrinkB=0,
            )
            ax.add_patch(arrow)
            if strong:
                # avoid label collision at the geometric center: place horizontal-
                # arrow labels at midpoint, sloped-arrow labels at 30 % / 70 % of
                # the line so up-slope and down-slope arrows don't share an x.
                slope = cy_a - cy_c
                if slope > 0.5:
                    frac = 0.30   # up-slope (e.g. Reflex → Smoothness): near source
                elif slope < -0.5:
                    frac = 0.70   # down-slope (e.g. VT → Bell-shape): near target
                else:
                    frac = 0.50
                xm = comp_x_hi + frac * (axis_x_lo - comp_x_hi)
                ym = cy_c + frac * (cy_a - cy_c)
                txt_color = color if d < 0 else "black"
                ax.text(xm, ym + 0.05, f"d={d:+.2f}",
                        ha="center", va="bottom",
                        fontsize=7, fontweight="bold", color=txt_color,
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.9, pad=0.8))

    # Title hint
    ax.text(5.0, 9.7,
            "Three components contribute non-redundantly to distinct task dimensions",
            ha="center", va="top", fontsize=9, fontweight="bold")


def draw_heatmap(ax, effects):
    n_comp = len(COMPONENTS)
    n_met  = len(METRICS)
    D = np.zeros((n_comp, n_met))
    P = np.zeros((n_comp, n_met))
    for ci in range(n_comp):
        for mi in range(n_met):
            D[ci, mi], P[ci, mi], _ = effects[ci][mi]

    vlim = 3.0
    im = ax.imshow(D, cmap="RdBu", vmin=-vlim, vmax=+vlim, aspect="auto")

    # Cell text: d value + significance
    for ci in range(n_comp):
        for mi in range(n_met):
            d = D[ci, mi]
            mark = sig_marker(P[ci, mi])
            txt = f"{d:+.2f}{mark}"
            color = "white" if abs(d) > 1.4 else "black"
            ax.text(mi, ci, txt, ha="center", va="center",
                    fontsize=7.5, color=color)

    ax.set_xticks(range(n_met))
    ax.set_xticklabels([m[1] for m in METRICS], fontsize=7.5)
    ax.set_yticks(range(n_comp))
    ax.set_yticklabels([c[0].replace("\n", " ") for c in COMPONENTS], fontsize=8)

    # Group brackets above metric labels
    ax2 = ax.secondary_xaxis("top")
    ax2.set_xticks([0.5, 2.5, 4.0])
    ax2.set_xticklabels(["Smoothness", "Accuracy", "Bell-shape"],
                        fontsize=8, fontweight="bold")
    ax2.tick_params(axis="x", length=0, pad=2)

    # Vertical separators between metric groups
    for x in [1.5, 3.5]:
        ax.axvline(x, color="white", linewidth=2.0)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Cohen's d  (+ : component improves axis)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title("Component-wise effect sizes\n($n = 20$ ablation set; reference = pure $\\lambda$ + visuomotor)",
                 fontsize=9, pad=22)


def main():
    with DATA.open() as f:
        d = json.load(f)
    raw = d["raw_per_seed"]
    effects = compute_effects(raw)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })

    # Bio Cyb full single-column width = 174 mm = 6.85 in (was 7.2 in / 183 mm).
    fig = plt.figure(figsize=(6.85, 6.09))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.35)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    draw_schematic(ax_a, effects)
    draw_heatmap(ax_b, effects)

    ax_a.text(-0.02, 1.02, "(a)", transform=ax_a.transAxes,
              fontsize=11, fontweight="bold", va="bottom")
    ax_b.text(-0.02, 1.18, "(b)", transform=ax_b.transAxes,
              fontsize=11, fontweight="bold", va="bottom")

    pdf_path = OUT / "fig4_orthogonal.pdf"
    png_path = OUT / "fig4_orthogonal.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

    # Console summary
    print("\n--- Signed Cohen's d (positive = component improves axis) ---")
    header = f"{'component':<28s}" + "".join(f"{m[1].replace(chr(10), ' '):>20s}" for m in METRICS)
    print(header)
    for ci, (label, _, _, _) in enumerate(COMPONENTS):
        row = f"{label.replace(chr(10), ' '):<28s}"
        for mi in range(len(METRICS)):
            d_v, p_v, _ = effects[ci][mi]
            row += f"  {d_v:+6.2f}{sig_marker(p_v):<3s}        "
        print(row[:28 + 20 * len(METRICS)])


if __name__ == "__main__":
    main()
