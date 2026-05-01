"""Fig 5: Controller architecture — signal-flow block diagram of the
neuromusculoskeletal reach controller.

Five vertically stacked stages (top → bottom):
    Target  →  M1 / IK  →  Cerebellum (virtual traj + visuomotor)
            →  Spinal α-MN pool (+ reflexes)  →  Muscle plant.

Colors mirror Fig 1–4:
    virtual trajectory  : #9ecae1
    visuomotor          : #4292c6
    reflex / cereb corr : #08519c
    other blocks        : #f0f0f0 (neutral)

Output: figures/fig5_architecture.{pdf,png}
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


# Component colors
C_VT     = "#9ecae1"   # virtual trajectory
C_VM     = "#4292c6"   # visuomotor
C_REFLEX = "#08519c"   # reflex / cerebellar correction
C_NEUTRAL = "#f0f0f0"  # M1 / α-MN / plant
C_TARGET  = "#fee391"  # target (visual input)

# FancyBboxPatch enlarges the visible box by `pad` in all directions, so
# arrow endpoints specified at the bounding-box coords land 0.04 units
# inside the visible edge. Add BOX_PAD to push arrow tips/tails out to
# the visible edge for clean abutment.
BOX_PAD = 0.04


def add_box(ax, x, y, w, h, text, facecolor, edgecolor="black",
            text_color="black", fontsize=8, fontweight="normal", lw=1.0,
            ha="center", text_x_pad=0.15):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=lw, edgecolor=edgecolor, facecolor=facecolor,
    )
    ax.add_patch(box)
    if ha == "left":
        text_x = x + text_x_pad
    elif ha == "right":
        text_x = x + w - text_x_pad
    else:
        text_x = x + w / 2
    ax.text(text_x, y + h / 2, text,
            ha=ha, va="center",
            fontsize=fontsize, fontweight=fontweight, color=text_color)
    return (x, y, w, h)


def add_arrow(ax, x1, y1, x2, y2, color="black", lw=1.4, ls="-",
              label=None, label_pos=0.5, label_dx=0.0, label_dy=0.10,
              label_fontsize=9, mutation_scale=14, alpha=1.0):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=mutation_scale,
        linewidth=lw, color=color, linestyle=ls, alpha=alpha,
        shrinkA=0, shrinkB=0,
    )
    ax.add_patch(arrow)
    if label is not None:
        xm = x1 + label_pos * (x2 - x1) + label_dx
        ym = y1 + label_pos * (y2 - y1) + label_dy
        ax.text(xm, ym, label, ha="center", va="center",
                fontsize=label_fontsize, color=color, style="italic",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.9, pad=0.6))


def add_region_label(ax, x_left, y_top, y_bot, label):
    ax.plot([x_left, x_left], [y_bot, y_top], color="#aaaaaa",
            linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(x_left - 0.15, (y_top + y_bot) / 2, label,
            ha="right", va="center", fontsize=8.5,
            fontweight="bold", color="#444444",
            rotation=90)


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
    })

    fig, ax = plt.subplots(figsize=(7.2, 8.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Layout reference (y top → bottom):
    #   12   target                      y = 11.0
    #   11   ─── CORTEX divider ───      y = 10.0
    #   10   M1 / IK                     y =  9.2
    #    9   ─── CEREBELLUM divider ─── y =  8.2
    #    8   virtual trajectory          y =  6.8 (with visuomotor on right)
    #    7   λ_eff merge                 y =  4.8
    #    6   ─── SPINAL divider ───      y =  4.5
    #    5   α-MN + reflexes             y =  2.6
    #    4   ─── PERIPHERY divider ───   y =  2.0
    #    3   muscle plant                y =  0.8

    # ── target ────────────────────────────────────────────────────────
    add_box(ax, 4.0, 11.05, 2.0, 0.7,
            "Target  $\\mathbf{x}_{\\rm target}$",
            facecolor=C_TARGET, fontsize=9, fontweight="bold")

    # ── divider: CORTEX ───────────────────────────────────────────────
    ax.axhline(10.7, xmin=0.05, xmax=0.95, color="#cccccc", lw=0.8)
    ax.text(0.15, 10.35, "CORTEX (M1)", fontsize=8.5,
            fontweight="bold", color="#444444")

    add_box(ax, 3.4, 9.55, 3.2, 0.85,
            "Inverse kinematics\n"
            "$\\boldsymbol{\\lambda}_{\\rm target} = \\mathbf{L}(\\mathbf{q}^{*}) - \\lambda_{0}\\mathbf{1}_{34}$",
            facecolor=C_NEUTRAL, fontsize=8.5)

    # target → IK
    add_arrow(ax, 5.0, 11.05 - BOX_PAD, 5.0, 10.40 + BOX_PAD, lw=1.6)

    # ── divider: CEREBELLUM ───────────────────────────────────────────
    ax.axhline(9.30, xmin=0.05, xmax=0.95, color="#cccccc", lw=0.8)
    ax.text(0.15, 8.95, "CEREBELLUM", fontsize=8.5,
            fontweight="bold", color="#444444")

    # virtual trajectory box (left/centre)
    add_box(ax, 1.4, 7.40, 3.6, 1.30,
            "Virtual trajectory\n"
            "$\\boldsymbol{\\lambda}(t)=\\boldsymbol{\\lambda}_{s}+s(\\tau)\\,(\\boldsymbol{\\lambda}_{\\rm target}-\\boldsymbol{\\lambda}_{s})$\n"
            "$s(\\tau)=10\\tau^{3}-15\\tau^{4}+6\\tau^{5}$",
            facecolor=C_VT, fontsize=7.5)

    # visuomotor box (right)
    add_box(ax, 6.4, 7.55, 3.0, 1.0,
            "Visuomotor\n"
            "(IK refresh 200 ms period)",
            facecolor=C_VM, text_color="white", fontsize=7.5)

    # arrow: M1 IK → virtual trajectory
    add_arrow(ax, 5.0, 9.55 - BOX_PAD, 4.4, 8.70 + BOX_PAD,
              label="$\\boldsymbol{\\lambda}_{\\rm target}$", label_pos=0.5,
              label_dx=0.45, label_dy=-0.05,
              lw=1.4)

    # arrow: visuomotor → virtual trajectory  (Δλ_target update)
    add_arrow(ax, 6.4 - BOX_PAD, 8.05, 5.0 + BOX_PAD, 8.05,
              color=C_VM, lw=1.4,
              label="$\\Delta\\boldsymbol{\\lambda}_{\\rm target}$",
              label_pos=0.5, label_dy=0.18)

    # cereb correction (arrow into λ_eff merge node)
    # λ_eff merge node
    merge_x, merge_y = 5.0, 6.40
    ax.add_patch(plt.Circle((merge_x, merge_y), 0.18,
                            facecolor="white", edgecolor="black", lw=1.2))
    ax.text(merge_x, merge_y, "+", ha="center", va="center",
            fontsize=11, fontweight="bold")

    add_arrow(ax, 3.8, 7.40 - BOX_PAD, 4.85, merge_y + 0.18,
              label="$\\boldsymbol{\\lambda}(t)$",
              label_pos=0.55, label_dx=-0.35, label_dy=-0.10,
              lw=1.4)

    # cereb correction Δλ_cereb (from right side, anatomically still cerebellum)
    add_box(ax, 6.4, 6.05, 3.0, 0.70,
            "Cerebellar correction\n$\\Delta\\boldsymbol{\\lambda}_{\\rm cereb}$",
            facecolor=C_REFLEX, text_color="white", fontsize=7.5)
    add_arrow(ax, 6.4 - BOX_PAD, 6.40, merge_x + 0.18, merge_y,
              color=C_REFLEX, lw=1.4)

    # output: λ_eff
    ax.text(merge_x + 0.55, merge_y - 0.15, "$\\boldsymbol{\\lambda}_{\\rm eff}$",
            fontsize=9, va="center", style="italic")

    # ── divider: SPINAL ───────────────────────────────────────────────
    ax.axhline(5.55, xmin=0.05, xmax=0.95, color="#cccccc", lw=0.8)
    ax.text(0.15, 5.20, "SPINAL CORD", fontsize=8.5,
            fontweight="bold", color="#444444")

    # α-MN block (text centred at x = 5; box width matched to text so the
    # reflex bracket can sit adjacent to the equation right edge)
    amn_x, amn_y, amn_w, amn_h = 3.0, 3.25, 4.0, 1.55
    add_box(ax, amn_x, amn_y, amn_w, amn_h,
            "$\\alpha$-motoneuron pool\n"
            "$\\;a_{\\rm base} = \\mathrm{clip}(c_{\\lambda}\\cdot\\max(\\mathbf{L}-\\boldsymbol{\\lambda}_{\\rm eff},\\,0),\\,0,\\,1)$\n"
            "$\\;\\;\\; +\\,\\Delta a_{\\rm Ia}\\;\\;(\\gamma\\!-\\!{\\rm modulated\\;stretch\\;reflex})$\n"
            "$\\;\\;\\; +\\,\\Delta a_{\\rm Ib}\\;\\;({\\rm Golgi\\;tendon})$\n"
            "$\\;\\;\\; +\\,\\Delta a_{\\rm RI}\\;\\;({\\rm reciprocal\\;inhibition})$",
            facecolor=C_NEUTRAL, fontsize=7.8)

    # arrow: λ_eff → α-MN
    add_arrow(ax, merge_x, merge_y - 0.18, merge_x, amn_y + amn_h + BOX_PAD,
              lw=1.6)

    # reflex side annotation: bracket placed just to the right of the
    # equation text (inside the box). Box right edge is at amn_x + amn_w;
    # the bracket sits ≈0.5 units inside that edge so it hugs the text
    bracket_x = amn_x + amn_w - 0.30
    bracket_top = amn_y + 0.90
    bracket_bot = amn_y + 0.23
    ax.plot([bracket_x, bracket_x + 0.15,
             bracket_x + 0.15, bracket_x],
            [bracket_bot, bracket_bot,
             bracket_top, bracket_top],
            color=C_REFLEX, lw=1.5)
    ax.text(bracket_x + 0.25, (bracket_top + bracket_bot) / 2,
            "spinal\nreflexes",
            fontsize=7.5, color=C_REFLEX, fontweight="bold",
            va="center", ha="left")

    # ── divider: PERIPHERY ────────────────────────────────────────────
    ax.axhline(2.55, xmin=0.05, xmax=0.95, color="#cccccc", lw=0.8)
    ax.text(0.15, 2.20, "PERIPHERY", fontsize=8.5,
            fontweight="bold", color="#444444")

    # plant
    plant_x, plant_y, plant_w, plant_h = 2.1, 0.95, 5.8, 0.95
    add_box(ax, plant_x, plant_y, plant_w, plant_h,
            "Muscle plant  (MyoSuite — 34 muscles → 20-DoF arm)\n"
            "→  $\\mathbf{q},\\,\\dot{\\mathbf{q}},\\,\\mathbf{x}_{\\rm tip}$",
            facecolor=C_NEUTRAL, fontsize=8.0)

    # arrow: α-MN → plant  (a_total)
    add_arrow(ax, 5.0, amn_y - BOX_PAD, 5.0, plant_y + plant_h + BOX_PAD,
              label="$\\mathbf{a}_{\\rm total}\\,(34)$",
              label_pos=0.55, label_dx=0.55, label_dy=0.0, lw=1.6)

    # ── feedback loops (right side) ───────────────────────────────────
    # tip_pos → visuomotor (long loop). The horizontal exit and the
    # vertical riser are plain lines; only the final segment that lands
    # on the visuomotor block carries an arrowhead.
    fb_x = 9.80
    ax.plot([plant_x + plant_w + BOX_PAD, fb_x],
            [plant_y + plant_h / 2, plant_y + plant_h / 2],
            color="#666666", lw=1.0)
    ax.plot([fb_x, fb_x], [plant_y + plant_h / 2, 8.05],
            color="#666666", lw=1.0)
    add_arrow(ax, fb_x, 8.05, 9.4 + BOX_PAD, 8.05, lw=1.0, color="#666666")
    ax.text(fb_x + 0.05, (plant_y + plant_h / 2 + 8.05) / 2,
            "$\\mathbf{x}_{\\rm tip}$\n(visual,\n100–200 ms)",
            fontsize=9, color="#666666", va="center", ha="left", style="italic")

    # L (proprio) → α-MN  (short loop, left side). Same as the right
    # loop: only the final segment into α-MN keeps the arrowhead.
    fb_x_l = 0.55
    ax.plot([plant_x - BOX_PAD, fb_x_l],
            [plant_y + plant_h / 2, plant_y + plant_h / 2],
            color="#666666", lw=1.0)
    ax.plot([fb_x_l, fb_x_l],
            [plant_y + plant_h / 2, amn_y + amn_h / 2],
            color="#666666", lw=1.0)
    add_arrow(ax, fb_x_l, amn_y + amn_h / 2,
              amn_x - BOX_PAD, amn_y + amn_h / 2, lw=1.0, color="#666666")
    ax.text(fb_x_l - 0.05, (plant_y + plant_h / 2 + amn_y + amn_h / 2) / 2,
            "$L,\\dot L$\n(proprio.,\n20 ms)",
            fontsize=9, color="#666666", va="center", ha="right", style="italic")

    # ── legend / colour key (bottom-left) ─────────────────────────────
    leg_x, leg_y = 0.20, 0.10
    leg_w, leg_h = 6.5, 0.65
    ax.add_patch(FancyBboxPatch((leg_x, leg_y), leg_w, leg_h,
                                 boxstyle="round,pad=0.04",
                                 facecolor="white",
                                 edgecolor="#aaaaaa", lw=0.8))
    items = [
        ("Virtual trajectory",      C_VT),
        ("Visuomotor",              C_VM),
        ("Reflex / cereb. corr.",   C_REFLEX),
    ]
    box_w, box_h = 0.40, 0.28
    text_gap = 0.08      # gap between colour box and its label
    inner_pad = 0.18     # padding inside the legend frame
    label_fontsize = 7.5

    # Measure each label width via a transient text object so the
    # entries (box + label) can be distributed with EQUAL inter-entry
    # gaps regardless of label length.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_widths = []
    for label, _ in items:
        tmp = ax.text(0, 0, label, fontsize=label_fontsize)
        bb_disp = tmp.get_window_extent(renderer=renderer)
        bb_data = bb_disp.transformed(ax.transData.inverted())
        label_widths.append(bb_data.width)
        tmp.remove()

    entry_widths = [box_w + text_gap + lw for lw in label_widths]
    free_space = leg_w - 2 * inner_pad - sum(entry_widths)
    gap = free_space / (len(items) - 1)

    bx = leg_x + inner_pad
    by = leg_y + 0.18
    for (label, c), entry_w in zip(items, entry_widths):
        ax.add_patch(FancyBboxPatch((bx, by), box_w, box_h,
                                     boxstyle="round,pad=0.02,rounding_size=0.05",
                                     facecolor=c, edgecolor="black", lw=0.7))
        ax.text(bx + box_w + text_gap, by + box_h / 2,
                label, fontsize=label_fontsize, va="center", ha="left")
        bx += entry_w + gap

    fig.tight_layout()
    pdf_path = OUT / "fig5_architecture.pdf"
    png_path = OUT / "fig5_architecture.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
