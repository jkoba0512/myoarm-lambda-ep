"""
EF 結果の包括的可視化・サマリースクリプト

生成物:
  results/summary_figures/ef_phase_progression.png  — 各 Phase の改善軌跡
  results/summary_figures/ef_module_contribution.png — モジュール寄与分解
  results/summary_figures/ef_conditions.png          — EF 内条件比較
  results/summary_figures/ef_energy_mae.png          — Energy vs MAE 散布図
  results/summary_figures/ef_final_paper.png         — 論文用統合図（2×3）
  results/paper_ef_table.md                          — 論文用 Markdown テーブル
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib import font_manager

# Use Noto Sans CJK JP for Japanese text rendering
_jp_font = font_manager.FontProperties(family="Noto Sans CJK JP")
matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]

ROOT    = Path(__file__).parents[1]
RES_DIR = ROOT / "results"
FIG_DIR = RES_DIR / "summary_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# データ読み込み
# ─────────────────────────────────────────────────────────────

def load(fname: str) -> dict:
    return json.loads((RES_DIR / fname).read_text())

e2e3 = load("franka_e2_e3_summary.json")
e4   = load("franka_e4_summary.json")
e5   = load("franka_e5_summary.json")
ef   = load("franka_ef_summary.json")

D_CONDITIONS = ["D0", "D1", "D5"]
D_LABELS     = ["D0\n(baseline)", "D1\n(低ゲイン kp×0.5)", "D5\n(トルク飽和 30Nm)"]

# ── 各 Phase の MAE_post（PD比較用ベースラインも含む）─────────
# E2/E3 は franka_e2_e3_summary.json から
E2_means = e2e3["E2"]["means_mae_post_mrad"]
E3_means = e2e3["E3"]["means_mae_post_mrad"]

# E4 は franka_e4_summary.json から
E4_means = e4["E4_hold"]["means_mae_post_mrad"]

# E5/EF は highlights から
def ef_mean(d: str, c: str) -> float:
    return ef["highlights"]["mae_post_mrad"][d][c]

def e5_mean(d: str, c: str) -> float:
    return e5["highlights"]["mae_post_mrad"][d][c]

def ef_energy(d: str, c: str) -> float:
    return ef["highlights"]["energy_post_J"][d][c]

def e5_energy(d: str, c: str) -> float:
    return e5["highlights"]["energy_post_J"][d][c]

def ef_cc(d: str, c: str) -> float:
    return ef["highlights"]["cc_ratio"][d][c]

def e5_cc(d: str, c: str) -> float:
    return e5["highlights"]["cc_ratio"][d][c]

# ─────────────────────────────────────────────────────────────
# 図1: Phase 改善軌跡
# ─────────────────────────────────────────────────────────────

def fig_phase_progression():
    phase_data = {
        "PD\nbaseline":   {d: E2_means["E2-PD"][d]       for d in D_CONDITIONS},
        "E2\n(+CC)":      {d: E2_means["E2-cc"][d]        for d in D_CONDITIONS},
        "E3\n(+Ia/Ib)":   {d: E3_means["E3-cc+ia_ib"][d]  for d in D_CONDITIONS},
        "E4\n(+MCA)":     {d: E4_means["E4-full"][d]      for d in D_CONDITIONS},
        "EF\n(+Cerebellum)": {d: ef_mean(d, "EF-full")    for d in D_CONDITIONS},
    }

    phases   = list(phase_data.keys())
    colors_d = {"D0": "#2196F3", "D1": "#FF9800", "D5": "#9C27B0"}
    markers  = {"D0": "o", "D1": "s", "D5": "^"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("Phase ごとの MAE_post 改善軌跡（11-seed 平均）", fontsize=13, fontweight="bold")

    for ax, d, dl in zip(axes, D_CONDITIONS, D_LABELS):
        vals = [phase_data[p][d] for p in phases]
        ax.plot(range(len(phases)), vals, color=colors_d[d], marker=markers[d],
                linewidth=2, markersize=8, zorder=3)
        ax.fill_between(range(len(phases)), vals, alpha=0.15, color=colors_d[d])
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.0f}", (i, v), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")
        # PD ベースライン水平線
        ax.axhline(vals[0], color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, fontsize=9)
        ax.set_ylabel("MAE_post [mrad]", fontsize=10)
        ax.set_title(dl, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, max(vals) * 1.25)

        # 改善率
        improv = (vals[0] - vals[-1]) / vals[0] * 100
        ax.text(0.97, 0.97, f"改善率\n{improv:.1f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

    plt.tight_layout()
    path = FIG_DIR / "ef_phase_progression.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"保存: {path}")


# ─────────────────────────────────────────────────────────────
# 図2: モジュール寄与分解（積み上げウォーターフォール）
# ─────────────────────────────────────────────────────────────

def fig_module_contribution():
    # D0 と D1 でウォーターフォール
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("モジュール別 MAE_post 寄与分解（対 PD ベースライン）", fontsize=13, fontweight="bold")

    for ax, d, dl in zip(axes, ["D0", "D1"], ["D0 (baseline)", "D1 (低ゲイン kp×0.5)"]):
        pd_val = E2_means["E2-PD"][d]

        steps = [
            ("PD",           pd_val,                       None),
            ("+CC\n(E2)",    E2_means["E2-cc"][d],         None),
            ("+Ia/Ib\n(E3)", E3_means["E3-cc+ia_ib"][d],   None),
            ("+MCA\n(E4)",   E4_means["E4-full"][d],        None),
            ("+Cerebellum\n(EF)", ef_mean(d, "EF-full"),   None),
        ]

        xs      = range(len(steps))
        bottoms = []
        for i, (label, val, _) in enumerate(steps):
            bottoms.append(val)

        colors_bar = ["#607D8B", "#2196F3", "#F44336", "#FF9800", "#4CAF50"]

        for i, (label, val, _) in enumerate(steps):
            if i == 0:
                ax.bar(i, val, color=colors_bar[i], alpha=0.85, zorder=3)
            else:
                prev = steps[i-1][1]
                delta = val - prev
                bottom = min(prev, val)
                height = abs(delta)
                color  = "#4CAF50" if delta < 0 else "#F44336"
                ax.bar(i, height, bottom=bottom, color=color, alpha=0.85, zorder=3)
                # 全体の棒（薄く）
                ax.bar(i, val, color=colors_bar[i], alpha=0.3, zorder=2)
            # 数値
            ax.text(i, val + pd_val * 0.015, f"{val:.0f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
            # 差分
            if i > 0:
                delta = val - steps[i-1][1]
                sign  = "▼" if delta < 0 else "▲"
                color2 = "#2E7D32" if delta < 0 else "#C62828"
                ax.text(i, val - pd_val * 0.04, f"{sign}{abs(delta):.0f}",
                        ha="center", va="top", fontsize=8, color=color2, fontweight="bold")

        ax.set_xticks(xs)
        ax.set_xticklabels([s[0] for s in steps], fontsize=9)
        ax.set_ylabel("MAE_post [mrad]", fontsize=10)
        ax.set_title(dl, fontsize=11)
        ax.set_ylim(0, pd_val * 1.3)
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(pd_val, color="gray", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = FIG_DIR / "ef_module_contribution.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"保存: {path}")


# ─────────────────────────────────────────────────────────────
# 図3: EF 条件比較（全指標）
# ─────────────────────────────────────────────────────────────

def fig_ef_conditions():
    conds  = ["EF-PD", "EF-E4", "EF-cereb", "EF-full"]
    colors = ["#607D8B", "#9C27B0", "#FF9800", "#4CAF50"]
    labels = ["EF-PD\n(baseline)", "EF-E4\n(MCA+CC+Ia/Ib)", "EF-cereb\n(CfC only)", "EF-full\n(complete)"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("EF 条件比較: 全指標 × D 条件（11-seed 平均）", fontsize=13, fontweight="bold")

    for col, (d, dl) in enumerate(zip(D_CONDITIONS, D_LABELS)):
        # MAE_post
        ax = axes[0][col]
        vals = [ef_mean(d, c) for c in conds]
        bars = ax.bar(range(4), vals, color=colors, alpha=0.85)
        ax.set_title(f"{dl}\nMAE_post [mrad]", fontsize=9)
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")
        vmax = max(vals)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+vmax*0.02,
                    f"{v:.0f}", ha="center", fontsize=7, fontweight="bold")

        # Energy_post
        ax = axes[1][col]
        vals = [ef_energy(d, c) for c in conds]
        bars = ax.bar(range(4), vals, color=colors, alpha=0.85)
        ax.set_title(f"{dl}\nEnergy_post [Nm²·s]", fontsize=9)
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")
        vmax = max(vals)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+vmax*0.01,
                    f"{v:.0f}", ha="center", fontsize=7)

    plt.tight_layout()
    path = FIG_DIR / "ef_conditions.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"保存: {path}")


# ─────────────────────────────────────────────────────────────
# 図4: Energy vs MAE 散布図（効率フロンティア）
# ─────────────────────────────────────────────────────────────

def fig_energy_mae():
    # 全 Phase の D0 / D1 データを散布
    points = {
        # (label, mae, energy, color, marker, size)
        "PD":              (E2_means["E2-PD"]["D0"],       e5_energy("D0","E5-PD"),   "#607D8B", "o", 80),
        "E2-cc":           (E2_means["E2-cc"]["D0"],       float("nan"),              "#2196F3", "s", 80),
        "E3-cc+ia_ib":     (E3_means["E3-cc+ia_ib"]["D0"], float("nan"),              "#F44336", "^", 80),
        "E4-full":         (ef_mean("D0","EF-E4"),         e5_energy("D0","E5-full"), "#FF9800", "D", 80),
        "EF-cereb":        (ef_mean("D0","EF-cereb"),      ef_energy("D0","EF-cereb"),"#03A9F4", "P", 100),
        "EF-full ★":       (ef_mean("D0","EF-full"),       ef_energy("D0","EF-full"), "#4CAF50", "*", 180),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Energy_post vs MAE_post: 効率フロンティア（D0・D1）", fontsize=13, fontweight="bold")

    for ax, d, dl in zip(axes, ["D0", "D1"], ["D0 (baseline)", "D1 (低ゲイン)"]):
        pts = {
            "PD":          (E2_means["E2-PD"][d],       e5_energy(d,"E5-PD")),
            "E4-full":     (ef_mean(d,"EF-E4"),         e5_energy(d,"E5-full")),
            "EF-cereb":    (ef_mean(d,"EF-cereb"),      ef_energy(d,"EF-cereb")),
            "EF-full ★":  (ef_mean(d,"EF-full"),       ef_energy(d,"EF-full")),
        }
        colors_map = {
            "PD":         "#607D8B",
            "E4-full":    "#FF9800",
            "EF-cereb":   "#03A9F4",
            "EF-full ★": "#4CAF50",
        }
        markers_map = {
            "PD":         "o",
            "E4-full":    "D",
            "EF-cereb":   "P",
            "EF-full ★": "*",
        }
        sizes_map = {
            "PD":         80,
            "E4-full":    80,
            "EF-cereb":   80,
            "EF-full ★": 180,
        }

        for label, (mae, energy) in pts.items():
            if np.isnan(energy):
                continue
            ax.scatter(mae, energy, color=colors_map[label], marker=markers_map[label],
                       s=sizes_map[label], zorder=4, label=label,
                       edgecolors="white", linewidth=0.8)
            offset_x = -30 if "EF-full" in label else 10
            ax.annotate(label, (mae, energy), textcoords="offset points",
                        xytext=(offset_x, 8), fontsize=8, fontweight="bold",
                        color=colors_map[label])

        # 左下が最良（低MAE・低エネルギー）を示す矢印
        ax.annotate("", xy=(300, 3000), xytext=(700, 5500),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
        ax.text(500, 4300, "最良方向", fontsize=9, color="gray", rotation=35)

        ax.set_xlabel("MAE_post [mrad]", fontsize=10)
        ax.set_ylabel("Energy_post [Nm²·s]", fontsize=10)
        ax.set_title(dl, fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIG_DIR / "ef_energy_mae.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"保存: {path}")


# ─────────────────────────────────────────────────────────────
# 図5: 論文用統合図（2×3）
# ─────────────────────────────────────────────────────────────

def fig_paper_final():
    fig = plt.figure(figsize=(18, 11))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    fig.suptitle(
        "Complete Neural Motor Hierarchy: Phase E1–EF Results Summary\n"
        "(Franka Panda Simulation, 11 seeds)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # ── (0,0)-(0,1): Phase progression ──────────────────────
    ax_prog = fig.add_subplot(gs[0, :2])
    phase_labels = ["PD", "E2\n(+CC)", "E3\n(+Ia/Ib)", "E4\n(+MCA)", "EF\n(+Cereb)"]
    phase_vals = {
        "D0": [E2_means["E2-PD"]["D0"], E2_means["E2-cc"]["D0"],
               E3_means["E3-cc+ia_ib"]["D0"], E4_means["E4-full"]["D0"], ef_mean("D0","EF-full")],
        "D1": [E2_means["E2-PD"]["D1"], E2_means["E2-cc"]["D1"],
               E3_means["E3-cc+ia_ib"]["D1"], E4_means["E4-full"]["D1"], ef_mean("D1","EF-full")],
        "D5": [E2_means["E2-PD"]["D5"], E2_means["E2-cc"]["D5"],
               E3_means["E3-cc+ia_ib"]["D5"], E4_means["E4-full"]["D5"], ef_mean("D5","EF-full")],
    }
    colors_d = {"D0": "#2196F3", "D1": "#FF9800", "D5": "#9C27B0"}
    markers_d = {"D0": "o", "D1": "s", "D5": "^"}
    x = np.arange(len(phase_labels))

    for d in D_CONDITIONS:
        ax_prog.plot(x, phase_vals[d], color=colors_d[d], marker=markers_d[d],
                     linewidth=2.2, markersize=8, label=f"{d}", zorder=3)
        ax_prog.fill_between(x, phase_vals[d], alpha=0.08, color=colors_d[d])

    ax_prog.set_xticks(x); ax_prog.set_xticklabels(phase_labels, fontsize=10)
    ax_prog.set_ylabel("MAE_post [mrad]", fontsize=10)
    ax_prog.set_title("(a) Phase 別 MAE_post 改善軌跡", fontsize=11, loc="left")
    ax_prog.legend(fontsize=9, title="Condition", loc="upper right")
    ax_prog.grid(True, alpha=0.3, axis="y")
    ax_prog.set_ylim(0, 2400)

    # ── (0,2): モジュール寄与（D1 ウォーターフォール）────────
    ax_wf = fig.add_subplot(gs[0, 2])
    d = "D1"
    wf_steps = [
        ("PD",      E2_means["E2-PD"][d]),
        ("+CC",     E2_means["E2-cc"][d]),
        ("+Ia/Ib",  E3_means["E3-cc+ia_ib"][d]),
        ("+MCA",    E4_means["E4-full"][d]),
        ("+Cereb",  ef_mean(d, "EF-full")),
    ]
    bar_colors_wf = ["#607D8B","#2196F3","#F44336","#FF9800","#4CAF50"]
    for i, (lbl, val) in enumerate(wf_steps):
        ax_wf.bar(i, val, color=bar_colors_wf[i], alpha=0.85, zorder=3)
        ax_wf.text(i, val + 30, f"{val:.0f}", ha="center", fontsize=8, fontweight="bold")
        if i > 0:
            delta = val - wf_steps[i-1][1]
            sign  = "▼" if delta < 0 else "▲"
            col2  = "#2E7D32" if delta < 0 else "#C62828"
            ax_wf.text(i, val - 80, f"{sign}{abs(delta):.0f}", ha="center",
                       fontsize=7, color=col2, fontweight="bold")
    ax_wf.set_xticks(range(5)); ax_wf.set_xticklabels([s[0] for s in wf_steps], fontsize=8)
    ax_wf.set_ylabel("MAE_post [mrad]", fontsize=9)
    ax_wf.set_title("(b) モジュール寄与 D1", fontsize=11, loc="left")
    ax_wf.set_ylim(0, max(v for _, v in wf_steps) * 1.25)
    ax_wf.grid(True, alpha=0.3, axis="y")

    # ── (1,0): EF 条件 × MAE D0/D1/D5 ──────────────────────
    ax_ef = fig.add_subplot(gs[1, 0])
    conds_ef = ["EF-PD", "EF-E4", "EF-cereb", "EF-full"]
    colors_ef = ["#607D8B", "#9C27B0", "#FF9800", "#4CAF50"]
    x_ef = np.arange(len(conds_ef))
    width = 0.25
    for i, (d, dl_short) in enumerate(zip(["D0","D1","D5"], ["D0","D1","D5"])):
        vals = [ef_mean(d, c) for c in conds_ef]
        ax_ef.bar(x_ef + (i-1)*width, vals, width, label=dl_short,
                  color=[f"#{int(c[1:3],16):02x}{int(c[3:5],16):02x}{int(c[5:7],16):02x}" for c in
                         ["#607D8B","#9C27B0","#FF9800","#4CAF50"]],
                  alpha=[0.5, 0.7, 0.9][i])
    # 色はグループ分けで
    for i, (d, alpha) in enumerate(zip(["D0","D1","D5"],[0.5,0.7,0.9])):
        vals = [ef_mean(d, c) for c in conds_ef]
        ax_ef.bar(x_ef + (i-1)*width, vals, width, label=d, alpha=alpha,
                  color=["#607D8B","#9C27B0","#FF9800","#4CAF50"])
    ax_ef.set_xticks(x_ef); ax_ef.set_xticklabels(conds_ef, fontsize=8, rotation=15)
    ax_ef.set_ylabel("MAE_post [mrad]", fontsize=9)
    ax_ef.set_title("(c) EF 条件比較", fontsize=11, loc="left")
    ax_ef.legend(fontsize=7, loc="upper right")
    ax_ef.grid(True, alpha=0.3, axis="y")

    # ── (1,1): Energy vs MAE (D0) ────────────────────────────
    ax_em = fig.add_subplot(gs[1, 1])
    pts_em = {
        "PD":         (E2_means["E2-PD"]["D0"],  e5_energy("D0","E5-PD"),   "#607D8B","o",60),
        "E4-full":    (ef_mean("D0","EF-E4"),     e5_energy("D0","E5-full"), "#FF9800","D",60),
        "EF-cereb":   (ef_mean("D0","EF-cereb"),  ef_energy("D0","EF-cereb"),"#03A9F4","P",80),
        "EF-full ★": (ef_mean("D0","EF-full"),   ef_energy("D0","EF-full"), "#4CAF50","*",150),
    }
    for lbl, (mae, ene, col, mk, sz) in pts_em.items():
        if np.isnan(ene): continue
        ax_em.scatter(mae, ene, c=col, marker=mk, s=sz, zorder=4, edgecolors="white", lw=0.8)
        ax_em.annotate(lbl, (mae, ene), textcoords="offset points",
                       xytext=(5, 5), fontsize=7.5, color=col, fontweight="bold")
    ax_em.set_xlabel("MAE_post [mrad]", fontsize=9)
    ax_em.set_ylabel("Energy_post [Nm²·s]", fontsize=9)
    ax_em.set_title("(d) Energy vs MAE (D0)", fontsize=11, loc="left")
    ax_em.grid(True, alpha=0.3)

    # ── (1,2): CC ratio 積み上げ ─────────────────────────────
    ax_cc = fig.add_subplot(gs[1, 2])
    conds_cc  = ["EF-PD", "EF-E4", "EF-cereb", "EF-full"]
    colors_cc = ["#607D8B", "#9C27B0", "#FF9800", "#4CAF50"]
    cc_vals_d0 = [ef_cc("D0", c) for c in conds_cc]
    cc_vals_d1 = [ef_cc("D1", c) for c in conds_cc]
    x_cc = np.arange(len(conds_cc))
    ax_cc.bar(x_cc - 0.18, cc_vals_d0, 0.35, label="D0", color=colors_cc, alpha=0.7)
    ax_cc.bar(x_cc + 0.18, cc_vals_d1, 0.35, label="D1", color=colors_cc, alpha=0.95)
    for i, (v0, v1) in enumerate(zip(cc_vals_d0, cc_vals_d1)):
        ax_cc.text(i-0.18, v0+0.01, f"{v0:.2f}", ha="center", fontsize=7)
        ax_cc.text(i+0.18, v1+0.01, f"{v1:.2f}", ha="center", fontsize=7)
    ax_cc.set_xticks(x_cc); ax_cc.set_xticklabels(conds_cc, fontsize=8, rotation=15)
    ax_cc.set_ylabel("CC ratio τ_cc/(|τ|+τ_cc)", fontsize=9)
    ax_cc.set_title("(e) 同時収縮率", fontsize=11, loc="left")
    ax_cc.legend(fontsize=8); ax_cc.grid(True, alpha=0.3, axis="y")
    ax_cc.set_ylim(0, 0.65)

    path = FIG_DIR / "ef_final_paper.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"保存: {path}")


# ─────────────────────────────────────────────────────────────
# Markdown テーブル
# ─────────────────────────────────────────────────────────────

def write_paper_table():
    lines = [
        "# Phase E2–EF 統合結果テーブル\n",
        f"Seeds: `{ef['seeds']}`\n",
        "## MAE_post [mrad]（外乱後平均絶対誤差）\n",
        "| 手法 | D0 (baseline) | D1 (低ゲイン) | D5 (飽和) | D0 改善率 |",
        "|---|---:|---:|---:|---:|",
    ]

    pd_d0 = E2_means["E2-PD"]["D0"]

    rows = [
        ("PD (baseline)",        E2_means["E2-PD"]["D0"],       E2_means["E2-PD"]["D1"],       E2_means["E2-PD"]["D5"]),
        ("E2: +VirtualCC",       E2_means["E2-cc"]["D0"],        E2_means["E2-cc"]["D1"],        E2_means["E2-cc"]["D5"]),
        ("E3: +Ia/Ib reflex",    E3_means["E3-cc+ia_ib"]["D0"],  E3_means["E3-cc+ia_ib"]["D1"],  E3_means["E3-cc+ia_ib"]["D5"]),
        ("E4: +MCA",             E4_means["E4-full"]["D0"],      E4_means["E4-full"]["D1"],      E4_means["E4-full"]["D5"]),
        ("EF-cereb (CfC only)",  ef_mean("D0","EF-cereb"),       ef_mean("D1","EF-cereb"),       ef_mean("D5","EF-cereb")),
        ("**EF-full (complete)**", ef_mean("D0","EF-full"),      ef_mean("D1","EF-full"),        ef_mean("D5","EF-full")),
    ]

    for label, d0, d1, d5 in rows:
        improv = (pd_d0 - d0) / pd_d0 * 100
        lines.append(f"| {label} | {d0:.1f} | {d1:.1f} | {d5:.1f} | {improv:+.1f}% |")

    lines += [
        "\n## Energy_post [Nm²·s]（外乱後エネルギー消費）\n",
        "| 手法 | D0 | D1 | D5 |",
        "|---|---:|---:|---:|",
    ]
    energy_rows = [
        ("PD",            e5_energy("D0","E5-PD"),    e5_energy("D1","E5-PD"),    e5_energy("D5","E5-PD")),
        ("E4-full",       e5_energy("D0","E5-full"),  e5_energy("D1","E5-full"),  e5_energy("D5","E5-full")),
        ("EF-cereb",      ef_energy("D0","EF-cereb"), ef_energy("D1","EF-cereb"), ef_energy("D5","EF-cereb")),
        ("**EF-full**",   ef_energy("D0","EF-full"),  ef_energy("D1","EF-full"),  ef_energy("D5","EF-full")),
    ]
    for label, d0, d1, d5 in energy_rows:
        lines.append(f"| {label} | {d0:.0f} | {d1:.0f} | {d5:.0f} |")

    lines += [
        "\n## 各モジュールの D0 MAE_post 寄与\n",
        "| モジュール追加 | D0 改善量 [mrad] | 累積 MAE_post [mrad] |",
        "|---|---:|---:|",
    ]
    contrib = [
        ("PD baseline",      0,                                        E2_means["E2-PD"]["D0"]),
        ("+VirtualCC (E2)",  E2_means["E2-PD"]["D0"]-E2_means["E2-cc"]["D0"],            E2_means["E2-cc"]["D0"]),
        ("+Ia/Ib (E3)",      E2_means["E2-cc"]["D0"]-E3_means["E3-cc+ia_ib"]["D0"],      E3_means["E3-cc+ia_ib"]["D0"]),
        ("+MCA (E4)",        E3_means["E3-cc+ia_ib"]["D0"]-E4_means["E4-full"]["D0"],    E4_means["E4-full"]["D0"]),
        ("+Cerebellum (EF)", E4_means["E4-full"]["D0"]-ef_mean("D0","EF-full"),           ef_mean("D0","EF-full")),
    ]
    for label, delta, cumul in contrib:
        sign = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        lines.append(f"| {label} | {sign} | {cumul:.1f} |")

    out = RES_DIR / "paper_ef_table.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"保存: {out}")


# ─────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== EF 可視化・サマリー生成 ===\n")
    fig_phase_progression()
    fig_module_contribution()
    fig_ef_conditions()
    fig_energy_mae()
    fig_paper_final()
    write_paper_table()
    print("\n完了。")
