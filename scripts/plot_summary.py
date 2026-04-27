"""
Phase C: 寄与分解整理 — 論文用サマリ図と寄与マトリクスの生成

入力: results/franka_master_summary.json
     results/franka_condition_sweep_summary.json
出力:
  results/summary_figures/
    fig1_ablation_mae.png          2A: アブレーション（静止保持 MAE）
    fig2_disturbance.png           2B: 外乱耐性（全強度、ピーク誤差）
    fig3_cyclic.png                2C: サイクリック（エンドポイント誤差）
    fig4_contribution_matrix.png   課題×モジュール 効果量マトリクス
    fig5_condition_sweeps.png      条件スイープ（2B/2C）
  results/franka_contribution_table.json

使い方:
  uv run python scripts/plot_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT        = Path(__file__).parents[1]
RESULTS_DIR = ROOT / "results"
SUMMARY_DIR = RESULTS_DIR / "summary_figures"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_JSON = RESULTS_DIR / "franka_master_summary.json"
SWEEP_JSON   = RESULTS_DIR / "franka_condition_sweep_summary.json"

# ── 共通スタイル ──────────────────────────────────────────────────────
COLORS = {
    "PD":               "#888888",
    "PD+CfC":           "#4477CC",
    "Full":             "#44AA44",
    "CPG+CfC":          "#CC7733",
    "CPG+CfC+LIF_FB":   "#44AA44",
}
CAPSIZE = 5
ELINEWIDTH = 1.5
ALPHA_BAR  = 0.85


def _bar_with_err(ax, x, mean, std, color, label, width=0.5):
    bar = ax.bar(x, mean, width, color=color, alpha=ALPHA_BAR, label=label,
                 zorder=3)
    ax.errorbar(x, mean, yerr=std, fmt="none", color="black",
                capsize=CAPSIZE, elinewidth=ELINEWIDTH, zorder=4)
    return bar


def _finish(ax, title, ylabel, xlabel_labels):
    ax.set_xticks(range(len(xlabel_labels)))
    ax.set_xticklabels(xlabel_labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)


# ──────────────────────────────────────────────────────────────────────
def fig1_ablation(data: dict) -> None:
    """2A: アブレーション（静止保持 MAE）"""
    d2a    = data["2a"]
    conds  = ["PD", "PD+CfC", "Full"]
    means  = [d2a[c]["static_mae_mrad"]["mean"]   for c in conds]
    stds   = [d2a[c]["static_mae_mrad"]["std"]    for c in conds]
    d_cfc  = d2a["effect_size"]["cohens_d_PD_vs_PDCfC"]
    d_full = d2a["effect_size"]["cohens_d_PD_vs_Full"]

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (cond, mean, std) in enumerate(zip(conds, means, stds)):
        _bar_with_err(ax, i, mean, std, COLORS[cond], cond)
        ax.text(i, mean + std + 1.5, f"{mean:.1f}", ha="center",
                va="bottom", fontsize=8)

    # Cohen's d annotation
    ax.annotate("", xy=(1, means[0] * 0.82), xytext=(0, means[0] * 0.82),
                arrowprops=dict(arrowstyle="<->", color="#555"))
    ax.text(0.5, means[0] * 0.84, f"d={d_cfc:.2f}", ha="center", fontsize=8, color="#333")
    ax.annotate("", xy=(2, means[0] * 0.72), xytext=(0, means[0] * 0.72),
                arrowprops=dict(arrowstyle="<->", color="#555"))
    ax.text(1.0, means[0] * 0.74, f"d={d_full:.2f}", ha="center", fontsize=8, color="#333")

    n = data["2a"]["PD"]["static_mae_mrad"]["n"]
    _finish(ax, f"Fig.1 Ablation: Static Holding MAE  (n={n} seeds)",
            "MAE [mrad]", conds)
    ax.set_ylim(0, max(means) + max(stds) + 20)
    plt.tight_layout()
    path = SUMMARY_DIR / "fig1_ablation_mae.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  保存: {path.name}")


# ──────────────────────────────────────────────────────────────────────
def fig2_disturbance(data: dict) -> None:
    """2B: 外乱耐性（3強度、ピーク誤差）"""
    d2b    = data["2b"]
    levels = ["light_30Nm", "medium_60Nm", "heavy_87Nm"]
    labels = ["Light\n(30 Nm)", "Medium\n(60 Nm)", "Heavy\n(87 Nm)"]
    conds  = ["PD+CfC", "Full"]
    w      = 0.3

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Fig.2 Disturbance Rejection: Peak Error & Recovery Time\n"
                 f"(n={d2b['light_30Nm']['PD+CfC']['peak_err_rad']['n']} seeds)",
                 fontsize=10)

    # 左: ピーク誤差
    ax = axes[0]
    x  = np.arange(len(levels))
    for j, cond in enumerate(conds):
        means = [d2b[lv][cond]["peak_err_rad"]["mean"] for lv in levels]
        stds  = [d2b[lv][cond]["peak_err_rad"]["std"]  for lv in levels]
        offset = (j - 0.5) * w
        bars = ax.bar(x + offset, means, w, color=COLORS[cond],
                      alpha=ALPHA_BAR, label=cond, zorder=3)
        ax.errorbar(x + offset, means, yerr=stds, fmt="none",
                    color="black", capsize=CAPSIZE, elinewidth=ELINEWIDTH, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Peak error [rad]", fontsize=10)
    ax.set_title("Peak Joint Error (J2)", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Cohen's d annotation（heavy のみ）
    d_heavy = d2b["heavy_87Nm"]["effect_size"]["cohens_d_peak_err_PDCfC_vs_Full"]
    ax.text(2, 0.05, f"d={d_heavy:.2f}", ha="center", fontsize=8, color="#555")

    # 右: 回復時間
    ax = axes[1]
    for j, cond in enumerate(conds):
        means = [d2b[lv][cond]["recovery_time_s"]["mean"] or 0 for lv in levels]
        stds  = [d2b[lv][cond]["recovery_time_s"]["std"]  or 0 for lv in levels]
        fr    = [d2b[lv][cond]["recovery_time_s"]["fail_rate"] or 0 for lv in levels]
        offset = (j - 0.5) * w
        ax.bar(x + offset, means, w, color=COLORS[cond],
               alpha=ALPHA_BAR, label=cond, zorder=3)
        ax.errorbar(x + offset, means, yerr=stds, fmt="none",
                    color="black", capsize=CAPSIZE, elinewidth=ELINEWIDTH, zorder=4)
        for k, (xi, fi) in enumerate(zip(x + offset, fr)):
            if fi > 0:
                ax.text(xi, means[k] + stds[k] + 0.02,
                        f"NR={fi:.0%}", ha="center", fontsize=7, color="red")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Recovery time [s]", fontsize=10)
    ax.set_title("Recovery Time  (NR=not recovered)", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = SUMMARY_DIR / "fig2_disturbance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  保存: {path.name}")


# ──────────────────────────────────────────────────────────────────────
def fig3_cyclic(data: dict) -> None:
    """2C: サイクリック動作（エンドポイント誤差、負荷前後）"""
    d2c   = data["2c"]
    conds = ["CPG+CfC", "CPG+CfC+LIF_FB"]
    d_eff = d2c["effect_size"]["cohens_d_ep_post_nofb_vs_fb"]
    n     = d2c["CPG+CfC"]["ep_err_post_mrad"]["n"]

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(conds))
    w = 0.3

    for j, phase in enumerate(["ep_err_pre_mrad", "ep_err_post_mrad"]):
        means  = [d2c[c][phase]["mean"] for c in conds]
        stds   = [d2c[c][phase]["std"]  for c in conds]
        offset = (j - 0.5) * w
        label  = "Pre-load" if j == 0 else "Post-load"
        alpha  = 0.4       if j == 0 else ALPHA_BAR
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.bar(x[i] + offset, m, w, color=COLORS[conds[i]],
                   alpha=alpha, label=f"{conds[i]} ({label})" if i == 0 else "",
                   zorder=3)
            ax.errorbar(x[i] + offset, m, yerr=s, fmt="none",
                        color="black", capsize=CAPSIZE, elinewidth=ELINEWIDTH, zorder=4)

    # post-load Cohen's d
    post_means = [d2c[c]["ep_err_post_mrad"]["mean"] for c in conds]
    post_stds  = [d2c[c]["ep_err_post_mrad"]["std"]  for c in conds]
    ymax = max(m + s for m, s in zip(post_means, post_stds)) + 20
    ax.annotate("", xy=(x[1] + 0.15, ymax - 5), xytext=(x[0] + 0.15, ymax - 5),
                arrowprops=dict(arrowstyle="<->", color="#555"))
    ax.text(0.5 + 0.15, ymax - 2,
            f"d={d_eff:.2f}  ({post_means[0]:.0f}→{post_means[1]:.0f} mrad)",
            ha="center", fontsize=8, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(["CPG+CfC", "CPG+CfC\n+LIF_FB"], fontsize=9)
    ax.set_ylabel("Endpoint error [mrad]", fontsize=10)
    ax.set_title(f"Fig.3 Cyclic Motion: Endpoint Error  (n={n} seeds)\n"
                 "Light bar=pre-load, dark bar=post-load", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ymax + 30)

    # 凡例
    patches = [
        mpatches.Patch(color=COLORS["CPG+CfC"],        alpha=0.4,       label="CPG+CfC  pre-load"),
        mpatches.Patch(color=COLORS["CPG+CfC"],        alpha=ALPHA_BAR, label="CPG+CfC  post-load"),
        mpatches.Patch(color=COLORS["CPG+CfC+LIF_FB"], alpha=0.4,       label="LIF_FB   pre-load"),
        mpatches.Patch(color=COLORS["CPG+CfC+LIF_FB"], alpha=ALPHA_BAR, label="LIF_FB   post-load"),
    ]
    ax.legend(handles=patches, fontsize=7, loc="upper right")

    plt.tight_layout()
    path = SUMMARY_DIR / "fig3_cyclic.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  保存: {path.name}")


# ──────────────────────────────────────────────────────────────────────
def fig4_contribution_matrix(data: dict) -> None:
    """課題×モジュール 効果量マトリクス（ヒートマップ）"""
    # 行: タスク、列: モジュール
    # 値: Cohen's d（PDに対するCfC、PD+CfCに対する反射弓、CPG+CfCに対するLIF_FB）
    tasks   = ["Static\nholding\n(2A/2D)", "Disturbance\nrejection\n(2B heavy)", "Cyclic\nmotion\n(2C)"]
    modules = ["CfC\nCerebellum", "Izhikevich\nReflex Arc", "LIF\nProprioceptor FB"]

    matrix = np.array([
        # CfC   Reflex   LIF_FB
        [data["2a"]["effect_size"]["cohens_d_PD_vs_PDCfC"],               0.0,   0.0],   # Static 2A
        [0.0,   data["2b"]["heavy_87Nm"]["effect_size"]["cohens_d_peak_err_PDCfC_vs_Full"], 0.0],   # Disturbance
        [0.0,   0.0,   data["2c"]["effect_size"]["cohens_d_ep_post_nofb_vs_fb"]],          # Cyclic
    ])

    # 2D でも CfC 効果を追加（平均）
    matrix[0, 0] = (data["2a"]["effect_size"]["cohens_d_PD_vs_PDCfC"] +
                    data["2d"]["effect_size"]["cohens_d_mae_PD_vs_Full"]) / 2

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=2.5, aspect="auto")

    # 値ラベル
    for i in range(len(tasks)):
        for j in range(len(modules)):
            v = matrix[i, j]
            color = "white" if v > 1.5 else "black"
            size_label = ("large" if v > 0.8 else "medium" if v > 0.5 else "small") if v > 0 else "—"
            ax.text(j, i, f"d={v:.2f}\n({size_label})",
                    ha="center", va="center", fontsize=10, color=color, fontweight="bold")

    ax.set_xticks(range(len(modules)))
    ax.set_xticklabels(modules, fontsize=10)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=10)
    ax.set_title("Fig.4 Contribution Matrix: Cohen's d per Task × Module\n"
                 "(green=large effect, yellow=medium, red=small/none)", fontsize=10)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cohen's d", fontsize=9)

    plt.tight_layout()
    path = SUMMARY_DIR / "fig4_contribution_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  保存: {path.name}")


# ──────────────────────────────────────────────────────────────────────
def fig5_condition_sweeps(data: dict) -> None:
    """条件スイープ（2B/2C）の論文用サマリ図。"""
    d2b = data["2b_condition_sweep"]
    d2c = data["2c_condition_sweep"]

    conds_2b = ["timing_early", "timing_mid", "timing_late", "torque_45", "torque_75"]
    labels_2b = ["Early", "Mid", "Late", "45 Nm", "75 Nm"]
    conds_2c = ["amp025_load20", "amp030_load25", "amp035_load30"]
    labels_2c = ["Amp0.25\nLoad20", "Amp0.30\nLoad25", "Amp0.35\nLoad30"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.suptitle("Fig.5 Condition Sweeps: Robustness Across Disturbance and Load Conditions\n"
                 "(n=11 seeds)", fontsize=10)

    # 左: 2B post-disturbance MAE
    ax = axes[0]
    x = np.arange(len(conds_2b))
    w = 0.34
    for j, cond_name in enumerate(["PD+CfC", "Full"]):
        means = [d2b[c][cond_name]["mae_post_mrad"]["mean"] for c in conds_2b]
        stds = [d2b[c][cond_name]["mae_post_mrad"]["std"] for c in conds_2b]
        offset = (j - 0.5) * w
        ax.bar(x + offset, means, w, color=COLORS[cond_name], alpha=ALPHA_BAR,
               label=cond_name, zorder=3)
        ax.errorbar(x + offset, means, yerr=stds, fmt="none", color="black",
                    capsize=CAPSIZE, elinewidth=ELINEWIDTH, zorder=4)

    for i, c in enumerate(conds_2b):
        pd_mean = d2b[c]["PD+CfC"]["mae_post_mrad"]["mean"]
        full_mean = d2b[c]["Full"]["mae_post_mrad"]["mean"]
        improvement = (pd_mean - full_mean) / pd_mean * 100.0
        d_eff = d2b[c]["effect_size"]["cohens_d_mae_post_PDCfC_vs_Full"]
        y = max(pd_mean, full_mean) + 8
        ax.text(i, y, f"-{improvement:.1f}%\nd={d_eff:.2f}",
                ha="center", va="bottom", fontsize=7, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_2b, fontsize=9)
    ax.set_ylabel("Post-disturbance MAE [mrad]", fontsize=10)
    ax.set_title("2B Disturbance Sweep", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper right")

    # 右: 2C post-load endpoint error
    ax = axes[1]
    x = np.arange(len(conds_2c))
    for j, cond_name in enumerate(["CPG+CfC", "CPG+CfC+LIF_FB"]):
        means = [d2c[c][cond_name]["ep_err_post_mrad"]["mean"] for c in conds_2c]
        stds = [d2c[c][cond_name]["ep_err_post_mrad"]["std"] for c in conds_2c]
        offset = (j - 0.5) * w
        ax.bar(x + offset, means, w, color=COLORS[cond_name], alpha=ALPHA_BAR,
               label=cond_name, zorder=3)
        ax.errorbar(x + offset, means, yerr=stds, fmt="none", color="black",
                    capsize=CAPSIZE, elinewidth=ELINEWIDTH, zorder=4)

    for i, c in enumerate(conds_2c):
        nofb = d2c[c]["CPG+CfC"]["ep_err_post_mrad"]["mean"]
        fb = d2c[c]["CPG+CfC+LIF_FB"]["ep_err_post_mrad"]["mean"]
        improvement = (nofb - fb) / nofb * 100.0
        d_eff = d2c[c]["effect_size"]["cohens_d_ep_err_post_nofb_vs_fb"]
        fail_fb = d2c[c]["CPG+CfC+LIF_FB"]["recovery_time_s"]["fail_rate"]
        y = max(nofb, fb) + 12
        ax.text(i, y, f"{improvement:+.1f}%\nd={d_eff:.2f}\nNR={fail_fb:.0%}",
                ha="center", va="bottom", fontsize=7, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_2c, fontsize=9)
    ax.set_ylabel("Post-load endpoint error [mrad]", fontsize=10)
    ax.set_title("2C Cyclic Motion Sweep", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    path = SUMMARY_DIR / "fig5_condition_sweeps.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  保存: {path.name}")


# ──────────────────────────────────────────────────────────────────────
def save_contribution_table(data: dict) -> None:
    """論文用寄与分解 JSON テーブルを保存する。"""
    n_seeds = len(data.get("seeds_used", []))

    table = {
        "n_seeds": n_seeds,
        "seeds":   data.get("seeds_used", []),
        "tasks": {
            "static_holding": {
                "metric":     "static_mae_mrad",
                "baseline":   {"condition": "PD",     **data["2a"]["PD"]["static_mae_mrad"]},
                "PD+CfC":     {**data["2a"]["PD+CfC"]["static_mae_mrad"],
                               "cohens_d_vs_PD": data["2a"]["effect_size"]["cohens_d_PD_vs_PDCfC"]},
                "Full":       {**data["2a"]["Full"]["static_mae_mrad"],
                               "cohens_d_vs_PD": data["2a"]["effect_size"]["cohens_d_PD_vs_Full"]},
                "module_verdict": {
                    "CfC_cerebellum":       "large effect (d>2)",
                    "izhikevich_reflex":    "not evaluated in isolation",
                    "lif_proprioceptor_fb": "not applicable (CPG off)",
                },
            },
            "disturbance_rejection_heavy": {
                "metric":  "peak_err_rad",
                "PD+CfC":  {**data["2b"]["heavy_87Nm"]["PD+CfC"]["peak_err_rad"]},
                "Full":    {**data["2b"]["heavy_87Nm"]["Full"]["peak_err_rad"],
                            "cohens_d_vs_PDCfC": data["2b"]["heavy_87Nm"]["effect_size"][
                                "cohens_d_peak_err_PDCfC_vs_Full"]},
                "module_verdict": {
                    "CfC_cerebellum":       "already included in PD+CfC baseline",
                    "izhikevich_reflex":    "small effect (d=0.13)",
                    "lif_proprioceptor_fb": "not applicable (CPG off)",
                },
            },
            "cyclic_motion": {
                "metric":        "ep_err_post_mrad",
                "CPG+CfC":       {**data["2c"]["CPG+CfC"]["ep_err_post_mrad"]},
                "CPG+CfC+LIF_FB": {**data["2c"]["CPG+CfC+LIF_FB"]["ep_err_post_mrad"],
                                   "cohens_d_vs_nofb": data["2c"]["effect_size"][
                                       "cohens_d_ep_post_nofb_vs_fb"]},
                "module_verdict": {
                    "CfC_cerebellum":       "already included in both conditions",
                    "izhikevich_reflex":    "disabled in 2C",
                    "lif_proprioceptor_fb": "small effect (d=0.28)",
                },
            },
        },
        "overall_verdict": {
            "CfC_cerebellum":       "PRIMARY CONTRIBUTOR — large effect (d≈2.1–2.6) on static MAE",
            "izhikevich_reflex":    "MINOR CONTRIBUTOR — small effect (d≈0.13) on disturbance peak",
            "lif_proprioceptor_fb": "MINOR CONTRIBUTOR — small effect (d≈0.28) on cyclic endpoint",
        },
    }

    out = RESULTS_DIR / "franka_contribution_table.json"
    with open(out, "w") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)
    print(f"  保存: {out.name}")


# ──────────────────────────────────────────────────────────────────────
def print_paper_table(data: dict) -> None:
    """論文用比較表をターミナルに出力する。"""
    n = len(data.get("seeds_used", []))
    print(f"\n{'='*65}")
    print(f"  論文用比較表  (n={n} seeds, mean ± std)")
    print(f"{'='*65}")

    print("\n[Table 1] Ablation: Static Holding MAE (mrad)")
    print(f"  {'Condition':<22} {'Mean':>8} {'±Std':>8}  {'d vs PD':>8}")
    print(f"  {'-'*50}")
    for cond, d_key in [("PD", None),
                         ("PD+CfC", "cohens_d_PD_vs_PDCfC"),
                         ("Full",   "cohens_d_PD_vs_Full")]:
        s = data["2a"][cond]["static_mae_mrad"]
        d = data["2a"]["effect_size"].get(d_key, "") if d_key else ""
        d_str = f"{d:.2f}" if d != "" else "—"
        print(f"  {cond:<22} {s['mean']:>8.1f} {s['std']:>8.1f}  {d_str:>8}")

    print("\n[Table 2] Disturbance Rejection: Peak Error (rad), Heavy (87 Nm)")
    print(f"  {'Condition':<12} {'Mean':>8} {'±Std':>8}  {'d vs PD+CfC':>12}")
    print(f"  {'-'*46}")
    for cond, d_key in [("PD+CfC", None), ("Full", "cohens_d_peak_err_PDCfC_vs_Full")]:
        s = data["2b"]["heavy_87Nm"][cond]["peak_err_rad"]
        d = data["2b"]["heavy_87Nm"]["effect_size"].get(d_key, "") if d_key else ""
        d_str = f"{d:.2f}" if d != "" else "—"
        print(f"  {cond:<12} {s['mean']:>8.3f} {s['std']:>8.3f}  {d_str:>12}")

    print("\n[Table 3] Cyclic Motion: Post-load Endpoint Error (mrad)")
    print(f"  {'Condition':<22} {'Mean':>8} {'±Std':>8}  {'d vs no-FB':>12}")
    print(f"  {'-'*56}")
    for cond, d_key in [("CPG+CfC", None),
                         ("CPG+CfC+LIF_FB", "cohens_d_ep_post_nofb_vs_fb")]:
        s = data["2c"][cond]["ep_err_post_mrad"]
        d = data["2c"]["effect_size"].get(d_key, "") if d_key else ""
        d_str = f"{d:.2f}" if d != "" else "—"
        print(f"  {cond:<22} {s['mean']:>8.1f} {s['std']:>8.1f}  {d_str:>12}")

    print(f"\n[Contribution Matrix]")
    print(f"  {'Module':<28} {'Static (d)':>10} {'Disturbance (d)':>16} {'Cyclic (d)':>11}")
    print(f"  {'-'*67}")
    print(f"  {'CfC Cerebellum':<28} {'≈2.3':>10} {'(baseline)':>16} {'(baseline)':>11}")
    d_ref  = data["2b"]["heavy_87Nm"]["effect_size"]["cohens_d_peak_err_PDCfC_vs_Full"]
    d_lif  = data["2c"]["effect_size"]["cohens_d_ep_post_nofb_vs_fb"]
    print(f"  {'Izhikevich Reflex Arc':<28} {'(n/a)':>10} {d_ref:>16.2f} {'(off)':>11}")
    print(f"  {'LIF Proprioceptor FB':<28} {'(off)':>10} {'(off)':>16} {d_lif:>11.2f}")

    print(f"\n  ★ CfC Cerebellum が主要貢献（d≈2.1–2.6）")
    print(f"  ★ 反射弓・LIF FB は小効果（d<0.3）")


# ──────────────────────────────────────────────────────────────────────
def main():
    with open(SUMMARY_JSON) as f:
        data = json.load(f)
    with open(SWEEP_JSON) as f:
        sweep_data = json.load(f)

    print("=== Phase C: 寄与分解整理 ===")
    print(f"  seeds: {data.get('seeds_used', [])}\n")

    fig1_ablation(data)
    fig2_disturbance(data)
    fig3_cyclic(data)
    fig4_contribution_matrix(data)
    fig5_condition_sweeps(sweep_data)
    save_contribution_table(data)
    print_paper_table(data)

    print(f"\n  → 全図: {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
