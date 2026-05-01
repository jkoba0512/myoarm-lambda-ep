"""Fig 3: 3D tip trajectories for representative seed.

Compares engineering (endpoint_pd) vs biological (λ + visuo, λ + visuo + reflex)
reach paths for seed 19. Shows straightness and final-error differences.

Output: figures/fig3_trajectories.{pdf,png}
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "results" / "experiment_myo_p15" / "f14_trajectories.npz"
TARGETS = ROOT / "results" / "experiment_myo_p15" / "f14_targets.json"
F16 = ROOT / "results" / "experiment_myo_p15" / "f16_n50.json"
OUT = ROOT / "figures"

REPRESENTATIVE_SEED = 27

CONDITIONS = [
    ("traj_endpoint_pd_reference", "endpoint_pd",
     "Endpoint PD", "#888888"),
    ("traj_F12_best_pure_λ_visuo", "F12 best (pure λ visuo)",
     "λ + visuo (pure)", "#4292c6"),
    ("traj_+_reflexes", "F12 best + reflexes",
     "λ + visuo + reflex", "#08519c"),
]


def path_metrics(pos):
    """Returns (path length, displacement, straightness)."""
    seg = np.diff(pos, axis=0)
    L = float(np.sum(np.linalg.norm(seg, axis=1)))
    D = float(np.linalg.norm(pos[-1] - pos[0]))
    return L, D, D / max(L, 1e-9)


def main():
    d = np.load(TRAJ, allow_pickle=True)
    seeds = d["seeds"].tolist()
    rep_idx = seeds.index(REPRESENTATIVE_SEED)

    with TARGETS.open() as f:
        targets = json.load(f)
    tip0, target = targets[str(REPRESENTATIVE_SEED)]
    tip0 = np.array(tip0)
    target = np.array(target)

    with F16.open() as f:
        f16 = json.load(f)

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

    fig = plt.figure(figsize=(8.4, 3.0))
    gs = fig.add_gridspec(1, 3, wspace=0.42, width_ratios=[1.5, 1.0, 1.0])
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_yz = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[0, 2])

    n_t = 150
    DT = 0.020
    t = np.arange(n_t) * DT

    print(f"--- Seed {REPRESENTATIVE_SEED} ---")
    print(f"  start={tip0.round(3)}, target={target.round(3)}, "
          f"dist={np.linalg.norm(target-tip0):.3f}m")

    for traj_key, f16_key, label, color in CONDITIONS:
        traj = np.array(d[traj_key].tolist(), dtype=float)
        pos = traj[rep_idx]

        L, D, S = path_metrics(pos)
        min_err = next(r["tip_err_min_mm"] for r in f16["raw_per_seed"][f16_key]
                       if r["seed"] == REPRESENTATIVE_SEED)

        ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=color,
                  linewidth=1.5, label=f"{label} (S={S:.2f})")
        ax_yz.plot(pos[:, 1], pos[:, 2], color=color, linewidth=1.5,
                   label=f"{label}\n(S={S:.2f})")

        dist_to_target = np.linalg.norm(pos - target[None, :], axis=1) * 1000
        ax_dist.plot(t, dist_to_target, color=color, linewidth=1.5,
                     label=label)

        print(f"  {label:25s}  S={S:.3f}  L={L:.3f}m  D={D:.3f}m  min_err={min_err:.1f}mm")

    ax3d.scatter([tip0[0]], [tip0[1]], [tip0[2]],
                 color="black", marker="o", s=40, zorder=5, label="Start")
    ax3d.scatter([target[0]], [target[1]], [target[2]],
                 color="red", marker="*", s=110, zorder=5, label="Target")
    ax_yz.scatter([tip0[1]], [tip0[2]],
                  color="black", marker="o", s=40, zorder=5, label="Start")
    ax_yz.scatter([target[1]], [target[2]],
                  color="red", marker="*", s=110, zorder=5, label="Target")

    ax3d.set_xlabel("x (m)", labelpad=-2)
    ax3d.set_ylabel("y (m)", labelpad=-2)
    ax3d.set_zlabel("z (m)", labelpad=-2)
    ax3d.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax3d.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax3d.zaxis.set_major_locator(MaxNLocator(nbins=4))
    ax3d.tick_params(axis="both", which="major", pad=-2)
    ax3d.set_title(r"$\mathbf{(a)}$" + "  3D reach path",
                    loc="left", pad=6)
    ax3d.legend(loc="upper left", fontsize=6, frameon=False)
    ax3d.view_init(elev=18, azim=-65)

    ax_yz.set_xlabel("y (m)")
    ax_yz.set_ylabel("z (m)")
    ax_yz.set_aspect("equal", adjustable="datalim")
    ax_yz.set_title(r"$\mathbf{(b)}$" + "  Sagittal projection (y-z)",
                     loc="left", pad=6)
    ax_yz.spines["top"].set_visible(False)
    ax_yz.spines["right"].set_visible(False)
    ax_yz.legend(loc="upper right", fontsize=6, frameon=False,
                  labelspacing=0.3)

    ax_dist.set_xlabel("Time (s)")
    ax_dist.set_ylabel("Distance to target (mm)")
    ax_dist.set_xlim(0, 2.0)
    ax_dist.set_ylim(bottom=0)
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)
    ax_dist.legend(loc="upper right", frameon=False)
    ax_dist.set_title(r"$\mathbf{(c)}$" + "  Distance to target",
                       loc="left", pad=6)

    pdf_path = OUT / "fig3_trajectories.pdf"
    png_path = OUT / "fig3_trajectories.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
