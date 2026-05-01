"""Fig 2: Tip-speed profiles.

3 conditions: endpoint_pd / F12 best (pure λ visuo) / F12 + reflexes.

Panel (a): Single representative seed (seed 19, vpr near group mean).
Panel (b): Mean speed time-warped to per-seed movement window
           (onset = first speed > 0.02 m/s; offset = first drop below 0.02
           after onset+5 samples). Reveals bell-shape.
Panel (c): Distribution of vel_peak_ratio across all n=50 seeds (F16).

dt = 0.020s (MyoSuite frame_skip=10).

Output: figures/fig2_velocity_profiles.{pdf,png}
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "results" / "experiment_myo_p15" / "f14_trajectories.npz"
F16 = ROOT / "results" / "experiment_myo_p15" / "f16_n50.json"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

DT = 0.020
SPEED_THRESH = 0.02
N_WARP = 100
REPRESENTATIVE_SEED = 27

CONDITIONS = [
    ("traj_endpoint_pd_reference", "endpoint_pd",
     "Endpoint PD", "#888888"),
    ("traj_F12_best_pure_λ_visuo", "F12 best (pure λ visuo)",
     "λ + visuo (pure)", "#4292c6"),
    ("traj_+_reflexes", "F12 best + reflexes",
     "λ + visuo + reflex", "#08519c"),
]


def per_seed_speed(traj_xyz):
    pos = savgol_filter(traj_xyz, window_length=11, polyorder=3, axis=1)
    vel = np.gradient(pos, DT, axis=1)
    return np.linalg.norm(vel, axis=2)


def warp_to_movement_window(speed_seed):
    onset_idx = next((i for i, s in enumerate(speed_seed) if s > SPEED_THRESH), None)
    if onset_idx is None:
        return np.full(N_WARP, np.nan), False
    offset_idx = next(
        (i for i in range(onset_idx + 5, len(speed_seed)) if speed_seed[i] < SPEED_THRESH),
        len(speed_seed) - 1,
    )
    seg = speed_seed[onset_idx : offset_idx + 1]
    if len(seg) < 5:
        return np.full(N_WARP, np.nan), False
    src_t = np.linspace(0, 1, len(seg))
    tgt_t = np.linspace(0, 1, N_WARP)
    return np.interp(tgt_t, src_t, seg), True


def main():
    d = np.load(TRAJ, allow_pickle=True)
    seeds = d["seeds"].tolist()
    n_seeds = len(seeds)
    rep_idx = seeds.index(REPRESENTATIVE_SEED)

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
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25),
                              gridspec_kw={"width_ratios": [1.2, 1.2, 0.9]})
    ax_rep, ax_warp, ax_dist = axes

    n_t = 150
    t_abs = np.arange(n_t) * DT

    print("--- Per-condition diagnostics ---")
    for traj_key, f16_key, label, color in CONDITIONS:
        traj = np.array(d[traj_key].tolist(), dtype=float)
        spd = per_seed_speed(traj)

        rep_speed = spd[rep_idx]
        ax_rep.plot(t_abs, rep_speed, color=color, linewidth=1.6, label=label)

        warped = []
        for i in range(n_seeds):
            w, ok = warp_to_movement_window(spd[i])
            if ok:
                warped.append(w)
        warped = np.array(warped)
        m_w = warped.mean(axis=0)
        sem_w = warped.std(axis=0, ddof=1) / np.sqrt(len(warped))
        nt_w = np.linspace(0, 1, N_WARP)
        ax_warp.plot(nt_w, m_w, color=color, linewidth=1.6, label=label)
        ax_warp.fill_between(nt_w, m_w - sem_w, m_w + sem_w,
                              color=color, alpha=0.2, linewidth=0)

        peak_idx = int(np.argmax(m_w))
        print(f"  {label:25s}  warped peak v={m_w[peak_idx]:.3f}  at t={peak_idx/(N_WARP-1):.2f}  "
              f"(n_valid_seeds={len(warped)}/{n_seeds})")

    ax_rep.set_xlabel("Time (s)")
    ax_rep.set_ylabel("Tip speed (m/s)")
    ax_rep.set_xlim(0, 2.0)
    ax_rep.set_ylim(bottom=0)
    ax_rep.legend(loc="upper right", frameon=False)
    ax_rep.set_title(
        r"$\mathbf{(a)}$" + f"  Seed {REPRESENTATIVE_SEED} ($n = 1$)",
        loc="left", pad=6,
    )

    ax_warp.axvspan(0.40, 0.50, color="#d4f1d4", alpha=0.6, zorder=0,
                     label="Human peak time")
    ax_warp.set_xlabel("Movement-normalized time")
    ax_warp.set_ylabel("Tip speed (m/s)")
    ax_warp.set_xlim(0, 1)
    ax_warp.set_ylim(bottom=0)
    ax_warp.legend(loc="upper right", frameon=False, fontsize=6.5)
    ax_warp.set_title(
        r"$\mathbf{(b)}$" + f"  Time-warped mean ($n = {n_seeds}$)",
        loc="left", pad=6,
    )

    rng = np.random.default_rng(0)
    for i, (traj_key, f16_key, label, color) in enumerate(CONDITIONS):
        vpr = np.array([row["vel_peak_ratio"] for row in f16["raw_per_seed"][f16_key]])
        x = i + rng.normal(0, 0.06, size=vpr.size)
        ax_dist.scatter(x, vpr, color=color, alpha=0.45, s=12,
                          edgecolor="none")
        ax_dist.hlines(vpr.mean(), i - 0.25, i + 0.25, color="black",
                        linewidth=1.5, zorder=3)

    ax_dist.axhspan(0.40, 0.50, color="#d4f1d4", alpha=0.6, zorder=0)
    ax_dist.set_xticks(range(len(CONDITIONS)))
    ax_dist.set_xticklabels([c[2] for c in CONDITIONS], rotation=15, ha="right")
    ax_dist.set_ylabel("Velocity peak ratio")
    ax_dist.set_ylim(0, 1)
    ax_dist.set_title(
        r"$\mathbf{(c)}$" + "  vpr distribution ($n = 50$)",
        loc="left", pad=6,
    )

    fig.tight_layout()

    pdf_path = OUT / "fig2_velocity_profiles.pdf"
    png_path = OUT / "fig2_velocity_profiles.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
