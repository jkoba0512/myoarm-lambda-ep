"""
Franka 条件スイープ用ドライバ。

まず 2B の外乱スイープ、次に 2C の負荷/周波数スイープを実行する。

使い方:
  .venv/bin/python scripts/run_franka_condition_sweeps.py --seeds 0 1 2
  .venv/bin/python scripts/run_franka_condition_sweeps.py --only-new  # 未実行条件のみ
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def already_done(sweep_dir: str, seed: int) -> bool:
    p = RESULTS / sweep_dir / f"seed{seed}" / "metrics.json"
    return p.exists()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--only-new", action="store_true",
                   help="既存の metrics.json があるものはスキップする")
    args = p.parse_args()

    py = sys.executable

    # ──────────────────────────────────────────────────
    # 2B 外乱スイープ
    # (name, torque_Nm, steps, dist_time_s, joint_idx)
    # ──────────────────────────────────────────────────
    disturbance_grid = [
        # --- タイミング変化 (joint=1, 60Nm) ---
        ("timing_early", 60.0, 20, 2.0, 1, 0.0),
        ("timing_mid",   60.0, 20, 3.0, 1, 0.0),
        ("timing_late",  60.0, 20, 4.0, 1, 0.0),
        # --- トルク変化 (joint=1, t=3s) ---
        ("torque_45",    45.0, 20, 3.0, 1, 0.0),
        ("torque_75",    75.0, 20, 3.0, 1, 0.0),
        # --- 外乱関節スイープ (medium 60Nm, t=3s) ---
        ("joint3_60Nm",  60.0, 20, 3.0, 3, 0.0),
        ("joint5_60Nm",  60.0, 20, 3.0, 5, 0.0),
        # --- 初期姿勢摂動 (joint=1, medium 60Nm, t=3s) ---
        ("qnoise_005",   60.0, 20, 3.0, 1, 0.05),
        ("qnoise_010",   60.0, 20, 3.0, 1, 0.10),
    ]

    # ──────────────────────────────────────────────────
    # 2C サイクリックスイープ
    # (name, amp, load_torque, load_time, load_joint, endpoint_joint, cpg_tau)
    # ──────────────────────────────────────────────────
    cyclic_grid = [
        # --- 振幅・負荷スイープ (cpg_tau=0.3 デフォルト) ---
        ("amp025_load20", 0.25, -20.0, 3.0, 1, 1, 0.3),
        ("amp030_load25", 0.30, -25.0, 3.0, 1, 1, 0.3),
        ("amp035_load30", 0.35, -30.0, 3.0, 1, 1, 0.3),
        # --- CPG 周波数スイープ (amp=0.3, load=-25Nm) ---
        ("tau_020",       0.30, -25.0, 3.0, 1, 1, 0.20),
        ("tau_025",       0.30, -25.0, 3.0, 1, 1, 0.25),
        ("tau_035",       0.30, -25.0, 3.0, 1, 1, 0.35),
    ]

    for seed in args.seeds:
        seed_s = str(seed)

        for name, torque, steps, tdist, joint, q_noise in disturbance_grid:
            sweep_dir = f"experiment_franka_2b/condition_sweep_2b/{name}"
            if args.only_new and already_done(sweep_dir, seed):
                print(f"  skip (exists): {sweep_dir}/seed{seed}")
                continue
            cmd = [
                py, str(SCRIPTS / "experiment_franka_2b.py"),
                "--seed", seed_s,
                "--disturbance-set", "custom",
                "--disturbance-name", name,
                "--disturbance-torque", str(torque),
                "--disturbance-steps", str(steps),
                "--disturbance-time", str(tdist),
                "--disturbance-joint", str(joint),
                "--sweep-name", f"condition_sweep_2b/{name}",
            ]
            if q_noise > 0.0:
                cmd += ["--q-init-noise", str(q_noise)]
            run(cmd)

        for name, amp, load_torque, load_time, load_joint, ep_joint, cpg_tau in cyclic_grid:
            sweep_dir = f"experiment_franka_2c/condition_sweep_2c/{name}"
            if args.only_new and already_done(sweep_dir, seed):
                print(f"  skip (exists): {sweep_dir}/seed{seed}")
                continue
            run([
                py, str(SCRIPTS / "experiment_franka_2c.py"),
                "--seed", seed_s,
                "--cpg-amplitude", str(amp),
                "--cpg-tau", str(cpg_tau),
                "--load-torque", str(load_torque),
                "--load-time", str(load_time),
                "--load-joint", str(load_joint),
                "--endpoint-joint", str(ep_joint),
                "--sweep-name", f"condition_sweep_2c/{name}",
            ])


if __name__ == "__main__":
    main()
