"""
Franka 条件スイープ用ドライバ。

2B の外乱スイープ、2C の負荷/周波数・位相整合スイープ、
Phase D のロバスト性スイープ、Phase A0/A3 の baseline 比較を実行する。

使い方:
  .venv/bin/python scripts/run_franka_condition_sweeps.py --seeds 0 1 2
  .venv/bin/python scripts/run_franka_condition_sweeps.py --only-new  # 未実行条件のみ
  .venv/bin/python scripts/run_franka_condition_sweeps.py --phases 2b 2c  # 特定フェーズのみ
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
    p.add_argument("--phases", nargs="+",
                   choices=["2b", "2c", "2c_phase_b", "phase_d", "phase_a"],
                   default=["2b", "2c", "2c_phase_b", "phase_d", "phase_a"],
                   help="実行するフェーズを指定。省略時は全て実行。")
    args = p.parse_args()

    py = sys.executable

    # ──────────────────────────────────────────────────
    # 2B 外乱スイープ (既存)
    # (name, torque_Nm, steps, dist_time_s, joint_idx, q_noise)
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
    # 2C サイクリックスイープ (既存)
    # (name, amp, load_torque, load_time, load_joint, endpoint_joint, cpg_tau, cpg_tau_r)
    # ──────────────────────────────────────────────────
    cyclic_grid = [
        # --- 振幅・負荷スイープ (cpg_tau=0.3 デフォルト) ---
        ("amp025_load20", 0.25, -20.0, 3.0, 1, 1, 0.3,  None),
        ("amp030_load25", 0.30, -25.0, 3.0, 1, 1, 0.3,  None),
        ("amp035_load30", 0.35, -30.0, 3.0, 1, 1, 0.3,  None),
        # --- CPG 周波数スイープ: tau_r = tau*2.0 連動 ---
        ("tau_020",       0.30, -25.0, 3.0, 1, 1, 0.20, None),
        ("tau_025",       0.30, -25.0, 3.0, 1, 1, 0.25, None),
        ("tau_035",       0.30, -25.0, 3.0, 1, 1, 0.35, None),
    ]

    # ──────────────────────────────────────────────────
    # Phase B: tau_r 固定スイープ (新規 B3-B8)
    # tau_r を固定し tau のみ変えて、tau と tau_r の交絡を除去する
    # ──────────────────────────────────────────────────
    cyclic_phase_b_grid = [
        # tau_r=0.50 固定
        ("tau_020_taur050", 0.30, -25.0, 3.0, 1, 1, 0.20, 0.50),
        ("tau_025_taur050", 0.30, -25.0, 3.0, 1, 1, 0.25, 0.50),
        ("tau_035_taur050", 0.30, -25.0, 3.0, 1, 1, 0.35, 0.50),
        # tau_r=0.60 固定
        ("tau_020_taur060", 0.30, -25.0, 3.0, 1, 1, 0.20, 0.60),
        ("tau_025_taur060", 0.30, -25.0, 3.0, 1, 1, 0.25, 0.60),
        ("tau_035_taur060", 0.30, -25.0, 3.0, 1, 1, 0.35, 0.60),
    ]

    # ──────────────────────────────────────────────────
    # Phase D: 反射弓ロバスト性 + Sim-to-Real 前段
    # (name, extra_args)  — experiment_franka_2b.py を流用
    # ──────────────────────────────────────────────────
    phase_d_grid = [
        ("D0_baseline",    []),
        ("D1_low_gain",    ["--pd-gain-scale", "0.5"]),
        ("D2_delay_5step", ["--obs-delay-steps", "5"]),
        ("D3_strong_dist", ["--disturbance-torque", "87.0"]),
        ("D4_obs_noise",   ["--obs-noise-std", "0.005"]),
        ("D5_torque_sat",  ["--torque-saturation", "30.0"]),
        ("D6_mass_scale",  ["--model-mass-scale", "1.1", "--model-friction-scale", "1.1"]),
    ]

    # ──────────────────────────────────────────────────
    # Phase A0/A3: CfC vs MLP/LSTM baseline
    # experiment_franka_2a.py を流用
    # ──────────────────────────────────────────────────
    phase_a_grid = [
        ("A0_cfc",  "cfc"),
        ("A3_mlp",  "mlp"),
        ("A3_lstm", "lstm"),
    ]

    for seed in args.seeds:
        seed_s = str(seed)

        # ── 2B 既存スイープ ────────────────────────────────────────
        if "2b" in args.phases:
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

        # ── 2C 既存スイープ ────────────────────────────────────────
        if "2c" in args.phases:
            for name, amp, load_torque, load_time, load_joint, ep_joint, cpg_tau, cpg_tau_r in cyclic_grid:
                sweep_dir = f"experiment_franka_2c/condition_sweep_2c/{name}"
                if args.only_new and already_done(sweep_dir, seed):
                    print(f"  skip (exists): {sweep_dir}/seed{seed}")
                    continue
                cmd = [
                    py, str(SCRIPTS / "experiment_franka_2c.py"),
                    "--seed", seed_s,
                    "--cpg-amplitude", str(amp),
                    "--cpg-tau", str(cpg_tau),
                    "--load-torque", str(load_torque),
                    "--load-time", str(load_time),
                    "--load-joint", str(load_joint),
                    "--endpoint-joint", str(ep_joint),
                    "--sweep-name", f"condition_sweep_2c/{name}",
                ]
                run(cmd)

        # ── Phase B: tau_r 固定スイープ (B3-B8) ───────────────────
        if "2c_phase_b" in args.phases:
            for name, amp, load_torque, load_time, load_joint, ep_joint, cpg_tau, cpg_tau_r in cyclic_phase_b_grid:
                sweep_dir = f"experiment_franka_2c/condition_sweep_2c/{name}"
                if args.only_new and already_done(sweep_dir, seed):
                    print(f"  skip (exists): {sweep_dir}/seed{seed}")
                    continue
                cmd = [
                    py, str(SCRIPTS / "experiment_franka_2c.py"),
                    "--seed", seed_s,
                    "--cpg-amplitude", str(amp),
                    "--cpg-tau", str(cpg_tau),
                    "--cpg-tau-r", str(cpg_tau_r),
                    "--load-torque", str(load_torque),
                    "--load-time", str(load_time),
                    "--load-joint", str(load_joint),
                    "--endpoint-joint", str(ep_joint),
                    "--save-phase-log",
                    "--sweep-name", f"condition_sweep_2c/{name}",
                ]
                run(cmd)

        # ── Phase D: ロバスト性スイープ ────────────────────────────
        if "phase_d" in args.phases:
            for d_name, extra_args in phase_d_grid:
                sweep_dir = f"experiment_franka_2b/condition_sweep_phase_d/{d_name}"
                if args.only_new and already_done(sweep_dir, seed):
                    print(f"  skip (exists): {sweep_dir}/seed{seed}")
                    continue
                cmd = [
                    py, str(SCRIPTS / "experiment_franka_2b.py"),
                    "--seed", seed_s,
                    "--disturbance-set", "custom",
                    "--disturbance-name", d_name,
                    "--disturbance-torque", "60.0",
                    "--disturbance-steps", "20",
                    "--disturbance-time", "3.0",
                    "--disturbance-joint", "1",
                    "--sweep-name", f"condition_sweep_phase_d/{d_name}",
                ] + extra_args
                run(cmd)

        # ── Phase A0/A3: CfC vs MLP/LSTM baseline ─────────────────
        if "phase_a" in args.phases:
            for a_name, compensator in phase_a_grid:
                sweep_dir = f"experiment_franka_2a/condition_sweep_phase_a/{a_name}"
                if args.only_new and already_done(sweep_dir, seed):
                    print(f"  skip (exists): {sweep_dir}/seed{seed}")
                    continue
                run([
                    py, str(SCRIPTS / "experiment_franka_2a.py"),
                    "--seed", seed_s,
                    "--compensator", compensator,
                    "--sweep-name", f"condition_sweep_phase_a/{a_name}",
                ])


if __name__ == "__main__":
    main()
