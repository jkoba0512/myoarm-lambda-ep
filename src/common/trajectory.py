"""
基準軌道生成

- sinusoidal_joint_trajectory: 関節空間での正弦波軌道
- TrajectoryPoint: 各時刻の参照値をまとめたデータクラス
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrajectoryPoint:
    q_ref: np.ndarray    # 関節角 (rad)
    dq_ref: np.ndarray   # 関節速度 (rad/s)
    ddq_ref: np.ndarray  # 関節加速度 (rad/s²)


def sinusoidal_joint_trajectory(
    t: float,
    amplitudes: np.ndarray,
    frequencies: np.ndarray,
    phases: np.ndarray,
    offsets: np.ndarray,
) -> TrajectoryPoint:
    """
    各関節独立の正弦波軌道。

    q_i(t) = offset_i + A_i * sin(2π f_i t + phase_i)
    """
    omega = 2 * np.pi * frequencies
    q = offsets + amplitudes * np.sin(omega * t + phases)
    dq = amplitudes * omega * np.cos(omega * t + phases)
    ddq = -amplitudes * omega**2 * np.sin(omega * t + phases)
    return TrajectoryPoint(q_ref=q, dq_ref=dq, ddq_ref=ddq)


def make_default_trajectory(n_joints: int = 3) -> dict:
    """実験 1-A 用デフォルト軌道パラメータ。"""
    return {
        "amplitudes": np.array([0.3, 0.4, 0.4])[:n_joints],
        "frequencies": np.array([0.5, 0.3, 0.4])[:n_joints],
        "phases":      np.array([0.0, np.pi / 4, np.pi / 2])[:n_joints],
        "offsets":     np.zeros(n_joints),
    }


def generate_trajectory_array(
    duration: float,
    dt: float,
    params: dict,
) -> list[TrajectoryPoint]:
    """時系列全体を一括生成してリストで返す。"""
    t_arr = np.arange(0.0, duration, dt)
    return [sinusoidal_joint_trajectory(t, **params) for t in t_arr]
