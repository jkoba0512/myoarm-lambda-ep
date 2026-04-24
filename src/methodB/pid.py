"""
PID コントローラ（関節空間）

各関節に独立した PID。積分アンチワインドアップ付き。
"""

from __future__ import annotations

import numpy as np


class PIDController:
    """
    τ_i = Kp_i * e_i + Kd_i * ė_i + Ki_i * ∫e_i dt

    Attributes
    ----------
    Kp, Kd, Ki : (n_joints,) ゲイン
    dt          : タイムステップ (s)
    windup_limit: 積分項クランプ (Nm)
    """

    def __init__(
        self,
        n_joints: int = 3,
        Kp: float | np.ndarray = 15.0,
        Kd: float | np.ndarray = 1.5,
        Ki: float | np.ndarray = 0.5,
        dt: float = 0.004,
        windup_limit: float = 3.0,
    ):
        ones = np.ones(n_joints)
        self.Kp = np.asarray(Kp) * ones
        self.Kd = np.asarray(Kd) * ones
        self.Ki = np.asarray(Ki) * ones
        self.dt = dt
        self.windup_limit = windup_limit
        self._integral = np.zeros(n_joints)

    def reset(self) -> None:
        self._integral[:] = 0.0

    def compute(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        q_ref: np.ndarray,
        dq_ref: np.ndarray,
    ) -> np.ndarray:
        e = q_ref - q
        de = dq_ref - dq
        self._integral += e * self.dt
        self._integral = np.clip(self._integral, -self.windup_limit, self.windup_limit)
        return self.Kp * e + self.Kd * de + self.Ki * self._integral
