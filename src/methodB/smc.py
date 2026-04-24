"""
スライディングモード制御（SMC）

スライディング面:  s_i = ė_i + λ_i * e_i
制御則:            τ_i = -k_s * s_i - k_r * tanh(s_i / φ)

tanh を使うことで sign によるチャタリングを抑制する。
"""

from __future__ import annotations

import numpy as np


class SMCController:
    """
    Parameters
    ----------
    lam     : スライディング面の帯域幅 λ (rad/s)
    k_s     : 等価制御ゲイン
    k_r     : リーチングゲイン
    phi     : boundary layer 厚さ（tanh のスケール）
    """

    def __init__(
        self,
        n_joints: int = 3,
        lam: float = 5.0,
        k_s: float = 3.0,
        k_r: float = 1.5,
        phi: float = 0.05,
    ):
        ones = np.ones(n_joints)
        self.lam = lam * ones
        self.k_s = k_s * ones
        self.k_r = k_r * ones
        self.phi = phi * ones

    def reset(self) -> None:
        pass  # SMC はステートレス

    def compute(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        q_ref: np.ndarray,
        dq_ref: np.ndarray,
    ) -> np.ndarray:
        e = q_ref - q
        de = dq_ref - dq
        s = de + self.lam * e
        tau = self.k_s * s + self.k_r * np.tanh(s / self.phi)
        return tau

    def sliding_surface(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        q_ref: np.ndarray,
        dq_ref: np.ndarray,
    ) -> np.ndarray:
        """可視化用：スライディング面 s の値を返す。"""
        e = q_ref - q
        de = dq_ref - dq
        return de + self.lam * e
