"""
真の Ia/Ib 反射弧（Phase E3）

生物学的背景:
  Ia 求心性線維（筋紡錘一次終末）:
    - 伸張量（Δq）と伸張速度（dq）の両方に感応
    - αモーターニューロンに興奮性シナプスを持つ → 伸張された筋を収縮させる
    - Ia 抑制性介在ニューロンを介して拮抗筋を抑制（相反抑制）
    - 生物学的潜時: ヒト上肢 20-25 ms（脊髄反射）

  Ib 求心性線維（ゴルジ腱器官）:
    - 筋腱の張力に感応（筋力に比例）
    - Ib 抑制性介在ニューロンを介して主動筋を抑制（自原性抑制）
    - 役割: 過剰な力の発生を防ぐ（腱の保護）

旧 IzhikevichReflexArc との違い:
  旧: 入力 = 位置誤差 err = q_target - q → 静止保持中でも常時発火 → D1 悪化
  新: 入力 = 伸張速度 dq（Ia）/ トルク τ（Ib） → 動的事象のみに応答 → 安定

遅延バッファ:
  生物学的潜時（~20 ms）を遅延バッファで再現。
  500 Hz 制御で 10 steps = 20 ms。デフォルト latency_steps=10。
"""

from __future__ import annotations
import numpy as np


class IaIbReflexArc:
    """
    Ia/Ib 求心性線維に基づく脊髄反射弓。

    Parameters
    ----------
    n_joints           : 関節数
    dt                 : タイムステップ [s]
    ia_latency_steps   : Ia 反射の遅延ステップ数（デフォルト 10 = 20 ms @500 Hz）
    ib_latency_steps   : Ib 反射の遅延ステップ数（デフォルト 12 = 24 ms @500 Hz）
    ia_vel_thresh      : Ia 発火の速度閾値 [rad/s]
    ia_gain            : Ia → 抵抗トルク変換ゲイン [Nm/(rad/s)]
    ia_reciprocal_gain : 相反抑制ゲイン（拮抗筋チャネル抑制の比率）
    ib_torque_thresh   : Ib 発火の力閾値 [Nm]
    ib_gain            : Ib → 抑制トルク変換ゲイン [Nm/Nm]
    """

    def __init__(
        self,
        n_joints:           int   = 7,
        dt:                 float = 0.002,
        ia_latency_steps:   int   = 10,
        ib_latency_steps:   int   = 12,
        ia_vel_thresh:      float = 0.5,
        ia_gain:            float = 20.0,
        ia_reciprocal_gain: float = 0.5,
        ib_torque_thresh:   float = 25.0,
        ib_gain:            float = 0.3,
    ):
        self.n_joints           = n_joints
        self.dt                 = dt
        self.ia_latency_steps   = ia_latency_steps
        self.ib_latency_steps   = ib_latency_steps
        self.ia_vel_thresh      = ia_vel_thresh
        self.ia_gain            = ia_gain
        self.ia_reciprocal_gain = ia_reciprocal_gain
        self.ib_torque_thresh   = ib_torque_thresh
        self.ib_gain            = ib_gain

        # 遅延バッファ（リングバッファ）
        self._ia_buf = np.zeros((ia_latency_steps, n_joints))
        self._ib_buf = np.zeros((ib_latency_steps, n_joints))
        self._ia_idx = 0
        self._ib_idx = 0

        # 診断用
        self._ia_active  = np.zeros(n_joints, dtype=bool)
        self._ib_active  = np.zeros(n_joints, dtype=bool)
        self._last_tau_ia = np.zeros(n_joints)
        self._last_tau_ib = np.zeros(n_joints)
        self._latency_measured: float | None = None

    # ------------------------------------------------------------------
    # リセット
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._ia_buf[:] = 0.0
        self._ib_buf[:] = 0.0
        self._ia_idx = 0
        self._ib_idx = 0
        self._ia_active[:]  = False
        self._ib_active[:]  = False
        self._last_tau_ia[:] = 0.0
        self._last_tau_ib[:] = 0.0
        self._latency_measured = None

    # ------------------------------------------------------------------
    # メインステップ
    # ------------------------------------------------------------------

    def step(
        self,
        dq:     np.ndarray,
        tau_ag: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        1 ステップ実行して Ia/Ib 反射トルクを計算する。

        Parameters
        ----------
        dq     : 現在の関節速度 (n,) [rad/s]（Ia の入力）
        tau_ag : 主動筋チャネルの推定トルク (n,) [Nm]（Ib の入力）

        Returns
        -------
        tau_ia : Ia 反射トルク (n,) [Nm]
                 符号: 伸張方向に抵抗（= -sign(dq) * amplitude）
        tau_ib : Ib 反射トルク (n,) [Nm]
                 符号: 主動筋を抑制（= -sign(tau_ag) * amplitude）
        """
        # ── Ia 経路 ────────────────────────────────────────────────
        # 1. 現在速度をバッファへ書き込み
        self._ia_buf[self._ia_idx] = dq
        # 2. 遅延後の速度を読み出し
        delayed_dq = self._ia_buf[
            (self._ia_idx - self.ia_latency_steps + 1) % self.ia_latency_steps
        ]
        self._ia_idx = (self._ia_idx + 1) % self.ia_latency_steps

        # 3. 閾値を超えた関節で発火
        ia_firing = np.abs(delayed_dq) > self.ia_vel_thresh
        self._ia_active[:] = ia_firing

        # 4. 伸張に抵抗するトルク（負の速度フィードバック）
        tau_ia = -self.ia_gain * delayed_dq * ia_firing.astype(float)

        # ── Ib 経路 ────────────────────────────────────────────────
        # 1. 現在の主動筋トルクをバッファへ書き込み
        self._ib_buf[self._ib_idx] = tau_ag
        # 2. 遅延後のトルクを読み出し
        delayed_tau = self._ib_buf[
            (self._ib_idx - self.ib_latency_steps + 1) % self.ib_latency_steps
        ]
        self._ib_idx = (self._ib_idx + 1) % self.ib_latency_steps

        # 3. 閾値を超えた関節で発火（過剰力の抑制）
        ib_firing = np.abs(delayed_tau) > self.ib_torque_thresh
        self._ib_active[:] = ib_firing

        # 4. 主動筋を抑制するトルク（力と逆向き）
        tau_ib = -self.ib_gain * delayed_tau * ib_firing.astype(float)

        self._last_tau_ia[:] = tau_ia
        self._last_tau_ib[:] = tau_ib
        return tau_ia, tau_ib

    # ------------------------------------------------------------------
    # 診断
    # ------------------------------------------------------------------

    def is_ia_active(self) -> np.ndarray:
        return self._ia_active.copy()

    def is_ib_active(self) -> np.ndarray:
        return self._ib_active.copy()

    def get_reflex_latency_ms(self) -> float:
        """Ia 遅延ステップ数からシミュレーション時間 [ms] を返す。"""
        return self.ia_latency_steps * self.dt * 1000.0

    def get_reciprocal_inhibition(self) -> np.ndarray:
        """
        相反抑制信号（0-1）を返す。
        拮抗筋チャネルにこの値を掛けることで相反抑制を実現する。
        active なら 1 - ia_reciprocal_gain まで抑制。
        """
        return 1.0 - self.ia_reciprocal_gain * self._ia_active.astype(float)
