"""
Izhikevich Fast-Spiking ニューロンによる反射弓

生物的対応:
  脊髄反射弓に動機づけられた高速誤差補正モジュール
  （Ia 求心性線維からの誤差入力を受け、興奮性トルクパルスを出力）

モデル（Izhikevich, 2003）:
  dv/dt = 0.04v² + 5v + 140 - u + I
  du/dt = a(bv - u)
  v ≥ 30 mV → 発火: v ← c, u ← u + d

  Fast-Spiking パラメータ: a=0.1, b=0.2, c=-65, d=2
  → 短い不応期、高周波発火が可能

内部タイムステップ:
  標準 Izhikevich モデルは 1ms 単位で定義されているため、
  シミュレーション dt=2ms の 1 ステップを 2 substep で積分する。

入力電流（デッドゾーン付き）:
  e_pos[i] = max(|Δq[i]| - θ_pos, 0)   位置誤差のデッドゾーン処理
  e_vel[i] = max(|Δdq[i]| - θ_vel, 0)  速度急変のデッドゾーン処理
  I[i] = k_pos · e_pos[i] + k_vel · e_vel[i]

出力（位置補正）:
  発火時: Δq_reflex[i] = gain · sign(Δq[i])   正しい方向へのパルス
  毎ステップ減衰: Δq_reflex ← decay · Δq_reflex

役割:
  - CfC（小脳）より高速（1-2 ステップ ≈ 2-4 ms）
  - 外乱閾値を超えた場合のみ発火（通常動作には干渉しない）

注意:
  生物学的 Ia 反射弓は拮抗筋の抑制を介した主動筋収縮促進を担うが、
  本実装は位置誤差方向への興奮性トルクパルスであり、
  機能的には「脊髄反射弓に動機づけられた高速誤差補正」として扱う。
"""

from __future__ import annotations

import numpy as np


class IzhikevichReflexArc:
    """
    n_joints 個の Fast-Spiking Izhikevich ニューロンによる反射弓。

    Parameters
    ----------
    n_joints       : 関節数
    dt             : シミュレーション タイムステップ [s]
    a, b, c, d     : Izhikevich パラメータ（Fast-Spiking デフォルト）
    theta_pos      : 位置誤差デッドゾーン [rad]
    theta_vel      : 速度急変デッドゾーン [rad/s]
    k_pos          : 位置誤差 → 入力電流ゲイン
    k_vel          : 速度急変 → 入力電流ゲイン
    gain           : 発火 → 位置補正ゲイン [rad/発火]
    max_correction : 最大補正量 [rad]（クリップ）
    decay          : 補正の毎ステップ減衰率 ∈ (0, 1)
    """

    def __init__(
        self,
        n_joints:       int   = 5,
        dt:             float = 0.002,
        a:              float = 0.1,
        b:              float = 0.2,
        c:              float = -65.0,
        d:              float = 2.0,
        theta_pos:      float = 0.15,
        theta_vel:      float = 1.0,
        k_pos:          float = 100.0,
        k_vel:          float = 25.0,
        gain:           float = 0.04,
        max_correction: float = 0.25,
        decay:          float = 0.7,
    ):
        self.n      = n_joints
        self.dt_ms  = dt * 1000.0   # [ms] Izhikevich モデルの単位に変換
        self.a = a; self.b = b; self.c = c; self.d = d

        self.theta_pos      = theta_pos
        self.theta_vel      = theta_vel
        self.k_pos          = k_pos
        self.k_vel          = k_vel
        self.gain           = gain
        self.max_correction = max_correction
        self.decay          = decay

        # ニューロン状態
        self.v = np.full(n_joints, c)          # 膜電位 [mV]
        self.u = b * np.full(n_joints, c)      # 回復変数

        # 出力バッファ（減衰中の補正量）
        self._output     = np.zeros(n_joints)  # 現在の補正出力 [rad]
        self._fired_prev = np.zeros(n_joints, dtype=bool)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """エピソード開始時にリセット。"""
        self.v[:]        = self.c
        self.u[:]        = self.b * self.c
        self._output[:]  = 0.0
        self._fired_prev[:] = False

    # ------------------------------------------------------------------
    def respond(
        self,
        q_error: np.ndarray,
        dq:      np.ndarray,
    ) -> np.ndarray:
        """
        関節誤差に対する反射補正量を返す。

        Parameters
        ----------
        q_error : q_cmd - q_actual (n,) [rad]
        dq      : 現在の関節速度 (n,) [rad/s]

        Returns
        -------
        delta_q : 位置補正 (n,) [rad]
        """
        # デッドゾーン付き入力電流
        e_pos = np.maximum(np.abs(q_error) - self.theta_pos, 0.0)
        e_vel = np.maximum(np.abs(dq)      - self.theta_vel, 0.0)
        I = self.k_pos * e_pos + self.k_vel * e_vel

        # Izhikevich 積分（2 substep × 1ms）
        fired = self._substep(I)
        fired = self._substep(I) | fired  # 2回目も OR で合算

        # 発火時: 方向に応じた補正を加算
        pulse = np.where(fired, self.gain * np.sign(q_error + 1e-9), 0.0)
        self._output += pulse

        # クリップ＋減衰
        self._output = np.clip(self._output, -self.max_correction, self.max_correction)
        self._output *= self.decay

        self._fired_prev = fired
        return self._output.copy()

    # ------------------------------------------------------------------
    def _substep(self, I: np.ndarray) -> np.ndarray:
        """
        Izhikevich モデル 1ms ステップ（Euler 法）。

        Parameters
        ----------
        I : 入力電流 (n,)

        Returns
        -------
        fired : 発火フラグ (n, bool)
        """
        dt = self.dt_ms  # [ms]

        dv = (0.04 * self.v**2 + 5.0 * self.v + 140.0 - self.u + I) * dt
        du = self.a * (self.b * self.v - self.u) * dt

        self.v += dv
        self.u += du

        fired = self.v >= 30.0
        self.v = np.where(fired, self.c, self.v)
        self.u = np.where(fired, self.u + self.d, self.u)

        return fired

    # ------------------------------------------------------------------
    def is_active(self) -> np.ndarray:
        """
        直前のステップで発火中の関節を返す（可視化用）。

        Returns
        -------
        bool array (n,)
        """
        return self._fired_prev.copy()
