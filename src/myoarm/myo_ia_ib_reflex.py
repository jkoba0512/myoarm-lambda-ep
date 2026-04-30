"""
MyoIaIbReflexArc — 筋骨格モデル用 Ia/Ib 反射弧。

Franka 版 IaIbReflexArc との違い:
  - 入力が「関節速度・関節トルク」ではなく
    「筋長変化率 (actuator_velocity)・筋力 (actuator_force)」
  - Ia 反射 (筋紡錘 → 急速伸張への応答): actuator_velocity > 0（筋が伸びている）
  - Ib 反射 (ゴルジ腱器官 → 過大張力への応答): |actuator_force| > Ib_threshold

信号の意味:
  - actuator_velocity > 0: 筋が伸張されている（ストレッチ）→ Ia 反射で筋収縮強化
  - actuator_force が大きい: 腱への過負荷 → Ib 反射で筋活性化を抑制

両者の合計が筋活性化への微小補正 (delta_act) として返される。
"""

from __future__ import annotations

from collections import deque

import numpy as np


class MyoIaIbReflexArc:
    """
    筋骨格用 Ia/Ib 反射弧。

    Parameters
    ----------
    n_muscles      : 筋数（34）
    dt             : タイムステップ [s]
    ia_latency_steps : Ia 反射遅延ステップ（脊髄処理 ~ 20 ms）
    ib_latency_steps : Ib 反射遅延ステップ
    K_ia           : Ia 反射ゲイン（伸張速度 → 活性化追加量）
    K_ib           : Ib 反射ゲイン（筋力超過 → 活性化抑制量）
    ib_threshold   : Ib 反射が発動する筋力閾値 [N] (絶対値)
    """

    def __init__(
        self,
        n_muscles:         int,
        dt:                float = 0.002,
        ia_latency_steps:  int   = 10,   # 20 ms
        ib_latency_steps:  int   = 10,
        K_ia:              float = 0.05,
        K_ib:              float = 0.03,
        ib_threshold:      float = 200.0,  # [N]
    ) -> None:
        self.n = n_muscles
        self.dt = dt
        self.K_ia = K_ia
        self.K_ib = K_ib
        self.ib_threshold = ib_threshold

        # 遅延バッファ（最小 1 ステップ）
        ia_lat = max(ia_latency_steps, 1)
        ib_lat = max(ib_latency_steps, 1)
        self._ia_buf: deque[np.ndarray] = deque(
            [np.zeros(n_muscles)] * ia_lat, maxlen=ia_lat
        )
        self._ib_buf: deque[np.ndarray] = deque(
            [np.zeros(n_muscles)] * ib_lat, maxlen=ib_lat
        )

    # ------------------------------------------------------------------

    def step(
        self,
        muscle_vel:   np.ndarray,  # actuator_velocity (n_muscles,) [m/s]
        muscle_force: np.ndarray,  # actuator_force    (n_muscles,) [N]  ※負値が多い
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        1 ステップ実行して Ia / Ib 反射補正量を返す。

        Returns
        -------
        delta_ia : (n_muscles,)  Ia 反射の筋活性化補正（正 = 収縮強化）
        delta_ib : (n_muscles,)  Ib 反射の筋活性化補正（負 = 過活性抑制）
        """
        # Ia 反射: 筋が伸張されているとき（vel > 0）にのみ応答
        ia_signal = self.K_ia * np.clip(muscle_vel, 0.0, None)

        # Ib 反射: |force| が閾値を超えた分だけ抑制
        force_excess = np.abs(muscle_force) - self.ib_threshold
        ib_signal = -self.K_ib * np.clip(force_excess, 0.0, None)

        # 遅延を適用して遅延済み信号を取り出す
        self._ia_buf.appendleft(ia_signal)
        delta_ia = self._ia_buf[-1].copy()

        self._ib_buf.appendleft(ib_signal)
        delta_ib = self._ib_buf[-1].copy()

        return delta_ia, delta_ib

    def reset(self) -> None:
        for _ in range(len(self._ia_buf)):
            self._ia_buf[_] = np.zeros(self.n)
        for _ in range(len(self._ib_buf)):
            self._ib_buf[_] = np.zeros(self.n)
        # deque は同じオブジェクトを参照するのでゼロ埋め
        for i in range(len(self._ia_buf)):
            self._ia_buf[i] = np.zeros(self.n)
        for i in range(len(self._ib_buf)):
            self._ib_buf[i] = np.zeros(self.n)
