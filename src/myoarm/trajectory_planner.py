"""
trajectory_planner.py — 最小ジャーク軌跡プランナー（小脳的 feedforward 計画）。

Flash & Hogan (1985) の最小ジャーク理論に基づき、
エンドポイント軌跡を事前計算して追従制御に使う。

神経科学的対応: 大脳皮質 + 小脳の協調による
「内部前向きモデル」ベースの軌跡計画。
"""

from __future__ import annotations

import numpy as np


class MinimumJerkPlanner:
    """
    最小ジャーク軌跡をリアルタイム生成するプランナー。

    使い方:
        planner = MinimumJerkPlanner(dt=0.005)
        planner.plan(p_start, p_target, T=1.0)
        for step in range(n_steps):
            p_ref, v_ref = planner.step()
    """

    def __init__(self, dt: float = 0.005) -> None:
        self.dt = dt
        self._p_start: np.ndarray | None = None
        self._p_end:   np.ndarray | None = None
        self._T:       float = 1.0
        self._t:       float = 0.0
        self._active:  bool  = False

    # ------------------------------------------------------------------

    def plan(
        self,
        p_start: np.ndarray,
        p_target: np.ndarray,
        T: float = 1.0,
        min_T: float = 0.3,
        max_T: float = 2.5,
        speed_gain: float = 0.9,
    ) -> None:
        """
        新たな軌跡を計画する。

        Parameters
        ----------
        p_start  : 開始位置 (3,)
        p_target : 目標位置 (3,)
        T        : 所要時間 [s]。指定がなければ距離から自動計算。
        min_T / max_T : T のクランプ範囲。
        speed_gain: 距離あたりの時間調整係数（小さいほど速い）。
        """
        dist = float(np.linalg.norm(p_target - p_start))
        if T <= 0:
            # 距離 × gain で自然な所要時間を推定 (human: ~0.5-1.5 s for 0.2-0.8 m)
            T = float(np.clip(dist * speed_gain / 0.5, min_T, max_T))
        else:
            T = float(np.clip(T, min_T, max_T))

        self._p_start = p_start.copy()
        self._p_end   = p_target.copy()
        self._T       = T
        self._t       = 0.0
        self._active  = True

    # ------------------------------------------------------------------

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        """1ステップ進め、(参照位置, 参照速度) を返す（後方互換）。"""
        p, v, _ = self.step_with_accel()
        return p, v

    def step_with_accel(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        1ステップ進め、(p_ref, v_ref, a_ref) を返す。
        feedforward 制御に使う加速度プロファイルも提供する。
        """
        if not self._active or self._p_end is None:
            p = self._p_end if self._p_end is not None else np.zeros(3)
            return p.copy(), np.zeros(3), np.zeros(3)

        tau = min(self._t / self._T, 1.0)
        delta = self._p_end - self._p_start

        # 最小ジャーク位置: p(τ) = p0 + (pf-p0)*(10τ³-15τ⁴+6τ⁵)
        s   = 10*tau**3 - 15*tau**4 + 6*tau**5
        # 速度: ṡ/T = (30τ²-60τ³+30τ⁴)/T
        ds  = (30*tau**2 - 60*tau**3 + 30*tau**4) / self._T
        # 加速度: s̈/T² = (60τ-180τ²+120τ³)/T²
        d2s = (60*tau - 180*tau**2 + 120*tau**3) / (self._T**2)

        p = self._p_start + delta * s
        v = delta * ds
        a = delta * d2s

        self._t += self.dt
        if self._t >= self._T:
            self._active = False

        return p, v, a

    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._active  = False
        self._t       = 0.0
        self._p_start = None
        self._p_end   = None

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def progress(self) -> float:
        """軌跡進捗 0-1。"""
        if not self._active or self._T <= 0:
            return 1.0
        return min(self._t / self._T, 1.0)
