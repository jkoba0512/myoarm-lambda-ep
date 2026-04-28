"""
InferiorOliveAnalog — 下オリーブ核の散発発火（〜1 Hz）を模擬する。

生体の下オリーブ核は予測誤差をプルキンエ細胞に登上線維として伝える。
発火頻度は約 1 Hz と非常に疎であることが電気生理学的に知られている。

mode:
  "sparse"       : 予測誤差に比例した確率で発火（デフォルト、生体に忠実）
  "continuous"   : 毎ステップ発火（F0-abstract の挙動に相当）
  "error_gated"  : 誤差が閾値を超えた場合のみ発火

F4 実験では firing_rate_hz と mode を変化させて学習効率を比較する。
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x: float) -> float:
    """数値安定版シグモイド関数。"""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    ex = np.exp(x)
    return ex / (1.0 + ex)


class InferiorOliveAnalog:
    """
    下オリーブ核アナログ。

    Parameters
    ----------
    dt              : タイムステップ [s]（FrankaEnv.dt に合わせる）
    firing_rate_hz  : 目標発火頻度 [Hz]（sparse モード時）
    gain            : 予測誤差 → 発火確率の sigmoid ゲイン
    mode            : "sparse" | "continuous" | "error_gated"
    error_gate_thresh : error_gated モードの誤差閾値 [rad]
    seed            : 乱数シード（sparse モード）
    """

    def __init__(
        self,
        dt:                float = 0.002,
        firing_rate_hz:    float = 1.0,
        gain:              float = 5.0,
        mode:              str   = "sparse",
        error_gate_thresh: float = 0.01,
        seed:              int   = 0,
    ) -> None:
        assert mode in ("sparse", "continuous", "error_gated"), f"Unknown mode: {mode}"
        self.dt                = dt
        self.firing_rate_hz    = firing_rate_hz
        self.gain              = gain
        self.mode              = mode
        self.error_gate_thresh = error_gate_thresh
        self._rng              = np.random.default_rng(seed)

        # ロギング用カウンタ
        self._fire_count:       int        = 0
        self._step_count:       int        = 0
        self._last_fire_step:   int        = 0
        self._fire_intervals:   list[int]  = []

    # ------------------------------------------------------------------

    def should_fire(self, prediction_error: np.ndarray) -> bool:
        """
        予測誤差に基づいて登上線維が発火するか判定する。

        Parameters
        ----------
        prediction_error : CfC の予測誤差ベクトル (n_joints,) [rad]

        Returns
        -------
        bool : 発火する場合 True
        """
        self._step_count += 1
        err_norm = float(np.linalg.norm(prediction_error))

        if self.mode == "continuous":
            fired = True
        elif self.mode == "error_gated":
            fired = err_norm > self.error_gate_thresh
        else:  # "sparse"
            p_fire = _sigmoid(err_norm * self.gain) * self.firing_rate_hz * self.dt
            fired = bool(self._rng.random() < p_fire)

        if fired:
            if self._fire_count > 0:
                self._fire_intervals.append(self._step_count - self._last_fire_step)
            self._fire_count    += 1
            self._last_fire_step = self._step_count

        return fired

    def reset(self) -> None:
        """エピソード開始時にカウンタをリセットする。"""
        self._fire_count      = 0
        self._step_count      = 0
        self._last_fire_step  = 0
        self._fire_intervals.clear()

    def get_stats(self) -> dict:
        """発火統計を返す（metrics.json に記録する用）。"""
        elapsed_s = self._step_count * self.dt
        return {
            "io_fire_count":          self._fire_count,
            "io_step_count":          self._step_count,
            "io_fire_rate_hz":        self._fire_count / max(elapsed_s, 1e-9),
            "io_fire_interval_mean":  (
                float(np.mean(self._fire_intervals))
                if self._fire_intervals else 0.0
            ),
        }
