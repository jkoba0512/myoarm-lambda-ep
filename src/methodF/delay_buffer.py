"""
DelayBuffer — 神経伝達遅延を模擬するリングバッファ。

delay_steps=0 のときパススルー（遅延なし）として動作する。
F1 / F2 実験で delay_steps を外部から変更することで遅延スイープが可能。

定数:
  PROPRIOCEPTIVE_DELAY_STEPS = 10  # 20 ms @ 500 Hz（固有受容遅延）
  CEREBELLAR_LOOP_DELAY_STEPS = 15 # 30 ms @ 500 Hz（小脳→視床→M1 ループ）
"""

from __future__ import annotations
from collections import deque

import numpy as np

PROPRIOCEPTIVE_DELAY_STEPS  = 10   # 20 ms @ 500 Hz
CEREBELLAR_LOOP_DELAY_STEPS = 15   # 30 ms @ 500 Hz


class DelayBuffer:
    """
    固定長リングバッファによる信号遅延。

    Parameters
    ----------
    delay_steps : 遅延ステップ数。0 でパススルー。
    shape       : バッファに格納する numpy 配列の shape（例: (7,)）。
    """

    def __init__(self, delay_steps: int, shape: tuple[int, ...] = (7,)) -> None:
        self.delay_steps = delay_steps
        self.shape = shape
        # delay_steps=0 でもバッファ長 1 を確保（deque の制約上）
        n = max(delay_steps, 1)
        self._buf: deque[np.ndarray] = deque(
            [np.zeros(shape) for _ in range(n)], maxlen=n
        )

    def push_and_get(self, value: np.ndarray) -> np.ndarray:
        """
        value をキューに追加し、delay_steps ステップ前の値を返す。
        delay_steps=0 のとき value をそのまま返す。
        """
        v = np.asarray(value, dtype=float).reshape(self.shape)
        if self.delay_steps == 0:
            return v.copy()
        self._buf.append(v.copy())
        return self._buf[0].copy()

    def reset(self) -> None:
        """エピソード開始時にバッファをゼロ初期化する。"""
        for arr in self._buf:
            arr[:] = 0.0
