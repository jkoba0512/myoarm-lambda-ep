"""
CerebellarSideLoop — 小脳サイドループ（前向きモデルのみ）。

解剖学的忠実実装における小脳の役割:
  - 前向きモデル（Forward model）のみを担う。逆モデルは担わない。
  - 入力: エファレンスコピー + DelayBuffer 経由の固有受容情報
  - 出力: 予測補正 → DelayBuffer（小脳ループ遅延 30 ms）経由で M1 へ

信号フロー（1 ステップ）:
  predict(q_del, dq_del, efcopy) [env.step() の前に呼ぶ]
    → CfCForwardModel.predict() で q̂(t+1) を計算
  update_and_get_correction(q_actual) [env.step() の後に呼ぶ]
    → CfCForwardModel.update(q_actual) で予測誤差・補正トルクを計算
    → InferiorOliveAnalog.should_fire(pred_error) で LTD 判定
    → 補正トルクを cereb_delay_buf に push して遅延済み値を返す

注意:
  - cereb_delay_buf は AnatomicalController が管理し、外から渡す。
  - 1 ステップに predict() → [env.step()] → update_and_get_correction()
    の順で呼ぶこと。
"""

from __future__ import annotations

import numpy as np

from methodB.cfc_forward_model import CfCForwardModel
from methodF.delay_buffer import DelayBuffer
from methodF.inferior_olive_analog import InferiorOliveAnalog


class CerebellarSideLoop:
    """
    小脳サイドループ（前向きモデル専用）。

    Parameters
    ----------
    cfc          : 訓練済み CfCForwardModel
    io_analog    : InferiorOliveAnalog（登上線維散発発火）
    cereb_delay_buf : 補正出力用 DelayBuffer（小脳ループ遅延、15 steps / 30 ms）
    """

    def __init__(
        self,
        cfc:            CfCForwardModel,
        io_analog:      InferiorOliveAnalog,
        cereb_delay_buf: DelayBuffer,
    ) -> None:
        self.cfc             = cfc
        self.io              = io_analog
        self.cereb_delay_buf = cereb_delay_buf

        n = cfc.n_joints
        self._pred_error:  np.ndarray = np.zeros(n)
        self._tau_raw:     np.ndarray = np.zeros(n)
        self._q_hat:       np.ndarray = np.zeros(n)
        self._last_fired:  bool       = False

    # ------------------------------------------------------------------

    def predict(
        self,
        q_del:   np.ndarray,
        dq_del:  np.ndarray,
        efcopy:  np.ndarray,
    ) -> np.ndarray:
        """
        env.step() の前に呼ぶ。
        遅延済み固有受容情報とエファレンスコピーから次状態を予測する。

        Returns
        -------
        q_hat : 予測次関節角 (n_joints,) [rad]（ロギング用）
        """
        self._q_hat = self.cfc.predict(q_del, dq_del, efcopy)
        return self._q_hat.copy()

    def update_and_get_correction(self, q_actual: np.ndarray) -> np.ndarray:
        """
        env.step() の後に呼ぶ。
        実際の次状態で予測誤差を更新し、遅延済み補正トルクを返す。

        Parameters
        ----------
        q_actual : env.step() 後の実際の関節角 (n_joints,) [rad]

        Returns
        -------
        tau_cereb_delayed : cereb_delay_buf を通過した補正トルク (n_joints,) [Nm]
                            AnatomicalController が次ステップの M1 に渡す。
        """
        # IO 発火判定（前ステップの予測誤差で判定）
        # continuous モード: 常に発火 → 毎ステップ学習
        # sparse/error_gated モード: 確率的/閾値発火 → 発火時のみ学習（H4 の肝）
        self._last_fired = self.io.should_fire(self._pred_error)

        # 予測誤差と補正トルクを更新（IO 発火がゲートするのはオンライン学習のみ）
        self.cfc.update(q_actual, allow_online_update=self._last_fired)
        self._pred_error = self.cfc.get_prediction_error().copy()
        self._tau_raw    = self.cfc.get_correction().copy()

        # 補正トルクを小脳ループ遅延バッファに通す
        return self.cereb_delay_buf.push_and_get(self._tau_raw)

    # ------------------------------------------------------------------

    def get_pred_error(self) -> np.ndarray:
        return self._pred_error.copy()

    def get_tau_raw(self) -> np.ndarray:
        return self._tau_raw.copy()

    def last_io_fired(self) -> bool:
        return self._last_fired

    def reset(self) -> None:
        """エピソード開始時にリセットする。"""
        self.cfc.reset()
        self.io.reset()
        self.cereb_delay_buf.reset()
        self._pred_error[:] = 0.0
        self._tau_raw[:]    = 0.0
        self._last_fired    = False
