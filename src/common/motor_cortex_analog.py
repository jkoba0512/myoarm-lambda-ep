"""
運動皮質アナログ（Motor Cortex Analog, Phase E4）

役割:
  - タスク状態（保持中・外乱後・振動中）を速度/誤差シグナルから推定
  - CPG 振幅・コ・コントラクションレベルを動的に変調
  - エファレンスコピー（遠心性コピー）を生成して下位層へ渡す

task_mode:
  "hold"      : CPG 振幅=0、コ・コントラクション=baseline
  "oscillate" : CPG 振幅=nominal、コ・コントラクション=低
  "disturbed" : 自動検出（速度閾値超過時）、コ・コントラクション=高（回復中に減衰）
"""

from __future__ import annotations

import numpy as np


class MotorCortexAnalog:
    """
    運動皮質アナログ（Phase E4）。

    Parameters
    ----------
    n_joints              : 関節数
    dt                    : タイムステップ [s]
    cc_hold               : 保持中のベースライン co-contraction [Nm]
    cc_disturbed          : 外乱後の一時的 co-contraction ブースト [Nm]
    cc_oscillate          : 振動中の低剛性 co-contraction [Nm]
    cc_decay              : 外乱状態の soft 減衰係数（1 step あたり）
    vel_disturb_thresh    : 外乱検出の速度閾値 [rad/s]
    cpg_amplitude_oscillate : 振動モード時の CPG 振幅 [rad]
    disturb_countdown_steps : 外乱モード持続 step 数（デフォルト 100 = 0.2 s @500Hz）
    """

    def __init__(
        self,
        n_joints:                 int   = 7,
        dt:                       float = 0.002,
        cc_hold:                  float = 2.0,
        cc_disturbed:             float = 8.0,
        cc_oscillate:             float = 0.5,
        cc_decay:                 float = 0.95,
        vel_disturb_thresh:       float = 0.3,
        cpg_amplitude_oscillate:  float = 0.3,
        disturb_countdown_steps:  int   = 100,
    ):
        self.n_joints                = n_joints
        self.dt                      = dt
        self.cc_hold                 = cc_hold
        self.cc_disturbed            = cc_disturbed
        self.cc_oscillate            = cc_oscillate
        self.cc_decay                = cc_decay
        self.vel_disturb_thresh      = vel_disturb_thresh
        self.cpg_amplitude_oscillate = cpg_amplitude_oscillate
        self.disturb_countdown_steps = disturb_countdown_steps

        # 外部からセットされるタスクモード
        self._task_mode_external: str = "hold"   # "hold" or "oscillate"
        # 実際に使われる内部状態
        self._current_mode: str = "hold"

        # 外乱カウンタ（正の間は "disturbed" モードをオーバーライド）
        self._disturb_countdown: int = 0
        # 外乱時 cc の現在レベル（ソフト減衰付き）
        self._cc_disturb_current: np.ndarray = np.zeros(n_joints)

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """エピソード開始時にリセット。"""
        self._disturb_countdown = 0
        self._cc_disturb_current[:] = 0.0
        self._current_mode = self._task_mode_external

    def set_task_mode(self, mode: str) -> None:
        """
        上位から保持・振動モードを設定する。

        Parameters
        ----------
        mode : "hold" または "oscillate"
        """
        assert mode in ("hold", "oscillate"), f"Unknown task_mode: {mode}"
        self._task_mode_external = mode

    def step(
        self,
        q:        np.ndarray,
        dq:       np.ndarray,
        q_target: np.ndarray,
    ) -> dict:
        """
        1 ステップ実行し、下位層への指令を返す。

        Parameters
        ----------
        q        : 現在関節角 (n,) [rad]
        dq       : 現在関節速度 (n,) [rad/s]
        q_target : 目標関節角 (n,) [rad]

        Returns
        -------
        dict with keys:
          "cc_target"       : co-contraction 指令 (n,) [Nm]
          "cpg_amplitude"   : CPG 振幅 [rad]
          "efference_copy"  : エファレンスコピー = cc_target（下位層への遠心性コピー）(n,)
          "task_mode"       : 現在のタスクモード文字列
        """
        # ── 外乱検出 ──────────────────────────────────────────────────
        if np.any(np.abs(dq) > self.vel_disturb_thresh):
            self._disturb_countdown = self.disturb_countdown_steps
            # 外乱 cc を最大値にセット（上限クリップ）
            self._cc_disturb_current[:] = self.cc_disturbed
        else:
            # ソフト減衰
            self._cc_disturb_current = np.minimum(
                self._cc_disturb_current * self.cc_decay,
                np.full(self.n_joints, self.cc_disturbed),
            )
            if self._disturb_countdown > 0:
                self._disturb_countdown -= 1

        # ── タスクモード決定 ──────────────────────────────────────────
        if self._disturb_countdown > 0:
            self._current_mode = "disturbed"
        else:
            self._current_mode = self._task_mode_external

        # ── cc_target と cpg_amplitude の計算 ────────────────────────
        if self._current_mode == "disturbed":
            # 外乱後: 高剛性（_cc_disturb_current はソフト減衰中）
            cc_target = self._cc_disturb_current.copy()
            cpg_amplitude = 0.0
        elif self._current_mode == "oscillate":
            cc_target = np.full(self.n_joints, self.cc_oscillate)
            cpg_amplitude = self.cpg_amplitude_oscillate
        else:  # "hold"
            cc_target = np.full(self.n_joints, self.cc_hold)
            cpg_amplitude = 0.0

        # エファレンスコピー = cc_target（下位への遠心性指令）
        efference_copy = cc_target.copy()

        return {
            "cc_target":      cc_target,
            "cpg_amplitude":  cpg_amplitude,
            "efference_copy": efference_copy,
            "task_mode":      self._current_mode,
        }
