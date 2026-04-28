"""
MotorCortexM1 — 一次運動野（M1）の逆モデルとエファレンスコピー生成。

解剖学的忠実実装における M1 の役割:
  1. 逆モデル（PD 制御）: q_target から τ_cortical（皮質脊髄路・直接路）を生成
  2. エファレンスコピー生成: τ_pd のコピーを τ_efcopy として小脳へ送る
  3. 小脳サイドループからの delayed 補正を受け取り τ_cortical に加算する

F3 実験ではエファレンスコピーに以下の操作が可能:
  - efcopy_enabled=False : エファレンスコピーを遮断（求心路遮断に相当）
  - efcopy_noise_std > 0 : エファレンスコピーにガウスノイズを付加
  - efcopy_delay_steps > 0: 別途 DelayBuffer を追加（M1→小脳経路の追加遅延）

F5 実験では inverse_model_loc を切り替えて逆モデルの所在を変更する:
  - "m1"         : M1 のみが逆モデルを担う（解剖学的忠実・デフォルト）
  - "cerebellum" : 逆モデルの出力を小脳からの補正に委ねる（Kawato 極端解釈）
  - "both"       : M1 と小脳の両方が逆モデルを担う（MOSAIC 的実装）
"""

from __future__ import annotations

import numpy as np

_KP_DEFAULT = np.array([50.0, 50.0, 50.0, 50.0, 10.0, 10.0, 10.0])
_KD_DEFAULT = np.array([ 7.0,  7.0,  7.0,  7.0,  1.5,  1.5,  1.5])


class MotorCortexM1:
    """
    一次運動野（M1）逆モデル。

    Parameters
    ----------
    n_joints           : 関節数
    kp                 : PD 比例ゲイン (n_joints,) [Nm/rad]
    kd                 : PD 微分ゲイン (n_joints,) [Nm·s/rad]
    efcopy_enabled     : False でエファレンスコピーをゼロにする（F3）
    efcopy_noise_std   : エファレンスコピーへのガウスノイズ標準偏差（F3）
    inverse_model_loc  : "m1" | "cerebellum" | "both"（F5）
    seed               : ノイズ乱数シード
    """

    def __init__(
        self,
        n_joints:          int              = 7,
        kp:                np.ndarray | None = None,
        kd:                np.ndarray | None = None,
        efcopy_enabled:    bool             = True,
        efcopy_noise_std:  float            = 0.0,
        inverse_model_loc: str              = "m1",
        seed:              int              = 0,
    ) -> None:
        assert inverse_model_loc in ("m1", "cerebellum", "both"), \
            f"Unknown inverse_model_loc: {inverse_model_loc}"

        self.n_joints          = n_joints
        self.kp                = (_KP_DEFAULT.copy() if kp is None
                                  else np.asarray(kp, dtype=float).copy())
        self.kd                = (_KD_DEFAULT.copy() if kd is None
                                  else np.asarray(kd, dtype=float).copy())
        self.efcopy_enabled    = efcopy_enabled
        self.efcopy_noise_std  = efcopy_noise_std
        self.inverse_model_loc = inverse_model_loc
        self._rng              = np.random.default_rng(seed)

        # 小脳サイドループからの delayed 補正（毎ステップ update される）
        self._cereb_feedback: np.ndarray = np.zeros(n_joints)

    # ------------------------------------------------------------------

    def receive_cerebellar_feedback(self, tau_cereb_delayed: np.ndarray) -> None:
        """
        小脳サイドループからの DelayBuffer 済み補正トルクを受け取る。
        AnatomicalController が毎ステップ呼ぶ。
        """
        self._cereb_feedback = np.asarray(tau_cereb_delayed, dtype=float).copy()

    def step(
        self,
        q_target: np.ndarray,
        q:        np.ndarray,
        dq:       np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        1 ステップ実行する。

        Parameters
        ----------
        q_target : 目標関節角 (n_joints,) [rad]
        q        : 現在関節角（生の値、遅延なし）(n_joints,) [rad]
        dq       : 現在関節角速度（生の値、遅延なし）(n_joints,) [rad/s]

        Returns
        -------
        tau_cortical : 脊髄への運動指令 (n_joints,) [Nm]
                       PD + cerebellar feedback（inverse_model_loc に依存）
        tau_efcopy   : 小脳へのエファレンスコピー (n_joints,) [Nm]
        """
        tau_pd = self.kp * (q_target - q) + self.kd * (-dq)

        # エファレンスコピー生成
        if not self.efcopy_enabled:
            tau_efcopy = np.zeros(self.n_joints)
        else:
            tau_efcopy = tau_pd.copy()
            if self.efcopy_noise_std > 0.0:
                tau_efcopy += self._rng.normal(0.0, self.efcopy_noise_std,
                                               self.n_joints)

        # 逆モデルの所在による皮質脊髄路トルクの計算
        if self.inverse_model_loc == "m1":
            # M1 が逆モデルを担う。小脳は補正のみ（解剖学的忠実）。
            tau_cortical = tau_pd + self._cereb_feedback
        elif self.inverse_model_loc == "cerebellum":
            # 逆モデルを小脳に委ねる。M1 の PD は補助のみ（Kawato 極端解釈）。
            # PD ゲインを 0.1 倍に下げて小脳補正を主力にする。
            tau_cortical = tau_pd * 0.1 + self._cereb_feedback
        else:  # "both"
            # M1 と小脳の両方が逆モデルを担う（MOSAIC 的実装）。
            tau_cortical = tau_pd + self._cereb_feedback

        return tau_cortical, tau_efcopy

    def reset(self) -> None:
        """エピソード開始時にリセットする。"""
        self._cereb_feedback[:] = 0.0
