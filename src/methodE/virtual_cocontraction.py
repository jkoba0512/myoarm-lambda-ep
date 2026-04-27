"""
仮想コ・コントラクション（Virtual Co-contraction）

生物学的背景:
  ヒトの随意運動では、主動筋と拮抗筋が同時収縮（co-contraction）することで
  関節剛性を独立制御できる。収縮レベルが高いほど関節は硬くなり外乱に強くなるが、
  エネルギー消費が増える。

物理的 QDD では 1 軸 1 アクチュエータだが、制御則の内部表現を 2 チャネルに分け、
仮想的なコ・コントラクションをエミュレートする。

内部表現:
  τ_ag  = max(0,  τ_net) + τ_cc   (主動筋チャネル)
  τ_ant = max(0, -τ_net) + τ_cc   (拮抗筋チャネル)
  τ_cmd = τ_ag - τ_ant = τ_net    (QDD への指令は変わらない)
  K_virtual ∝ τ_cc                (関節剛性の指標)

追加インピーダンス:
  τ_stiffness = K_virt(τ_cc) × (q_target - q)   追加バネ
  τ_damping   = B_virt(τ_cc) × (-dq)             追加ダンパ
  τ_virtual   = τ_stiffness + τ_damping

  τ_total = τ_net + τ_virtual  ← これが実際の QDD への指令

co-contraction レベルの決まり方:
  - 近目標（err 小）: τ_cc_hold  (安定保持のための高剛性)
  - 外乱検出（|dq| 大）: τ_cc_dist (外乱応答の一時的高剛性)
"""

from __future__ import annotations
import numpy as np


class VirtualCocontraction:
    """
    仮想コ・コントラクションによる可変インピーダンス制御。

    Parameters
    ----------
    n_joints        : 関節数
    k_virtual_gain  : co-contraction → 追加剛性変換ゲイン [Nm/rad per Nm of τ_cc]
    b_virtual_gain  : co-contraction → 追加減衰変換ゲイン [(Nm·s/rad) per Nm of τ_cc]
    tau_cc_hold_max : 近目標での最大 co-contraction [Nm]
    tau_cc_dist_max : 外乱検出時の最大 co-contraction [Nm]
    err_sigma       : 近目標 co-contraction の位置誤差スケール [rad]
                      err < err_sigma でフル有効化
    vel_thresh      : 外乱検出速度閾値 [rad/s]
    dist_decay      : 外乱応答 co-contraction の減衰率（1 step あたり）
    """

    def __init__(
        self,
        n_joints:        int   = 7,
        k_virtual_gain:  float = 0.3,
        b_virtual_gain:  float = 0.1,
        tau_cc_hold_max: float = 5.0,
        tau_cc_dist_max: float = 10.0,
        err_sigma:       float = 0.15,
        vel_thresh:      float = 0.5,
        dist_decay:      float = 0.90,
    ):
        self.n_joints        = n_joints
        self.k_virtual_gain  = k_virtual_gain
        self.b_virtual_gain  = b_virtual_gain
        self.tau_cc_hold_max = tau_cc_hold_max
        self.tau_cc_dist_max = tau_cc_dist_max
        self.err_sigma       = err_sigma
        self.vel_thresh      = vel_thresh
        self.dist_decay      = dist_decay

        self._cc_dist = np.zeros(n_joints)  # 外乱応答成分（減衰付き）

    def reset(self) -> None:
        self._cc_dist[:] = 0.0

    def step(
        self,
        q:        np.ndarray,
        dq:       np.ndarray,
        q_target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        co-contraction レベルと追加インピーダンストルクを計算する。

        Parameters
        ----------
        q        : 現在関節角 (n,) [rad]
        dq       : 現在関節速度 (n,) [rad/s]
        q_target : 目標関節角 (n,) [rad]

        Returns
        -------
        tau_virtual : 追加インピーダンストルク (n,) [Nm]
        tau_cc      : co-contraction レベル (n,) [Nm]（診断用）
        """
        err = q_target - q

        # 近目標 co-contraction: 誤差が小さいほど高い
        cc_hold = self.tau_cc_hold_max * np.exp(
            -(err ** 2) / (2.0 * self.err_sigma ** 2)
        )

        # 外乱検出 co-contraction: 速度スパイクで増加、その後減衰
        # cc_dist は tau_cc_dist_max を超えないように上限を設ける（蓄積防止）
        disturbance_mask = np.abs(dq) > self.vel_thresh
        self._cc_dist = np.minimum(
            self._cc_dist * self.dist_decay
            + self.tau_cc_dist_max * disturbance_mask.astype(float),
            self.tau_cc_dist_max,
        )

        tau_cc = cc_hold + self._cc_dist  # 総 co-contraction レベル

        # 追加インピーダンス: stiffness + damping
        K_virt = self.k_virtual_gain * tau_cc
        B_virt = self.b_virtual_gain * tau_cc

        tau_virtual = K_virt * err + B_virt * (-dq)
        return tau_virtual, tau_cc

    # ------------------------------------------------------------------
    # 2 チャネル内部表現（診断用）
    # ------------------------------------------------------------------

    @staticmethod
    def decompose(tau_net: np.ndarray, tau_cc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        τ_net と τ_cc から主動筋/拮抗筋チャネルを返す（診断・可視化用）。

        Returns
        -------
        tau_ag  : 主動筋トルク (n,)
        tau_ant : 拮抗筋トルク (n,)
        """
        tau_ag  = np.maximum(0.0, tau_net)  + tau_cc
        tau_ant = np.maximum(0.0, -tau_net) + tau_cc
        return tau_ag, tau_ant
