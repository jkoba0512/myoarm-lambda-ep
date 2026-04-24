"""
Franka Panda MuJoCo 環境ラッパー（トルク制御）

特徴:
  - 7 DoF、motor アクチュエータ（直接トルク指令 [Nm]）
  - Joint 1-4: ±87 Nm、Joint 5-7: ±12 Nm
  - timestep: 0.002 s (500 Hz)
  - 逆動力学（inverse_dynamics）でトルク教師データを生成可能

position actuator ベースの環境との違い:
  - step(tau) でトルクを指令（位置指令ではない）
  - 重力・コリオリ補償が必要（CfC 小脳の役割）
  - 外乱に対してバックドライバブル（QDD アームを模倣）
"""

from __future__ import annotations

import numpy as np
import mujoco
from pathlib import Path

MODEL_PATH = str(
    Path(__file__).parents[2] / "sim/models/franka_emika_panda/panda_torque.xml"
)

N_JOINTS = 7  # アーム関節数
# Franka Panda トルク上限 [Nm]
TAU_LIMIT = np.array([87, 87, 87, 87, 12, 12, 12], dtype=np.float64)


class FrankaEnv:
    """
    Franka Panda トルク制御シミュレーション環境。

    step(tau) でトルクを直接指令する。
    重力・コリオリ力は MuJoCo が内部で計算するが、
    補償しない限り関節は重力で落下する点に注意。
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data  = mujoco.MjData(self.model)
        # 関節可動域 (7, 2)
        self.jnt_range = np.array(
            [self.model.jnt_range[i] for i in range(N_JOINTS)]
        )

    # ------------------------------------------------------------------
    # 基本 API
    # ------------------------------------------------------------------

    def reset(
        self,
        q0: np.ndarray | None = None,
        dq0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        環境リセット。

        Parameters
        ----------
        q0  : 初期関節角 (7,) [rad]。None なら中立姿勢。
        dq0 : 初期関節速度 (7,) [rad/s]。None なら 0。
        """
        mujoco.mj_resetData(self.model, self.data)
        if q0 is not None:
            self.data.qpos[:N_JOINTS] = q0
        if dq0 is not None:
            self.data.qvel[:N_JOINTS] = dq0
        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def step(self, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        トルク指令を 1 ステップ適用して進める。

        Parameters
        ----------
        tau : 関節トルク (7,) [Nm]

        Returns
        -------
        q, dq : 次ステップの関節角・速度
        """
        tau_clipped = np.clip(tau, -TAU_LIMIT, TAU_LIMIT)
        self.data.ctrl[:N_JOINTS] = tau_clipped
        mujoco.mj_step(self.model, self.data)
        return self.get_state()

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """現在の関節角・速度 (q, dq) を返す。"""
        q  = self.data.qpos[:N_JOINTS].copy()
        dq = self.data.qvel[:N_JOINTS].copy()
        return q, dq

    def get_ee_pos(self) -> np.ndarray:
        """エンドエフェクタ位置（link7 先端）を返す [m]。"""
        # link7 の site "attachment_site" を取得
        # 存在しない場合は link7 の body pos で代替
        try:
            site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
            )
            return self.data.site_xpos[site_id].copy()
        except Exception:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "link7"
            )
            return self.data.xpos[body_id].copy()

    def inverse_dynamics(
        self,
        q:   np.ndarray,
        dq:  np.ndarray,
        ddq: np.ndarray,
    ) -> np.ndarray:
        """
        逆動力学計算: M(q)·ddq + C(q,dq) + G(q) → τ

        訓練データ収集（CfC 小脳の教師信号）に使用。
        """
        data_tmp = mujoco.MjData(self.model)
        data_tmp.qpos[:N_JOINTS] = q
        data_tmp.qvel[:N_JOINTS] = dq
        data_tmp.qacc[:N_JOINTS] = ddq
        mujoco.mj_inverse(self.model, data_tmp)
        return data_tmp.qfrc_inverse[:N_JOINTS].copy()

    def apply_disturbance(
        self,
        tau_dist: np.ndarray,
        duration_steps: int,
    ) -> None:
        """
        外乱トルクを duration_steps ステップ印加する。

        Parameters
        ----------
        tau_dist       : 外乱トルク (7,) [Nm]
        duration_steps : 印加ステップ数
        """
        for _ in range(duration_steps):
            self.data.ctrl[:N_JOINTS] = np.clip(
                self.data.ctrl[:N_JOINTS] + tau_dist, -TAU_LIMIT, TAU_LIMIT
            )
            mujoco.mj_step(self.model, self.data)

    @property
    def dt(self) -> float:
        return self.model.opt.timestep

    @property
    def time(self) -> float:
        return self.data.time

    @property
    def ctrl_range(self) -> np.ndarray:
        """関節可動域 (7, 2) [rad]。"""
        return self.jnt_range.copy()
