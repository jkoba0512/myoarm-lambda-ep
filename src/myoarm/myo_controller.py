"""
MyoArmController — 解剖学的忠実神経運動制御コントローラ（MyoSuite 版）。

Franka AnatomicalController との対応関係:
  制御出力  : τ [Nm] (7 関節) → 筋活性化 a ∈ [0,1]^34
  固有受容  : q/dq (7 関節) → q/dq (20 関節) + 筋長/筋速度 (34 筋)
  M1 逆モデル: 関節トルク出力 → エンドポイント PD → 筋活性化 (疑似逆行列)
  小脳補正  : Δτ (関節空間) → Δa (筋空間) via J_act^+
  反射弧    : 関節速度/トルク → 筋長速度/筋力 (新 MyoIaIbReflexArc)
  相反抑制  : 新規追加 (ReciprocalInhibition)

信号フロー（1 ステップ）:
  1. q_del, dq_del = prop_delay(q, dq)
  2. a_efcopy (エファレンスコピー, 前ステップの a_base)
  3. cerebellum.predict(q_del, dq_del, a_efcopy)  [env.step() 前]
  4. a_base = endpoint_PD(reach_err) → J_act^+ → [0,1]^34
  5. Δa_cereb_delayed を a_base に加算
  6. Δa_ia, Δa_ib = reflex(muscle_vel, muscle_force)
  7. Δa_ri = reciprocal_inhibition(a_base)
  8. a_total = clip(a_base + Δa_cereb_delayed + Δa_ia + Δa_ib + Δa_ri, 0, 1)
  ----- env.step(a_total) -----
  9. cerebellum.update_and_get_correction(q_actual)
     → Δq_err → J_act^+ → Δa_cereb_raw → cereb_delay_buf

呼び出し順（実験スクリプト）:
  a, info = ctrl.step(q, dq, reach_err, muscle_vel, muscle_force, m, d, site_id)
  env.step(a)
  ctrl.update_cerebellum(q_actual, m, d)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from methodB.cfc_forward_model import CfCForwardModel
from methodF.delay_buffer import DelayBuffer, PROPRIOCEPTIVE_DELAY_STEPS, CEREBELLAR_LOOP_DELAY_STEPS
from methodF.inferior_olive_analog import InferiorOliveAnalog
from myoarm.reciprocal_inhibition import ReciprocalInhibition
from myoarm.myo_ia_ib_reflex import MyoIaIbReflexArc

# ── 定数 ─────────────────────────────────────────────────────────────
_N_JOINTS  = 20
_N_MUSCLES = 34
_SITE_NAME = "IFtip"
_JACOBIAN_RECOMPUTE_INTERVAL = 50  # steps


# ──────────────────────────────────────────────────────────────────────
# ヤコビアン計算ユーティリティ (Phase 1-0 の baseline から移植)
# ──────────────────────────────────────────────────────────────────────

def _compute_actuator_jacobian(
    m: mujoco.MjModel, d: mujoco.MjData, delta: float = 0.05
) -> np.ndarray:
    """筋活性化 → 関節トルク ヤコビアン J_act (nv × nu) を数値微分で計算。"""
    mujoco.mj_forward(m, d)
    qfrc0 = d.qfrc_actuator.copy()
    act0  = d.act.copy()
    J = np.zeros((m.nv, m.nu))
    for i in range(m.nu):
        d.act[:] = act0
        d.act[i] += delta
        mujoco.mj_forward(m, d)
        J[:, i] = (d.qfrc_actuator - qfrc0) / delta
    d.act[:] = act0
    mujoco.mj_forward(m, d)
    return J


def _endpoint_jacobian(
    m: mujoco.MjModel, d: mujoco.MjData, site_id: int
) -> np.ndarray:
    """IFtip の位置ヤコビアン (3 × nv)。"""
    Jp = np.zeros((3, m.nv))
    Jr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, Jp, Jr, site_id)
    return Jp


# ──────────────────────────────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MyoArmConfig:
    """MyoArmController の全パラメータ。"""

    # 遅延
    prop_delay_steps:  int   = PROPRIOCEPTIVE_DELAY_STEPS   # 10 steps = 20 ms
    cereb_delay_steps: int   = CEREBELLAR_LOOP_DELAY_STEPS  # 15 steps = 30 ms

    # エンドポイント PD (M1)
    Kp_ee:   float = 80.0
    Kd_ee:   float = 15.0
    Ki_ee:   float = 2.0
    act_bias: float = 0.15

    # 小脳補正ゲイン
    K_cereb: float = 0.3   # Δq_err → Δa_cereb のスケール

    # 下オリーブ核
    io_mode:           str   = "sparse"
    io_firing_rate_hz: float = 1.0
    io_gain:           float = 5.0

    # Ia/Ib 反射
    K_ia:          float = 0.05
    K_ib:          float = 0.03
    ib_threshold:  float = 200.0

    # 相反抑制
    K_ri:         float = 0.5
    ri_threshold: float = 0.3

    # CfC
    cfc_hidden_units: int   = 64
    cfc_device:       str   = "cpu"

    # タイムステップ
    dt: float = 0.002


# ──────────────────────────────────────────────────────────────────────
# コントローラ本体
# ──────────────────────────────────────────────────────────────────────

class MyoArmController:
    """
    解剖学的忠実神経運動制御コントローラ（MyoSuite / myoArm 用）。

    Parameters
    ----------
    config       : MyoArmConfig
    muscle_names : list[str]  — アクチュエータ名リスト (m.nu 個)
    seed         : 乱数シード
    """

    def __init__(
        self,
        config:       MyoArmConfig | None = None,
        muscle_names: list[str] | None    = None,
        seed:         int                 = 0,
    ) -> None:
        if config is None:
            config = MyoArmConfig()
        self.config = config

        if muscle_names is None:
            muscle_names = [f"muscle_{i}" for i in range(_N_MUSCLES)]
        self.muscle_names = muscle_names

        n_j = _N_JOINTS
        n_m = _N_MUSCLES
        dt  = config.dt

        # ── 遅延バッファ ─────────────────────────────────────────────
        self._prop_q   = DelayBuffer(config.prop_delay_steps,  shape=(n_j,))
        self._prop_dq  = DelayBuffer(config.prop_delay_steps,  shape=(n_j,))
        # 小脳ループ遅延は筋活性化補正空間 (n_m,) で適用
        self._cereb_buf = DelayBuffer(config.cereb_delay_steps, shape=(n_m,))

        # ── 下オリーブ核アナログ ─────────────────────────────────────
        self._io = InferiorOliveAnalog(
            dt=dt,
            firing_rate_hz=config.io_firing_rate_hz,
            gain=config.io_gain,
            mode=config.io_mode,
            seed=seed,
        )

        # ── CfC 前向きモデル（小脳） ─────────────────────────────────
        # efcopy として a_base の最初の n_j 要素を使用（Franka との次元互換性）
        # K_cereb は n_j=20 次元で初期化（Franka デフォルトの 7 次元ではない）
        K_cereb_myo = np.full(n_j, config.K_cereb)
        self._cfc = CfCForwardModel(
            n_joints=n_j,
            hidden_units=config.cfc_hidden_units,
            device=config.cfc_device,
            K_cereb=K_cereb_myo,
        )
        self._pred_error = np.zeros(n_j)

        # ── 脊髄反射弧 ───────────────────────────────────────────────
        self._reflex = MyoIaIbReflexArc(
            n_muscles=n_m,
            dt=dt,
            K_ia=config.K_ia,
            K_ib=config.K_ib,
            ib_threshold=config.ib_threshold,
        )

        # ── 相反抑制 ─────────────────────────────────────────────────
        self._ri = ReciprocalInhibition(
            muscle_names=muscle_names,
            K_ri=config.K_ri,
            threshold=config.ri_threshold,
        )

        # ── ヤコビアンキャッシュ ─────────────────────────────────────
        self._J_act_pinv: np.ndarray | None = None
        self._J_act_step_count = 0
        self._site_id: int | None = None

        # ── 内部ステート ─────────────────────────────────────────────
        self._a_cereb_delayed = np.zeros(n_m)
        self._a_efcopy        = np.zeros(n_j)   # 前ステップの efcopy (n_j 次元)
        self._integral_ee     = np.zeros(3)      # エンドポイント積分項
        self._prev_tip        = np.zeros(3)

    # ------------------------------------------------------------------
    # 初期化・リセット
    # ------------------------------------------------------------------

    def initialize(self, m: mujoco.MjModel, d: mujoco.MjData) -> None:
        """env.reset() 直後に 1 度呼ぶ。site_id を取得し Jacobian を計算する。"""
        self._site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, _SITE_NAME)
        self._refresh_jacobian(m, d, force=True)

    def reset(self) -> None:
        self._prop_q.reset()
        self._prop_dq.reset()
        self._cereb_buf.reset()
        self._cfc.reset()
        self._io.reset()
        self._reflex.reset()
        self._ri.reset()
        self._J_act_pinv      = None
        self._J_act_step_count = 0
        self._a_cereb_delayed[:] = 0.0
        self._a_efcopy[:] = 0.0
        self._integral_ee[:] = 0.0
        self._prev_tip[:] = 0.0

    def load_cfc(self, path: str | Path) -> None:
        self._cfc.load(str(path))
        # load() は保存時の K_cereb を復元するが MyoArm は 20 次元が必要
        self._cfc.K_cereb = np.full(_N_JOINTS, self.config.K_cereb)

    # ------------------------------------------------------------------
    # ヤコビアン管理
    # ------------------------------------------------------------------

    def _refresh_jacobian(
        self,
        m: mujoco.MjModel,
        d: mujoco.MjData,
        force: bool = False,
    ) -> None:
        if force or (
            self._J_act_pinv is None
            or self._J_act_step_count % _JACOBIAN_RECOMPUTE_INTERVAL == 0
        ):
            J = _compute_actuator_jacobian(m, d)
            self._J_act_pinv = np.linalg.pinv(J)
        self._J_act_step_count += 1

    # ------------------------------------------------------------------
    # メインステップ (env.step() の前に呼ぶ)
    # ------------------------------------------------------------------

    def step(
        self,
        q:            np.ndarray,   # 関節角 (20,) [rad]
        dq:           np.ndarray,   # 関節速度 (20,) [rad/s]
        reach_err:    np.ndarray,   # 手先誤差 (3,) [m]
        tip_pos:      np.ndarray,   # 手先位置 (3,) [m]
        muscle_vel:   np.ndarray,   # 筋長変化率 (34,) [m/s]
        muscle_force: np.ndarray,   # 筋力 (34,) [N]
        m:            mujoco.MjModel,
        d:            mujoco.MjData,
    ) -> tuple[np.ndarray, dict]:
        """
        1 ステップ実行して筋活性化指令を返す。

        Returns
        -------
        a_total : (34,)  筋活性化 [0, 1]
        info    : ロギング用辞書
        """
        cfg = self.config

        # ── 1. ヤコビアン更新 ─────────────────────────────────────────
        self._refresh_jacobian(m, d)

        # ── 2. 固有受容遅延 ───────────────────────────────────────────
        q_del  = self._prop_q.push_and_get(q)
        dq_del = self._prop_dq.push_and_get(dq)

        # ── 3. 小脳前向きモデル予測（env.step() 前） ──────────────────
        # efcopy として前ステップの基本指令の最初の n_j 次元を使用
        self._cfc.predict(q_del, dq_del, self._a_efcopy)

        # ── 4. M1 エンドポイント PID → 基本筋活性化 ─────────────────
        tip_vel = tip_pos - self._prev_tip
        self._prev_tip = tip_pos.copy()
        self._integral_ee = np.clip(self._integral_ee + reach_err, -2.0, 2.0)

        F_ee = (
            cfg.Kp_ee * reach_err
            - cfg.Kd_ee * tip_vel
            + cfg.Ki_ee * self._integral_ee
        )

        assert self._site_id is not None, "initialize() を先に呼ぶこと"
        J_ee  = _endpoint_jacobian(m, d, self._site_id)       # (3, nv)
        tau_pd = J_ee.T @ F_ee                                 # (nv,)
        tau_grav = d.qfrc_bias.copy()                          # (nv,)
        tau_desired = tau_pd + tau_grav

        a_base = np.clip(
            self._J_act_pinv @ tau_desired + cfg.act_bias,
            0.0, 1.0,
        )

        # efcopy 更新（次ステップの小脳予測に使用）
        self._a_efcopy = a_base[:_N_JOINTS].copy()

        # ── 5. 小脳補正を適用（前ステップ遅延済み） ──────────────────
        a_after_cereb = a_base + self._a_cereb_delayed

        # ── 6. 脊髄反射弧 ─────────────────────────────────────────────
        da_ia, da_ib = self._reflex.step(muscle_vel, muscle_force)

        # ── 7. 相反抑制 ───────────────────────────────────────────────
        da_ri = self._ri.inhibit(a_after_cereb)

        # ── 8. 合算・クリッピング ─────────────────────────────────────
        a_total = np.clip(
            a_after_cereb + da_ia + da_ib + da_ri,
            0.0, 1.0,
        ).astype(np.float32)

        info = {
            "a_base":          a_base,
            "a_cereb_delayed": self._a_cereb_delayed.copy(),
            "da_ia":           da_ia,
            "da_ib":           da_ib,
            "da_ri":           da_ri,
            "q_del":           q_del.copy(),
            "dq_del":          dq_del.copy(),
        }
        return a_total, info

    # ------------------------------------------------------------------
    # 小脳更新 (env.step() の後に呼ぶ)
    # ------------------------------------------------------------------

    def update_cerebellum(
        self,
        q_actual: np.ndarray,   # env.step() 後の関節角 (20,)
        m: mujoco.MjModel,
        d: mujoco.MjData,
    ) -> None:
        """
        env.step() 後の実際の状態で小脳を更新し、補正を遅延バッファへ送る。
        """
        # IO 発火判定で学習をゲート
        fired = self._io.should_fire(self._pred_error)
        self._cfc.update(q_actual, allow_online_update=fired)
        self._pred_error = self._cfc.get_prediction_error().copy()

        # Δq 誤差 → 関節空間補正トルク → 筋活性化補正
        delta_q_err = self._cfc.get_prediction_error()               # (n_j,)
        tau_cereb   = self.config.K_cereb * delta_q_err              # (n_j→nv,)
        tau_full    = np.zeros(_N_JOINTS)
        tau_full[:] = tau_cereb

        da_cereb_raw = self._J_act_pinv @ tau_full                    # (n_m,)
        self._a_cereb_delayed = self._cereb_buf.push_and_get(da_cereb_raw)

    # ------------------------------------------------------------------

    def get_io_stats(self) -> dict:
        return self._io.get_stats()

    def get_pred_error(self) -> np.ndarray:
        return self._pred_error.copy()
