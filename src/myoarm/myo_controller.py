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
from myoarm.trajectory_planner import MinimumJerkPlanner

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


def _compute_moment_arm(
    m: mujoco.MjModel, d: mujoco.MjData, delta: float = 1e-3
) -> np.ndarray:
    """筋長に対する関節角の感度 R = dL/dq (nu × nv) を数値微分で計算。

    λ-EP の小脳補正で `Δλ = -K × R @ delta_q_err` を計算するために使う。
    """
    saved_qpos = d.qpos.copy()
    saved_qvel = d.qvel.copy()
    mujoco.mj_forward(m, d)
    L0 = d.actuator_length.copy()
    R = np.zeros((m.nu, m.nv))
    for j in range(m.nv):
        d.qpos[:] = saved_qpos
        d.qpos[j] += delta
        mujoco.mj_forward(m, d)
        R[:, j] = (d.actuator_length - L0) / delta
    d.qpos[:] = saved_qpos
    d.qvel[:] = saved_qvel
    mujoco.mj_forward(m, d)
    return R


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

    # γ 運動ニューロン変調 (target 接近で Ia 反射ゲインを増す → 減速)
    # 文献: Prochazka 1989, Loeb 1984 (preflexes)
    # K_ia_eff = K_ia * (1 + gamma_gain * exp(-||reach_err|| / gamma_scale))
    # gamma_gain=0 で無効化 (旧挙動)
    gamma_gain:    float = 0.0   # target で何倍に増幅するか
    gamma_scale:   float = 0.10  # 距離スケール [m]

    # 下行性 damping 変調 (target 接近で endpoint Kd を増す → 慣性ブレーキ)
    # 文献: Burdet et al. 2001 Nature; Franklin & Wolpert 2011 — 下行性指令が
    #   脊髄の速度フィードバックゲインを上げ、co-contraction 様の impedance 増加
    # Kd_eff = Kd_ee * (1 + Kd_proximity_gain * exp(-||reach_err|| / Kd_proximity_scale))
    # Kd_proximity_gain=0 で無効化 (旧挙動)
    Kd_proximity_gain:  float = 0.0
    Kd_proximity_scale: float = 0.10  # 距離スケール [m]

    # 制御モード: "endpoint_pd" (旧来) or "lambda_ep" (Feldman λ-model)
    # λ-model (Feldman 1966, 1986; Bizzi & Mussa-Ivaldi):
    #   target = equilibrium。下行性指令が各筋の閾値長 λᵢ を設定し、
    #   a_i = clip(c_lambda × max(L_i - λ_i, 0), 0, 1) で活性化生成。
    #   target = IK で q_target を求め、L_target = mj_forward(q_target).actuator_length
    #   λ = L_target - lambda_offset
    # この機構では Kp_ee, Kd_ee, Jacobian 投影は使わない。
    control_mode:       str   = "endpoint_pd"
    c_lambda:           float = 5.0     # 活性化/伸長 ゲイン [1/m]
    lambda_offset:      float = 0.005   # target での baseline 伸長 [m] (~5mm)
    ik_max_iter:        int   = 30
    ik_damping:         float = 0.01    # damped LS の正則化

    # virtual trajectory hypothesis (Feldman 1998 review):
    # 下行性指令は静的な λ_target でなく λ(t) の時系列。
    # λ_start から λ_target へ最小ジャーク補間: λ(τ) = λ_s + s(τ)(λ_t - λ_s)
    # s(τ) = 10τ³ - 15τ⁴ + 6τ⁵  (Flash & Hogan 1985)
    # 時間スケール T = clip(target_dist × speed_gain / 0.5, 0.3, 2.5) [s]
    lambda_trajectory:        bool  = False
    lambda_traj_speed_gain:   float = 1.2

    # task-space virtual trajectory (Morasso 1981, Flash & Hogan 1985):
    # 筋空間でなく 3D task 空間で minimum-jerk 軌跡を生成し、各 waypoint で IK
    # を実行して λ(t) 配列を pre-compute する。これにより手先軌跡が straight に
    # なることを期待 (ヒト reach の主要 invariance)。
    # 注意: lambda_trajectory=True が前提。task_space_trajectory=True で task 空間版に切替。
    task_space_trajectory:    bool  = False

    # 小脳補正の出力先 (Kawato/Wolpert/Ito):
    #   "joint"  : 既存。delta_q_err → tau → J_act⁺ → muscle 活性化補正
    #              (PD 制御では機能するが λ-EP では不適切)
    #   "lambda" : delta_q_err → moment_arm @ delta_q_err → Δλ で λ_target を補正
    #              (λ-EP と整合、descending λ 指令を直接修正する小脳出力)
    cereb_correction_target:  str   = "joint"
    K_cereb_lambda:           float = 1.0   # スケール (m/rad 相当)

    # 視覚運動フィードバック (Saunders & Knill 2003; Sarlegna & Sainburg 2014):
    # tip_pos の周期的な観測から IK を再実行し λ_target を更新する。
    # ヒト視覚運動 latency ~100-200ms に相当 (period_steps × dt)。
    # 更新時は現在の λ を新しい λ_start として連続性を維持。
    visuomotor_feedback:      bool  = False
    visuomotor_period_steps:  int   = 10    # 10 × 0.020 = 200ms

    # 相反抑制
    K_ri:         float = 0.5
    ri_threshold: float = 0.3

    # 軌跡計画 (feedforward)
    use_traj_plan:     bool  = False  # True で軌跡計画を有効化
    traj_mode:         str   = "vel_scale"  # "vel_scale" | "feedforward"
    traj_speed_gain:   float = 1.2    # dist × gain / 0.5 [s] で T を自動計算
    traj_plan_dist_th: float = 0.05   # 再計画する最小距離 [m]
    traj_dt:           float = 0.020  # 実制御周期 (myoArm: frame_skip=10 × 0.002s = 0.020s)
    vel_scale_min:     float = 0.10   # 速度スケールの最小値 (最終補正用)
    # feedforward モード専用 (traj_mode="feedforward" のとき有効)
    K_ff:              float = 10.0   # 加速度→力の変換ゲイン [kg]（実効エンドポイント質量）
    Kp_traj:           float = 8.0    # feedforward モードでの位置補正ゲイン (小さく設定)
    Kd_traj:           float = 50.0   # feedforward モードでの速度補正ゲイン

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
        self._moment_arm: np.ndarray | None = None   # (nu, nv) [m/rad]
        self._J_act_step_count = 0
        self._site_id: int | None = None

        # ── λ-mode 用 cerebellar correction バッファ ─────────────────
        # cereb_correction_target="lambda" のとき (n_m,) の Δλ を遅延バッファに格納
        self._lambda_cereb_buf = DelayBuffer(config.cereb_delay_steps, shape=(n_m,))
        self._lambda_cereb_delayed = np.zeros(n_m)

        # ── ステップカウンタ (visuomotor feedback で使用) ─────────────
        self._step_count: int = 0

        # ── 軌跡プランナー ────────────────────────────────────────────
        # traj_dt は実際のシミュレーション dt (myoArm: 0.005s) に合わせる
        self._planner = MinimumJerkPlanner(dt=config.traj_dt)

        # ── 内部ステート ─────────────────────────────────────────────
        self._a_cereb_delayed = np.zeros(n_m)
        self._a_efcopy        = np.zeros(n_j)   # 前ステップの efcopy (n_j 次元)
        self._integral_ee     = np.zeros(3)      # エンドポイント積分項
        self._prev_tip        = np.zeros(3)

        # ── λ-model 用ステート ──────────────────────────────────────
        self._lambda:        np.ndarray | None = None   # (n_m,)  現在の λ
        self._lambda_start:  np.ndarray | None = None   # (n_m,)  episode 開始時 λ
        self._lambda_target: np.ndarray | None = None   # (n_m,)  target 平衡 λ
        self._last_target:   np.ndarray | None = None   # (3,)    最後に λ を計算した target
        self._traj_t:        float = 0.0                 # virtual traj 経過時間 [s]
        self._traj_T:        float = 1.0                 # virtual traj 所要時間 [s]
        # task-space VT 用 (pre-computed waypoints)
        self._lambda_array:  np.ndarray | None = None   # (N+1, n_m)
        self._tip_start:     np.ndarray | None = None

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
        self._planner.reset()
        self._J_act_pinv      = None
        self._moment_arm      = None
        self._J_act_step_count = 0
        self._a_cereb_delayed[:] = 0.0
        self._lambda_cereb_buf.reset()
        self._lambda_cereb_delayed[:] = 0.0
        self._a_efcopy[:] = 0.0
        self._integral_ee[:] = 0.0
        self._prev_tip[:] = 0.0
        self._lambda = None
        self._lambda_start = None
        self._lambda_target = None
        self._last_target = None
        self._traj_t = 0.0
        self._traj_T = 1.0
        self._lambda_array = None
        self._tip_start = None
        self._step_count = 0

    def load_cfc(self, path: str | Path) -> None:
        self._cfc.load(str(path))
        # load() は保存時の K_cereb を復元するが MyoArm は 20 次元が必要
        self._cfc.K_cereb = np.full(_N_JOINTS, self.config.K_cereb)

    # ------------------------------------------------------------------
    # λ-model: IK で q_target を求めて L_target を計算
    # ------------------------------------------------------------------

    def _compute_lambda(
        self,
        m: mujoco.MjModel,
        d: mujoco.MjData,
        target_pos: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """damped LS IK で q_target を求め、L_target = actuator_length(q_target) から λ を返す。

        Returns
        -------
        lambda_vec : (n_m,) [m]  各筋の閾値長
        info       : {iters, residual_mm, q_target}
        """
        cfg = self.config
        site_id = self._site_id
        assert site_id is not None

        saved_qpos = d.qpos.copy()
        saved_qvel = d.qvel.copy()

        # 初期 IK 線形化用の damped LS
        damp_sq = cfg.ik_damping ** 2
        for it in range(cfg.ik_max_iter):
            mujoco.mj_forward(m, d)
            p = d.site_xpos[site_id].copy()
            err = target_pos - p
            err_norm = float(np.linalg.norm(err))
            if err_norm < 1e-3:
                break
            Jp = _endpoint_jacobian(m, d, site_id)              # (3, nv)
            H  = Jp @ Jp.T + damp_sq * np.eye(3)
            dq = Jp.T @ np.linalg.solve(H, err)                 # (nv,)
            new_qpos = d.qpos + dq
            if m.jnt_limited.any():
                lo = m.jnt_range[:, 0]
                hi = m.jnt_range[:, 1]
                # 制限ありの関節のみクリップ (ない関節は ±inf)
                limited = m.jnt_limited.astype(bool)
                new_qpos = np.where(limited, np.clip(new_qpos, lo, hi), new_qpos)
            d.qpos[:] = new_qpos

        q_target = d.qpos.copy()
        mujoco.mj_forward(m, d)
        residual = float(np.linalg.norm(d.site_xpos[site_id] - target_pos))
        L_target = d.actuator_length.copy()

        # 復元
        d.qpos[:] = saved_qpos
        d.qvel[:] = saved_qvel
        mujoco.mj_forward(m, d)

        lambda_vec = L_target - cfg.lambda_offset
        return lambda_vec, {
            "iters": it + 1,
            "residual_mm": residual * 1000.0,
            "q_target": q_target,
        }

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
            # λ-mode の cerebellar correction で使う moment arm
            self._moment_arm = _compute_moment_arm(m, d)
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

        # ── 4. M1 → 基本筋活性化 (control_mode で分岐) ────────────────
        tip_vel = tip_pos - self._prev_tip
        self._prev_tip = tip_pos.copy()
        dist_to_target = float(np.linalg.norm(reach_err))

        # kd_factor は info 用に常に計算 (endpoint_pd 以外では無効)
        if cfg.Kd_proximity_gain > 0.0:
            kd_factor = 1.0 + cfg.Kd_proximity_gain * float(
                np.exp(-dist_to_target / cfg.Kd_proximity_scale)
            )
        else:
            kd_factor = 1.0
        Kd_ee_eff = cfg.Kd_ee * kd_factor

        ik_info = None
        if cfg.control_mode == "lambda_ep":
            # ── λ-model (Feldman EP-hypothesis) ──────────────────────
            # virtual trajectory: λ(t) = λ_start + s(τ)(λ_target - λ_start)
            #                     s(τ) = 10τ³-15τ⁴+6τ⁵ (Flash & Hogan 1985)
            # 静的モード: λ = λ_target 固定 (lambda_trajectory=False)
            # 視覚運動フィードバック: 周期的に IK を再実行し λ_target 更新
            target_pos = tip_pos + reach_err
            initial_compute = (self._lambda_target is None or self._last_target is None
                               or np.linalg.norm(target_pos - self._last_target) > 0.01)
            visuomotor_update = (cfg.visuomotor_feedback
                                 and self._lambda is not None
                                 and self._step_count > 0
                                 and self._step_count % cfg.visuomotor_period_steps == 0)

            if initial_compute:
                # 初回 (target が 1cm 以上動いたとき)
                L_now = d.actuator_length.copy()
                self._lambda_start  = (L_now - cfg.lambda_offset).copy()
                self._tip_start = tip_pos.copy()
                self._lambda_target, ik_info = self._compute_lambda(m, d, target_pos)
                self._traj_t = 0.0
                dist_to_tgt = float(np.linalg.norm(target_pos - tip_pos))
                self._traj_T = float(np.clip(
                    dist_to_tgt * cfg.lambda_traj_speed_gain / 0.5, 0.3, 2.5
                ))
                self._last_target = target_pos.copy()

                # task-space VT: 各 waypoint で IK を実行し λ 配列を pre-compute
                if cfg.task_space_trajectory:
                    N = max(int(self._traj_T / cfg.traj_dt), 4)
                    self._lambda_array = np.zeros((N+1, _N_MUSCLES))
                    for i in range(N+1):
                        tau_i = i / N
                        s_i = 10*tau_i**3 - 15*tau_i**4 + 6*tau_i**5
                        p_ref = self._tip_start + s_i * (target_pos - self._tip_start)
                        lam_i, _ = self._compute_lambda(m, d, p_ref)
                        self._lambda_array[i] = lam_i
            elif visuomotor_update:
                # 視覚運動フィードバック: 現在の q から IK を再実行して λ_target のみ
                # を更新する。λ_start と traj_t は変更しない (trajectory は連続)。
                # 理由: traj_t をリセットすると s(τ)≈0 で λ ≈ λ_start に戻り、
                #       駆動力がゼロになって運動が停滞する。
                if cfg.task_space_trajectory and self._lambda_array is not None:
                    # task-space VT の visuomotor: 残り waypoints を current state から
                    # 再計算する (current_idx 以降を上書き)
                    N = len(self._lambda_array) - 1
                    current_idx = min(int((self._traj_t / self._traj_T) * N), N)
                    p_now = tip_pos.copy()
                    for i in range(current_idx, N+1):
                        # 残り区間で task-space min-jerk
                        tau_i = (i - current_idx) / max(N - current_idx, 1)
                        s_i = 10*tau_i**3 - 15*tau_i**4 + 6*tau_i**5
                        p_ref = p_now + s_i * (target_pos - p_now)
                        lam_i, _ = self._compute_lambda(m, d, p_ref)
                        self._lambda_array[i] = lam_i
                    ik_info = {"iters": 0, "residual_mm": 0.0, "q_target": np.zeros(_N_JOINTS)}
                else:
                    self._lambda_target, ik_info = self._compute_lambda(m, d, target_pos)

            if cfg.task_space_trajectory and self._lambda_array is not None:
                # task-space VT: pre-computed waypoints から lookup
                N = len(self._lambda_array) - 1
                tau = min(self._traj_t / self._traj_T, 1.0)
                idx = min(int(tau * N), N)
                self._lambda = self._lambda_array[idx].copy()
                self._traj_t += cfg.traj_dt
            elif cfg.lambda_trajectory:
                tau = min(self._traj_t / self._traj_T, 1.0)
                s = 10*tau**3 - 15*tau**4 + 6*tau**5
                self._lambda = self._lambda_start + s * (
                    self._lambda_target - self._lambda_start
                )
                self._traj_t += cfg.traj_dt
            else:
                self._lambda = self._lambda_target

            # 小脳補正を λ 空間で加える (cereb_correction_target="lambda")
            if cfg.cereb_correction_target == "lambda":
                lambda_eff = self._lambda + self._lambda_cereb_delayed
            else:
                lambda_eff = self._lambda

            L_now = d.actuator_length.copy()
            stretch = np.maximum(L_now - lambda_eff, 0.0)
            a_base = np.clip(cfg.c_lambda * stretch, 0.0, 1.0)
            F_ee = np.zeros(3)  # 未使用 (info 用 placeholder)
        else:
            # ── endpoint_pd (旧来) ───────────────────────────────────
            if cfg.use_traj_plan:
                if not self._planner.is_active and dist_to_target > cfg.traj_plan_dist_th:
                    target_pos = tip_pos + reach_err
                    self._planner.plan(
                        p_start=tip_pos,
                        p_target=target_pos,
                        T=0,
                        speed_gain=cfg.traj_speed_gain,
                    )

                if cfg.traj_mode == "feedforward":
                    p_ref, v_ref, a_ref = self._planner.step_with_accel()
                    pos_err = p_ref - tip_pos
                    vel_err = v_ref - tip_vel
                    self._integral_ee = np.clip(self._integral_ee + pos_err, -2.0, 2.0)
                    F_ee = (
                        cfg.K_ff * a_ref
                        + cfg.Kp_traj * pos_err
                        + cfg.Kd_traj * vel_err
                        + cfg.Ki_ee * self._integral_ee
                    )
                else:
                    tau = self._planner.progress
                    vel_scale = max(30 * tau**2 * (1-tau)**2 / 1.875, cfg.vel_scale_min)
                    _ = self._planner.step()
                    scaled_err = reach_err * vel_scale
                    self._integral_ee = np.clip(self._integral_ee + scaled_err, -2.0, 2.0)
                    F_ee = (
                        cfg.Kp_ee * scaled_err
                        - Kd_ee_eff * tip_vel
                        + cfg.Ki_ee * self._integral_ee
                    )
            else:
                self._integral_ee = np.clip(self._integral_ee + reach_err, -2.0, 2.0)
                F_ee = (
                    cfg.Kp_ee * reach_err
                    - Kd_ee_eff * tip_vel
                    + cfg.Ki_ee * self._integral_ee
                )

            assert self._site_id is not None, "initialize() を先に呼ぶこと"
            J_ee  = _endpoint_jacobian(m, d, self._site_id)
            tau_pd = J_ee.T @ F_ee
            tau_grav = d.qfrc_bias.copy()
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

        # γ 運動ニューロン変調: target 接近で Ia ゲインを増幅 (ブレーキ機構)
        if cfg.gamma_gain > 0.0:
            dist = float(np.linalg.norm(reach_err))
            gamma_factor = 1.0 + cfg.gamma_gain * float(np.exp(-dist / cfg.gamma_scale))
            da_ia = da_ia * gamma_factor

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
            "gamma_factor":    (1.0 + cfg.gamma_gain *
                                float(np.exp(-dist_to_target / cfg.gamma_scale))
                                if cfg.gamma_gain > 0 else 1.0),
            "kd_factor":       kd_factor,
            "control_mode":    cfg.control_mode,
            "ik_info":         ik_info,  # None if no IK ran this step
            "stretch_max":     (float(np.max(np.maximum(d.actuator_length - self._lambda, 0.0)))
                                if cfg.control_mode == "lambda_ep" and self._lambda is not None
                                else 0.0),
            "lambda_cereb_norm": float(np.linalg.norm(self._lambda_cereb_delayed)),
            "step_count":      self._step_count,
        }
        self._step_count += 1
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

        cereb_correction_target に応じて補正先を分ける:
          - "joint" : 旧来。J_act⁺ で muscle 活性化補正
          - "lambda": moment_arm @ delta_q_err で λ 直接補正 (Kawato/Wolpert 流)
        """
        cfg = self.config

        # IO 発火判定で学習をゲート
        fired = self._io.should_fire(self._pred_error)
        self._cfc.update(q_actual, allow_online_update=fired)
        self._pred_error = self._cfc.get_prediction_error().copy()

        delta_q_err = self._cfc.get_prediction_error()               # (n_j,)

        if cfg.cereb_correction_target == "lambda":
            # λ 空間補正: predicted_q が actual より大きい (delta_q_err>0) → 筋がもっと
            # 短くなるべき位置に向かいたい。muscle i の "predicted L" は
            # L_actual + R[i,:] @ delta_q_err。stretch を増やすには λ を下げる。
            # → Δλ = -K × R @ delta_q_err
            assert self._moment_arm is not None, "moment_arm not initialized"
            dL_corr = -cfg.K_cereb_lambda * (self._moment_arm @ delta_q_err)  # (n_m,)
            self._lambda_cereb_delayed = self._lambda_cereb_buf.push_and_get(dL_corr)
            # joint 経路は無効化
            self._a_cereb_delayed = self._cereb_buf.push_and_get(np.zeros(_N_MUSCLES))
        else:
            # 旧来 joint 経路
            tau_cereb   = cfg.K_cereb * delta_q_err
            tau_full    = np.zeros(_N_JOINTS)
            tau_full[:] = tau_cereb
            da_cereb_raw = self._J_act_pinv @ tau_full
            self._a_cereb_delayed = self._cereb_buf.push_and_get(da_cereb_raw)
            # λ 経路は無効化
            self._lambda_cereb_delayed = self._lambda_cereb_buf.push_and_get(
                np.zeros(_N_MUSCLES)
            )

    # ------------------------------------------------------------------

    def get_io_stats(self) -> dict:
        return self._io.get_stats()

    def get_pred_error(self) -> np.ndarray:
        return self._pred_error.copy()
