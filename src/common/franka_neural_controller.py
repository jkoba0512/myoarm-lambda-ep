"""
神経系統合コントローラ（Franka Panda トルク制御用）

論文スコープ: 運動計画を除く脊髄〜小脳レベルの神経系を統合。
位置制御版コントローラのトルク制御対応版。

コンポーネント:
  1. LIFProprioceptor      : LIF 固有受容器（関節角・速度 → スパイク発火率）
  2. MatsuokaOscillator     : 脊髄 CPG（リズム生成 + 固有受容器 FB）
  3. CfCGravityCompensator  : 小脳（重力・コリオリ力の学習補償）
  4. IzhikevichReflexArc    : 反射弓（外乱時の高速トルク補正）

信号フロー:
  q_ref(t) [従来法からの目標軌道]
    → CPG（Matsuoka + LIF-FB）→ q_target = q_ref + q_cpg
    → PD: τ_pd = Kp*(q_target - q) + Kd*(0 - dq)
    → τ_total = clip(τ_pd + τ_comp + τ_reflex, -TAU_LIMIT, TAU_LIMIT) → FrankaEnv

  q(t), dq(t) ← FrankaEnv
    → LIF → r_q, r_dq → CPG（FB）
                       → Izhikevich → Δq_reflex → τ_reflex = Kp_r * Δq
    → CfCGravityCompensator(q, dq) → τ_comp

アブレーション:
  use_proprioceptor / use_reflex / use_cerebellum フラグで各コンポーネントを制御。
"""

from __future__ import annotations

import numpy as np

from methodD.cpg import MatsuokaOscillator
from methodB.cfc_compensator import CfCGravityCompensator
from methodE.lif_proprioceptor import LIFProprioceptor
from methodE.izhikevich_reflex import IzhikevichReflexArc

# Franka Panda トルク上限 [Nm]
_TAU_LIMIT = np.array([87, 87, 87, 87, 12, 12, 12], dtype=np.float64)

# デフォルト PD ゲイン（関節別）
# Kp を下げることで重力・コリオリ補償（小脳 CfC）の寄与が顕在化する。
# Joint 1-4: Kp=50, Kd=7  / Joint 5-7: Kp=10, Kd=1.5
_KP_DEFAULT = np.array([50.0, 50.0, 50.0, 50.0, 10.0, 10.0, 10.0])
_KD_DEFAULT = np.array([ 7.0,  7.0,  7.0,  7.0,  1.5,  1.5,  1.5])


class FrankaNeuralController:
    """
    Franka Panda トルク制御用神経系統合コントローラ。

    Parameters
    ----------
    dt                 : タイムステップ [s]（FrankaEnv.dt に合わせる）
    q_range            : 関節可動域 (7, 2)。None なら FrankaEnv のデフォルト。
    kp                 : PD 比例ゲイン (7,) [Nm/rad]
    kd                 : PD 微分ゲイン (7,) [Nm·s/rad]
    cpg_params         : MatsuokaOscillator への追加キーワード引数
    lif_params         : LIFProprioceptor への追加キーワード引数
    reflex_params      : IzhikevichReflexArc への追加キーワード引数
    use_proprioceptor  : LIF 固有受容器を使用するか
    use_reflex         : Izhikevich 反射弓を使用するか
    use_cerebellum     : CfCGravityCompensator を使用するか
    cpg_alpha_fb       : CPG 固有受容器フィードバックゲイン
    cfc_hidden_units   : CfCGravityCompensator の hidden_units
    device             : 'cpu' or 'cuda'
    """

    N_JOINTS = 7

    def __init__(
        self,
        dt:                float           = 0.002,
        q_range:           np.ndarray | None = None,
        kp:                np.ndarray | None = None,
        kd:                np.ndarray | None = None,
        cpg_params:        dict | None      = None,
        lif_params:        dict | None      = None,
        reflex_params:     dict | None      = None,
        use_proprioceptor: bool             = True,
        use_reflex:        bool             = True,
        use_cerebellum:    bool             = True,
        cpg_alpha_fb:      float            = 0.3,
        cfc_hidden_units:  int              = 64,
        device:            str              = "cpu",
    ):
        n = self.N_JOINTS
        self.dt     = dt
        self.device = device

        self.use_proprioceptor = use_proprioceptor
        self.use_reflex        = use_reflex
        self.use_cerebellum    = use_cerebellum

        # Franka デフォルト可動域（panda_torque.xml の joint range に準拠）
        if q_range is None:
            q_range = np.array([
                [-2.8973,  2.8973],
                [-1.7628,  1.7628],
                [-2.8973,  2.8973],
                [-3.0718, -0.0698],
                [-2.8973,  2.8973],
                [-0.0175,  3.7525],
                [-2.8973,  2.8973],
            ])
        self.q_range = q_range

        self.kp = kp if kp is not None else _KP_DEFAULT.copy()
        self.kd = kd if kd is not None else _KD_DEFAULT.copy()
        self.tau_limit = _TAU_LIMIT.copy()

        # ── 1. 脊髄 CPG ─────────────────────────────────────────
        _cpg_kw = dict(
            n_joints=n,
            dt=dt,
            alpha_fb=cpg_alpha_fb if use_proprioceptor else 0.0,
        )
        if cpg_params:
            _cpg_kw.update(cpg_params)
        self.cpg = MatsuokaOscillator(**_cpg_kw)

        # ── 2. LIF 固有受容器 ────────────────────────────────────
        _lif_kw = dict(n_joints=n, dt=dt, q_range=q_range)
        if lif_params:
            _lif_kw.update(lif_params)
        self.proprioceptor = LIFProprioceptor(**_lif_kw)

        # ── 3. Izhikevich 反射弓 ─────────────────────────────────
        _ref_kw = dict(n_joints=n, dt=dt)
        if reflex_params:
            _ref_kw.update(reflex_params)
        self.reflex = IzhikevichReflexArc(**_ref_kw)

        # ── 4. 小脳 CfCGravityCompensator ────────────────────────
        self.cerebellum: CfCGravityCompensator | None = None
        if use_cerebellum:
            self.cerebellum = CfCGravityCompensator(
                n_joints=n,
                hidden_units=cfc_hidden_units,
                device=device,
            )

    # ------------------------------------------------------------------
    # リセット
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """エピソード開始時に全コンポーネントをリセット。"""
        self.cpg.reset()
        self.proprioceptor.reset()
        self.reflex.reset()
        if self.cerebellum is not None:
            self.cerebellum.reset()

    # ------------------------------------------------------------------
    # 1 ステップ制御
    # ------------------------------------------------------------------

    def step(
        self,
        q:      np.ndarray,
        dq:     np.ndarray,
        q_ref:  np.ndarray,
        cpg_I:  np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        1 ステップ実行し、最終トルク指令と診断情報を返す。

        Parameters
        ----------
        q      : 現在の関節角 (7,) [rad]
        dq     : 現在の関節速度 (7,) [rad/s]
        q_ref  : 目標軌道からの参照関節角 (7,) [rad]
        cpg_I  : CPG への外部ドライブ (7,)。None なら定常入力 1.0。

        Returns
        -------
        tau_total : 最終トルク指令 (7,) [Nm]（TAU_LIMIT クリップ済み）
        info      : 診断辞書
        """
        info: dict = {
            "r_q":           np.zeros(self.N_JOINTS),
            "r_dq":          np.zeros(self.N_JOINTS),
            "q_cpg":         np.zeros(self.N_JOINTS),
            "q_target":      np.zeros(self.N_JOINTS),
            "tau_pd":        np.zeros(self.N_JOINTS),
            "tau_comp":      np.zeros(self.N_JOINTS),
            "tau_reflex":    np.zeros(self.N_JOINTS),
            "reflex_active": np.zeros(self.N_JOINTS, dtype=bool),
            "tau_sys":       None,
        }

        # ── 1. 固有受容器 ─────────────────────────────────────────
        r_q = None
        if self.use_proprioceptor:
            r_q, r_dq = self.proprioceptor.encode(q, dq)
            info["r_q"]  = r_q
            info["r_dq"] = r_dq

        # ── 2. CPG（固有受容器 FB を注入）────────────────────────
        q_cpg = self.cpg.step(I=cpg_I, r_q=r_q)
        q_target = q_ref + q_cpg
        q_target = np.clip(q_target, self.q_range[:, 0], self.q_range[:, 1])
        info["q_cpg"]    = q_cpg
        info["q_target"] = q_target

        # ── 3. PD フィードバックトルク ────────────────────────────
        tau_pd = self.kp * (q_target - q) + self.kd * (-dq)
        info["tau_pd"] = tau_pd

        # ── 4. 小脳 CfC 重力補償 ─────────────────────────────────
        tau_comp = np.zeros(self.N_JOINTS)
        if self.use_cerebellum and self.cerebellum is not None:
            tau_comp = self.cerebellum.predict(q, dq)
            info["tau_sys"] = self.cerebellum.get_tau_sys()
        info["tau_comp"] = tau_comp

        # ── 5. 反射弓 ─────────────────────────────────────────────
        tau_reflex = np.zeros(self.N_JOINTS)
        if self.use_reflex:
            q_error = q_target - q
            delta_q_reflex = self.reflex.respond(q_error, dq)
            # Δq → トルク変換（PD 比例ゲイン使用）
            tau_reflex = self.kp * delta_q_reflex
            info["reflex_active"] = self.reflex.is_active()
        info["tau_reflex"] = tau_reflex

        # ── 6. 統合・クリップ ─────────────────────────────────────
        tau_total = np.clip(
            tau_pd + tau_comp + tau_reflex,
            -self.tau_limit,
            self.tau_limit,
        )

        return tau_total, info

    # ------------------------------------------------------------------
    # 小脳 CfC の訓練・保存・読み込み
    # ------------------------------------------------------------------

    def train_cerebellum(
        self,
        env,
        n_trajectories: int  = 200,
        seq_len:        int  = 50,
        n_epochs:       int  = 300,
        verbose:        bool = True,
    ) -> list[float]:
        """
        FrankaEnv からランダム軌道データを収集し、CfCGravityCompensator を訓練する。

        Parameters
        ----------
        env            : FrankaEnv インスタンス
        n_trajectories : 収集する軌道数
        seq_len        : 各軌道のシーケンス長 [ステップ]
        n_epochs       : 訓練エポック数
        verbose        : 進捗表示
        """
        if self.cerebellum is None:
            raise RuntimeError("use_cerebellum=False のため小脳が初期化されていません。")

        q_seqs, dq_seqs, tau_seqs = CfCGravityCompensator.collect_sequence_data(
            env,
            n_trajectories=n_trajectories,
            seq_len=seq_len,
        )
        return self.cerebellum.fit(
            q_seqs, dq_seqs, tau_seqs,
            n_epochs=n_epochs,
            verbose=verbose,
        )

    def save_cerebellum(self, path: str) -> None:
        if self.cerebellum is None:
            raise RuntimeError("小脳が初期化されていません。")
        self.cerebellum.save(path)

    def load_cerebellum(self, path: str) -> None:
        if self.cerebellum is None:
            raise RuntimeError("小脳が初期化されていません。")
        self.cerebellum.load(path)
