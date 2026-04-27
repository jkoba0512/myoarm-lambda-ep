"""
神経系統合コントローラ（Franka Panda トルク制御用）

論文スコープ: 運動計画を除く脊髄〜小脳レベルの神経系を統合。
位置制御版コントローラのトルク制御対応版。

コンポーネント:
  1. LIFProprioceptor      : LIF 固有受容器（関節角・速度 → スパイク発火率）
  2. MatsuokaOscillator     : 脊髄 CPG（リズム生成 + 固有受容器 FB）
  3. CfCGravityCompensator  : 小脳（重力・コリオリ力の学習補償）[旧]
     CfCForwardModel        : 小脳 Forward model（順動力学 + エファレンスコピー）[E1]
  4. IzhikevichReflexArc    : 反射弓（位置誤差ベース）[旧]
     IaIbReflexArc          : 真の Ia/Ib 反射弓（速度・力依存）[E3]
  5. VirtualCocontraction   : 仮想コ・コントラクション（可変インピーダンス）[E2]

アブレーション:
  use_proprioceptor / use_reflex / use_cerebellum / use_cocontraction フラグで制御。
  use_ia_ib_reflex=True で旧 Izhikevich 反射弓を IaIb 反射弓に切り替える。
"""

from __future__ import annotations

import numpy as np

from methodD.cpg import MatsuokaOscillator
from methodB.cfc_compensator import CfCGravityCompensator
from methodB.cfc_forward_model import CfCForwardModel
from methodE.lif_proprioceptor import LIFProprioceptor
from methodE.izhikevich_reflex import IzhikevichReflexArc
from methodE.virtual_cocontraction import VirtualCocontraction
from methodE.ia_ib_reflex import IaIbReflexArc
from common.motor_cortex_analog import MotorCortexAnalog

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
    reflex_params      : IzhikevichReflexArc または IaIbReflexArc への追加キーワード引数
    cocontraction_params: VirtualCocontraction への追加キーワード引数
    ia_ib_params       : IaIbReflexArc への追加キーワード引数
    use_proprioceptor  : LIF 固有受容器を使用するか
    use_reflex         : 反射弓を使用するか（True なら use_ia_ib_reflex で種類を選択）
    use_cerebellum     : CfC 補償器を使用するか
    use_forward_model  : True なら CfCForwardModel（順動力学）を使用する
                         False なら CfCGravityCompensator（逆動力学, 旧実装）を使用する
    use_cocontraction  : VirtualCocontraction（仮想コ・コントラクション）を使用するか [E2]
    use_ia_ib_reflex   : True なら IaIbReflexArc（Ia/Ib ベース）[E3]
                         False なら IzhikevichReflexArc（位置誤差ベース, 旧実装）
    use_motor_cortex   : MotorCortexAnalog（運動皮質アナログ）を使用するか [E4]
    motor_cortex_params: MotorCortexAnalog への追加キーワード引数
    cpg_alpha_fb       : CPG 固有受容器フィードバックゲイン
    cfc_hidden_units   : CfC モデルの hidden_units
    K_cereb            : Forward model の補正ゲイン (7,) [Nm/rad]。None でデフォルト。
    online_lr          : Forward model のオンライン学習率（0.0 = 無効）
    device             : 'cpu' or 'cuda'
    """

    N_JOINTS = 7

    def __init__(
        self,
        dt:                  float             = 0.002,
        q_range:             np.ndarray | None = None,
        kp:                  np.ndarray | None = None,
        kd:                  np.ndarray | None = None,
        cpg_params:          dict | None       = None,
        lif_params:          dict | None       = None,
        reflex_params:       dict | None       = None,
        cocontraction_params: dict | None      = None,
        ia_ib_params:        dict | None       = None,
        use_proprioceptor:   bool              = True,
        use_reflex:          bool              = True,
        use_cerebellum:      bool              = True,
        use_forward_model:   bool              = False,
        use_cocontraction:   bool              = False,
        use_ia_ib_reflex:    bool              = False,
        use_motor_cortex:    bool              = False,
        motor_cortex_params: dict | None       = None,
        cpg_alpha_fb:        float             = 0.3,
        cfc_hidden_units:    int               = 64,
        K_cereb:             np.ndarray | None = None,
        online_lr:           float             = 0.0,
        device:              str               = "cpu",
    ):
        n = self.N_JOINTS
        self.dt     = dt
        self.device = device

        self.use_proprioceptor = use_proprioceptor
        self.use_reflex        = use_reflex
        self.use_cerebellum    = use_cerebellum
        self.use_forward_model = use_forward_model
        self.use_cocontraction = use_cocontraction
        self.use_ia_ib_reflex  = use_ia_ib_reflex
        self.use_motor_cortex  = use_motor_cortex

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

        # ── 3. 反射弓 ─────────────────────────────────────────────
        # use_ia_ib_reflex=True  : IaIbReflexArc（E3: 速度・力依存, 生物学的）
        # use_ia_ib_reflex=False : IzhikevichReflexArc（旧: 位置誤差依存）
        self.reflex:      IzhikevichReflexArc | None = None
        self.reflex_ia_ib: IaIbReflexArc | None      = None
        if use_reflex:
            if use_ia_ib_reflex:
                _ref_kw = dict(n_joints=n, dt=dt)
                if ia_ib_params:
                    _ref_kw.update(ia_ib_params)
                self.reflex_ia_ib = IaIbReflexArc(**_ref_kw)
            else:
                _ref_kw = dict(n_joints=n, dt=dt)
                if reflex_params:
                    _ref_kw.update(reflex_params)
                self.reflex = IzhikevichReflexArc(**_ref_kw)

        # ── 4a. 仮想コ・コントラクション（E2）─────────────────────
        self.cocontraction: VirtualCocontraction | None = None
        if use_cocontraction:
            _cc_kw: dict = dict(n_joints=n)
            if cocontraction_params:
                _cc_kw.update(cocontraction_params)
            self.cocontraction = VirtualCocontraction(**_cc_kw)

        # ── 4b. 運動皮質アナログ（E4）────────────────────────────
        self.motor_cortex: MotorCortexAnalog | None = None
        if use_motor_cortex:
            _mca_kw: dict = dict(n_joints=n, dt=dt)
            if motor_cortex_params:
                _mca_kw.update(motor_cortex_params)
            self.motor_cortex = MotorCortexAnalog(**_mca_kw)

        # ── 4c. 小脳 ─────────────────────────────────────────────
        # use_forward_model=True  : CfCForwardModel（順動力学, Phase E1〜）
        # use_forward_model=False : CfCGravityCompensator（逆動力学, 旧実装）
        self.cerebellum: CfCGravityCompensator | None = None
        self.cerebellum_fwd: CfCForwardModel | None   = None
        if use_cerebellum:
            if use_forward_model:
                self.cerebellum_fwd = CfCForwardModel(
                    n_joints=n,
                    hidden_units=cfc_hidden_units,
                    device=device,
                    K_cereb=K_cereb,
                    online_lr=online_lr,
                )
            else:
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
        if self.reflex is not None:
            self.reflex.reset()
        if self.reflex_ia_ib is not None:
            self.reflex_ia_ib.reset()
        if self.cocontraction is not None:
            self.cocontraction.reset()
        if self.motor_cortex is not None:
            self.motor_cortex.reset()
        if self.cerebellum is not None:
            self.cerebellum.reset()
        if self.cerebellum_fwd is not None:
            self.cerebellum_fwd.reset()

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
            "r_q":              np.zeros(self.N_JOINTS),
            "r_dq":             np.zeros(self.N_JOINTS),
            "q_cpg":            np.zeros(self.N_JOINTS),
            "q_target":         np.zeros(self.N_JOINTS),
            "tau_pd":           np.zeros(self.N_JOINTS),
            "tau_comp":         np.zeros(self.N_JOINTS),
            "tau_reflex":       np.zeros(self.N_JOINTS),
            "tau_virtual":      np.zeros(self.N_JOINTS),  # E2: 仮想インピーダンス
            "tau_cc":           np.zeros(self.N_JOINTS),  # E2: co-contraction レベル
            "reflex_active":    np.zeros(self.N_JOINTS, dtype=bool),
            "ia_active":        np.zeros(self.N_JOINTS, dtype=bool),  # E3: Ia 発火
            "ib_active":        np.zeros(self.N_JOINTS, dtype=bool),  # E3: Ib 発火
            "reflex_latency_ms": None,
            "tau_sys":          None,
            # Forward model 固有フィールド（use_forward_model=False では None）
            "q_hat":            None,
            "pred_error":       None,
            # E4: 運動皮質アナログ
            "mca_task_mode":    None,
            "mca_cc_target":    None,
            "mca_cpg_amplitude": None,
        }

        # ── 0. 運動皮質アナログ（E4）─────────────────────────────
        # MCA を先に呼び出し、CPG 振幅と cc_target を取得する。
        # cocontraction が有効な場合は cc_target で tau_cc を上書きする。
        mca_cc_target    : np.ndarray | None = None
        mca_cpg_amplitude: float | None      = None
        if self.motor_cortex is not None:
            mca_out = self.motor_cortex.step(q, dq, q_ref)
            mca_cc_target     = mca_out["cc_target"]
            mca_cpg_amplitude = mca_out["cpg_amplitude"]
            info["mca_task_mode"]     = mca_out["task_mode"]
            info["mca_cc_target"]     = mca_cc_target.copy()
            info["mca_cpg_amplitude"] = mca_cpg_amplitude

        # ── 1. 固有受容器 ─────────────────────────────────────────
        r_q = None
        if self.use_proprioceptor:
            r_q, r_dq = self.proprioceptor.encode(q, dq)
            info["r_q"]  = r_q
            info["r_dq"] = r_dq

        # ── 2. CPG（固有受容器 FB を注入）────────────────────────
        # E4: MCA が cpg_amplitude を動的に上書きする
        if mca_cpg_amplitude is not None:
            self.cpg.amplitude = mca_cpg_amplitude
        q_cpg = self.cpg.step(I=cpg_I, r_q=r_q)
        q_target = q_ref + q_cpg
        q_target = np.clip(q_target, self.q_range[:, 0], self.q_range[:, 1])
        info["q_cpg"]    = q_cpg
        info["q_target"] = q_target

        # ── 3. PD フィードバックトルク ────────────────────────────
        tau_pd = self.kp * (q_target - q) + self.kd * (-dq)
        info["tau_pd"] = tau_pd

        # ── 4. 小脳 ──────────────────────────────────────────────
        tau_comp = np.zeros(self.N_JOINTS)
        if self.use_cerebellum:
            if self.use_forward_model and self.cerebellum_fwd is not None:
                # Forward model: (q_t, dq_t, τ_pd_t) → q̂(t+1)
                # τ_cereb = K × (q_target − q̂)
                #
                # 機能: 近目標域（静止保持）での高精度補正
                #   実効 kp_eff = kp + K_cereb（50+25=75）として機能し、
                #   近目標の定常誤差を削減（103 mrad vs PD-only 158 mrad）。
                #
                # 制限: 大外乱後の回復は対象外。
                #   displaced 位置の重力トルクが大きく、3秒で0.1 rad以下に収束不可。
                #   大外乱への対応は Phase E3 反射弓（IzhikevichReflexArc）が担う。
                q_hat = self.cerebellum_fwd.predict(q, dq, tau_pd)
                tau_comp = self.cerebellum_fwd.K_cereb * (q_target - q_hat)
                info["q_hat"]      = q_hat
                info["pred_error"] = q_target - q_hat  # 予測される将来の位置誤差
                info["tau_sys"]    = self.cerebellum_fwd.get_tau_sys()
            elif self.cerebellum is not None:
                # 旧逆動力学モード
                tau_comp = self.cerebellum.predict(q, dq)
                info["tau_sys"] = self.cerebellum.get_tau_sys()
        info["tau_comp"] = tau_comp

        # ── 5. 仮想コ・コントラクション（E2）+ E4 MCA オーバーライド ─
        tau_virtual = np.zeros(self.N_JOINTS)
        if self.cocontraction is not None:
            tau_virtual, tau_cc = self.cocontraction.step(q, dq, q_target)
            # E4: MCA の cc_target で上書き（追加インピーダンスを再計算）
            if mca_cc_target is not None:
                # K_virt / B_virt はデフォルトゲインを使って再計算
                k_gain = self.cocontraction.k_virtual_gain
                b_gain = self.cocontraction.b_virtual_gain
                err    = q_target - q
                tau_virtual = k_gain * mca_cc_target * err + b_gain * mca_cc_target * (-dq)
                tau_cc      = mca_cc_target
            info["tau_virtual"] = tau_virtual
            info["tau_cc"]      = tau_cc
        elif mca_cc_target is not None:
            # cocontraction モジュールなしで MCA のみ有効な場合
            # シンプルな追加インピーダンス（デフォルトゲイン）
            k_gain = 0.3
            b_gain = 0.1
            err    = q_target - q
            tau_virtual = k_gain * mca_cc_target * err + b_gain * mca_cc_target * (-dq)
            info["tau_virtual"] = tau_virtual
            info["tau_cc"]      = mca_cc_target

        # ── 6. 反射弓 ─────────────────────────────────────────────
        tau_reflex = np.zeros(self.N_JOINTS)
        if self.use_reflex:
            if self.reflex_ia_ib is not None:
                # E3: Ia/Ib 反射弓
                # τ_ag を主動筋チャネルとして渡す（co-contraction 込み）
                tau_net_so_far = tau_pd + tau_comp + tau_virtual
                tau_ag, _ = VirtualCocontraction.decompose(
                    tau_net_so_far,
                    info["tau_cc"],
                )
                tau_ia, tau_ib = self.reflex_ia_ib.step(dq, tau_ag)
                tau_reflex = tau_ia + tau_ib
                info["ia_active"]         = self.reflex_ia_ib.is_ia_active()
                info["ib_active"]         = self.reflex_ia_ib.is_ib_active()
                info["reflex_latency_ms"] = self.reflex_ia_ib.get_reflex_latency_ms()
            elif self.reflex is not None:
                # 旧 Izhikevich 反射弓
                q_error = q_target - q
                delta_q_reflex = self.reflex.respond(q_error, dq)
                tau_reflex = self.kp * delta_q_reflex
                info["reflex_active"] = self.reflex.is_active()
        info["tau_reflex"] = tau_reflex

        # ── 7. 統合・クリップ ─────────────────────────────────────
        tau_total = np.clip(
            tau_pd + tau_comp + tau_virtual + tau_reflex,
            -self.tau_limit,
            self.tau_limit,
        )

        return tau_total, info

    # ------------------------------------------------------------------
    # 運動皮質アナログ（E4）制御 API
    # ------------------------------------------------------------------

    def set_task_mode(self, mode: str) -> None:
        """
        MCA のタスクモードを外部から設定する。
        use_motor_cortex=True のときのみ有効。

        Parameters
        ----------
        mode : "hold" または "oscillate"
        """
        if self.motor_cortex is not None:
            self.motor_cortex.set_task_mode(mode)

    # ------------------------------------------------------------------
    # Forward model 更新（env.step() 後に呼ぶ）
    # ------------------------------------------------------------------

    def update_cerebellum(self, q_actual: np.ndarray) -> None:
        """
        実際の次状態 q_actual を使って Forward model の予測誤差を更新する。
        use_forward_model=True のときのみ有効。env.step() の直後に呼ぶこと。

        Parameters
        ----------
        q_actual : env.step() 後の実際の関節角 q(t+1) [rad]
        """
        if self.use_forward_model and self.cerebellum_fwd is not None:
            self.cerebellum_fwd.update(q_actual)

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
        if self.use_forward_model:
            if self.cerebellum_fwd is None:
                raise RuntimeError("Forward model が初期化されていません。")
            self.cerebellum_fwd.save(path)
        else:
            if self.cerebellum is None:
                raise RuntimeError("小脳が初期化されていません。")
            self.cerebellum.save(path)

    def load_cerebellum(self, path: str) -> None:
        if self.use_forward_model:
            if self.cerebellum_fwd is None:
                raise RuntimeError("Forward model が初期化されていません。")
            self.cerebellum_fwd.load(path)
        else:
            if self.cerebellum is None:
                raise RuntimeError("小脳が初期化されていません。")
            self.cerebellum.load(path)

    def train_cerebellum_forward(
        self,
        env,
        n_trajectories: int  = 200,
        seq_len:        int  = 50,
        n_epochs:       int  = 300,
        verbose:        bool = True,
        seed:           int  = 42,
    ) -> list[float]:
        """
        CfCForwardModel を訓練する（use_forward_model=True 時専用）。
        env.step() の実軌跡から (q_t, dq_t, τ_t) → q_{t+1} を学習する。
        """
        if self.cerebellum_fwd is None:
            raise RuntimeError("use_forward_model=False のため Forward model がありません。")
        rng = np.random.default_rng(seed)
        q_seqs, dq_seqs, tau_seqs, q_next_seqs = \
            CfCForwardModel.collect_forward_data(
                env, n_trajectories=n_trajectories, seq_len=seq_len, rng=rng
            )
        return self.cerebellum_fwd.fit(
            q_seqs, dq_seqs, tau_seqs, q_next_seqs,
            n_epochs=n_epochs, verbose=verbose, seed=seed,
        )
