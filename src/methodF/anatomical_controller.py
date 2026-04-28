"""
AnatomicalController — 解剖学的に忠実な神経運動制御コントローラ。

設計原則:
  - EF-full（F0-abstract）を一切変更せず、独立した新クラスとして実装する。
  - F0-faithful の基本設定: 小脳サイドループ・感覚遅延・散発的 IO エラー信号。
  - AnatomicalConfig の設定値を変えるだけで F1〜F5 の全実験条件に対応できる。

信号フロー（1 ステップ）:
  1. prop_delay_{q,dq}.push_and_get(q, dq)   → q_del, dq_del
  2. m1.receive_cerebellar_feedback(τ_cereb_prev_delayed)
  3. τ_cortical, τ_efcopy = m1.step(q_target, q, dq)
  4. cerebellum.predict(q_del, dq_del, τ_efcopy)   [env.step() 前]
  --------- env.step(τ_total) ---------
  5. τ_cereb_delayed = cerebellum.update_and_get_correction(q_actual)
                        ↑ 次ステップの M1 フィードバック用に保存
  6. τ_reflex = reflex.step(dq_del, τ_cortical)
  7. τ_cpg    = cpg.step(q_del, dq_del, spike_rates)
  8. τ_total  = τ_cortical + τ_reflex + τ_cpg (clip to τ_limit)

呼び出し順（実験スクリプト）:
  q, dq = env.get_state()
  tau, info = ctrl.step(q, dq, q_target)
  env.step(tau)
  q_actual, _ = env.get_state()
  ctrl.update_cerebellum(q_actual)        # ← env.step() の後に必ず呼ぶ
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from methodB.cfc_forward_model import CfCForwardModel
from methodD.cpg import MatsuokaOscillator
from methodE.ia_ib_reflex import IaIbReflexArc
from methodE.lif_proprioceptor import LIFProprioceptor
from methodE.virtual_cocontraction import VirtualCocontraction
from methodF.delay_buffer import (
    DelayBuffer,
    PROPRIOCEPTIVE_DELAY_STEPS,
    CEREBELLAR_LOOP_DELAY_STEPS,
)
from methodF.inferior_olive_analog import InferiorOliveAnalog
from methodF.motor_cortex_m1 import MotorCortexM1
from methodF.cerebellar_side_loop import CerebellarSideLoop

# Franka Panda トルク上限 [Nm]
_TAU_LIMIT = np.array([87, 87, 87, 87, 12, 12, 12], dtype=np.float64)

_KP_DEFAULT = np.array([50.0, 50.0, 50.0, 50.0, 10.0, 10.0, 10.0])
_KD_DEFAULT = np.array([ 7.0,  7.0,  7.0,  7.0,  1.5,  1.5,  1.5])


@dataclass
class AnatomicalConfig:
    """AnatomicalController の全パラメータ。"""

    # 遅延設定 ------------------------------------------------------------------
    prop_delay_steps:   int   = PROPRIOCEPTIVE_DELAY_STEPS   # 固有受容遅延（10 = 20 ms）
    cereb_delay_steps:  int   = CEREBELLAR_LOOP_DELAY_STEPS  # 小脳ループ遅延（15 = 30 ms）

    # 下オリーブ核 ---------------------------------------------------------------
    io_mode:            str   = "sparse"   # "sparse" | "continuous" | "error_gated"
    io_firing_rate_hz:  float = 1.0
    io_gain:            float = 5.0

    # M1 逆モデル ----------------------------------------------------------------
    inverse_model_loc:  str   = "m1"       # "m1" | "cerebellum" | "both"（F5）
    efcopy_enabled:     bool  = True       # False で遮断（F3）
    efcopy_noise_std:   float = 0.0        # ノイズ付加（F3）

    # PD ゲイン ------------------------------------------------------------------
    kp: np.ndarray = field(default_factory=lambda: _KP_DEFAULT.copy())
    kd: np.ndarray = field(default_factory=lambda: _KD_DEFAULT.copy())

    # CfC モデル -----------------------------------------------------------------
    cfc_hidden_units:   int   = 64
    cfc_device:         str   = "cpu"

    # タイムステップ -------------------------------------------------------------
    dt:                 float = 0.002


class AnatomicalController:
    """
    解剖学的忠実コントローラ（F0-faithful）。

    Parameters
    ----------
    config : AnatomicalConfig
    seed   : 乱数シード（IO・ノイズ）
    """

    N_JOINTS = 7

    def __init__(self, config: AnatomicalConfig | None = None, seed: int = 0) -> None:
        if config is None:
            config = AnatomicalConfig()
        self.config = config
        n = self.N_JOINTS
        dt = config.dt

        # ── 遅延バッファ ─────────────────────────────────────────────
        self._prop_delay_q  = DelayBuffer(config.prop_delay_steps,  shape=(n,))
        self._prop_delay_dq = DelayBuffer(config.prop_delay_steps,  shape=(n,))
        self._cereb_delay   = DelayBuffer(config.cereb_delay_steps, shape=(n,))

        # ── 下オリーブ核アナログ ─────────────────────────────────────
        self._io = InferiorOliveAnalog(
            dt=dt,
            firing_rate_hz=config.io_firing_rate_hz,
            gain=config.io_gain,
            mode=config.io_mode,
            seed=seed,
        )

        # ── M1 逆モデル ──────────────────────────────────────────────
        self._m1 = MotorCortexM1(
            n_joints=n,
            kp=config.kp,
            kd=config.kd,
            efcopy_enabled=config.efcopy_enabled,
            efcopy_noise_std=config.efcopy_noise_std,
            inverse_model_loc=config.inverse_model_loc,
            seed=seed,
        )

        # ── 小脳サイドループ ─────────────────────────────────────────
        cfc = CfCForwardModel(
            n_joints=n,
            hidden_units=config.cfc_hidden_units,
            device=config.cfc_device,
        )
        self._cerebellum = CerebellarSideLoop(
            cfc=cfc,
            io_analog=self._io,
            cereb_delay_buf=self._cereb_delay,
        )

        # ── 脊髄層（共通） ───────────────────────────────────────────
        # IaIbReflexArc は latency_steps=1（実質 0 遅延）で使用。
        # 固有受容遅延は prop_delay_buf で既に適用済みなので追加遅延不要。
        # latency_steps=0 だと内部バッファが空になり IndexError になるため 1 を指定。
        self._reflex = IaIbReflexArc(n_joints=n, dt=dt, ia_latency_steps=1,
                                     ib_latency_steps=1)
        self._cpg = MatsuokaOscillator(n_joints=n, dt=dt, alpha_fb=0.0,
                                       amplitude=0.0)
        self._lif = LIFProprioceptor(n_joints=n, dt=dt)
        self._cc  = VirtualCocontraction(n_joints=n)

        # 内部ステート -----------------------------------------------------------
        self._tau_cereb_delayed: np.ndarray = np.zeros(n)
        self._q_del:   np.ndarray = np.zeros(n)
        self._dq_del:  np.ndarray = np.zeros(n)
        self._efcopy:  np.ndarray = np.zeros(n)

    # ------------------------------------------------------------------
    # CfC モデルの読み込み
    # ------------------------------------------------------------------

    def load_cfc(self, path: str | Path) -> None:
        """事前学習済み CfC モデルをロードする。"""
        self._cerebellum.cfc.load(str(path))

    # ------------------------------------------------------------------
    # エピソードリセット
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._prop_delay_q.reset()
        self._prop_delay_dq.reset()
        self._cereb_delay.reset()
        self._cerebellum.reset()
        self._m1.reset()
        self._reflex.reset()
        self._cpg.reset()
        self._lif.reset()
        self._cc.reset()
        self._tau_cereb_delayed[:] = 0.0
        self._q_del[:] = 0.0
        self._dq_del[:] = 0.0
        self._efcopy[:] = 0.0

    # ------------------------------------------------------------------
    # メインステップ（env.step() の前に呼ぶ）
    # ------------------------------------------------------------------

    def step(
        self,
        q:        np.ndarray,
        dq:       np.ndarray,
        q_target: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """
        1 ステップ実行してトルク指令を返す。
        env.step() の前に呼ぶこと。

        Returns
        -------
        tau_total : 関節トルク指令 (n_joints,) [Nm]
        info      : ロギング用辞書
        """
        n = self.N_JOINTS

        # ── 1. 固有受容遅延 ──────────────────────────────────────────
        self._q_del  = self._prop_delay_q.push_and_get(q)
        self._dq_del = self._prop_delay_dq.push_and_get(dq)

        # ── 2. M1: 小脳フィードバックを受け取ってから指令を生成 ──────
        self._m1.receive_cerebellar_feedback(self._tau_cereb_delayed)
        tau_cortical, tau_efcopy = self._m1.step(q_target, q, dq)
        self._efcopy = tau_efcopy.copy()

        # ── 3. 小脳: 前向きモデル予測（env.step() 前） ───────────────
        self._cerebellum.predict(self._q_del, self._dq_del, tau_efcopy)

        # ── 4. 脊髄反射弓（遅延済み固有受容情報を入力） ─────────────
        tau_pd_approx = self.config.kp * (q_target - self._q_del) \
                        + self.config.kd * (-self._dq_del)
        tau_ia, tau_ib = self._reflex.step(self._dq_del, tau_pd_approx)
        tau_reflex = tau_ia + tau_ib

        # ── 5. LIF 固有受容器 → CPG フィードバック ───────────────────
        spike_rates = self._lif.encode(self._q_del, self._dq_del)
        tau_cpg     = self._cpg.step(r_q=spike_rates)

        # ── 6. 仮想コ・コントラクション ──────────────────────────────
        tau_impedance, _ = self._cc.step(q, dq, q_target)

        # ── 7. トルク合算・クリッピング ──────────────────────────────
        tau_total = tau_cortical + tau_reflex + tau_cpg + tau_impedance
        tau_total = np.clip(tau_total, -_TAU_LIMIT, _TAU_LIMIT)

        info = {
            "tau_cortical":      tau_cortical,
            "tau_cereb_delayed": self._tau_cereb_delayed.copy(),
            "tau_reflex":        tau_reflex,
            "tau_cpg":           tau_cpg,
            "tau_impedance":     tau_impedance,
            "tau_efcopy":        tau_efcopy,
            "q_del":             self._q_del.copy(),
            "dq_del":            self._dq_del.copy(),
            "prop_delay_steps":  self.config.prop_delay_steps,
            "cereb_delay_steps": self.config.cereb_delay_steps,
            "io_fired":          False,   # update_cerebellum() で更新される
        }

        return tau_total, info

    # ------------------------------------------------------------------
    # 小脳更新（env.step() の後に呼ぶ）
    # ------------------------------------------------------------------

    def update_cerebellum(self, q_actual: np.ndarray) -> None:
        """
        env.step() 後の実際の次状態で小脳を更新する。
        必ず env.step() の直後に呼ぶこと。

        Parameters
        ----------
        q_actual : env.step() 後の実際の関節角 (n_joints,) [rad]
        """
        tau_delayed = self._cerebellum.update_and_get_correction(q_actual)
        self._tau_cereb_delayed = tau_delayed.copy()

    # ------------------------------------------------------------------
    # ロギング補助
    # ------------------------------------------------------------------

    def get_io_stats(self) -> dict:
        """IO 発火統計を返す（metrics.json 用）。"""
        return self._io.get_stats()

    def get_pred_error(self) -> np.ndarray:
        return self._cerebellum.get_pred_error()

    def get_cereb_correction(self) -> np.ndarray:
        return self._tau_cereb_delayed.copy()
