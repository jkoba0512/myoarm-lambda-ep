"""
LIF（Leaky Integrate-and-Fire）固有受容器

生物的対応:
  Ia 求心性線維（筋紡錘）→ 関節角・角速度をスパイク列に変換

モデル:
  τ_m · dV/dt = -(V - V_rest) + I(t)
  V ≥ V_th → スパイク発火、V ← V_reset（t_ref 間は不応期）

符号化方式:
  レート符号化 (rate coding)
    - 発火率 = スライディングウィンドウ内のスパイク数 / ウィンドウ幅
    - 符号（関節が正方向か負方向か）は元信号から保持
    - 出力 r_q, r_dq ∈ [-1, 1]（正規化済み）

入力電流:
  I_q[i]  = I_bias + k_q  · (q[i] - q_mid[i]) / q_half_range[i]
  I_dq[i] = I_bias + k_dq · dq[i] / dq_max

Note:
  - 位置: 静止姿勢 q_rest からの偏差 |q - q_rest| を符号化
    k_q [rad^-1] を使って偏差 0.2 rad → I=1.0（発火閾値）となるよう設定
  - 速度: |dq| を直接 k_dq で符号化（1 rad/s → I≈1.0）
  - v_rest=0, v_th=1 の無次元 LIF モデルを採用
"""

from __future__ import annotations

import numpy as np


class LIFProprioceptor:
    """
    n_joints 個の関節に対する LIF 固有受容器。

    各関節に「位置ニューロン」「速度ニューロン」の 2 種類を持つ。
    合計 2 × n_joints 個の LIF ニューロン。

    Parameters
    ----------
    n_joints   : 関節数
    tau_m      : 膜時定数 [s]
    v_rest     : 静止電位 [mV]
    v_th       : 発火閾値 [mV]
    v_reset    : リセット電位 [mV]
    t_ref      : 不応期 [s]
    dt         : タイムステップ [s]
    k_q        : 位置→電流ゲイン（V_th - V_rest = 16 で最大偏差→境界発火）
    k_dq       : 速度→電流ゲイン
    I_bias     : バイアス電流（静止時の基礎発火を設定する場合）
    dq_max     : 速度の正規化上限 [rad/s]
    q_range    : 関節可動域 (n_joints, 2)。None なら ±π を仮定。
    window     : レート推定ウィンドウ幅 [ステップ数]
    """

    def __init__(
        self,
        n_joints: int = 5,
        tau_m:    float = 0.010,
        v_rest:   float = 0.0,    # 無次元化: 静止電位 = 0
        v_th:     float = 1.0,    # 無次元化: 閾値 = 1
        v_reset:  float = 0.0,    # 無次元化: リセット = 0
        t_ref:    float = 0.004,  # 不応期 [s]
        dt:       float = 0.002,
        k_q:      float = 5.0,    # [rad^-1]: 偏差 0.2 rad → I=1.0（閾値）
        k_dq:     float = 1.5,    # [s/rad]: 速度 0.7 rad/s → I=1.0（閾値）
        I_bias:   float = 0.0,
        q_rest:   np.ndarray | None = None,   # 静止姿勢 (n,) [rad]。None なら 0。
        q_range:  np.ndarray | None = None,   # 可動域 (n,2)。出力クリップ用。
        window:   int = 20,
    ):
        self.n      = n_joints
        self.tau_m  = tau_m
        self.v_rest = v_rest
        self.v_th   = v_th
        self.v_reset = v_reset
        self.t_ref  = t_ref
        self.dt     = dt
        self.k_q    = k_q
        self.k_dq   = k_dq
        self.I_bias = I_bias
        self.window = window

        # 静止姿勢（偏差計算の基準）
        self.q_rest = np.zeros(n_joints) if q_rest is None else np.array(q_rest)

        # 可動域（出力スケーリング用のみ）
        if q_range is None:
            q_range = np.tile([-np.pi, np.pi], (n_joints, 1))
        self.q_half_range = (q_range[:, 1] - q_range[:, 0]) / 2.0  # (n,)

        # LIF 状態
        self.V_q   = np.full(n_joints, v_rest)   # 位置ニューロン膜電位
        self.V_dq  = np.full(n_joints, v_rest)   # 速度ニューロン膜電位
        self.ref_q  = np.zeros(n_joints)          # 残り不応期 [s]
        self.ref_dq = np.zeros(n_joints)

        # スパイク履歴バッファ（レート推定用）
        self._buf_q  = np.zeros((window, n_joints), dtype=np.int8)
        self._buf_dq = np.zeros((window, n_joints), dtype=np.int8)
        self._buf_idx = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """エピソード開始時に状態をリセット。"""
        self.V_q[:]   = self.v_rest
        self.V_dq[:]  = self.v_rest
        self.ref_q[:]  = 0.0
        self.ref_dq[:] = 0.0
        self._buf_q[:] = 0
        self._buf_dq[:] = 0
        self._buf_idx = 0

    # ------------------------------------------------------------------
    def encode(
        self,
        q:  np.ndarray,
        dq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        1 ステップ進め、符号付き正規化発火率を返す。

        Parameters
        ----------
        q  : 関節角 (n,) [rad]
        dq : 関節速度 (n,) [rad/s]

        Returns
        -------
        r_q  : 位置発火率 ∈ [-1, 1] (n,)
        r_dq : 速度発火率 ∈ [-1, 1] (n,)
        """
        # --- 入力電流（絶対偏差ベース）---
        q_dev = q - self.q_rest                               # 静止姿勢からの偏差 [rad]
        I_q  = self.I_bias + self.k_q  * np.abs(q_dev)       # 偏差 0.2 rad → I=1.0
        I_dq = self.I_bias + self.k_dq * np.abs(dq)          # 速度 0.7 rad/s → I=1.0

        # --- LIF 1 ステップ ---
        spiked_q,  self.V_q,  self.ref_q  = self._lif_step(self.V_q,  self.ref_q,  I_q)
        spiked_dq, self.V_dq, self.ref_dq = self._lif_step(self.V_dq, self.ref_dq, I_dq)

        # --- スパイク履歴更新 ---
        self._buf_q [self._buf_idx] = spiked_q.astype(np.int8)
        self._buf_dq[self._buf_idx] = spiked_dq.astype(np.int8)
        self._buf_idx = (self._buf_idx + 1) % self.window

        # --- レート計算（発火率 ∈ [0, 1]）---
        rate_q  = self._buf_q.sum(axis=0)  / self.window    # (n,)
        rate_dq = self._buf_dq.sum(axis=0) / self.window

        # --- 符号付き出力 ---
        r_q  = rate_q  * np.sign(q_dev + 1e-9)
        r_dq = rate_dq * np.sign(dq + 1e-9)

        return r_q, r_dq

    # ------------------------------------------------------------------
    def _lif_step(
        self,
        V:   np.ndarray,
        ref: np.ndarray,
        I:   np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        LIF 1 ステップ（Euler 法）。

        Parameters
        ----------
        V   : 現在の膜電位 (n,)
        ref : 残り不応期 [s] (n,)
        I   : 入力電流 (n,)

        Returns
        -------
        spiked : 発火フラグ (n, bool)
        V_new  : 更新後の膜電位 (n,)
        ref_new: 更新後の不応期 (n,)
        """
        in_ref = ref > 0.0

        # 膜電位更新（不応期中はリセット電位に固定）
        dV = (-(V - self.v_rest) + I) * self.dt / self.tau_m
        V_new = np.where(in_ref, self.v_reset, V + dV)

        # 発火判定（不応期中は発火しない）
        spiked = (~in_ref) & (V_new >= self.v_th)

        # リセット
        V_new  = np.where(spiked, self.v_reset, V_new)
        ref_new = np.where(spiked, self.t_ref, np.maximum(ref - self.dt, 0.0))

        return spiked, V_new, ref_new

    # ------------------------------------------------------------------
    def get_spike_raster(self) -> dict:
        """
        可視化用: スパイク履歴バッファを返す。

        Returns
        -------
        dict with keys:
          "q"   : (window, n_joints) int8 スパイク列（位置）
          "dq"  : (window, n_joints) int8 スパイク列（速度）
          "head": 現在のバッファインデックス
        """
        return {
            "q":    self._buf_q.copy(),
            "dq":   self._buf_dq.copy(),
            "head": self._buf_idx,
        }
