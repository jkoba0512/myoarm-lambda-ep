"""
CfC 順動力学モデル（小脳 Forward model）

旧 CfCGravityCompensator との違い:
  旧: (q, dq)          → τ    逆動力学。各ステップ独立 → MLP と等価
  新: (q, dq, τ_cmd)   → Δq   順動力学。時系列メモリが Δq 予測に直接寄与

CfC 液体特性の意義:
  入力依存の時定数 τ_CfC により、外乱後（Δq 大）は高速適応、
  定常保持時（Δq ≈ 0）は緩慢な積分となる。生物学的小脳の挙動に対応。

ts の扱い:
  現行は ts=1.0 固定（ncps batch_size>1 バグ回避）。
  ts=dt/T_ref (T_ref=1.0s) への切り替えは ncps バグ修正後に実施する。

制御ループでの使い方:
  # env.step() の前:
  q_hat      = cerebellum.predict(q, dq, tau_efference)
  tau_cereb  = cerebellum.get_correction()   # 前ステップの誤差から生成

  # env.step() の後:
  cerebellum.update(q_actual)   # 予測誤差を計算し次ステップの補正を更新
"""

from __future__ import annotations
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ncps.torch import CfC
from ncps.wirings import AutoNCP


# デフォルト補正ゲイン [Nm/rad]（Kp の約 0.5 倍）
_K_CEREB_DEFAULT = np.array([25.0, 25.0, 25.0, 25.0, 5.0, 5.0, 5.0])


class CfCForwardModel:
    """
    小脳 Forward model: (q_t, dq_t, τ_t) → Δq_hat = q(t+1) − q(t)

    Parameters
    ----------
    n_joints        : 関節数
    hidden_units    : AutoNCP の units 数
    device          : 'cpu' or 'cuda'
    K_cereb         : 予測誤差 → 補正トルク変換ゲイン (n_joints,) [Nm/rad]
    online_lr       : オンライン学習率（0.0 = 無効）
    online_interval : オンライン重み更新を行うステップ間隔
    """

    def __init__(
        self,
        n_joints:        int              = 7,
        hidden_units:    int              = 64,
        device:          str              = "cpu",
        K_cereb:         np.ndarray | None = None,
        online_lr:       float            = 0.0,
        online_interval: int              = 20,
    ):
        self.n_joints = n_joints
        self.device   = torch.device(device)

        wiring     = AutoNCP(units=hidden_units, output_size=n_joints)
        self.model = CfC(input_size=n_joints * 3, units=wiring).to(self.device)
        self.h: torch.Tensor | None = None

        self.K_cereb = (_K_CEREB_DEFAULT.copy() if K_cereb is None
                        else np.asarray(K_cereb, dtype=float).copy())

        self.online_lr       = online_lr
        self.online_interval = online_interval
        self._online_opt: torch.optim.Optimizer | None = (
            torch.optim.Adam(self.model.parameters(), lr=online_lr)
            if online_lr > 0 else None
        )

        # 正規化パラメータ（fit() で設定）
        self._x_mean = np.zeros(n_joints * 3)
        self._x_std  = np.ones(n_joints * 3)
        self._y_mean = np.zeros(n_joints)
        self._y_std  = np.ones(n_joints)

        # ステップ間バッファ（predict() → update() で使用）
        self._last_q:         np.ndarray | None = None
        self._last_delta_hat: np.ndarray | None = None
        self._last_x_norm:    np.ndarray | None = None
        self._correction:     np.ndarray        = np.zeros(n_joints)
        self._pred_error:     np.ndarray        = np.zeros(n_joints)
        self._step_count:     int               = 0

        # τ_sys 診断フック
        self._t_interp_buf:  list[float] = []
        self._last_t_interp: float | None = None
        self._hooks_active:  bool         = True
        self._hooks:         list         = []
        self._register_tau_hooks()

    # ------------------------------------------------------------------
    # τ_sys 診断フック
    # ------------------------------------------------------------------

    def _register_tau_hooks(self) -> None:
        def make_hook(ref: "CfCForwardModel"):
            def hook(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
                if ref._hooks_active:
                    ref._t_interp_buf.append(out.detach().cpu().mean().item())
            return hook
        for cell in self.model.rnn_cell._layers:
            h = cell.sigmoid.register_forward_hook(make_hook(self))
            self._hooks.append(h)

    # ------------------------------------------------------------------
    # エピソードリセット
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.h                = None
        self._last_q          = None
        self._last_delta_hat  = None
        self._last_x_norm     = None
        self._correction[:]   = 0.0
        self._pred_error[:]   = 0.0
        self._step_count      = 0
        self._t_interp_buf.clear()
        self._last_t_interp = None

    # ------------------------------------------------------------------
    # 推論 API
    # ------------------------------------------------------------------

    def get_correction(self) -> np.ndarray:
        """前ステップの予測誤差から計算した補正トルク τ_cereb を返す [Nm]。"""
        return self._correction.copy()

    def get_prediction_error(self) -> np.ndarray:
        """直前ステップの予測誤差 Δq_actual − Δq_hat [rad]。"""
        return self._pred_error.copy()

    def get_tau_sys(self) -> float | None:
        return self._last_t_interp

    def predict(
        self,
        q:             np.ndarray,
        dq:            np.ndarray,
        tau_efference: np.ndarray,
    ) -> np.ndarray:
        """
        (q_t, dq_t, τ_efference_t) から q̂(t+1) を予測して返す。

        update() が呼ばれると予測誤差から補正トルクを更新する。
        env.step() の前に呼ぶこと。

        Returns
        -------
        q_hat : 予測次状態 q(t+1) [rad]
        """
        self._t_interp_buf.clear()
        self._last_q = q.copy()

        x_raw  = np.concatenate([q, dq, tau_efference])
        x_norm = (x_raw - self._x_mean) / self._x_std
        self._last_x_norm = x_norm.copy()

        x = torch.tensor(x_norm, dtype=torch.float32, device=self.device)
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, 3n)

        self.model.eval()
        with torch.no_grad():
            delta_norm, self.h = self.model(x, self.h)

        if self._t_interp_buf:
            self._last_t_interp = float(np.mean(self._t_interp_buf))

        delta_np = delta_norm.squeeze().cpu().numpy()
        self._last_delta_hat = delta_np * self._y_std + self._y_mean

        return q + self._last_delta_hat  # q̂(t+1)

    def update(self, q_actual: np.ndarray,
               allow_online_update: bool = True) -> None:
        """
        実際の次状態 q_actual を受け取り、補正トルクを更新する。
        env.step() の直後に呼ぶこと。

        Parameters
        ----------
        q_actual            : env.step() 後の実際の関節角 q(t+1) [rad]
        allow_online_update : False にするとオンライン学習をスキップ（IO 発火ゲート用）
        """
        if self._last_q is None or self._last_delta_hat is None:
            return

        delta_actual    = q_actual - self._last_q
        error           = delta_actual - self._last_delta_hat
        self._pred_error[:] = error
        self._correction[:] = self.K_cereb * error

        if allow_online_update and \
                self._online_opt is not None and self._last_x_norm is not None:
            self._step_count += 1
            if self._step_count % self.online_interval == 0:
                self._online_update(delta_actual)

    def _online_update(self, delta_actual: np.ndarray) -> None:
        """直前の入力と実際の Δq で 1 回だけ勾配更新する。"""
        y_t = torch.tensor(
            (delta_actual - self._y_mean) / self._y_std,
            dtype=torch.float32, device=self.device,
        )
        x_t = torch.tensor(
            self._last_x_norm, dtype=torch.float32, device=self.device,
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, 3n)

        self.model.train()
        self._online_opt.zero_grad()
        pred, _ = self.model(x_t, None)
        loss = nn.MSELoss()(pred.squeeze(), y_t)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self._online_opt.step()
        self.model.eval()

    # ------------------------------------------------------------------
    # 訓練データ収集
    # ------------------------------------------------------------------

    @staticmethod
    def collect_forward_data(
        env,
        n_trajectories: int                      = 200,
        seq_len:        int                      = 50,
        rng:            np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        実際の env.step() を使い遷移データ (q_t, dq_t, τ_t) → q_{t+1} を収集する。
        PD ベースのトルク＋探索ノイズで多様な軌道を生成する。

        Returns
        -------
        q_seqs, dq_seqs, tau_seqs : (N, T, n_joints)
        q_next_seqs               : (N, T, n_joints)  ノイズなし次関節角
        """
        if rng is None:
            rng = np.random.default_rng(42)

        from common.franka_env import TAU_LIMIT, N_JOINTS

        jnt_range = env.ctrl_range
        n  = N_JOINTS
        kp = np.array([50., 50., 50., 50., 10., 10., 10.])
        kd = np.array([ 7.,  7.,  7.,  7.,  1.5, 1.5, 1.5])

        q_seqs, dq_seqs, tau_seqs, q_next_seqs = [], [], [], []

        for _ in range(n_trajectories):
            q0 = rng.uniform(jnt_range[:, 0] * 0.5, jnt_range[:, 1] * 0.5)
            env.reset(q0=q0)
            q_tgt = rng.uniform(jnt_range[:, 0] * 0.5, jnt_range[:, 1] * 0.5)

            q_seq, dq_seq, tau_seq, q_next_seq = [], [], [], []
            for step in range(seq_len):
                q_cur, dq_cur = env.get_state()

                tau = kp * (q_tgt - q_cur) + kd * (-dq_cur)
                tau += rng.normal(0.0, 2.0, n)
                tau = np.clip(tau, -TAU_LIMIT * 0.5, TAU_LIMIT * 0.5)

                q_seq.append(q_cur.copy())
                dq_seq.append(dq_cur.copy())
                tau_seq.append(tau.copy())

                env.step(tau)
                q_next_seq.append(env._raw_state()[0].copy())

                if step % 15 == 14:
                    q_tgt = rng.uniform(jnt_range[:, 0] * 0.5, jnt_range[:, 1] * 0.5)

            q_seqs.append(q_seq)
            dq_seqs.append(dq_seq)
            tau_seqs.append(tau_seq)
            q_next_seqs.append(q_next_seq)

        return (
            np.array(q_seqs,       dtype=np.float32),
            np.array(dq_seqs,      dtype=np.float32),
            np.array(tau_seqs,     dtype=np.float32),
            np.array(q_next_seqs,  dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # 訓練
    # ------------------------------------------------------------------

    def fit(
        self,
        q_seqs:      np.ndarray,
        dq_seqs:     np.ndarray,
        tau_seqs:    np.ndarray,
        q_next_seqs: np.ndarray,
        n_epochs:    int   = 300,
        batch_size:  int   = 32,
        lr:          float = 1e-3,
        verbose:     bool  = True,
        seed:        int   = 42,
    ) -> list[float]:
        """
        Forward model を教師あり学習する。
        目標: Δq = q(t+1) − q(t) の正規化残差予測（ts=1.0 固定）。
        """
        N, T, _ = q_seqs.shape
        delta_seqs = q_next_seqs - q_seqs  # (N, T, n)

        X_raw = np.concatenate([q_seqs, dq_seqs, tau_seqs], axis=-1)
        Y_raw = delta_seqs

        self._x_mean = X_raw.reshape(-1, self.n_joints * 3).mean(axis=0)
        self._x_std  = X_raw.reshape(-1, self.n_joints * 3).std(axis=0) + 1e-8
        self._y_mean = Y_raw.reshape(-1, self.n_joints).mean(axis=0)
        self._y_std  = Y_raw.reshape(-1, self.n_joints).std(axis=0) + 1e-8

        X = torch.tensor((X_raw - self._x_mean) / self._x_std, dtype=torch.float32)
        Y = torch.tensor((Y_raw - self._y_mean) / self._y_std, dtype=torch.float32)

        _gen   = torch.Generator().manual_seed(seed)
        loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size,
                            shuffle=True, generator=_gen)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * 0.01
        )
        loss_fn      = nn.MSELoss()
        loss_history = []

        self._hooks_active = False
        self.model.train()
        try:
            for epoch in range(1, n_epochs + 1):
                epoch_loss = 0.0
                for xb, yb in loader:
                    xb = xb.to(self.device); yb = yb.to(self.device)
                    optimizer.zero_grad()
                    pred, _ = self.model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item() * len(xb)
                scheduler.step()
                epoch_loss /= N
                loss_history.append(epoch_loss)
                if verbose and epoch % 50 == 0:
                    print(f"  Epoch {epoch:4d}/{n_epochs}  loss={epoch_loss:.6f}")
        finally:
            self._hooks_active = True

        return loss_history

    # ------------------------------------------------------------------
    # 保存 / 読み込み
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "model_state": self.model.state_dict(),
            "x_mean":  self._x_mean.tolist(),
            "x_std":   self._x_std.tolist(),
            "y_mean":  self._y_mean.tolist(),
            "y_std":   self._y_std.tolist(),
            "K_cereb": self.K_cereb.tolist(),
        }, path)
        print(f"Saved CfC Forward model → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        self._x_mean = np.array(ckpt["x_mean"])
        self._x_std  = np.array(ckpt["x_std"])
        self._y_mean = np.array(ckpt["y_mean"])
        self._y_std  = np.array(ckpt["y_std"])
        if "K_cereb" in ckpt:
            self.K_cereb = np.array(ckpt["K_cereb"])
        print(f"Loaded CfC Forward model ← {path}")
