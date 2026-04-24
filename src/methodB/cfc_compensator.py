"""
CfC 重力・摩擦補償器（小脳型逆モデル）

修正点:
  [Fix 1] τ_sys アクセス: CfCCell.sigmoid に forward hook を登録し
          t_interp = σ(t_a·Δt + t_b) を記録（LTC の τ_sys に相当）
  [Fix 2] ts の一貫性: ncps は batch_size>1 で timespans を渡すとブロードキャスト
          エラーが発生するバグがある。ts=1.0（デフォルト）を訓練・推論で統一する。
          物理的な dt は訓練データの時系列に暗黙的にエンコードされているため
          ts の絶対値は不変であれば良い。
  [Fix 3] シーケンス訓練: T=1 独立サンプルでなく連続軌道シーケンスで訓練し
          訓練・推論の隠れ状態伝播を一致させる
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ncps.torch import CfC
from ncps.wirings import AutoNCP


class CfCGravityCompensator:
    """
    Parameters
    ----------
    n_joints     : アクティブ関節数
    hidden_units : AutoNCP の units 数
    device       : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        n_joints: int = 3,
        hidden_units: int = 32,
        device: str = "cpu",
    ):
        self.n_joints = n_joints
        self.device = torch.device(device)

        wiring = AutoNCP(units=hidden_units, output_size=n_joints)
        self.model = CfC(input_size=n_joints * 2, units=wiring).to(self.device)
        self.h: torch.Tensor | None = None

        # 正規化パラメータ（訓練後に設定）
        self._x_mean = np.zeros(n_joints * 2)
        self._x_std = np.ones(n_joints * 2)
        self._y_mean = np.zeros(n_joints)
        self._y_std = np.ones(n_joints)

        # [Fix 1] τ_sys フック
        self._hooks_active = True   # 訓練中は False に切り替える
        self._t_interp_buf: list[float] = []
        self._last_t_interp: float | None = None
        self._hooks: list = []
        self._register_tau_hooks()

    # ------------------------------------------------------------------
    # τ_sys フック（Fix 1）
    # ------------------------------------------------------------------

    def _register_tau_hooks(self) -> None:
        """
        WiredCfCCell 内の各 CfCCell.sigmoid に forward フックを登録する。
        CfCCell.forward では
            t_interp = sigmoid(t_a * ts + t_b)
        として sigmoid が 1 回だけ呼ばれるため、このフックで t_interp を取得できる。
        t_interp ∈ (0,1) は LTC の τ_sys に対応する実効適応レート：
          値が大きい → 高速応答（小さい τ_sys）
          値が小さい → 緩慢応答（大きい τ_sys）
        """
        def make_hook(self_ref: "CfCGravityCompensator"):
            def hook(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
                if self_ref._hooks_active:
                    # 全ニューロンの t_interp 平均をバッファに追記
                    self_ref._t_interp_buf.append(
                        out.detach().cpu().mean().item()
                    )
            return hook

        for cell in self.model.rnn_cell._layers:
            h = cell.sigmoid.register_forward_hook(make_hook(self))
            self._hooks.append(h)

    # ------------------------------------------------------------------
    # 推論（Fix 1, Fix 2）
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """エピソード開始時に隠れ状態と τ_sys バッファをリセット。"""
        self.h = None
        self._t_interp_buf.clear()
        self._last_t_interp = None

    def predict(self, q: np.ndarray, dq: np.ndarray, dt: float = 0.004) -> np.ndarray:
        """
        1 ステップの補償トルクを返す。隠れ状態 h を更新する。

        Parameters
        ----------
        q, dq : 関節角・速度 (n_joints,)
        dt    : 引数として受け取るが timespans には使わない（[Fix 2] 参照）
        """
        # [Fix 2] timespans=None (ts=1.0) で統一。訓練と一致させる。
        self._t_interp_buf.clear()

        x_raw = np.concatenate([q, dq])
        x_norm = (x_raw - self._x_mean) / self._x_std
        x = torch.tensor(x_norm, dtype=torch.float32, device=self.device)
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, 6)

        self.model.eval()
        with torch.no_grad():
            y_norm, self.h = self.model(x, self.h)  # ts=1.0 default

        # [Fix 1] t_interp を記録
        if self._t_interp_buf:
            self._last_t_interp = float(np.mean(self._t_interp_buf))

        y_np = y_norm.squeeze().cpu().numpy()
        return y_np * self._y_std + self._y_mean

    def get_tau_sys(self) -> float | None:
        """
        直前の predict() での t_interp 平均値を返す。
        t_interp = σ(t_a·dt + t_b) ∈ (0,1)
          → 大: 高速応答（把持・外乱時）
          → 小: 緩慢応答（自由運動時）
        """
        return self._last_t_interp

    # ------------------------------------------------------------------
    # 訓練データ収集（Fix 3: シーケンスデータ）
    # ------------------------------------------------------------------

    @staticmethod
    def collect_sequence_data(
        env,
        n_trajectories: int = 150,
        seq_len: int = 30,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ランダム正弦波軌道から時系列 (q, dq, τ_inv) を収集する。
        mj_inverse に ddq を渡して動的補償トルクを取得する。

        Returns
        -------
        q_seqs, dq_seqs, tau_seqs : shape (N_traj, seq_len, n_joints)
        """
        if rng is None:
            rng = np.random.default_rng(42)

        dt = env.dt
        n = env.ctrl_range.shape[0]  # active joints (env に合わせて自動取得)
        jnt_range = env.ctrl_range   # (n, 2)

        q_seqs, dq_seqs, tau_seqs = [], [], []

        for _ in range(n_trajectories):
            # ランダム軌道パラメータ
            A = rng.uniform(0.1, 0.5, n)
            f = rng.uniform(0.2, 1.5, n)
            phi = rng.uniform(0.0, 2 * np.pi, n)
            q0 = rng.uniform(jnt_range[:, 0] * 0.5, jnt_range[:, 1] * 0.5)
            omega = 2 * np.pi * f

            q_seq, dq_seq, tau_seq = [], [], []
            for step in range(seq_len):
                t = step * dt
                q = q0 + A * np.sin(omega * t + phi)
                dq = A * omega * np.cos(omega * t + phi)
                ddq = -A * omega**2 * np.sin(omega * t + phi)
                q = np.clip(q, jnt_range[:, 0], jnt_range[:, 1])

                tau = env.inverse_dynamics(q, dq, ddq)
                q_seq.append(q)
                dq_seq.append(dq)
                tau_seq.append(tau)

            q_seqs.append(q_seq)
            dq_seqs.append(dq_seq)
            tau_seqs.append(tau_seq)

        return (
            np.array(q_seqs, dtype=np.float32),    # (N, T, 3)
            np.array(dq_seqs, dtype=np.float32),
            np.array(tau_seqs, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # 訓練（Fix 2, Fix 3）
    # ------------------------------------------------------------------

    def fit(
        self,
        q_seqs: np.ndarray,
        dq_seqs: np.ndarray,
        tau_seqs: np.ndarray,
        dt: float = 0.004,  # 現在は未使用（[Fix 2]: ts=1.0 で統一）
        n_epochs: int = 200,
        batch_size: int = 32,
        lr: float = 1e-3,
        verbose: bool = True,
        seed: int = 42,
    ) -> list[float]:
        """
        シーケンスデータで教師あり学習（Fix 3）。
        ts=1.0（default）で訓練し推論と一致させる（Fix 2）。

        Parameters
        ----------
        q_seqs, dq_seqs, tau_seqs : shape (N, T, n_joints)
        """
        N, T, _ = q_seqs.shape

        X_raw = np.concatenate([q_seqs, dq_seqs], axis=-1)  # (N, T, 6)
        Y_raw = tau_seqs                                      # (N, T, 3)

        # 正規化パラメータをシーケンス全体から計算
        self._x_mean = X_raw.reshape(-1, self.n_joints * 2).mean(axis=0)
        self._x_std = X_raw.reshape(-1, self.n_joints * 2).std(axis=0) + 1e-8
        self._y_mean = Y_raw.reshape(-1, self.n_joints).mean(axis=0)
        self._y_std = Y_raw.reshape(-1, self.n_joints).std(axis=0) + 1e-8

        X = torch.tensor(
            (X_raw - self._x_mean) / self._x_std, dtype=torch.float32
        )  # (N, T, 6)
        Y = torch.tensor(
            (Y_raw - self._y_mean) / self._y_std, dtype=torch.float32
        )  # (N, T, 3)

        dataset = TensorDataset(X, Y)
        _gen = torch.Generator().manual_seed(seed)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=_gen)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * 0.01
        )
        loss_fn = nn.MSELoss()
        loss_history = []

        # [Fix 1] 訓練中はフックを無効化
        # [Fix 2] timespans=None (ts=1.0) で推論と一致
        self._hooks_active = False
        self.model.train()
        try:
            for epoch in range(1, n_epochs + 1):
                epoch_loss = 0.0
                for xb, yb in loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)

                    optimizer.zero_grad()
                    pred, _ = self.model(xb)  # ts=1.0 default, 推論と一致
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
            self._hooks_active = True  # 必ず復元

        return loss_history

    # ------------------------------------------------------------------
    # 保存 / 読み込み
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        # numpy 配列を float リストに変換して weights_only 互換で保存
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "x_mean": self._x_mean.tolist(),
                "x_std":  self._x_std.tolist(),
                "y_mean": self._y_mean.tolist(),
                "y_std":  self._y_std.tolist(),
            },
            path,
        )
        print(f"Saved CfC compensator → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        self._x_mean = np.array(ckpt["x_mean"])
        self._x_std  = np.array(ckpt["x_std"])
        self._y_mean = np.array(ckpt["y_mean"])
        self._y_std  = np.array(ckpt["y_std"])
        print(f"Loaded CfC compensator ← {path}")
