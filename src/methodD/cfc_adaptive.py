"""
CfC 適応層（手法 D: CPG + CfC）

CPG が生成した基本リズムに対して、感覚フィードバックから
位相・振幅補正を生成する CfC ネットワーク。

入力:  [q(n), dq(n), cpg_state(2n)] = 4n 次元
出力:  correction(n) [rad]  ← CPG 出力に加算する位置補正

訓練:
  CPG 単独で動いたときの残差誤差を収集し、
  その誤差を 0 に近づける補正を学習する（教師あり）。

推論:
  オンラインで隠れ状態 h を伝播しながら補正値を出力。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ncps.torch import CfC
from ncps.wirings import AutoNCP


class CfCAdaptiveLayer:
    """
    Parameters
    ----------
    n_joints     : 制御する関節数
    hidden_units : AutoNCP の units
    device       : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        n_joints:     int = 5,
        hidden_units: int = 32,
        device:       str = "cpu",
    ):
        self.n_joints = n_joints
        self.device = torch.device(device)

        # 入力: [q, dq, cpg_x1, cpg_x2] = 4 * n_joints
        input_size = 4 * n_joints
        wiring = AutoNCP(units=hidden_units, output_size=n_joints)
        self.model = CfC(input_size=input_size, units=wiring).to(self.device)
        self.h: torch.Tensor | None = None

        self._x_mean = np.zeros(input_size)
        self._x_std  = np.ones(input_size)
        self._y_mean = np.zeros(n_joints)
        self._y_std  = np.ones(n_joints)

        # τ_sys フック（Fix 1 と同様）
        self._hooks_active = True
        self._t_interp_buf: list[float] = []
        self._last_t_interp: float | None = None
        self._hooks: list = []
        self._register_tau_hooks()

    # ------------------------------------------------------------------
    # τ_sys フック
    # ------------------------------------------------------------------

    def _register_tau_hooks(self) -> None:
        def make_hook(ref: "CfCAdaptiveLayer"):
            def hook(m: nn.Module, inp: tuple, out: torch.Tensor) -> None:
                if ref._hooks_active:
                    ref._t_interp_buf.append(out.detach().cpu().mean().item())
            return hook

        for cell in self.model.rnn_cell._layers:
            h = cell.sigmoid.register_forward_hook(make_hook(self))
            self._hooks.append(h)

    # ------------------------------------------------------------------
    # 推論
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.h = None
        self._t_interp_buf.clear()
        self._last_t_interp = None

    def predict(
        self,
        q:         np.ndarray,
        dq:        np.ndarray,
        cpg_state: np.ndarray,
    ) -> np.ndarray:
        """
        CPG 出力への位置補正値を返す。

        Parameters
        ----------
        q, dq      : 関節角・速度 (n_joints,)
        cpg_state  : CPG の [x1, x2] (2*n_joints,)
        """
        self._t_interp_buf.clear()

        x_raw = np.concatenate([q, dq, cpg_state])
        x_norm = (x_raw - self._x_mean) / self._x_std
        x = torch.tensor(x_norm, dtype=torch.float32, device=self.device)
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, 4n)

        self.model.eval()
        with torch.no_grad():
            y_norm, self.h = self.model(x, self.h)

        if self._t_interp_buf:
            self._last_t_interp = float(np.mean(self._t_interp_buf))

        y_np = y_norm.squeeze().cpu().numpy()
        return y_np * self._y_std + self._y_mean

    def get_tau_sys(self) -> float | None:
        return self._last_t_interp

    # ------------------------------------------------------------------
    # 訓練データ収集
    # ------------------------------------------------------------------

    @staticmethod
    def collect_correction_data(
        env,
        cpg: "MatsuokaOscillator",  # type: ignore[name-defined]
        n_episodes: int = 80,
        episode_steps: int = 200,
        q_offset: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, ...]:
        """
        CPG 単独で動かし、参照軌道との残差（補正すべき量）を収集。

        q_ref(t) = q_offset + cpg_output(t)
        q_actual(t) = シミュレーション上の関節角

        Returns
        -------
        X_seqs : (N_ep, T, 4n)  入力シーケンス
        Y_seqs : (N_ep, T, n)   補正目標（参照 - 実際）
        """
        import mujoco

        if rng is None:
            rng = np.random.default_rng(42)
        if q_offset is None:
            q_offset = np.zeros(cpg.n)

        model = env.model
        n = cpg.n

        X_seqs, Y_seqs = [], []

        for ep in range(n_episodes):
            # CPG 初期化（エピソードごとにランダムシード）
            cpg.reset(seed=ep)
            mujoco.mj_resetData(model, env.data)
            env.data.qpos[:n] = q_offset.copy()
            mujoco.mj_forward(model, env.data)

            X_ep, Y_ep = [], []
            for _ in range(episode_steps):
                q   = env.data.qpos[:n].copy()
                dq  = env.data.qvel[:n].copy()
                cpg_out   = cpg.step()
                cpg_state = cpg.state  # (2n,)

                q_ref = q_offset + cpg_out
                correction_target = q_ref - q  # 補正すべき量

                x = np.concatenate([q, dq, cpg_state])
                X_ep.append(x)
                Y_ep.append(correction_target)

                # position actuator に q_ref を指令（補正なし）
                env.data.ctrl[:n] = np.clip(q_ref, -3.14, 3.14)
                mujoco.mj_step(model, env.data)

            X_seqs.append(X_ep)
            Y_seqs.append(Y_ep)

        return (
            np.array(X_seqs, dtype=np.float32),  # (N, T, 4n)
            np.array(Y_seqs, dtype=np.float32),  # (N, T, n)
        )

    # ------------------------------------------------------------------
    # 訓練
    # ------------------------------------------------------------------

    def fit(
        self,
        X_seqs:     np.ndarray,
        Y_seqs:     np.ndarray,
        n_epochs:   int   = 200,
        batch_size: int   = 16,
        lr:         float = 1e-3,
        verbose:    bool  = True,
    ) -> list[float]:
        """シーケンスデータで教師あり学習。"""
        N, T, _ = X_seqs.shape

        self._x_mean = X_seqs.reshape(-1, X_seqs.shape[-1]).mean(0)
        self._x_std  = X_seqs.reshape(-1, X_seqs.shape[-1]).std(0) + 1e-8
        self._y_mean = Y_seqs.reshape(-1, Y_seqs.shape[-1]).mean(0)
        self._y_std  = Y_seqs.reshape(-1, Y_seqs.shape[-1]).std(0) + 1e-8

        X = torch.tensor((X_seqs - self._x_mean) / self._x_std, dtype=torch.float32)
        Y = torch.tensor((Y_seqs - self._y_mean) / self._y_std, dtype=torch.float32)

        loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * 0.01
        )
        loss_fn = nn.MSELoss()
        loss_history = []

        self._hooks_active = False
        self.model.train()
        try:
            for epoch in range(1, n_epochs + 1):
                epoch_loss = 0.0
                for xb, yb in loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
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
            "x_mean": self._x_mean.tolist(),
            "x_std":  self._x_std.tolist(),
            "y_mean": self._y_mean.tolist(),
            "y_std":  self._y_std.tolist(),
        }, path)
        print(f"Saved CfC adaptive layer → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        self._x_mean = np.array(ckpt["x_mean"])
        self._x_std  = np.array(ckpt["x_std"])
        self._y_mean = np.array(ckpt["y_mean"])
        self._y_std  = np.array(ckpt["y_std"])
        print(f"Loaded CfC adaptive layer ← {path}")
