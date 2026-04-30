"""
train_myo_cfc.py — MyoArm 用 CfC 前向きモデルの事前学習。

タスク:
  (q_t, dq_t, a_t[:n_joints]) → Δq_t = q_{t+1} - q_t を学習する。
  CfCForwardModel.fit() の API に合わせてシーケンス形式で渡す。

出力:
  results/myo_cfc_data/cfc_model.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from methodB.cfc_forward_model import CfCForwardModel

DATA_DIR   = ROOT / "results" / "myo_cfc_data"
MODEL_PATH = DATA_DIR / "cfc_model.pt"

N_JOINTS     = 20
HIDDEN_UNITS = 64
SEQ_LEN      = 50   # スライディングウィンドウ長
N_EPOCHS     = 200
BATCH_SIZE   = 32
LR           = 1e-3


def load_as_sequences(seq_len: int = SEQ_LEN) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    フラットな収集データをスライディングウィンドウでシーケンスに変換する。

    Returns
    -------
    q_seqs, dq_seqs, efcopy_seqs, q_next_seqs : (N, seq_len, n_joints)
    """
    data   = np.load(DATA_DIR / "train_data.npz")
    q      = data["q"].astype(np.float32)      # (total, 20)
    dq     = data["dq"].astype(np.float32)
    a      = data["a"].astype(np.float32)      # (total, 34)
    q_next = data["q_next"].astype(np.float32)

    efcopy = a[:, :N_JOINTS]  # (total, 20) — 最初の 20 筋をエファレンスコピーとして使用

    total = len(q)
    n_seqs = (total - seq_len) // (seq_len // 2)  # 50% オーバーラップ
    stride = seq_len // 2

    q_seqs:      list = []
    dq_seqs:     list = []
    ef_seqs:     list = []
    q_next_seqs: list = []

    for i in range(n_seqs):
        start = i * stride
        end   = start + seq_len
        if end > total:
            break
        q_seqs.append(q[start:end])
        dq_seqs.append(dq[start:end])
        ef_seqs.append(efcopy[start:end])
        q_next_seqs.append(q_next[start:end])

    return (
        np.stack(q_seqs),
        np.stack(dq_seqs),
        np.stack(ef_seqs),
        np.stack(q_next_seqs),
    )


def main() -> None:
    print("=== MyoArm CfC 事前学習 ===")

    q_seqs, dq_seqs, ef_seqs, q_next_seqs = load_as_sequences()
    print(f"  sequences: {q_seqs.shape[0]}  seq_len: {q_seqs.shape[1]}")
    print(f"  q shape: {q_seqs.shape}  ef shape: {ef_seqs.shape}")

    cfc = CfCForwardModel(
        n_joints=N_JOINTS,
        hidden_units=HIDDEN_UNITS,
        device="cpu",
    )

    # fit() の tau_seqs 引数に efcopy (a[:n_joints]) を渡す
    losses = cfc.fit(
        q_seqs=q_seqs,
        dq_seqs=dq_seqs,
        tau_seqs=ef_seqs,
        q_next_seqs=q_next_seqs,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        verbose=True,
    )

    cfc.save(str(MODEL_PATH))
    print(f"\n  final loss: {losses[-1]:.6f}")
    print(f"  saved → {MODEL_PATH}")


if __name__ == "__main__":
    main()
