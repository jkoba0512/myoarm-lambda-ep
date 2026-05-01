"""
train_myo_cfc_lambda.py — λ-EP rollout データで CfC 前向きモデルを再訓練。

既存の `train_myo_cfc.py` と同じアーキ・ハイパラだが、データパスのみ
results/myo_cfc_data_lambda/ に切り替える。

タスク:
  (q_t, dq_t, a_t[:n_joints]) → q_{t+1} を学習する。

出力:
  results/myo_cfc_data_lambda/cfc_model.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from methodB.cfc_forward_model import CfCForwardModel

DATA_DIR   = ROOT / "results" / "myo_cfc_data_lambda"
MODEL_PATH = DATA_DIR / "cfc_model.pt"

N_JOINTS     = 20
HIDDEN_UNITS = 64
SEQ_LEN      = 50
N_EPOCHS     = 200
BATCH_SIZE   = 32
LR           = 1e-3


def load_as_sequences(seq_len: int = SEQ_LEN) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    data   = np.load(DATA_DIR / "train_data.npz")
    q      = data["q"].astype(np.float32)
    dq     = data["dq"].astype(np.float32)
    a      = data["a"].astype(np.float32)
    q_next = data["q_next"].astype(np.float32)

    efcopy = a[:, :N_JOINTS]

    total = len(q)
    stride = seq_len // 2
    n_seqs = (total - seq_len) // stride

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
    print("=== λ-EP CfC 再訓練 ===")
    print(f"  data: {DATA_DIR / 'train_data.npz'}")

    q_seqs, dq_seqs, ef_seqs, q_next_seqs = load_as_sequences()
    print(f"  sequences: {q_seqs.shape[0]}  seq_len: {q_seqs.shape[1]}")
    print(f"  q shape: {q_seqs.shape}  ef shape: {ef_seqs.shape}")

    cfc = CfCForwardModel(
        n_joints=N_JOINTS,
        hidden_units=HIDDEN_UNITS,
        device="cpu",
    )

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
