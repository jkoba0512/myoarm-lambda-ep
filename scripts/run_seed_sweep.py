"""
seed sweep: 実験 2A-2D を複数 seed で再実行する。

使い方:
  uv run python scripts/run_seed_sweep.py              # seed 0-9
  uv run python scripts/run_seed_sweep.py --seeds 0 1 2
  uv run python scripts/run_seed_sweep.py --seeds 42   # 単一 seed
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT    = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"


def run(script: str, seed: int) -> bool:
    cmd = [sys.executable, str(SCRIPTS / script), "--seed", str(seed)]
    print(f"\n  >>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)),
                   help="実行する seed のリスト（デフォルト: 0-9）")
    args = p.parse_args()

    seeds   = args.seeds
    failed  = []

    print(f"=== seed sweep: seeds={seeds} ===")

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"  seed = {seed}")
        print(f"{'='*50}")

        # 2A: 小脳訓練 + アブレーション
        if not run("experiment_franka_2a.py", seed):
            print(f"  [FAILED] 2A seed={seed}")
            failed.append((seed, "2a"))
            continue  # 2A 失敗なら 2B/2C/2D もスキップ

        # 2B, 2C, 2D は 2A のモデルに依存
        for exp in ["2b", "2c", "2d"]:
            if not run(f"experiment_franka_{exp}.py", seed):
                print(f"  [FAILED] {exp.upper()} seed={seed}")
                failed.append((seed, exp))

    print(f"\n{'='*50}")
    print("=== sweep 完了 ===")
    if failed:
        print(f"  失敗: {failed}")
    else:
        print(f"  全 {len(seeds)} seeds 成功")
    print(f"\n次のステップ: uv run python scripts/aggregate_results.py")


if __name__ == "__main__":
    main()
