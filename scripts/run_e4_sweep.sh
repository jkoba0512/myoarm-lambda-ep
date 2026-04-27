#!/bin/bash
set -e

SEEDS="0 1 2 3 4 5 6 7 8 9 42"
SWEEP="${1:-sweep11}"

echo "=== E4 sweep: sweep_name=${SWEEP} ==="

pids=()
for seed in $SEEDS; do
    uv run python scripts/experiment_franka_e4.py --seed "$seed" --sweep-name "$SWEEP" &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "=== E4 sweep complete. Aggregating... ==="
uv run python scripts/aggregate_e4.py --sweep-name "$SWEEP"
