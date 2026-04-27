#!/bin/bash
# E5 実験 11-seed parallel sweep
set -e

SEEDS="0 1 2 3 4 5 6 7 8 9 42"
SWEEP="sweep11"

echo "=== Phase E5 sweep (${SWEEP}) ==="
pids=()
for seed in $SEEDS; do
    uv run python scripts/experiment_franka_e5.py --seed "$seed" --sweep-name "$SWEEP" &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
echo "E5 sweep complete."

echo ""
echo "=== Aggregation ==="
uv run python scripts/aggregate_e5.py --sweep-name "$SWEEP"
echo "Done."
