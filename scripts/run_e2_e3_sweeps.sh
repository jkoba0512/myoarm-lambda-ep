#!/bin/bash
# E2/E3 実験 11-seed parallel sweep
set -e

SEEDS="0 1 2 3 4 5 6 7 8 9 42"
SWEEP="sweep11"

echo "=== Phase E2 sweep (${SWEEP}) ==="
pids=()
for seed in $SEEDS; do
    uv run python scripts/experiment_franka_e2.py --seed "$seed" --sweep-name "$SWEEP" &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
echo "E2 sweep complete."

echo ""
echo "=== Phase E3 sweep (${SWEEP}) ==="
pids=()
for seed in $SEEDS; do
    uv run python scripts/experiment_franka_e3.py --seed "$seed" --sweep-name "$SWEEP" &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
echo "E3 sweep complete."

echo ""
echo "=== Aggregation ==="
uv run python scripts/aggregate_e2_e3.py --sweep-name "$SWEEP"
echo "Done."
