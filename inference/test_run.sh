#!/usr/bin/env bash
# inference/test_run.sh — self-evaluate the trained baseline on the
# last 2 days of 2022 (the 2-window slice referenced throughout the
# report and submission README).
#
# Thin wrapper around scripts/self_eval.py. The latter loads the
# checkpoint, runs inference on the cluster's data slice, and prints
# per-zone MAPE in physical MWh space.
#
# Usage (from the repo root, inside the cs137 conda env):
#   bash inference/test_run.sh
#
# Optional overrides:
#   CKPT=path/to/other.pt bash inference/test_run.sh
#   N_DAYS=4 bash inference/test_run.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="${CKPT:-runs/cnn_transformer_baseline/checkpoints/best.pt}"
NORM_STATS="${NORM_STATS:-runs/cnn_transformer_baseline/norm_stats.pt}"
N_DAYS="${N_DAYS:-2}"
YEAR="${YEAR:-2022}"

cd "$REPO_ROOT"

echo "=========================================="
echo " Self-evaluation"
echo " Checkpoint : $CKPT"
echo " Norm stats : $NORM_STATS"
echo " Window     : last $N_DAYS days of $YEAR"
echo "=========================================="

python scripts/self_eval.py \
    --ckpt "$CKPT" \
    --norm_stats "$NORM_STATS" \
    --year "$YEAR" \
    --n_days "$N_DAYS"
