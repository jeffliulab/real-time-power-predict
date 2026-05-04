#!/usr/bin/env bash
# inference/test_run_supplementary.sh — one-command runner for the
# supplementary foundation-model ensemble (baseline ⊕ c1 ⊕ c2).
#
# Pre-requisite: scripts/run_hpc_dump.sh has been run once (1 DUO push) and
# pretrained_models/baseline/dump/ now contains:
#   - baseline_preds_test_2022_last2d.json
#   - baseline_preds_val_2022_last14d.json
#   - demand_2019_2022_hourly.csv
#
# This script then:
#   1. Runs c1 (Chronos-Bolt-base zero-shot) on test + val windows.
#   2. Runs the 2-way ensemble (baseline ⊕ c1).
#   3. If c2 fine-tuned ckpt exists at runs/chronos_c2/finetuned/,
#      also runs c2 inference + 3-way ensemble.
#
# All steps run on local CPU. No HPC, no DUO.
#
# Usage:
#   bash inference/test_run_supplementary.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-.venv/bin/python}"
DUMP_DIR="pretrained_models/baseline/dump"
DEMAND_CSV="$DUMP_DIR/demand_2019_2022_hourly.csv"
BASELINE_TEST="$DUMP_DIR/baseline_preds_test_2022_last2d.json"
BASELINE_VAL="$DUMP_DIR/baseline_preds_val_2022_last14d.json"

# --- Sanity check ---
for f in "$BASELINE_TEST" "$BASELINE_VAL" "$DEMAND_CSV"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing $f" >&2
        echo "Have you run scripts/run_hpc_dump.sh yet?" >&2
        exit 1
    fi
done

echo "=========================================="
echo " Supplementary ensemble pipeline"
echo " Repo  : $REPO_ROOT"
echo " Python: $PY"
echo "=========================================="

# --- c1: Chronos-Bolt-base zero-shot ---
mkdir -p runs/chronos_c1
echo ""
echo "[1/3] c1: Chronos-Bolt-base zero-shot inference (test + val)"

C1_TEST="runs/chronos_c1/preds_test_2022_last2d.json"
C1_VAL="runs/chronos_c1/preds_val_2022_last14d.json"

$PY inference/run_chronos_zeroshot.py \
    --demand_csv "$DEMAND_CSV" \
    --split test \
    --n_test_days 2 \
    --out "$C1_TEST" \
    --device cpu

$PY inference/run_chronos_zeroshot.py \
    --demand_csv "$DEMAND_CSV" \
    --split val \
    --val_start "2022-12-16" --val_days 14 \
    --out "$C1_VAL" \
    --device cpu

# --- 2-way ensemble (baseline + c1) ---
mkdir -p runs/foundation_ensemble
echo ""
echo "[2/3] 2-way ensemble (baseline ⊕ c1) — alpha grid search on val"

$PY inference/foundation_ensemble.py \
    --baseline_test "$BASELINE_TEST" \
    --baseline_val "$BASELINE_VAL" \
    --c1_test "$C1_TEST" \
    --c1_val "$C1_VAL" \
    --out "runs/foundation_ensemble/eval_test_2022_last2d_2way.json"

# --- 3-way ensemble (if c2 ckpt exists) ---
C2_DIR="runs/chronos_c2"
C2_TEST="$C2_DIR/preds_test_2022_last2d.json"
C2_VAL="$C2_DIR/preds_val_2022_last14d.json"

if [[ -d "$C2_DIR/finetuned" ]]; then
    echo ""
    echo "[3/3] c2 fine-tuned ckpt found — running c2 inference + 3-way ensemble"
    $PY inference/run_chronos_zeroshot.py \
        --demand_csv "$DEMAND_CSV" \
        --model_card "$C2_DIR/finetuned" \
        --split test --n_test_days 2 \
        --out "$C2_TEST" --device cpu
    $PY inference/run_chronos_zeroshot.py \
        --demand_csv "$DEMAND_CSV" \
        --model_card "$C2_DIR/finetuned" \
        --split val --val_start "2022-12-16" --val_days 14 \
        --out "$C2_VAL" --device cpu
    $PY inference/foundation_ensemble.py \
        --baseline_test "$BASELINE_TEST" \
        --baseline_val "$BASELINE_VAL" \
        --c1_test "$C1_TEST"  --c1_val "$C1_VAL" \
        --c2_test "$C2_TEST"  --c2_val "$C2_VAL" \
        --per_zone \
        --out "runs/foundation_ensemble/eval_test_2022_last2d_3way.json"
else
    echo ""
    echo "[3/3] c2 ckpt not found at $C2_DIR/finetuned — skipping (run finetune_chronos.py first to enable 3-way)"
fi

echo ""
echo "Done. Final reports:"
echo "  runs/foundation_ensemble/eval_test_2022_last2d_2way.json"
[[ -d "$C2_DIR/finetuned" ]] && echo "  runs/foundation_ensemble/eval_test_2022_last2d_3way.json"
