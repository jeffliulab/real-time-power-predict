#!/bin/bash
#
# Sweep Chronos-Bolt ablations for the foundation-model section of the report.
# All inference is local CPU; no training.
#
# Sweeps:
#   C1 — context length: 168 / 336 / 672 / 1344 hours, model_card = base
#   C3 — model size:    tiny / mini / small / base, context_len = 672
# Each (config, split) pair is one run; total ~7 base-size runs reused for both
# axes if context_len=672 + size=base is the shared anchor.
#
# Quantile-aggregation ablation (C2) and alpha-stability ablation (C4) are
# computed offline by aggregating the dumped quantile_preds JSONs — no extra
# inference needed.
#
# Usage:
#   bash scripts/run_chronos_ablations.sh
#
# Outputs:
#   runs/chronos_ablation/<sweep>/<config>_<split>.json

set -e
set -o pipefail

ROOT="/Users/macbookpro/Local_Root/predict-power"
PY="$ROOT/.venv/bin/python"
DEMAND_CSV="$ROOT/pretrained_models/baseline/dump/demand_2019_2022_hourly.csv"
SCRIPT="$ROOT/inference/run_chronos_ablation.py"

OUT_DIR="$ROOT/runs/chronos_ablation"
mkdir -p "$OUT_DIR/ctx" "$OUT_DIR/size"

run_one() {
    local sweep="$1"  config="$2"  split="$3"  ctx="$4"  model_card="$5"
    local out="$OUT_DIR/$sweep/${config}_${split}.json"
    if [ -f "$out" ]; then
        echo "[skip] $out already exists"
        return 0
    fi
    echo "============================================================"
    echo "  $sweep / $config / $split   (ctx=$ctx, model=$model_card)"
    echo "============================================================"
    if [ "$split" = "test" ]; then
        "$PY" "$SCRIPT" \
            --demand_csv "$DEMAND_CSV" \
            --model_card "$model_card" \
            --context_len "$ctx" \
            --split test \
            --year 2022 --n_test_days 2 \
            --aggregation median \
            --out "$out" 2>&1
    else
        "$PY" "$SCRIPT" \
            --demand_csv "$DEMAND_CSV" \
            --model_card "$model_card" \
            --context_len "$ctx" \
            --split val \
            --val_start 2022-12-16 --val_days 14 \
            --aggregation median \
            --out "$out" 2>&1
    fi
    echo
}

echo "###################################################"
echo "# C1 — context length sweep (model = base)        #"
echo "###################################################"
for CTX in 168 336 672 1344; do
    run_one ctx "ctx${CTX}" test "$CTX" amazon/chronos-bolt-base
    run_one ctx "ctx${CTX}" val  "$CTX" amazon/chronos-bolt-base
done

echo "###################################################"
echo "# C3 — model size sweep (context = 672)           #"
echo "###################################################"
for SZ in tiny mini small; do
    run_one size "size_${SZ}" test 672 "amazon/chronos-bolt-${SZ}"
    run_one size "size_${SZ}" val  672 "amazon/chronos-bolt-${SZ}"
done
# 'base' anchor already in ctx/ctx672_*.json; copy/symlink for size sweep
ln -sf ../ctx/ctx672_test.json "$OUT_DIR/size/size_base_test.json" 2>/dev/null || true
ln -sf ../ctx/ctx672_val.json  "$OUT_DIR/size/size_base_val.json"  2>/dev/null || true

echo
echo "ALL ABLATIONS COMPLETE"
ls -la "$OUT_DIR/ctx" "$OUT_DIR/size"
