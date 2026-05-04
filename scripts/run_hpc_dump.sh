#!/usr/bin/env bash
# scripts/run_hpc_dump.sh — bundled HPC step for the supplementary
# foundation-model ensemble. Runs ONCE, costs ONE DUO push.
#
# What this does (all in a single ssh session):
#   1. Uploads scripts/dump_baseline_preds.py via heredoc (so you do NOT
#      need to git push first).
#   2. Activates the cs137 conda env on HPC.
#   3. Runs scripts/dump_baseline_preds.py to:
#        - run baseline inference on test (last 2 days of 2022) and val
#          (14 days before test) windows
#        - write baseline_preds_{test,val}_*.json (preds + truth)
#        - bundle 2019-2022 demand history → demand_2019_2022_hourly.csv
#   4. tars the dump dir and streams it back via stdout.
#
# All logs (git pull, conda activate, python output) go to STDERR so STDOUT
# is the clean tarball. Local script extracts to pretrained_models/baseline/dump/.
#
# Triggers: ONE ssh session ⇒ ONE DUO push (when no ControlMaster alive).
# Have your phone ready. Re-running with mux already active should not
# require additional DUO (per CLAUDE.md mux notes).
#
# Usage:
#   bash scripts/run_hpc_dump.sh
#
# Env overrides:
#   HPC_HOST    (default: pliu07@login-prod.pax.tufts.edu)
#   HPC_REPO    (default: /cluster/tufts/c26sp1cs0137/pliu07/predict-power)
#   CONDA_ACT   (default: source /cluster/tufts/c26sp1cs0137/pliu07/conda_envs/cs137/bin/activate)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HPC_HOST="${HPC_HOST:-pliu07@login-prod.pax.tufts.edu}"
HPC_REPO="${HPC_REPO:-/cluster/tufts/c26sp1cs0137/pliu07/predict-power}"
HPC_PYTHON="${HPC_PYTHON:-/cluster/tufts/c26sp1cs0137/pliu07/conda_envs/cs137/bin/python}"

LOCAL_DUMP_DIR="$REPO_ROOT/pretrained_models/baseline/dump"
DUMP_SCRIPT_LOCAL="$REPO_ROOT/scripts/dump_baseline_preds.py"

if [[ ! -f "$DUMP_SCRIPT_LOCAL" ]]; then
    echo "ERROR: $DUMP_SCRIPT_LOCAL not found" >&2
    exit 1
fi

mkdir -p "$LOCAL_DUMP_DIR"
TARBALL="$LOCAL_DUMP_DIR/_remote_dump.tgz"

echo "==========================================" >&2
echo " HPC bundled dump (1 ssh = 1 DUO push)" >&2
echo " Host : $HPC_HOST" >&2
echo " Repo : $HPC_REPO" >&2
echo " Local: $LOCAL_DUMP_DIR" >&2
echo "==========================================" >&2
echo "" >&2
echo "Have your phone ready for DUO push (~5s window after ssh prompt)." >&2
echo "" >&2

# Run remote script: upload script via stdin → cat to file → run → tar back
# Single ssh session — all output streams documented above.
{
    echo "=== Script content (will be cat'd to remote scripts/dump_baseline_preds.py) ==="
    echo "[size: $(wc -c < "$DUMP_SCRIPT_LOCAL") bytes, $(wc -l < "$DUMP_SCRIPT_LOCAL") lines]"
} >&2

# Pack the dump script into a tar so we can stream it via stdin without escaping issues.
CM_SOCK="${CM_SOCK:-$HOME/.ssh/cm-${HPC_HOST}:22}"
SSH_OPTS=()
if [[ -S "$CM_SOCK" ]]; then
    echo "[local] reusing existing ControlMaster at $CM_SOCK" >&2
    SSH_OPTS+=("-o" "ControlPath=$CM_SOCK" "-o" "ControlMaster=no")
fi

tar -C "$REPO_ROOT" -czf - scripts/dump_baseline_preds.py | \
ssh "${SSH_OPTS[@]}" "$HPC_HOST" "
    set -euo pipefail
    cd '$HPC_REPO'

    # Receive uploaded script tar via stdin → extract it
    tar xzf - >&2

    echo '[remote] using python at' '$HPC_PYTHON' >&2

    echo '[remote] running dump_baseline_preds.py' >&2
    $HPC_PYTHON scripts/dump_baseline_preds.py >&2

    echo '[remote] tar-ing dump dir back to local stdout' >&2
    cd runs/cnn_transformer_baseline/dump
    tar czf - .
" > "$TARBALL"

REMOTE_BYTES="$(wc -c < "$TARBALL")"
echo "" >&2
echo "Received tarball: $TARBALL ($REMOTE_BYTES bytes)" >&2
echo "Extracting to $LOCAL_DUMP_DIR ..." >&2
tar -C "$LOCAL_DUMP_DIR" -xzf "$TARBALL"
rm "$TARBALL"

echo "" >&2
echo "Files in $LOCAL_DUMP_DIR:" >&2
ls -la "$LOCAL_DUMP_DIR" >&2

if [[ -f "$LOCAL_DUMP_DIR/summary.json" ]]; then
    echo "" >&2
    echo "Dump summary:" >&2
    cat "$LOCAL_DUMP_DIR/summary.json" >&2
fi
echo "" >&2
echo "Done. Next: bash inference/test_run_supplementary.sh" >&2
