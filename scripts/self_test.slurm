#!/bin/bash
#SBATCH -c 4
#SBATCH -t 0-1:00:00
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -J self_test
#SBATCH -o /cluster/tufts/c26sp1cs0137/pliu07/assignment3/runs/self_test_%j.out
#SBATCH -e /cluster/tufts/c26sp1cs0137/pliu07/assignment3/runs/self_test_%j.err

# GPU-partition replacement for the TA's test_run.sh, which is currently
# broken on the batch partition (modules class/default + cs137/2026spring
# fail to load there as of 2026-04-21).
#
# Usage:
#   sbatch scripts/self_test.slurm <MODEL_NAME> [N_DAYS]
# Example:
#   sbatch scripts/self_test.slurm part1-models/pangliu 2

module load class/default
module load cs137/2026spring
PYTHON=python

MODEL_NAME="${1:-part1-models/pangliu}"
N_DAYS="${2:-2}"

cd /cluster/tufts/c26sp1cs0137/data/assignment3_data/evaluation

echo "=========================================="
echo "Self-test: $MODEL_NAME for $N_DAYS days"
echo "Node  : $(hostname)"
echo "GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $PYTHON"
echo "Date  : $(date)"
echo "=========================================="

$PYTHON --version
$PYTHON -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

$PYTHON evaluate.py "$MODEL_NAME" "$N_DAYS"
