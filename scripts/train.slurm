#!/bin/bash
#SBATCH -c 4
#SBATCH -t 0-24:00:00
#SBATCH --mem=48G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o runs/%x_%j.out
#SBATCH -e runs/%x_%j.err
#SBATCH -J energy_forecast

# Usage (from project root):
#   sbatch scripts/train.slurm
#   sbatch scripts/train.slurm --history_len 48 --grid_size 6
#   sbatch scripts/train.slurm --epochs 50 --lr 5e-4

module load class/default
module load cs137/2026spring

export PYTHONUNBUFFERED=1

cd /cluster/tufts/c26sp1cs0137/pliu07/assignment3

EXTRA_ARGS="${@}"

echo "=========================================="
echo "Model: cnn_transformer"
echo "Extra args: $EXTRA_ARGS"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "=========================================="

python training/train.py --model cnn_transformer $EXTRA_ARGS
