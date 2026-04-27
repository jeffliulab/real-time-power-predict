#!/bin/bash
#SBATCH -c 4
#SBATCH -t 0-1:00:00
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -J self_eval
#SBATCH -o /cluster/tufts/c26sp1cs0137/pliu07/assignment3/runs/self_eval_%j.out
#SBATCH -e /cluster/tufts/c26sp1cs0137/pliu07/assignment3/runs/self_eval_%j.err

# Our-code evaluator. Bypasses the TA's evaluate.py entirely so we can
# verify any model checkpoint while the course modules (class/default,
# cs137/2026spring) are missing from the cluster.
#
# Uses our persistent conda env at /cluster/.../conda_envs/cs137/ which
# has torch 2.3.1+cu118 + bundled cuDNN 8.9, numpy/pandas/matplotlib.
#
# Usage:
#   sbatch scripts/self_eval.slurm                                # default: cnn_transformer_baseline best.pt, 2 days
#   sbatch scripts/self_eval.slurm <CKPT_PATH> <N_DAYS>           # explicit ckpt path and days
#   sbatch scripts/self_eval.slurm runs/cnn_encoder_decoder/checkpoints/best_epoch6_snapshot.pt 2

ENV_PATH=/cluster/tufts/c26sp1cs0137/pliu07/conda_envs/cs137
export PATH="$ENV_PATH/bin:$PATH"
unset LD_LIBRARY_PATH

cd /cluster/tufts/c26sp1cs0137/pliu07/assignment3
export PYTHONPATH="$PWD:$PYTHONPATH"

CKPT="${1:-runs/cnn_transformer_baseline/checkpoints/best.pt}"
N_DAYS="${2:-2}"
# Derive norm_stats path from ckpt: runs/<model>/checkpoints/X.pt → runs/<model>/norm_stats.pt
NORM_STATS="$(dirname "$(dirname "$CKPT")")/norm_stats.pt"

echo "=========================================="
echo "Self-eval (our code, not TA evaluate.py)"
echo "Node  : $(hostname)"
echo "GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(which python)"
echo "Ckpt  : $CKPT"
echo "Norm  : $NORM_STATS"
echo "Days  : $N_DAYS"
echo "Date  : $(date)"
echo "=========================================="

python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

python scripts/self_eval.py \
    --ckpt "$CKPT" \
    --norm_stats "$NORM_STATS" \
    --n_days "$N_DAYS" \
    --device cpu
