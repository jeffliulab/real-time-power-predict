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
# verify Part 1 while the course modules (class/default, cs137/2026spring)
# are missing from the cluster. Uses the Tufts-AI anaconda module which
# ships torch 2.3.1+cu121.

module load anaconda/2023.07.tuftsai

cd /cluster/tufts/c26sp1cs0137/pliu07/assignment3

echo "=========================================="
echo "Self-eval (our code, not TA evaluate.py)"
echo "Node  : $(hostname)"
echo "GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(which python)"
echo "Date  : $(date)"
echo "=========================================="

python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

python scripts/self_eval.py \
    --ckpt runs/cnn_transformer_baseline/checkpoints/best.pt \
    --norm_stats runs/cnn_transformer_baseline/norm_stats.pt \
    --n_days "${1:-2}" \
    --device cpu
