#!/bin/bash
#SBATCH -c 4
#SBATCH -t 0-1:00:00
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -J attn_maps
#SBATCH -o /cluster/tufts/c26sp1cs0137/pliu07/assignment3/runs/attn_maps_%j.out
#SBATCH -e /cluster/tufts/c26sp1cs0137/pliu07/assignment3/runs/attn_maps_%j.err

# Part 3 Track A — geographic attention map extraction.
# Loads baseline best.pt, runs forward on 5 representative 2022 samples
# with return_attn=True, slices future-tabular → history-spatial
# attention, plots 4 figures into runs/cnn_transformer_baseline/figures/attention/.

ENV_PATH=/cluster/tufts/c26sp1cs0137/pliu07/conda_envs/cs137
export PATH="$ENV_PATH/bin:$PATH"
unset LD_LIBRARY_PATH

cd /cluster/tufts/c26sp1cs0137/pliu07/assignment3
export PYTHONPATH="$PWD:$PYTHONPATH"

echo '=========================================='
echo 'Part 3 — Attention Map Extraction'
echo "Node  : $(hostname)"
echo "GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date  : $(date)"
echo '=========================================='

python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

python scripts/attention_maps.py \
    --ckpt runs/cnn_transformer_baseline/checkpoints/best.pt \
    --norm_stats runs/cnn_transformer_baseline/norm_stats.pt \
    --year 2022 \
    --n_samples 5 \
    --out_dir runs/cnn_transformer_baseline/figures/attention \
    --device cpu
