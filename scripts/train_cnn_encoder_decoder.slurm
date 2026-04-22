#!/bin/bash
#SBATCH -c 4
#SBATCH -t 0-24:00:00
#SBATCH --mem=48G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o runs/%x_%j.out
#SBATCH -e runs/%x_%j.err
#SBATCH -J cnn_encdec

# Usage (from project root on HPC):
#   sbatch scripts/train_cnn_encoder_decoder.slurm
#   sbatch scripts/train_cnn_encoder_decoder.slurm --epochs 20 --batch_size 2
#   sbatch scripts/train_cnn_encoder_decoder.slurm --use_future_weather_xattn
#
# Parameter budget (smoke-test on HPC before first real run):
#   4 encoder + 2 decoder (default) ≈ 2.28 M params (baseline was 1.75 M)
#   3 encoder + 2 decoder           ≈ 2.08 M (closer to baseline)
#   3 encoder + 1 decoder           ≈ 1.82 M (essentially parity)
# To adjust: pass e.g. `--n_encoder_layers 3 --n_decoder_layers 1` as extra args.

module load class/default
module load cs137/2026spring

export PYTHONUNBUFFERED=1

cd /cluster/tufts/c26sp1cs0137/pliu07/assignment3

EXTRA_ARGS="${@}"

echo "=========================================="
echo "Model: cnn_encoder_decoder"
echo "Extra args: $EXTRA_ARGS"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "=========================================="

python training/train.py \
    --model cnn_encoder_decoder \
    --epochs 18 \
    --batch_size 4 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --scheduler cosine \
    --history_len 24 \
    --embed_dim 128 \
    --grid_size 8 \
    --n_encoder_layers 4 \
    --n_decoder_layers 2 \
    --n_heads 4 \
    --dropout 0.1 \
    --warmup_steps 500 \
    --patience 5 \
    --train_years 2019 2020 2021 \
    --val_years 2022 \
    $EXTRA_ARGS
