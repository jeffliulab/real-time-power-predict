#!/bin/bash
#SBATCH -c 2
#SBATCH -t 0-0:10:00
#SBATCH --mem=16G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -J smoke_test
#SBATCH -o /cluster/tufts/c26sp1cs0137/pliu07/assignment3/runs/smoke_%j.out
#SBATCH -e /cluster/tufts/c26sp1cs0137/pliu07/assignment3/runs/smoke_%j.err

# Param-count + 1-batch forward smoke test for cnn_encoder_decoder.
# Uses anaconda/2023.07.tuftsai since the course modules are still
# missing cluster-wide.

module load anaconda/2023.07.tuftsai

cd /cluster/tufts/c26sp1cs0137/pliu07/assignment3

echo "=========================================="
echo "Smoke test: cnn_encoder_decoder"
echo "Node  : $(hostname)"
echo "Date  : $(date)"
echo "=========================================="

python -c '
import sys
sys.path.insert(0, ".")
import torch

# Import torch first to avoid BLAS conflict with numpy in this env
from models import create_model, MODEL_DEFAULTS

for n_enc, n_dec, label in [(4, 2, "default 4+2"), (3, 2, "3+2"), (3, 1, "3+1")]:
    m = create_model(
        "cnn_encoder_decoder",
        n_encoder_layers=n_enc,
        n_decoder_layers=n_dec,
    )
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  {label}: total={total:,} trainable={trainable:,} ({total/1e6:.2f} M)")

print("\nForward pass smoke test (batch=1, default config, CPU)...")
m = create_model("cnn_encoder_decoder").eval()
B, S = 1, 24
hist_w = torch.zeros(B, S, 450, 449, 7)
hist_e = torch.zeros(B, S, 8)
hist_c = torch.zeros(B, S, 44)
fut_w  = torch.zeros(B, 24, 450, 449, 7)
fut_c  = torch.zeros(B, 24, 44)

with torch.no_grad():
    out = m(hist_w, hist_e, hist_c, fut_w, fut_c)

print(f"  Output shape: {tuple(out.shape)} (expected (1, 24, 8))")
print(f"  Output dtype: {out.dtype}")
assert out.shape == (1, 24, 8), "shape mismatch!"
print("  PASS")
'
