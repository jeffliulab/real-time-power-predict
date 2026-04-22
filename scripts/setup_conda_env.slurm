#!/bin/bash
#SBATCH -c 4
#SBATCH -t 0-1:00:00
#SBATCH --mem=16G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -J setup_env
#SBATCH -o /cluster/tufts/c26sp1cs0137/pliu07/assignment3/runs/setup_env_%j.out
#SBATCH -e /cluster/tufts/c26sp1cs0137/pliu07/assignment3/runs/setup_env_%j.err

# One-shot: build a persistent conda env for all CS 137 assignment 3 training.
# Env path: /cluster/tufts/c26sp1cs0137/pliu07/conda_envs/cs137/
#
# Strategy:
#   - Use miniforge3/conda from the cluster's shared tools
#   - Python 3.10 (matches Part 1's course-module baseline)
#   - torch 2.3.1 built for CUDA 11.8 — matches cuda/11.7 runtime module
#     and bundles its own cuDNN (no dependency on Tufts' broken cuDNN mods)
#   - numpy, pandas, matplotlib — what training/dataset/report code needs
#
# After this runs, subsequent SLURM scripts load:
#     module load anaconda/2023.07.tuftsai
#     source activate /cluster/tufts/c26sp1cs0137/pliu07/conda_envs/cs137
# and skip all cuda/cudnn modules (torch bundles its own cuDNN).

set -eo pipefail

ENV_PATH=/cluster/tufts/c26sp1cs0137/pliu07/conda_envs/cs137

# Bootstrap conda
module load anaconda/2023.07.tuftsai

cd /cluster/tufts/c26sp1cs0137/pliu07

echo "=========================================="
echo "Creating conda env at: $ENV_PATH"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# Fresh install — delete if exists
if [ -d "$ENV_PATH" ]; then
    echo "Env exists; removing first"
    rm -rf "$ENV_PATH"
fi

# Create env with python 3.10
conda create -y -p "$ENV_PATH" python=3.10 pip

# Activate (use source activate for batch mode)
source activate "$ENV_PATH"

which python
python --version

echo ""
echo "=== Installing torch 2.3.1 (cu118) ==="
pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "=== Installing scientific stack ==="
pip install --no-cache-dir numpy pandas matplotlib scipy tqdm

echo ""
echo "=== Verification ==="
python - <<'PY'
import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"cuda version torch was built with: {torch.version.cuda}")
print(f"cudnn version bundled: {torch.backends.cudnn.version()}")

if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
    # Real backward-pass test — the one that's been failing with tuftsai torch
    x = torch.randn(4, 128, device='cuda', requires_grad=True)
    y = torch.nn.functional.linear(x, torch.randn(64, 128, device='cuda'))
    y.sum().backward()
    print("forward+backward: OK")
    # Transformer-like test (nn.MultiheadAttention)
    mha = torch.nn.MultiheadAttention(128, 4, batch_first=True).cuda()
    q = torch.randn(2, 10, 128, device='cuda', requires_grad=True)
    out, _ = mha(q, q, q)
    out.sum().backward()
    print("MultiheadAttention forward+backward: OK")
PY

echo ""
echo "=== Env size ==="
du -sh "$ENV_PATH"

echo ""
echo "Done. Env is at $ENV_PATH"
