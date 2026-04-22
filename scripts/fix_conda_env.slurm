#!/bin/bash
#SBATCH -c 2
#SBATCH -t 0-0:30:00
#SBATCH --mem=8G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -J fix_env
#SBATCH -o /cluster/tufts/c26sp1cs0137/pliu07/assignment3/runs/fix_env_%j.out
#SBATCH -e /cluster/tufts/c26sp1cs0137/pliu07/assignment3/runs/fix_env_%j.err

# Top up the previously-built env: pandas failed to build from source
# (pandas 2.3.3 requires cython build toolchain we don't have). Install
# a pinned earlier version that ships manylinux2014 wheels for cp310.

set -eo pipefail

ENV_PATH=/cluster/tufts/c26sp1cs0137/pliu07/conda_envs/cs137

module load anaconda/2023.07.tuftsai
source activate "$ENV_PATH"

echo "=========================================="
echo "Env: $ENV_PATH"
echo "Python: $(which python) ($(python --version))"
echo "=========================================="

# Downgrade numpy to 1.26.x first so pandas 2.2.x wheels install cleanly.
# torch 2.3.1 requires numpy <2 for some ops anyway.
pip install --no-cache-dir "numpy<2" "pandas==2.2.3" "matplotlib==3.8.4" "scipy" "tqdm"

echo ""
echo "=== Verification ==="
python - <<'PY'
import torch, numpy, pandas, matplotlib, scipy
print(f"torch    : {torch.__version__}")
print(f"numpy    : {numpy.__version__}")
print(f"pandas   : {pandas.__version__}")
print(f"matplotlib: {matplotlib.__version__}")
print(f"scipy    : {scipy.__version__}")
print(f"cuda     : {torch.cuda.is_available()}  (builtwith={torch.version.cuda})")
print(f"cudnn    : {torch.backends.cudnn.version()}")

if torch.cuda.is_available():
    print(f"device   : {torch.cuda.get_device_name(0)}")
    # real backward pass test
    x = torch.randn(4, 128, device='cuda', requires_grad=True)
    y = torch.nn.functional.linear(x, torch.randn(64, 128, device='cuda'))
    y.sum().backward()
    print("linear forward+backward: OK")
    mha = torch.nn.MultiheadAttention(128, 4, batch_first=True).cuda()
    q = torch.randn(2, 10, 128, device='cuda', requires_grad=True)
    out, _ = mha(q, q, q)
    out.sum().backward()
    print("MHA forward+backward   : OK")
PY

echo ""
echo "Env is ready. Size:"
du -sh "$ENV_PATH"
