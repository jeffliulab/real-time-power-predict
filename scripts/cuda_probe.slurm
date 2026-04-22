#!/bin/bash
#SBATCH -c 2
#SBATCH -t 0-0:10:00
#SBATCH --mem=16G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -J cuda_probe
#SBATCH -o /cluster/tufts/c26sp1cs0137/pliu07/assignment3/runs/cuda_probe_%j.out
#SBATCH -e /cluster/tufts/c26sp1cs0137/pliu07/assignment3/runs/cuda_probe_%j.err

# Probe module combos to find one that lets torch run CUDA ops on A100.
# Each combo runs in its own subshell to keep module envs isolated.

cd /cluster/tufts/c26sp1cs0137/pliu07/assignment3

PY_SNIPPET='
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        x = torch.randn(128, 128, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print("matmul ok, norm =", y.norm().item())
        print("CUDA OPS WORK")
    except Exception as e:
        print("CUDA OP FAIL:", type(e).__name__, e)
else:
    print("cuda not available")
'

(
    echo "=========================================="
    echo "Combo 1: anaconda/2023.07.tuftsai only"
    echo "=========================================="
    module load anaconda/2023.07.tuftsai 2>&1
    which python
    python -c "$PY_SNIPPET" 2>&1 | tail -10
)

echo ""
(
    echo "=========================================="
    echo "Combo 2: anaconda/2023.07.tuftsai + cuda/12.2"
    echo "=========================================="
    module load anaconda/2023.07.tuftsai 2>&1
    module load cuda/12.2 2>&1
    which python
    python -c "$PY_SNIPPET" 2>&1 | tail -10
)

echo ""
(
    echo "=========================================="
    echo "Combo 3: anaconda/2023.07.tuftsai + cuda/11.7"
    echo "=========================================="
    module load anaconda/2023.07.tuftsai 2>&1
    module load cuda/11.7 2>&1
    which python
    python -c "$PY_SNIPPET" 2>&1 | tail -10
)

echo ""
(
    echo "=========================================="
    echo "Combo 4: pytorchgpu/1 only"
    echo "=========================================="
    module load pytorchgpu/1 2>&1
    which python
    python -c "$PY_SNIPPET" 2>&1 | tail -10
)
