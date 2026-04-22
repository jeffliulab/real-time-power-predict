<div align="center">

[![en](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![zh](https://img.shields.io/badge/lang-中文-red.svg)](README_zh.md)

<h1>ISO-NE Day-Ahead Energy Demand Forecasting</h1>

<p>
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Status-In%20Progress-yellow" alt="Status">
</p>

<p>
  Multi-modal CNN-Transformer for <strong>24-hour per-zone electricity demand</strong> forecasting across the ISO New England grid.
</p>

</div>

---

## Task

Given a historical window ending at time `t`, predict hourly MWh demand for **all 8 ISO-NE load zones** over the next 24 hours `t+1 … t+24`. Inputs fuse high-dimensional weather maps (`450×449×7`) with tabular sequential data (past demand + calendar features). Assumes perfect future weather forecasts, matching real-world grid-operation pipelines.

See [docs/assignment.pdf](docs/assignment.pdf) for the full spec.

| Component | Detail |
|-----------|--------|
| **Weather input** | `(B, S+24, 450, 449, 7)` — hourly HRRR-style reanalysis tensors over New England |
| **Energy input** | `(B, S, 8)` — MWh per zone (ME, NH, VT, CT, RI, SEMA, WCMA, NEMA_BOST) |
| **Calendar** | One-hot hour/dow/month + holiday flag (44-d), for all `S+24` timesteps |
| **Output** | `(B, 24, 8)` — predicted MWh per zone for the next 24 h |
| **Metric** | Average MAPE across the 24-h horizon and all zones |

---

## Current Status

| Part | Weight | Due | Status |
|---|---|---|---|
| Part 1 — Baseline CNN-Transformer patch architecture | 40 | Apr 15 | ✅ Submitted, test MAPE **5.24 %** on 2 days of 2022, independently verified ([runs/cnn_transformer/](runs/cnn_transformer/)) |
| Part 2 — Architecture search (beat the baseline) | 30 | **Apr 22** | 🔨 Code ready: `cnn_encoder_decoder` w/ cross-attention ([docs/part2_report.md](docs/part2_report.md)), pending HPC training |
| Part 3 — Model diagnosis OR independent study | 30 | May 1 | ⏳ Not started (preliminary plan: Track A — geographic attention maps) |
| Report + presentation | — | May 1 / May 4 | ⏳ Not started |

See [docs/progress.md](docs/progress.md) for the detailed work plan and [docs/hpc-evaluation-structure.md](docs/hpc-evaluation-structure.md) for where Part 1 lives on the course cluster.

---

## Part 1 — Baseline (Hybrid CNN-Transformer, 1.75 M params)

Architecture follows the assignment spec (Figure 2):

1. **Spatial tokens** — each `(450, 449, 7)` weather snapshot passes through a ResBlock-based CNN (5 stride-2 stages + adaptive pool → `G×G` grid, default `G=8`) yielding `P = G²` tokens of dim `D = 128` per timestep.
2. **Historical tabular tokens** — `Linear(demand + 44-d calendar → D)` per historical hour.
3. **Future tabular tokens** — `Linear(learned_demand_mask + calendar → D)` for the 24 prediction hours.
4. **Sequence assembly** — per hour, concatenate `P` spatial tokens + 1 tabular token. Add learnable spatial pos-embed (shared across time) + temporal pos-embed (shared across tokens within an hour) + learnable tabular-type embedding. Flatten to length `(S+24)·(P+1) = 48·65 = 3120`.
5. **Transformer encoder** — pre-norm, 4 layers, 4 heads, GELU MLP.
6. **Prediction** — slice the 24 future tabular tokens → `MLP(D → D/2 → 8)`.

Source: [models/cnn_transformer.py](models/cnn_transformer.py).

### Training configuration

| Setting | Value |
|---------|-------|
| Optimizer | AdamW, lr=1e-3, wd=1e-4 |
| Scheduler | CosineAnnealingLR |
| Loss | MSE (on z-score-normalized targets) |
| History length `S` | 24 hours |
| Grid `G` | 8 → P = 64 spatial tokens |
| Epochs | 15 (14 completed within 24 h time limit on A100-40GB) |
| Train / val | 2019–2020 / 2021 |

### Training curve

![training curves](runs/cnn_transformer/figures/training_curves.png)

Best val MAPE 6.92 % at epoch 12 (local val on 2021). Held-out test MAPE 5.24 % on the final 2 days of 2022 via the course evaluation harness.

---

## Project Structure

```
real-time-power-predict/
├── README.md                        # This file
├── README_zh.md                     # Chinese version
├── .gitignore
├── docs/
│   ├── assignment.pdf               # Course handout
│   ├── progress.md                  # Current status & remaining work
│   └── hpc-evaluation-structure.md  # Layout of /cluster/.../assignment3_data/evaluation/ on HPC
├── models/
│   ├── __init__.py                  # Model registry
│   └── cnn_transformer.py           # Part 1 baseline
├── training/
│   ├── train.py                     # Training entry point
│   └── data_preparation/
│       └── dataset.py               # Dataset with LRU weather cache
├── evaluation/
│   └── pangliu/                     # Course evaluator wrapper (get_model + adapt_inputs)
│       └── model.py
├── runs/
│   └── cnn_transformer/             # Part 1 artifacts
│       ├── config.json
│       ├── norm_stats.pt
│       ├── logs/training_log.csv
│       └── figures/training_curves.png
│       # Checkpoints (best.pt, latest.pt) excluded from git — pull from HPC.
└── scripts/
    └── train.slurm                  # SLURM job for Tufts HPC gpu partition
```

---

## Quick Start

### Sync code to the cluster and train

```bash
# From your laptop — rsync the repo to HPC
rsync -avz --exclude=__pycache__ --exclude=.git ./ tufts-login:/cluster/tufts/c26sp1cs0137/pliu07/assignment3/

# On the cluster — submit training job
ssh tufts-login
cd /cluster/tufts/c26sp1cs0137/pliu07/assignment3
sbatch scripts/train.slurm --epochs 15 --train_years 2019 2020 --val_years 2021
```

### Pull results back

```bash
rsync -avz --exclude=checkpoints tufts-login:/cluster/tufts/c26sp1cs0137/pliu07/assignment3/runs/ ./runs/
```

### Run the course evaluator

Submissions live at `/cluster/tufts/c26sp1cs0137/data/assignment3_data/evaluation/part1-models/<team>/` — see [docs/hpc-evaluation-structure.md](docs/hpc-evaluation-structure.md).

```bash
# On HPC
cd /cluster/tufts/c26sp1cs0137/data/assignment3_data/evaluation
sbatch -J part1-models/pangliu test_run.sh 2   # 2 = number of test days
```

---

## Data

All data lives on the Tufts HPC at `/cluster/tufts/c26sp1cs0137/data/assignment3_data/` (~278 GB). The training script points at that path by default.

| Item | Shape / type | Coverage |
|---|---|---|
| Weather tensors | `(450, 449, 7)` per hour | Hourly, 2019–2023 |
| Energy demand CSVs | 8 zones, hourly MWh | 2019–2023, UTC |
| Test set (held-out) | same format | 2024 |

All timestamps are aligned in UTC to match the meteorological inputs. We z-score-normalize weather and energy inputs using statistics from 500 random training samples (cached in `runs/cnn_transformer/norm_stats.pt`).

---

## Acknowledgments

- **Compute**: Tufts Research Technology HPC (NVIDIA A100-40GB)
- **Course**: Tufts CS 137 — Deep Neural Networks, Spring 2026
