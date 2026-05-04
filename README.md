<div align="center">

[![en](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![zh](https://img.shields.io/badge/lang-中文-red.svg)](README_zh.md)

<h1>ISO-NE Day-Ahead Energy Demand Forecasting</h1>

<p>
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.3-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Status-Training%20%2F%20In%20Progress-yellow" alt="Status">
</p>

<p>
  Multi-modal CNN-Transformer for <strong>24-hour per-zone electricity demand</strong> forecasting across the ISO New England grid, fusing high-resolution weather rasters with historical demand and calendar features.
</p>

<p>
  <em>Live demo and HF model weights coming after Part 2 training completes.</em>
</p>

</div>

---

## Highlights

- **Three trained models** (state of v1.2 release):
  - **Part 1 baseline** — single-encoder CNN-Transformer (1.75 M params), **test MAPE 5.24 %** on last 2 days of 2022 ⭐ best
  - **Part 2 v1** — encoder-decoder, history-only cross-attention (2.29 M params), test MAPE **6.82 %**
  - **Part 2 v2** — encoder-decoder, history + future cross-attention (2.42 M params), test MAPE **9.27 %** ⚠ **undertrained** (13 of 24 planned epochs, ~54 %; 3 SLURM TIMEOUTs)
- All three checkpoints + cards available at [`pretrained_models/`](pretrained_models/) (HF-style layout).
- **Self-contained pipeline**: data prep → training → evaluation → inference → real-time demo (Part 3)
- **Faithful evaluation**: independent reproduction of the TA evaluator harness, byte-for-byte matching MAPE numbers
- **Reproducibility from public sources**: dataset can be rebuilt from ISO Express + NOAA HRRR via [`scripts/data_preparation/`](scripts/data_preparation/) if the cluster mirror disappears.

---

## Table of Contents

- [Highlights](#highlights)
- [Task Definition](#task-definition)
- [Data and Normalization](#data-and-normalization)
- [Architectures](#architectures)
- [Results](#results)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Status](#status)
- [Acknowledgments](#acknowledgments)

---

## Task Definition

Given a historical window ending at time `t`, predict hourly MWh demand for **all 8 ISO-NE load zones** over the next 24 hours `t+1 … t+24`. Inputs fuse high-dimensional weather maps (`450×449×7`) with tabular sequential data (past demand + calendar features). Assumes perfect future weather forecasts, matching real-world grid-operation pipelines.

See [docs/assignment.pdf](docs/assignment.pdf) for the full spec.

| Component | Detail |
|-----------|--------|
| **Weather input** | `(B, S+24, 450, 449, 7)` — hourly HRRR-style reanalysis tensors over New England |
| **Energy input** | `(B, S, 8)` — MWh per zone (ME, NH, VT, CT, RI, SEMA, WCMA, NEMA_BOST) |
| **Calendar** | One-hot hour/dow/month + holiday flag (44-d), for all `S+24` timesteps |
| **Output** | `(B, 24, 8)` — predicted MWh per zone for the next 24 h |
| **Metric** | Average MAPE across the 24-h horizon and all zones, in raw MWh space |

---

## Data and Normalization

| Property | Value |
|---|---|
| **Source** | ISO-NE zonal load CSVs + 7-channel HRRR-style weather tensors (Tufts HPC) |
| **Region** | New England, 450×449 grid, ~3 km resolution |
| **Coverage** | 2019 – 2023 hourly |
| **Train** | 2019 + 2020 (baseline) / 2019-2021 (Part 2) |
| **Val** | 2021 (baseline) / 2022 (Part 2) |
| **Test (TA)** | last 2 days of 2022 |

**Normalization pipeline** (single source of truth — both models share the same chain):

1. `compute_norm_stats()` ([dataset.py](training/data_preparation/dataset.py)) — z-score from 500 random training samples (cached in `runs/<model>/norm_stats.pt`)
2. Train-time: inputs **and** targets normalized → MSE loss in z-score space
3. MAPE computed in **physical MWh space** after de-normalizing predictions
4. Eval wrappers ([part1-baseline](evaluation/part1-baseline/model.py) / [part2-encoder-decoder](evaluation/part2-encoder-decoder/model.py)) re-apply the ckpt-embedded `norm_stats` so the TA evaluator sees raw MWh

---

## Architectures

### Part 1 — Baseline ([cnn_transformer_baseline.py](models/cnn_transformer_baseline.py), 1.75 M params)

Hybrid CNN-Transformer, follows assignment Figure 2:

```
Weather (B,S+24,450,449,7) → Shared ResBlock CNN → 8×8 spatial token grid (P=64, D=128)
    +
Tabular tokens (1 per hour): Linear(demand+calendar → D)
    +
Spatial pos-embed × Temporal pos-embed × Tabular type-embed
    ↓
Single 4-layer Transformer encoder (self-attention over 3120 tokens)
    ↓
Slice 24 future tabular tokens → MLP(128→64→8) → (B, 24, 8) MWh
```

### Part 2 — Encoder-Decoder ([cnn_encoder_decoder.py](models/cnn_encoder_decoder.py), 2.29 M params)

Day-ahead forecasting is a translation-style task: known future covariates act as **queries**, past observations as **memory**. We split the architecture accordingly:

```
              Encoder (4 layers self-attn over history-only)
              S × (P+1) = 1560 tokens →  mem_hist
                        ↓
24 decoder queries (seeded from future calendar embedding)
              ↓
              Decoder (2 layers: self-attn → cross-attn → MLP)
                        ↓
              MLP(128→64→8) → (B, 24, 8) MWh
```

**Why encoder-decoder.** Single-encoder baseline entangles past observations and future queries in one self-attention pool. The dedicated decoder gives the model a clean inductive bias: "future = query, past = memory". Encoder attention cost drops ~4× (1560² vs 3120²); the savings get reinvested in more epochs / a larger grid.

**Optional ablation** (`--use_future_weather_xattn`): adds a second cross-attention branch in each decoder block that attends to the **future** weather spatial tokens (24 × 64 = 1536 KV). Restores information parity with baseline.

Full report: [docs/part2_report.md](docs/part2_report.md).

---

## Results

### Test MAPE — last 2 days of 2022 (TA-defined slice)

| Model | Params | Overall | ME | NH | VT | CT | RI | SEMA | WCMA | NEMA_BOST |
|---|---|---|---|---|---|---|---|---|---|---|
| **Baseline (Part 1)** ⭐ | 1.75 M | **5.24 %** | 2.31 | 3.69 | 5.95 | 7.28 | 5.27 | 5.44 | 5.87 | 6.09 |
| Part 2 v1 (history-only xattn, epoch-6 best) | 2.29 M | 6.82 % | 3.22 | 5.67 | 5.85 | 9.56 | 7.45 | 7.22 | 7.38 | 8.24 |
| Part 2 v2 (hist+future xattn, **undertrained**) | 2.42 M | 9.27 % | 5.91 | 8.51 | 7.78 | 12.53 | 8.25 | 7.38 | 11.60 | 12.20 |

⚠ **v2 caveat.** v2 reached only 13 of 24 planned epochs (~54 %) before the 3-job SLURM chain TIMEOUT'd 3 times. Its 9.27 % is a *partial measurement* — at epoch 13 the cosine LR is still ~9 × 10⁻⁴, well above the small-LR fine-tuning regime where baseline made its big drop. Both v1 and v2 share the same chained-resume LR-scheduler reset bug; details in the [v2 model card](pretrained_models/v2_encoder_decoder_xattn/README.md).

### Training trajectory (val MAPE, lower = better)

| Epoch | Baseline (val 2021) | v1 (val 2022) | v2 (val 2022, undertrained) |
|---|---|---|---|
| 0 | 11.22 % | 10.08 % | _(log overwritten on resume)_ |
| 4 | 9.16 % | 8.70 % | _(log overwritten on resume)_ |
| 6 | 8.76 % | **8.63 %** ✓ | _(log overwritten on resume)_ |
| 8 | — | 9.95 % | 8.95 % (LR reset to 1e-3) |
| 13 (final for baseline / v2) | **6.92 %** | 10.50 % | **8.72 %** ✓ |

Both encoder-decoder variants stall after the LR-scheduler reset that fires on every chained `--resume`; baseline ran continuously for 14 epochs and reached the small-LR fine-tuning regime.

---

## Project Structure

```
real-time-power-predict/
├── README.md / README_zh.md         # English / Chinese overview
├── CLAUDE.md                        # Repo-specific operating rules
├── .gitignore
├── docs/
│   ├── assignment.pdf               # Course handout
│   ├── progress.md                  # Work plan & status
│   ├── part2_report.md              # Part 2 technical report
│   ├── part3_references.md          # Part 3 reading list & track plan
│   └── hpc-evaluation-structure.md  # Tufts HPC evaluator layout
├── models/
│   ├── __init__.py                  # Registry: create_model, MODEL_DEFAULTS
│   ├── cnn_transformer_baseline.py  # Part 1 baseline (encoder-only, 1.75M)
│   └── cnn_encoder_decoder.py       # Part 2 encoder-decoder (2.29M)
├── training/
│   ├── train.py                     # Training entry point (shared by both models)
│   └── data_preparation/dataset.py  # Dataset + LRU weather cache + z-score norm
├── evaluation/
│   ├── part1-baseline/              # Part 1 eval wrapper → HPC part1-models/pangliu/
│   └── part2-encoder-decoder/       # Part 2 eval wrapper → HPC part2-models/pangliu/
├── inference/
│   └── predict.py                   # CLI inference (offline forecast)
├── space/                           # HF Spaces real-time demo (Part 3 work)
│   ├── app.py                       # Gradio UI
│   ├── model_utils.py               # Self-contained model loading
│   ├── iso_ne_fetch.py              # ISO Express real-time data fetcher (placeholder)
│   ├── requirements.txt             # Space dependencies
│   └── README.md                    # Deployment notes
├── tests/
│   └── smoke_test.py                # Param count + forward+backward sanity check
├── runs/
│   ├── model_registry.json          # Central record of all trained models
│   ├── cnn_transformer_baseline/    # Part 1 artifacts
│   └── cnn_encoder_decoder/         # Part 2 artifacts (after training)
└── scripts/
    ├── train.slurm                          # Part 1 training SLURM
    ├── train_cnn_encoder_decoder.slurm      # Part 2 training SLURM (24h, any GPU)
    ├── self_eval.py + self_eval.slurm       # Model-agnostic MAPE evaluator
    ├── self_test.slurm + smoke_test.slurm   # Sanity checks
    ├── cuda_probe.slurm                     # GPU module compatibility probe
    ├── setup_conda_env.slurm + fix_conda_env.slurm   # Build the cs137 conda env
    └── hf_upload.py                          # Push checkpoint to HF Hub
```

---

## Quick Start

### 1. Sync code to HPC and train

```bash
# Local → HPC
rsync -avz --exclude=__pycache__ --exclude=.git --exclude=runs --exclude=data \
    ./ tufts-login:/cluster/tufts/c26sp1cs0137/pliu07/assignment3/

# Submit chain (3 × 24 h, any GPU, --resume from latest.pt)
ssh tufts-login
cd /cluster/tufts/c26sp1cs0137/pliu07/assignment3
sbatch scripts/train_cnn_encoder_decoder.slurm \
    --resume runs/cnn_encoder_decoder/checkpoints/latest.pt
```

### 2. Pull results back

```bash
rsync -avz --exclude=checkpoints \
    tufts-login:/cluster/tufts/c26sp1cs0137/pliu07/assignment3/runs/ \
    ./runs/
# Pull specific checkpoint when ready
rsync -avz tufts-login:/cluster/.../checkpoints/best.pt \
    ./runs/cnn_encoder_decoder/checkpoints/
```

### 3. Run TA evaluator (independent verification)

```bash
ssh tufts-login
cd /cluster/tufts/c26sp1cs0137/pliu07/assignment3
sbatch scripts/self_eval.slurm runs/cnn_encoder_decoder/checkpoints/best.pt 2
```

Submission folder for grading: `/cluster/.../evaluation/part2-models/pangliu/` (per Piazza 4/16) — see [docs/hpc-evaluation-structure.md](docs/hpc-evaluation-structure.md).

### 4. Run smoke test locally (CPU OK)

```bash
python -m tests.smoke_test
```

### 5. CLI inference on a saved sample

```bash
python -m inference.predict \
    --checkpoint runs/cnn_encoder_decoder/checkpoints/best.pt \
    --sample tests/sample_input.pt
```

---

## Status

| Part | Weight | Due | Status |
|---|---|---|---|
| Part 1 — Baseline CNN-Transformer | 40 | Apr 15 | ✅ Done, test MAPE **5.24 %** |
| Part 2 v1 — Encoder-decoder (history-only xattn) | 30 | Apr 22 | ✅ Done, test MAPE **6.82 %** (best.pt @ epoch 6) |
| Part 2 v2 — Encoder-decoder (hist+future xattn) | — | — | ⚠ **Undertrained** — best.pt @ epoch 13/24 (~54 %), test MAPE **9.27 %**; 3 SLURM TIMEOUTs (5/1, 5/2, 5/3); see [v2 card](pretrained_models/v2_encoder_decoder_xattn/README.md) |
| Part 3 Track A — Geographic attention maps | 30 | May 1 | ✅ figures rendered on HPC and pulled |
| Part 3-2 (bonus) — Real-time deployment study | — | — | ✅ Live at [`huggingface.co/spaces/jeffliulab/predict-power`](https://huggingface.co/spaces/jeffliulab/predict-power) |
| Report + presentation | — | May 1 / May 4 | ✅ Submitted (`v1.1`); v1.2 release with figures + v2 result is in flight |

---

## Acknowledgments

- **Compute**: Tufts Research Technology HPC (NVIDIA A100-80GB / 40GB / P100)
- **Course**: Tufts CS 137 — Deep Neural Networks, Spring 2026
- **Sibling project**: [real_time_weather_forecasting](https://github.com/jeffliulab/real_time_weather_forecasting) (assignment 2) — shared HRRR data plumbing and project layout patterns
