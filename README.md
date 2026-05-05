<div align="center">

[![en](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![zh](https://img.shields.io/badge/lang-中文-red.svg)](README_zh.md)

<h1>ISO-NE Day-Ahead Energy Demand Forecasting</h1>

<p>
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.5-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <a href="https://huggingface.co/spaces/jeffliulab/predict-power">
    <img src="https://img.shields.io/badge/🤗%20Live%20demo-HF%20Space-yellow" alt="Live demo">
  </a>
  <img src="https://img.shields.io/badge/Submission-Final%20(CS--137)-brightgreen" alt="Status">
</p>

<p>
  Multi-modal CNN-Transformer for <strong>24-hour per-zone electricity demand</strong> forecasting across the 8 ISO New England load zones, blended in a per-zone weighted ensemble with a zero-shot foundation model (Chronos-Bolt-base). Headline test MAPE: <strong>4.33 %</strong> on the last 2 days of 2022 — a 17 % relative reduction over the 1.75 M-parameter baseline alone (5.24 %).
</p>

<p>
  <strong>🌐 Live demo</strong>: <a href="https://huggingface.co/spaces/jeffliulab/predict-power">huggingface.co/spaces/jeffliulab/predict-power</a> — runs on real ISO-NE per-zone load + real HRRR weather, with a <strong>7-day rolling backtest</strong> refreshed daily by GitHub Actions cron.
</p>

</div>

---

## Highlights

This repository contains the **final submission** for Tufts CS-137 Spring 2026 (graded individually as a solo project). Four contributions:

- **Part 1 — Baseline CNN-Transformer** (1.75 M params): single-encoder over 7-channel HRRR rasters + per-zone demand + 44-d calendar one-hots → **5.24 % test MAPE** ([model code](space/models/cnn_transformer_baseline.py)).
- **Part 2a — Encoder-decoder family** (2.29 / 2.42 M params): two architecture-search variants. v1 (history-only cross-attention) hits 6.82 % test MAPE; v2 (hist+future cross-attention) is undertrained at 9.27 %. Both lose to the baseline.
- **Part 2b — Foundation-model ensemble** ⭐: per-zone late fusion of the baseline with **Chronos-Bolt-base** (Amazon, 205 M params, Apache-2.0, used **zero-shot** with no fine-tuning) → **4.33 % test MAPE** (–0.91 pp / 17 % relative). The deployed mini variant (Chronos-Bolt-mini, 21 M params) reaches **4.21 %** on the same slice.
- **Part 3 — Attention diagnostics**: 4 figures with 4 geographic-orientation correctness checks (slice direction, row-major reshape, compass orientation, aggregate east-vs-west sanity). Reveals the surprising result that 7 of 8 prediction-head zones share a global attention pattern rather than zone-localising.
- **Real-time deployment** ⭐: a public HuggingFace Space pulling **live HRRR analyses + live HRRR forecasts + live ISO-NE per-zone demand** every click, with a strict-discipline 7-day backtest that uses only data available at each forecast time. Backed by a separate **public auxiliary data repo** that runs the rolling backtest under GitHub Actions cron and pushes JSON/CSV the Space loads at startup.

All checkpoints + model cards are in [`pretrained_models/`](pretrained_models/) (HF-style layout). The full report is at [`report/submission/submission_report.pdf`](report/submission/submission_report.pdf); the slides are at [`report/slides/slides.pdf`](report/slides/slides.pdf).

---

## Architecture (3-repo split)

```
┌─────────────────────────────────────────────────────────────────────┐
│  github.com/jeffliulab/real-time-power-predict      (this repo)     │
│  Code source-of-truth: training, models, inference,                 │
│  scripts/, attention diagnostics, report sources, Space app code.   │
│  ─ HF Space syncs from main branch on every push.                   │
└────────────────────────┬────────────────────────────────────────────┘
                         │ actions/checkout
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ github.com/jeffliulab/new-england-real-time-power-predict-data      │
│ (public, automation-only repo)                                      │
│ ─ .github/workflows/refresh.yml runs daily at 04:00 UTC             │
│ ─ Runs scripts/build_rolling_backtest.py from this repo with        │
│   8-way parallel HRRR fetch + ISO-NE 5-min zonal + Chronos-Bolt-mini│
│ ─ Commits data/backtest_rolling_7d.json + data/iso_ne_30d.csv +     │
│   data/last_built.json on changes                                   │
└────────────────────────┬────────────────────────────────────────────┘
                         │ HTTPS GET on Space startup
                         │ raw.githubusercontent.com/.../main/data/*
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ huggingface.co/spaces/jeffliulab/predict-power                      │
│ ─ Real-time tab: real HRRR + real ISO-NE per-zone, live ensemble    │
│ ─ Backtest tab: 7-day rolling cache from the data repo (instant)    │
│ ─ About tab: dynamic MAPE summary (refreshed every cron run)        │
└─────────────────────────────────────────────────────────────────────┘
```

Why split: HF auto-syncs `real-time-power-predict` → Space, so committing daily backtest data to the main repo would force a Docker rebuild every day for a ~200 KB JSON. The split keeps the main repo's `git log` clean (only real code changes) and lets third parties `curl` the latest backtest data without cloning anything.

---

## Task Definition

Given a window ending at time `t`, predict hourly MWh demand for **all 8 ISO-NE load zones** over `t+1 … t+24`. See [`docs/assignment.pdf`](docs/assignment.pdf) for the full spec.

| Component | Detail |
|-----------|--------|
| **Weather input** | `(B, S+24, 450, 449, 7)` — hourly HRRR f00 analyses (training); HRRR f01..f24 forecasts substituted at deployment |
| **Energy input** | `(B, S, 8)` — hourly per-zone MWh (ME, NH, VT, CT, RI, SEMA, WCMA, NEMA_BOST) |
| **Calendar** | `(B, S+24, 44)` — hour-of-day (24) + day-of-week (7) + month (12) + US-holiday flag (1) one-hots |
| **Output** | `(B, 24, 8)` — predicted MWh per zone for the next 24 h |
| **Metric** | MAPE in physical MWh space, averaged over horizon and zones |
| **Test slice** | last 2 days of 2022 (matches cluster's `inference/test_run.sh` default) |

---

## Data and Normalization

| Property | Value |
|---|---|
| **Demand source** | ISO-NE Energy/Load/Demand reports portal (per-zone hourly archive) |
| **Weather source** | NOAA HRRR f00 analyses on AWS S3 `noaa-hrrr-bdp-pds` |
| **Region** | New England, 40.5–47.5°N, –74 to –66°W, regridded to 450×449 (~1.6 km/cell) |
| **Coverage** | 2019 – 2023 hourly |
| **Train (Part 1)** | 2019–2020 (~17 K samples) |
| **Train (Part 2)** | 2019–2021 (~26 K samples) |
| **Val (Part 1)** | 2021 |
| **Val (Part 2)** | 2022 |
| **Test (self-eval)** | last 2 days of 2022 (2022-12-30, 2022-12-31) |
| **Test (TA-side)** | held-out 2024 slice |

**Four-step normalization** ([`training/data_preparation/dataset.py`](training/data_preparation/dataset.py)) — single source of truth shared across all model variants:

1. `compute_norm_stats()` — z-score from 500 random training samples; cached in `runs/<model>/norm_stats.pt` and bundled into each checkpoint.
2. Train-time: weather AND demand inputs z-scored before the forward pass; targets also z-scored → MSE in z-space.
3. MAPE computed in **physical MWh space** after de-normalization.
4. Eval wrappers re-apply the checkpoint-embedded `norm_stats`, so the TA evaluator and the HF Space both see raw MWh in/out.

See the report's *Dataset* section for the per-channel and per-zone z-score statistics used at training time.

---

## Architectures

### Part 1 — Baseline ([`cnn_transformer_baseline.py`](space/models/cnn_transformer_baseline.py), 1.75 M params)

```
Weather (B, S+24, 450, 449, 7) → Shared ResBlock CNN → 8×8 spatial token grid (P=64, D=128)
    +
Tabular tokens (1 per hour): Linear(demand+calendar → D)
    +
Spatial pos-embed × Temporal pos-embed × Tabular type-embed
    ↓
Single 4-layer Transformer encoder (self-attention over 3120 tokens)
    ↓
Slice 24 future tabular tokens → MLP(128→64→8) → (B, 24, 8) MWh
```

### Part 2a — Encoder-decoder ([`cnn_encoder_decoder.py`](models/cnn_encoder_decoder.py), 2.29 M params)

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

Day-ahead forecasting is a translation-style task: known future covariates are queries, past observations are memory. Encoder attention cost drops ~4× (1560² vs 3120²); the savings get reinvested in epochs / larger grids. Optional `--use_future_weather_xattn` adds a second cross-attention to the future weather spatial tokens (24 × 64 = 1536 KV) — the *v2* variant.

### Part 2b — Foundation-model ensemble ([`space/model_utils.py`](space/model_utils.py))

Per-zone late fusion of the trained baseline with **Chronos-Bolt** ([Ansari et al. 2024](https://arxiv.org/abs/2403.07815), Amazon, Apache-2.0), used **zero-shot** with no fine-tuning. The Chronos leg sees only 720 hours of per-zone demand history (no weather, no calendar); the baseline leg sees the full multimodal input.

$$
\hat y_z = \alpha_z \cdot \hat y_z^{\text{baseline}} + (1 - \alpha_z) \cdot \hat y_z^{\text{Chronos}}
$$

The per-zone weights $\alpha_z$ are tuned on a 14-day validation window in 2022. Headline numbers from the ablation (see report's *Foundation-model ensemble* section): **4.33 % test MAPE** with Chronos-Bolt-base (205 M params), **4.21 %** with Chronos-Bolt-mini (21 M params, deployed in the live Space).

### Part 3 — Attention diagnostics ([`scripts/attention_maps.py`](scripts/attention_maps.py))

Slices the trained baseline's encoder attention to expose the future-tabular → history-spatial weights, reshapes to an 8×8 spatial grid, and overlays on the New England basemap with cartopy. Includes 4 orientation-correctness checks (slice direction, row-major reshape, compass orientation, east-vs-west aggregate sanity). Four diagnostic days (`mild`, `heat-wave`, `cold-snap`, `extreme`, `holiday`) illustrate aggregate / per-hour / per-zone / extreme-vs-mild patterns. Surprising finding: 7 of 8 prediction-head zones share a global attention pattern rather than zone-localising.

---

## Results (test slice: 2022-12-30 → 2022-12-31)

| Variant | Params | Overall MAPE | ME | NH | VT | CT | RI | SEMA | WCMA | NEMA_BOST |
|---|---|---|---|---|---|---|---|---|---|---|
| Part 1 baseline ⭐ | 1.75 M | **5.24 %** | 2.31 | 3.69 | 5.95 | 7.28 | 5.27 | 5.44 | 5.87 | 6.09 |
| Part 2a v1 (history-only xattn) | 2.29 M | 6.82 % | 3.22 | 5.67 | 5.85 | 9.56 | 7.45 | 7.22 | 7.38 | 8.24 |
| Part 2a v2 (hist+future xattn, undertrained ⚠) | 2.42 M | 9.27 % | 5.91 | 8.51 | 7.78 | 12.53 | 8.25 | 7.38 | 11.60 | 12.20 |
| **Part 2b ensemble (baseline + Chronos-Bolt-base)** ⭐ | 1.75 M + 205 M | **4.33 %** | — | — | — | — | — | — | — | — |
| **Part 2b ensemble (baseline + Chronos-Bolt-mini, deployed)** | 1.75 M + 21 M | **4.21 %** | — | — | — | — | — | — | — | — |

⚠ **v2 caveat.** v2 reached only 13 of 24 planned epochs (~54 %) before the 3-job SLURM chain TIMEOUT'd 3 times. Its 9.27 % is a partial measurement; both v1 and v2 share the same chained-resume LR-scheduler reset bug. Details in the [v2 model card](pretrained_models/v2_encoder_decoder_xattn/README.md).

---

## Real-time Deployment ⭐

The HF Space at [`jeffliulab/predict-power`](https://huggingface.co/spaces/jeffliulab/predict-power) runs the full ensemble on **truly real, live** ISO-NE inputs — no synthetic weather, no proportionally-split system demand. Three tabs:

1. **Real-time forecast** — every click pulls 24 h of HRRR f00 analyses + 24 h of HRRR forecast hours from a long cycle (00/06/12/18 UTC, since shorter cycles only go to f18) + the latest 24 contiguous hours of ISO-NE 5-minute zonal load. Runs the baseline + Chronos-Bolt-mini and renders the per-zone weighted ensemble. Cold start ~3-5 min (HRRR fetch + Chronos load); subsequent clicks ~10-30 s.
2. **Backtest (last 7 days)** — 7 daily forecasts on the most recent 7 fully-published days, with strict-discipline windowing (at each forecast time T, only data available before T is used). Per-zone × per-model MAPE table + overall MAPE bars.
3. **About** — dynamic MAPE summary (auto-refreshed) + the disclosure on training→deployment distribution drift + figures.

### Live performance — auto-refreshed daily

```bash
curl -s https://raw.githubusercontent.com/jeffliulab/new-england-real-time-power-predict-data/main/data/last_built.json
```

Or open the Space's About tab to see the same numbers rendered, plus the per-zone breakdown in the Backtest tab. **Numbers are not hardcoded anywhere.**

### Why live MAPE is higher than the 4.21 % headline (and why this is the honest answer)

The trained baseline was fit on **2019–2022** weather + per-zone demand. Deployed today it sees **2026** inputs, and 3+ years of New-England-grid evolution (utility-scale BTM solar, EVs, post-COVID load patterns) has shifted the per-zone distribution — most visibly in the dense southern coastal zones (RI / SEMA / WCMA) where rooftop solar is heaviest.

### Live performance validation

To rule out a pipeline bug, we re-ran the live pipeline on **2022-12-30** (which is *in the training window* and where the cluster reports 6.54 % MAPE) using the training-CSV demand for that day:

| Pipeline | MAPE on 2022-12-30 |
|---|---|
| Cluster's stored prediction (headline-source) | 6.54 % |
| Live HF Space pipeline rerun | **6.41 %** |
| Element-wise diff | mean -5 MW, max abs 94 MW (3.6 % of cluster pred) |

The two pipelines reproduce each other. The deployed numbers therefore measure honest training→2026 distribution drift, not bugs.

### Reproducing the cron locally

```bash
python scripts/build_rolling_backtest.py \
    --output-dir /tmp/backtest \
    --parallel 8
```

Requires `herbie-data + cfgrib + xarray + eccodes + libeccodes-dev/-tools` (see `scripts/build_rolling_backtest.requirements.txt`).

---

## Project Structure

```
real-time-power-predict/
├── README.md / README_zh.md
├── CLAUDE.md
├── docs/
│   ├── assignment.pdf
│   ├── progress.md
│   └── ...
├── models/
│   └── cnn_encoder_decoder.py             # Part 2 architecture
├── space/                                 # HF Space source (auto-synced to HF)
│   ├── app.py                             # Gradio Blocks UI: live + backtest + about
│   ├── iso_ne_zonal.py                    # ISO-NE 5-min zonal endpoint (cookie-prime)
│   ├── iso_ne_fetch.py                    # High-level demand fetcher + cache + fallback
│   ├── hrrr_fetch.py                      # HRRR realtime: Herbie + KDTree-IDW + /tmp cache
│   ├── model_utils.py                     # Baseline + Chronos-Bolt-mini + per-zone ensemble
│   ├── calendar_features.py               # 44-d calendar one-hots
│   ├── models/cnn_transformer_baseline.py # Part 1 architecture (also used by training)
│   ├── checkpoints/best.pt + norm_stats.pt
│   ├── assets/                            # backtest_fallback.json + figures
│   ├── about.md                           # About-tab static prose
│   ├── packages.txt                       # libeccodes-dev / libeccodes-tools (apt-style)
│   └── requirements.txt
├── training/
│   ├── train.py
│   └── data_preparation/dataset.py        # 4-step normalization pipeline source-of-truth
├── evaluation/
│   ├── part1-baseline/                    # Tufts evaluator wrapper
│   └── part2-encoder-decoder/
├── inference/
│   └── predict.py                         # CLI offline inference
├── pretrained_models/                     # HF-style model cards + checkpoints
├── runs/
│   ├── model_registry.json
│   └── cnn_transformer_baseline/ ...
├── report/
│   ├── tex/submission.tex                 # Final report source
│   ├── submission/submission_report.pdf   # Final report PDF
│   ├── slides/slides.pdf + slides.pptx
│   └── figures/                           # All report figures (architecture, attention, ...)
└── scripts/
    ├── data_preparation/                  # Offline ISO-NE + HRRR fetchers (training data)
    ├── build_rolling_backtest.py          # Cron-driven 7-day backtest builder
    ├── build_rolling_backtest.requirements.txt
    ├── attention_maps.py                  # Part 3 diagnostics + 4 figure generators
    ├── train.slurm                        # Part 1 SLURM
    ├── train_cnn_encoder_decoder.slurm    # Part 2 SLURM
    ├── self_eval.py + self_eval.slurm     # MAPE evaluator
    └── ...
```

---

## Quick Start

### Run the live demo locally (CPU, no GPU needed)

```bash
cd space
pip install -r requirements.txt
sudo apt-get install libeccodes-dev libeccodes-tools     # for cfgrib/HRRR
python app.py
# Gradio launches at http://localhost:7860
```

The Space loads `BACKTEST` from the public data repo on startup; if your machine has no internet, drop `space/assets/backtest_fallback.json` next to `app.py` and it'll use the bundled snapshot.

### Smoke-test the model pipeline

```bash
python -m tests.smoke_test
```

### Re-run the rolling backtest builder (~10–15 min on 8 cores)

```bash
pip install -r scripts/build_rolling_backtest.requirements.txt
python scripts/build_rolling_backtest.py --output-dir /tmp/backtest --parallel 8
```

### Re-train Part 1 baseline on Tufts HPC

```bash
sbatch scripts/train.slurm
```

### Re-train Part 2 encoder-decoder

```bash
sbatch scripts/train_cnn_encoder_decoder.slurm \
    --resume runs/cnn_encoder_decoder/checkpoints/latest.pt
```

### Run TA evaluator (independent verification)

```bash
sbatch scripts/self_eval.slurm runs/cnn_encoder_decoder/checkpoints/best.pt 2
```

---

## Reproducibility

- **Random seeds**: not set in the training pipeline, so the headline 5.24 % / 4.33 % / 6.82 % numbers are not bit-reproducible across re-training runs. The numbers in this README are pinned to the specific checkpoints in [`runs/cnn_transformer_baseline/checkpoints/best.pt`](runs/cnn_transformer_baseline/checkpoints/best.pt) and [`runs/cnn_encoder_decoder/checkpoints/best.pt`](runs/cnn_encoder_decoder/checkpoints/best.pt), both bundled with the submission.
- **Datasheet**: see *Dataset Documentation* in the report.
- **Public data sources**: ISO-NE Energy/Load/Demand reports portal + NOAA HRRR S3 mirror are both no-auth public; the live demo and the rolling-backtest cron use them directly.

---

## AI-tool disclosure

The author used **Claude Code** (Anthropic; models: Opus 4.7 and Sonnet 4.6) as a coding and writing assistant throughout the project. Specifically: (i) code scaffolding (training-loop boilerplate, SLURM scripts, Gradio app stub, the rolling-backtest cron); (ii) figure-rendering code (matplotlib generators in `scripts/figures/` and the TikZ source for the architecture diagrams); (iii) LaTeX boilerplate; (iv) prose copy-editing on the report and `slides.md`. AI was **not** used for: research design, model architecture choices, training-result interpretation, the LR-scheduler-reset-bug diagnosis, the geographic-orientation correctness checks, or any final scientific claim. All design decisions, all numerical results, and all interpretation reflect the author's own judgment and responsibility.

---

## Acknowledgments

- **Compute**: Tufts Research Technology HPC (NVIDIA A100-80GB / 40GB / P100)
- **Course**: Tufts CS-137 — Deep Neural Networks, Spring 2026
- **Public data**: ISO New England + NOAA HRRR
- **Foundation model**: Chronos-Bolt by Amazon (Apache-2.0)

**Author**: Pang Liu (`pliu07`) · solo submission · all components attributable to a single contributor (see report's *Contribution Statement*).
