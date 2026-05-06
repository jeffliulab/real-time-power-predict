<div align="center">

[![en](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![zh](https://img.shields.io/badge/lang-中文-red.svg)](README_zh.md)

<h1>ISO-NE Day-Ahead Energy Demand Forecasting</h1>

<p>
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.5-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/release-v1.5-brightgreen" alt="v1.5">
  <a href="https://huggingface.co/spaces/jeffliulab/predict-power">
    <img src="https://img.shields.io/badge/🤗%20Live%20demo-HF%20Space-yellow" alt="Live demo">
  </a>
  <a href="https://github.com/jeffliulab/new-england-real-time-power-predict-data">
    <img src="https://img.shields.io/badge/📦%20Data%20repo-cron--refreshed-blue" alt="Data repo">
  </a>
</p>

<p>
  Multi-modal CNN-Transformer + Chronos-Bolt-mini zero-shot ensemble for <strong>24-hour per-zone electricity demand</strong> forecasting across the 8 ISO New England load zones. Headline test MAPE: <strong>4.33 %</strong> on the last 2 days of 2022 (baseline 5.24 % alone). Deployed publicly with a daily-refreshed 7-day rolling backtest, used as the case study for the workshop paper at <a href="docs/paper.pdf">docs/paper.pdf</a>.
</p>

<p>
  <strong>🌐 Live demo</strong>: <a href="https://huggingface.co/spaces/jeffliulab/predict-power">huggingface.co/spaces/jeffliulab/predict-power</a> — real ISO-NE per-zone load + real HRRR weather, refreshed daily.<br>
  <strong>📄 Workshop paper</strong>: <a href="docs/paper.pdf">docs/paper.pdf</a> — full LaTeX source at <a href="report/arxiv/"><code>report/arxiv/</code></a>.<br>
  <strong>📦 Auxiliary data repo</strong>: <a href="https://github.com/jeffliulab/new-england-real-time-power-predict-data">jeffliulab/new-england-real-time-power-predict-data</a> — GitHub Actions cron rebuilds the rolling backtest daily at 04:00 UTC.
</p>

</div>

---

## Highlights

- **Trained baseline** (1.75 M params): single-encoder CNN-Transformer over 7-channel HRRR rasters + 24-h per-zone demand history + 44-d calendar one-hots → **5.24 % test MAPE** on the 2-day 2022 self-eval slice ([model code](space/models/cnn_transformer_baseline.py)).
- **Foundation-model ensemble** ⭐: per-zone weighted late fusion of the trained baseline with **Chronos-Bolt** ([Amazon, Apache-2.0](https://huggingface.co/amazon/chronos-bolt-mini), used **zero-shot** with no fine-tuning) → **4.33 % test MAPE** with the 205 M-param `chronos-bolt-base`, **4.21 %** with the 21 M-param `chronos-bolt-mini` (deployed in the live Space). The ensemble uses fixed per-zone weights $\alpha_z$ tuned on a 14-day 2022 validation window.
- **Real-time deployment** ⭐: a public HuggingFace Space pulling **live HRRR analyses + live HRRR forecasts + live ISO-NE per-zone demand** every click. Strict-discipline 7-day rolling backtest (only data available at each forecast time T is used). Backed by a separate **public auxiliary data repo** that runs the rolling backtest under GitHub Actions cron and pushes JSON the Space loads at startup.
- **Two-window deployment-drift study** (workshop paper): comparing W1 = 2025-05 vs W2 = 2026-04/05, the trained baseline degrades from **17.97 %** to **25.17 %** MAPE year-over-year, while the weather-agnostic Chronos zero-shot model only drifts from **9.62 %** to **13.45 %**. Per-zone breakdown shows the drift is sharply localised to MA + RI zones, consistent with state-level rooftop / behind-the-meter solar buildout.
- **Geographic attention diagnostics**: 4 figures with 4 orientation-correctness checks. Surprising finding — 7 of 8 prediction-head zones share a global attention pattern rather than zone-localising. Compressed into the workshop paper's appendix.

---

## Architecture (3-repo split)

```
┌─────────────────────────────────────────────────────────────────────┐
│  github.com/jeffliulab/real-time-power-predict      (this repo)     │
│  Code source-of-truth: training, models, inference,                 │
│  scripts/, attention diagnostics, paper sources, Space app code.    │
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

Why split: HF auto-syncs `real-time-power-predict` → Space, so committing daily backtest data to the main repo would force a Docker rebuild every day for a ~200 KB JSON. The split keeps the main repo's `git log` clean and lets third parties `curl` the latest backtest data without cloning anything.

---

## Task Definition

Given a window ending at time `t`, predict hourly MWh demand for **all 8 ISO-NE load zones** over `t+1 … t+24`.

| Component | Detail |
|-----------|--------|
| **Weather input** | `(B, S+24, 450, 449, 7)` — hourly HRRR f00 analyses (training); HRRR f01..f24 forecasts substituted at deployment |
| **Energy input** | `(B, S, 8)` — hourly per-zone MWh (ME, NH, VT, CT, RI, SEMA, WCMA, NEMA_BOST) |
| **Calendar** | `(B, S+24, 44)` — hour-of-day (24) + day-of-week (7) + month (12) + US-holiday flag (1) one-hots |
| **Output** | `(B, 24, 8)` — predicted MWh per zone for the next 24 h |
| **Metric** | MAPE in physical MWh space, averaged over horizon and zones; bootstrap 95 % CIs over (forecast-day, zone) pairs |
| **Test slice** | last 2 days of 2022 (2022-12-30, 2022-12-31), plus rolling 7-day live windows in 2025+ |

---

## Data and Normalization

| Property | Value |
|---|---|
| **Demand source** | ISO-NE Energy/Load/Demand reports portal (per-zone hourly archive); for live deployment, the 5-min `fiveminuteestimatedzonalload` CSV with the cookie-prime trick described in the paper's Appendix B |
| **Weather source** | NOAA HRRR f00 analyses on AWS S3 `s3://noaa-hrrr-bdp-pds/` via the `herbie-data` library |
| **Region** | New England, 40.5–47.5°N, –74 to –66°W, regridded to 450 × 449 (~1.6 km/cell) via Delaunay barycentric interpolation |
| **Coverage** | 2019 – 2023 hourly for the training corpus; 2025+ for the live deployment |
| **Train (baseline)** | 2019–2020 (~17 K samples) |
| **Validation** | 2021 (~8 K samples) |
| **Test (training-window self-eval)** | last 2 days of 2022 |
| **Test (live deployment)** | rolling 7-day windows (May 2025, May 2026 in the workshop paper) |

**Four-step normalization** ([`training/data_preparation/dataset.py`](training/data_preparation/dataset.py)) — single source of truth shared across all model variants:

1. `compute_norm_stats()` — z-score from 500 random training samples; cached in `runs/<model>/norm_stats.pt` and bundled into each checkpoint.
2. Train-time: weather AND demand inputs z-scored before the forward pass; targets also z-scored → MSE in z-space.
3. MAPE computed in **physical MWh space** after de-normalization.
4. Eval wrappers re-apply the checkpoint-embedded `norm_stats`, so the deployed pipeline serves predictions on the same MWh scale as the training-time evaluation.

---

## Architectures

### Baseline ([`cnn_transformer_baseline.py`](space/models/cnn_transformer_baseline.py), 1.75 M params)

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

### Foundation-model ensemble ([`space/model_utils.py`](space/model_utils.py))

Per-zone late fusion of the trained baseline with **Chronos-Bolt-mini** ([Ansari et al. 2024](https://arxiv.org/abs/2403.07815), Amazon, Apache-2.0), used **zero-shot** with no fine-tuning. The Chronos leg processes each zone independently as a univariate series: 720 hours of past per-zone demand history go in, a 24-hour median quantile forecast comes out per zone. No weather, no calendar, no zone label.

$$
\hat y_z = \alpha_z \cdot \hat y_z^{\text{baseline}} + (1 - \alpha_z) \cdot \hat y_z^{\text{Chronos}}
$$

The per-zone weights $\alpha_z$ are tuned on a 14-day validation window (2022-12-16 → 12-29) and hard-coded at deployment. Zones with $\alpha_z = 0$ (CT, SEMA, NEMA_BOST) drop the baseline entirely; zones with high $\alpha_z$ (VT = 0.80) are baseline-dominant.

### Geographic attention diagnostics ([`scripts/attention_maps.py`](scripts/attention_maps.py))

Slices the trained baseline's encoder attention to expose the future-tabular → history-spatial weights, reshapes to an 8×8 spatial grid, and overlays on the New England basemap with cartopy. Includes 4 orientation-correctness checks (slice direction, row-major reshape, compass orientation, east-vs-west aggregate sanity). Five diagnostic days span mild / heat-wave / cold-snap / extreme / holiday regimes.

---

## Results

### Training-window self-evaluation (2022-12-30 → 2022-12-31)

| Variant | Params | Overall MAPE | ME | NH | VT | CT | RI | SEMA | WCMA | NEMA_BOST |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline ⭐ | 1.75 M | **5.24 %** | 2.31 | 3.69 | 5.95 | 7.28 | 5.27 | 5.44 | 5.87 | 6.09 |
| **Ensemble (baseline + Chronos-Bolt-base)** ⭐ | 1.75 M + 205 M | **4.33 %** | — | — | — | — | — | — | — | — |
| **Ensemble (baseline + Chronos-Bolt-mini, deployed)** | 1.75 M + 21 M | **4.21 %** | — | — | — | — | — | — | — | — |

### Two-window deployment-drift case study (workshop paper §4)

Overall MAPE [95 % bootstrap CI] across two rolling 7-day windows, with naive baselines as a floor:

| Model | W1: 2025-05-01..07 | W2: 2026-04-28..05-04 |
|---|---|---|
| Baseline (CNN-Transformer + HRRR) | 17.97 % [13.65–22.84] | **25.17 %** [18.98–32.12] |
| Chronos-Bolt-mini (zero-shot) | 9.62 % [8.31–11.07] | **13.45 %** [10.75–16.39] |
| Ensemble (per-zone α) | 11.09 % [9.04–13.64] | **14.51 %** [11.23–18.07] |
| Persistence-1d | 9.97 % [8.23–11.78] | 15.34 % [11.79–19.35] |
| Persistence-7d | 12.80 % [10.44–15.08] | 15.50 % [12.64–18.68] |
| Climatological mean (4-week) | 13.20 % [10.90–15.80] | 15.77 % [12.69–19.23] |

The W1→W2 baseline jump (+7.2 pp) exceeds the within-window bootstrap CI band; the weather-agnostic Chronos drifts much less; per-zone breakdown localises the drift to MA + RI zones (workshop paper Figure 3).

### Pipeline-equivalence validation

Re-running the deployed pipeline on **2022-12-30** (a date inside the training window) reproduces the cluster's stored prediction to within 0.13 percentage points: cluster MAPE 6.54 % [5.02–7.92] vs live 6.41 % [4.99–7.69], paired diff -0.13 pp [-0.23, +0.00]. Element-wise differences are bounded by 94 MW (3.6 % of the cluster's prediction at the worst point). The deployed pipeline is therefore certified equivalent to the training-time evaluation; subsequent live-deployment numbers measure honest input-distribution change, not bugs.

---

## Real-time Deployment

The HF Space at [`jeffliulab/predict-power`](https://huggingface.co/spaces/jeffliulab/predict-power) serves three tabs:

1. **Real-time forecast** — every click pulls 24 h of HRRR f00 analyses + 24 h of HRRR forecasts from the latest long cycle (00/06/12/18 UTC, since shorter cycles only go to f18) + the latest 24 contiguous hours of ISO-NE 5-minute zonal load. Runs the baseline + Chronos-Bolt-mini and renders the per-zone weighted ensemble. Cold start ~3-5 min (HRRR fetch + Chronos load); subsequent clicks ~10-30 s.
2. **Backtest (last 7 days)** — 7 daily forecasts on the most recent 7 fully-published days, with strict-discipline windowing. Per-zone × per-model MAPE table + overall MAPE bars.
3. **About** — dynamic MAPE summary (auto-refreshed each cron run).

Live performance is auto-refreshed daily by GitHub Actions in the auxiliary data repo:

```bash
curl -s https://raw.githubusercontent.com/jeffliulab/new-england-real-time-power-predict-data/main/data/last_built.json
```

---

## v1.5 release contents

This is the canonical post-coursework public release. All artifacts are pinned at the `v1.5` git tag.

| Artifact | Location | Type |
|---|---|---|
| Trained baseline (1.75 M params, 5.24 % test MAPE) | https://huggingface.co/jeffliulab/predict-power-baseline | model |
| Live demo (real-time + 7-day rolling backtest) | https://huggingface.co/spaces/jeffliulab/predict-power | demo |
| Daily-refreshed backtest data (cron, GitHub Actions) | https://github.com/jeffliulab/new-england-real-time-power-predict-data | data |
| Workshop paper PDF | [docs/paper.pdf](docs/paper.pdf) | paper |
| arXiv source (LaTeX) | [report/arxiv/](report/arxiv/) | paper source |
| Experimental data (4 JSONs) | [report/arxiv/data/](report/arxiv/data/) | data |
| Trained checkpoint (in-repo copy) | `pretrained_models/baseline/best.pt`, `space/checkpoints/best.pt` | model |
| Training-CSV demand (2019-2022 hourly per-zone) | `pretrained_models/baseline/dump/demand_2019_2022_hourly.csv` | data |

---

## Project Structure

```
real-time-power-predict/
├── README.md / README_zh.md
├── CLAUDE.md                                # Repo-specific operating rules
├── docs/
│   └── paper.pdf                            # Workshop paper PDF (mirror of report/arxiv/paper.pdf)
├── report/
│   └── arxiv/                               # Workshop paper LaTeX source + figures + experimental JSON
│       ├── paper.tex, paper.pdf, preamble.tex, refs.bib
│       ├── sec_*.tex (intro / setup / validation / drift / btm / discussion)
│       ├── appendix_*.tex (A–H)
│       ├── data/                            # Experimental outputs (multi-window, validation, BTM, forecast-shift)
│       └── figures/                         # All paper figures
├── space/                                   # HF Space source (auto-synced to HF on push)
│   ├── app.py                               # Gradio Blocks UI: Real-time / Backtest / About tabs
│   ├── iso_ne_zonal.py                      # ISO-NE 5-min zonal endpoint (cookie-prime)
│   ├── iso_ne_fetch.py                      # High-level demand fetcher + cache + fallback
│   ├── hrrr_fetch.py                        # HRRR realtime: Herbie + Delaunay barycentric regrid + /tmp cache
│   ├── model_utils.py                       # Baseline + Chronos-Bolt-mini + per-zone ensemble
│   ├── calendar_features.py                 # 44-d calendar one-hots
│   ├── models/cnn_transformer_baseline.py   # Baseline architecture (also used by training)
│   ├── checkpoints/best.pt + norm_stats.pt
│   ├── assets/                              # backtest_fallback.json + figures
│   ├── about.md                             # About-tab static prose
│   ├── packages.txt                         # libeccodes-dev / libeccodes-tools (apt-style)
│   └── requirements.txt
├── training/
│   ├── train.py
│   └── data_preparation/dataset.py          # 4-step normalization pipeline source-of-truth
├── inference/
│   └── predict.py                           # CLI offline inference
├── pretrained_models/                       # HF-style model cards + checkpoint mirrors
├── runs/
│   ├── model_registry.json
│   └── cnn_transformer_baseline/ ...        # Training-run artifacts (logs, checkpoints, figures)
├── models/                                  # Encoder-decoder source (lineage marker; not in workshop paper)
├── tests/
│   └── smoke_test.py                        # Sanity check (param count, forward pass)
└── scripts/
    ├── data_preparation/                    # Public ISO-NE + HRRR offline fetchers (training-data construction)
    ├── baselines/                           # Naive baselines (workshop paper §4)
    ├── experiments/                         # historical_drift_sweep.py (multi-window experiment)
    ├── validation/                          # reproduce_dec30_2022.py + future_weather_shift_quantify.py
    ├── figures/                             # bootstrap_mape.py + render_*.py for paper figures
    ├── attention_maps.py                    # Attention diagnostics
    ├── build_rolling_backtest.py            # Cron-driven 7-day backtest builder (used by data repo)
    └── build_rolling_backtest.requirements.txt
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

The Space loads `BACKTEST` from the public data repo on startup; without internet, drop `space/assets/backtest_fallback.json` next to `app.py` and it'll use the bundled snapshot.

### Smoke-test the model pipeline

```bash
python -m tests.smoke_test
```

### Reproduce the rolling-backtest cron locally

```bash
pip install -r scripts/build_rolling_backtest.requirements.txt
python scripts/build_rolling_backtest.py --output-dir /tmp/backtest --parallel 8
```

### Reproduce the workshop paper's multi-window experiment

```bash
python scripts/experiments/historical_drift_sweep.py
# ~75 min per window; outputs report/arxiv/data/multi_window_results.json
```

### Reproduce the pipeline-equivalence validation

```bash
python scripts/validation/reproduce_dec30_2022.py
python scripts/validation/augment_validation_with_ci.py
# Expected MAPE: live 6.41 % vs cluster 6.54 %, diff ≤ 0.13 pp
```

### Re-train the baseline

```bash
python training/train.py --model cnn_transformer_baseline \
    --epochs 14 --batch_size 16 --lr 3e-4
# Requires NVIDIA GPU (training was performed on A100-80GB / 40GB)
```

---

## Reproducibility

- **Random seeds**: not set in the training pipeline, so the headline 5.24 % / 4.33 % numbers are not bit-reproducible across re-training runs. The numbers in this README and the paper are pinned to the specific checkpoint at `pretrained_models/baseline/best.pt` (equivalently `space/checkpoints/best.pt`), shipped at the v1.5 tag.
- **Datasheet**: see Appendix C of the workshop paper.
- **Public data sources**: ISO-NE Energy/Load/Demand reports portal + NOAA HRRR S3 mirror are both no-auth public; the live demo and the rolling-backtest cron use them directly.
- **Numeric claims → source data**: every MAPE number in the paper traces to a JSON file in `report/arxiv/data/` produced by a script under `scripts/`. See [report/arxiv/README.md](report/arxiv/README.md) for the full mapping.

---

## Acknowledgments

- **Public data**: ISO New England (per-zone 5-minute zonal load) + NOAA HRRR (3 km mesoscale weather)
- **Foundation model**: Chronos-Bolt by Amazon (Apache-2.0)
- **Compute**: NVIDIA A100-80GB / 40GB / P100 for training; HuggingFace Spaces cpu-basic for serving
- **Author**: Pang Liu, Independent Researcher, [`jeff.pang.liu@gmail.com`](mailto:jeff.pang.liu@gmail.com)
