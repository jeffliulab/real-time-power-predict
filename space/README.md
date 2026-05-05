---
title: ISO-NE Energy Demand Forecasting
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.30.0
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
short_description: Real-time day-ahead demand forecasting for ISO New England
---

# ⚡ Multi-Modal Deep Learning for Energy Demand Forecasting

Live demo of the trained CNN-Transformer baseline (1.75 M params) from our CS-137 final project (Tufts, Spring 2026), blended in a per-zone weighted ensemble with **Chronos-Bolt-mini** (Amazon, 21 M params, zero-shot on demand history).

**All inputs are now fully real** — no synthetic weather, no proportionally-split system demand:

- HRRR f00 weather analyses for the past 24 h (NOAA AWS S3, public)
- HRRR f01..f24 forecast for the future 24 h (most recent long cycle ≤ T-2h)
- True per-zone load from ISO-NE's public 5-minute zonal estimated load feed
- Calendar features (deterministic from timestamps)

Headline offline number: **5.24 % MAPE** baseline / **4.21 % MAPE** ensemble (with future analyses at training time — see disclosure in `about.md`). Live MAPE will be modestly worse because deployment substitutes HRRR forecasts for the future window.

## What it does

1. **Real-time tab**: every click pulls real ISO-NE per-zone demand + real HRRR weather and runs the ensemble. Expect ~3-5 min on the very first click of a fresh Space (cold HRRR cache + Chronos load), then ~10-30 s on subsequent clicks within the same uptime session.
2. **Backtest tab**: 7 daily forecasts on the most recent fully-published days, with full predict-vs-truth comparisons + per-zone MAPE table. Refreshed daily by a GitHub Actions cron in the [auxiliary data repo](https://github.com/jeffliulab/new-england-real-time-power-predict-data).

## Links

- 📄 [Final report (PDF)](https://github.com/jeffliulab/real-time-power-predict/blob/main/report/final_report.pdf)
- 💻 [GitHub repository](https://github.com/jeffliulab/real-time-power-predict)
- 👤 Author: **Pang Liu** · `pliu07` · Tufts CS-137

## Local development

```bash
cd space
pip install -r requirements.txt
python app.py     # http://localhost:7860
```

## File map

| File | Purpose |
|---|---|
| `app.py` | Gradio Blocks UI + Real-time / Backtest / About tabs |
| `iso_ne_fetch.py` | High-level demand fetcher: live ISO-NE 5-min → hourly + bundled CSV fallback + 30-day data-repo cache |
| `iso_ne_zonal.py` | Low-level ISO-NE 5-minute zonal CSV fetcher (cookie-prime) |
| `hrrr_fetch.py` | Real-time HRRR weather fetcher (Herbie + AWS S3 + KDTree-based regrid + `/tmp` cache) |
| `calendar_features.py` | 44-d calendar one-hot encoder |
| `model_utils.py` | Model loading + inference + Chronos ensemble |
| `models/cnn_transformer_baseline.py` | Baseline architecture (1.75 M params) |
| `checkpoints/best.pt` | Trained baseline weights (~20 MB) |
| `checkpoints/norm_stats.pt` | z-score statistics (weather + energy) |
| `assets/` | Figures shown in the *About* tab + bundled fallback samples |
| `assets/backtest_fallback.json` | Last-known-good backtest snapshot (used if data repo unreachable) |
| `about.md` | Demo explanation rendered in the UI |
| `packages.txt` | apt-style packages: `libeccodes-dev`, `libeccodes-tools` (for cfgrib) |

## No secrets required

The Space pulls real data from public, no-auth endpoints:
- ISO-NE: `https://www.iso-ne.com/transform/csv/fiveminuteestimatedzonalload?start=...&end=...` (with browser-cookie prime; see `iso_ne_zonal.py`)
- HRRR: `s3://noaa-hrrr-bdp-pds/hrrr.{date}/conus/...` via the Herbie library

The Backtest tab loads pre-built JSON from the auxiliary data repo
[`new-england-real-time-power-predict-data`](https://github.com/jeffliulab/new-england-real-time-power-predict-data),
also public; no auth needed.
