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

Live demo of the trained **CNN-Transformer baseline** from CS-137 Assignment 3 (Tufts University, Spring 2026). The baseline reaches **5.24 % MAPE** on the 2022 self-evaluation slice (last 2 days of 2022) when given real HRRR weather inputs.

## What it does

1. Pick a target datetime (UTC) — defaults to "now".
2. The Space fetches the last 24 hours of ISO New England system demand from the public ISO Express data feed and splits it into the 8 load zones using a fixed proportion vector estimated from 2022 historical zonal reports. (If the live feed is unreachable, it falls back to a bundled 24-hour CSV from 2022.)
3. Calendar features (hour-of-day, day-of-week, month, US-holiday flag) are computed for the past 24 h and the next 24 h.
4. The trained baseline runs forward and produces a 24-hour per-zone demand forecast in MWh.
5. You see two plots: an 8-panel per-zone history+forecast chart and a sorted bar of next-hour predicted demand.

## ⚠ Demo limitation — synthetic weather inputs

This Space substitutes **zeros** (training-mean weather in z-score space) for the model's weather raster channel. The cluster runs that hit **5.24 % MAPE** used real HRRR rasters. Calendar features and the recent demand pattern still drive the output, so the forecast shape is preserved, but absolute accuracy is degraded vs. the cluster.

The full real-weather pipeline (live HRRR fetch + per-zone real-time demand) is documented in the report and tracked as future work.

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
| `app.py` | Gradio Blocks UI + request handler |
| `iso_ne_fetch.py` | Live ISO-NE demand fetch (with CSV fallback) |
| `calendar_features.py` | 44-d calendar one-hot encoder |
| `model_utils.py` | Checkpoint loading + inference + denormalization |
| `models/cnn_transformer_baseline.py` | Baseline architecture (1.75 M params) |
| `checkpoints/best.pt` | Trained baseline weights (~20 MB) |
| `checkpoints/norm_stats.pt` | z-score statistics for de-/normalization |
| `assets/` | Figures shown in the *About* tab |
| `about.md` | Demo explanation rendered in the UI |
