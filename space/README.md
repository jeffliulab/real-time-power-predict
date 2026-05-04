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

Live demo of two models from our CS-137 final project (Tufts, Spring 2026):

1. **Part 1 baseline** — CNN-Transformer (1.75 M params), reaches **5.24 % MAPE** with real HRRR weather on the 2022 self-eval slice.
2. **Ensemble (Baseline ⊕ Chronos-Bolt-mini, zero-shot, per-zone α)** — adds the 21 M-param Amazon foundation model on demand history alone (no weather, no fine-tuning) and reaches **4.21 % MAPE** in offline evaluation.

## What it does

1. **Real-time tab**: every click pulls the most recent 24 h of ISO New England system demand from the [EIA Open Data API](https://www.eia.gov/opendata/) (`respondent=ISNE`, `type=D`), splits it into the 8 ISO-NE zones via fixed proportions, and runs the chosen model on it. The Space holds a personal `EIA_API_KEY` as a Secret; if EIA is unreachable we fall through to an ISO-NE legacy endpoint and finally to a bundled 2022 sample.
2. **Backtest tab**: 7 pre-computed daily forecasts (Dec 25–31, 2022 at 00:00 UTC) with all three models side-by-side and a per-zone MAPE table. The baseline curves there were computed on the Tufts HPC cluster with **real HRRR weather**, so this tab reaches the headline accuracy that the live tab can't get without weather inputs.

## ⚠ Demo limitation — synthetic weather inputs (live tab)

The live tab substitutes **zeros** (training-mean weather in z-score space) for the baseline's weather raster channels because real-time HRRR isn't accessible from the Space. Calendar features (hour-of-day, day-of-week, month, holiday flag) and the recent demand pattern still drive the output, so the forecast shape is preserved, but absolute accuracy is lower than the cluster's 5.24 %. **Ensemble** mode largely closes the gap because Chronos-Bolt-mini doesn't need weather at all.

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
| `iso_ne_fetch.py` | Live demand fetch: EIA API → ISO-NE legacy → bundled CSV |
| `calendar_features.py` | 44-d calendar one-hot encoder |
| `model_utils.py` | Baseline + Chronos-Bolt-mini loading, inference, per-zone ensemble |
| `models/cnn_transformer_baseline.py` | Baseline architecture (1.75 M params) |
| `checkpoints/best.pt` | Trained baseline weights (~20 MB) |
| `checkpoints/norm_stats.pt` | z-score statistics for de-/normalization |
| `assets/backtest_2022_last7d.json` | 7-day cached forecasts shown in the Backtest tab |
| `assets/` | Figures shown in the *About* tab |
| `about.md` | Demo explanation rendered in the UI |

## Secrets

| Name | Purpose |
|---|---|
| `EIA_API_KEY` | Personal EIA Open Data key for live ISO-NE demand. Free; register at https://www.eia.gov/opendata/register.php. Without this secret the Space still works — it just falls through to the ISO-NE legacy endpoint and (if that also fails) a bundled 2022 sample. |
