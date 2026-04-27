# Space — Real-time ISO-NE Demand Forecast Demo

Hugging Face Spaces deployment for the trained Part 2 model.

**Status: scaffold — full real-time pipeline pending Part 3 work.**

## What this demo will do (when complete)

1. Click **Run Forecast**
2. Fetch the last 24 hours of ISO-NE actual zonal demand from
   ISO Express API (https://www.iso-ne.com/isoexpress/web-services-api)
3. Fetch the last 24h + next 24h HRRR weather analysis/forecast for the
   New England region (450×449×7 tensor) from NOAA AWS S3 via Herbie
4. Pull 24h calendar features (hour-of-day, day-of-week, month, holiday)
5. Feed all into the trained encoder-decoder CNN-Transformer
6. Display:
   - Per-zone 24h forecast lines (8 ISO-NE zones)
   - Total system load forecast
   - Comparison vs. ISO-NE's own forecast (if API exposes it)
   - Side-by-side weather map for context (HRRR temperature)

## Files (planned)

| File | Purpose |
|---|---|
| `app.py` | Gradio UI |
| `iso_ne_fetch.py` | Fetch real-time ISO-NE demand from ISO Express |
| `weather_fetch.py` | Fetch HRRR weather (history + 24h forecast) — port from real_time_weather_forecasting |
| `model_utils.py` | Load `best.pt`, run inference, denormalize |
| `visualization.py` | 8-zone forecast plots + maps |
| `models/` | Copy of `models/cnn_encoder_decoder.py` + `cnn_transformer_baseline.py` (HF Spaces needs self-contained code) |
| `checkpoints/best.pt` | Trained model weights (≤ 100 MB) — copied from `runs/cnn_encoder_decoder/checkpoints/` after training completes |
| `requirements.txt` | Gradio, torch, herbie-data, requests, plotly |
| `packages.txt` | apt-level deps if any |

## Deployment

After model training is complete:
```bash
python scripts/deploy_space.py    # uploads space/ to HF Spaces
python scripts/hf_upload.py       # uploads checkpoint to HF Hub
```

## ISO-NE data sources (research notes)

- **ISO Express API** (free, no auth required for some endpoints):
  https://www.iso-ne.com/isoexpress/web-services
  - `/genfuelmix/current.json` — fuel mix
  - `/loadhrly/current/json` — system load (5-min, 1-hr aggregates)
  - Per-zone load: needs ISO-NE account
- **Alternative**: scrape https://www.iso-ne.com/isoexpress/ HTML
- **Historical demand CSVs** (what we trained on): same format from ISO-NE's monthly archives

## Weather (reuse from sibling project)

Already implemented in `real_time_weather_forecasting/space/hrrr_fetch.py`:
- Uses `herbie-data` Python library
- Fetches HRRR analysis from NOAA AWS S3
- Regrids to 450×449 grid

We need the same 7-channel subset our model was trained on (TMP, RH, UGRD,
VGRD, GUST, DSWRF, APCP_1hr — channels 0-6 of the 42-channel HRRR input
the weather project uses).

## Developer notes

- HF Spaces free tier: 16 GB RAM, 2 vCPU, no GPU. Inference is fast on
  CPU since model is only ~2.4 M params.
- Cold start: HF Spaces sleeps after inactivity, first request takes ~30s
  to wake.
- Streaming logs: `gr.update(...)` for progressive UI feedback during
  the ~30s data fetch.
