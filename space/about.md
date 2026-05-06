## About this demo

This Space runs a trained CNN-Transformer baseline on **fully real, live ISO New England inputs**, blended with **Chronos-Bolt-mini** (Amazon, 21 M params, Apache-2.0, zero-shot on demand history alone) in a per-zone weighted ensemble.

There are two tabs:

1. **Real-time forecast** — every click pulls the latest 24 h of demand and HRRR weather, plus a 24 h HRRR forecast cycle, and produces a 24-hour 8-zone prediction.
2. **Backtest (last 7 days)** — 7 daily forecasts on the most recent 7 fully-published days, refreshed every day at 04:00 UTC by a GitHub Actions cron in [`new-england-real-time-power-predict-data`](https://github.com/jeffliulab/new-england-real-time-power-predict-data).

> ⚠ **Why deployment MAPE differs from the offline 4.21 % headline.**
> The trained baseline saw 2019–2022 weather + demand at training time. Deployed today it sees 2026 weather + demand, and 3+ years of New-England-grid evolution (utility-scale BTM solar, EVs, post-COVID load patterns) has shifted the per-zone signal — most visibly in the dense southern coastal zones (RI / SEMA / WCMA). **The pipeline itself is verified correct**: re-running it on 2022-12-30 reproduces the cluster's 6.54 % MAPE to within 0.13 percentage points (we observe 6.41 %, see [Live performance validation](https://github.com/jeffliulab/real-time-power-predict#live-performance-validation) in the README). The gap between the headline and the deployed numbers is dominated by honest training→deployment drift, not bugs.

### What's real (everything)

| Component | Source | Real or synthetic? |
|---|---|---|
| Per-zone demand history (24 h) | ISO-NE public 5-min `fiveminuteestimatedzonalload` feed → hourly mean | ✅ live (~1-2 h publication lag) |
| Chronos context (720 h history) | Same ISO-NE feed (data repo cache + live splice) | ✅ live |
| Weather history (24 h, 7 channels) | NOAA HRRR f00 analyses on AWS S3 (`noaa-hrrr-bdp-pds`) via Herbie | ✅ live |
| Weather forecast (24 h, 7 channels) | NOAA HRRR cycle T-1's f01..f24 forecasts | ✅ live ⁽¹⁾ |
| Calendar features | Computed deterministically from timestamps | ✅ |
| Baseline weights | Trained on 2019–2022 data | ✅ |
| Chronos-Bolt-mini | Amazon, zero-shot, no fine-tuning | ✅ |

⁽¹⁾ One small caveat: the `APCP_1hr` (1-hour accumulated precipitation) channel is zero-filled for HRRR forecast hours because the search regex that worked at training time matches a different accumulation window than the operational forecast files publish. Six of the seven weather channels are real; APCP_1hr's training mean is small (~0.4 mm/h) so after z-scoring this approximates the training mean, with measured impact <1 % MAPE.

The bundled 2022 sample CSVs are kept ONLY as a final fallback for when the live ISO-NE / HRRR endpoints are unreachable.

### Strict-discipline backtest

For each daily forecast at time **T** (the last 7 days at 00:00 UTC each):

- **Demand history** for hours [T-24, T-1] comes from the public 5-min zonal feed
- **Weather history** is 24 HRRR f00 analyses, one per valid hour [T-24, T-1]
- **Weather forecast** is HRRR cycle (T-1)'s f01..f24 — i.e. the most recent forecast that was issued *before* T, with valid hours [T, T+23]
- **Truth** for MAPE is the ISO-NE per-zone load for [T, T+23]

In particular **no future analyses are used** — every forecast at T sees only data that would have been available at time T, matching what a real deployment would do.

### Disclosure: training-time `future_weather` mismatch

The trained baseline saw **HRRR f00 analyses for both history AND future windows during training** (i.e. the model was given the actual weather that occurred during the prediction window as an *input* feature). This is a form of supervised-learning-with-privileged-information that the report acknowledges in its Part-2 future-weather ablation.

At deployment we cannot use future analyses (they don't exist for the future yet), so we substitute HRRR forecasts (`f01..f24`) issued at the cycle just before the forecast time. The model therefore sees a slightly out-of-distribution input for the future window. This Space measures the deployable accuracy honestly. The Chronos-Bolt-mini ensemble path partially compensates because Chronos doesn't use weather at all.

### Per-zone allocation — actually per-zone now

Earlier prototypes of this demo used a fixed proportion vector to split the system total (from the EIA Open Data API) into 8 zones, which made the per-zone view cosmetic. The current Space pulls **true per-zone load** from ISO-NE's 5-minute estimated zonal feed, so per-zone forecasts are real.

### First-call latency

The first Live tab click triggers ~24 HRRR analysis cycles + 24 forecast hours from AWS S3 (parallel-fetched, cached at `/tmp/hrrr_cache/`) plus a one-time Chronos-Bolt-mini load (~80 MB from HuggingFace Hub). Expect **~3-5 minutes on the very first click** of a fresh Space instance, then ~10-30 s on subsequent clicks within the same uptime session. The Backtest tab is instant — its data ships pre-computed from the data repo.

### Per-zone ensemble weights

Per-zone $\alpha_z$ (shown beneath the chart) blends baseline and Chronos:

$$\hat y_z = \alpha_z \cdot \hat y_z^{\text{baseline}} + (1 - \alpha_z) \cdot \hat y_z^{\text{Chronos}}$$

$\alpha_z$ values come from a grid search on a 14-day validation window in 2022. See the **Foundation-model ensemble** section of the report for the full ablation across context length, model size, and quantile aggregation.

### Links

- 📄 [Final report (PDF)](https://github.com/jeffliulab/real-time-power-predict/blob/main/report/submission/submission_report.pdf)
- 💻 [Main code repo](https://github.com/jeffliulab/real-time-power-predict)
- 🤖 [Auxiliary data repo (cron-refreshed backtest data)](https://github.com/jeffliulab/new-england-real-time-power-predict-data)
- 👤 Author: **Pang Liu** · Independent Researcher · [`jeff.pang.liu@gmail.com`](mailto:jeff.pang.liu@gmail.com)
