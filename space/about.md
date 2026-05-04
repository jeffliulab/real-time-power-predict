## About this demo

This Space runs two models from our CS-137 final project on **live ISO New England demand history**:

1. **Baseline only** — the Part 1 CNN-Transformer (1.75 M params). Reaches **5.24 % MAPE** with real HRRR weather on the 2022 self-evaluation slice; in this Space the weather inputs are synthetic so accuracy is degraded.
2. **Ensemble (Baseline + Chronos-Bolt-mini)** — late-fusion of the baseline with [Chronos-Bolt-mini](https://huggingface.co/amazon/chronos-bolt-mini) (Amazon, 21 M params, Apache-2.0), used **zero-shot on demand history only** — no weather, no fine-tuning. Reaches **4.21 % MAPE** on the same offline slice and is the recommended path for this demo.

The Model selector at the top of the page switches between them.

### What's real vs. synthetic

| Component | Baseline only | Ensemble | Cluster runs |
|---|---|---|---|
| Baseline weights | ✅ | ✅ | ✅ |
| Calendar features | ✅ | ✅ | ✅ |
| Demand history | ✅ live ISO-NE (or 2022 fallback) | ✅ | ✅ |
| **Weather inputs to baseline** | ❌ zeros (training-mean) | ❌ zeros (training-mean) | ✅ real HRRR rasters |
| Chronos-Bolt-mini (zero-shot, demand only) | — | ✅ | — |

In Baseline-only mode, the forecast is degraded vs. the cluster's **5.24 %** MAPE because real weather is replaced with z-score zeros. Calendar features (hour, day-of-week, month, holiday flag) and the recent demand pattern still drive the output, so the shape of the forecast (daily double-peak, weekend/weekday differences) is preserved.

In Ensemble mode, Chronos-Bolt-mini receives 720 hours (4 weeks) of recent per-zone demand and outputs a zero-shot 24-hour forecast for each zone. Per-zone weights $\alpha_z$ (shown beneath the chart) control the blend: $\alpha_z = 1$ keeps only the baseline; $\alpha_z = 0$ keeps only Chronos. The values come from a grid search on a 14-day validation window (2022-12-16 → 12-29) and are hard-coded in this Space — see Table 10 of the report for the underlying ablation.

### Per-zone allocation

ISO-NE's public data feed publishes *system-level* demand at 5-minute granularity. We split that total into 8 zones using fixed proportions estimated from 2022 historical zonal load reports. Per-zone real-time data requires an authenticated ISO Express account.

### What this is for

This is a **technical demonstration** of the trained models' input/output pipelines, not a production forecasting service. The full pipeline (live HRRR weather + authenticated per-zone real-time demand + the Chronos-Bolt-mini foundation-model ensemble) is documented in the report and tracked as future work in the GitHub repo.

### First-call latency

If the **Ensemble** mode hasn't been used yet on this Space instance, the first request will trigger a one-time download of Chronos-Bolt-mini weights ($\sim$80 MB from HuggingFace Hub). Expect $\sim$30 s the first time and $\sim$5 s on subsequent requests. Baseline-only mode is always $\sim$2 s.

### Links

- 📄 [Final report (PDF)](https://github.com/jeffliulab/real-time-power-predict/blob/main/report/final_report.pdf)
- 💻 [GitHub repository](https://github.com/jeffliulab/real-time-power-predict)
- 👤 Author: **Pang Liu** · `pliu07` · Tufts CS-137
