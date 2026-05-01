## About this demo

This Space runs the trained **Part 1 CNN-Transformer baseline** from our CS-137 final project on **live ISO New England demand history**. The baseline reaches **5.24 % MAPE** on the 2022 self-evaluation slice (last 2 days of 2022) when given real HRRR weather inputs.

### What's real vs. synthetic

| Component | This demo | Cluster runs |
|---|---|---|
| Model weights | ✅ trained on real HRRR + ISO-NE | ✅ |
| Calendar features | ✅ derived from request timestamp | ✅ |
| Demand history | ✅ live ISO-NE (or 2022 fallback) | ✅ |
| **Weather inputs** | ❌ **zeros (training-mean) — synthetic** | ✅ real HRRR rasters |

The forecast quality you see here is degraded vs. the cluster's **5.24 %** MAPE because real weather is replaced with z-score zeros. Calendar features (hour, day-of-week, month, holiday flag) and the recent demand pattern still drive the output, so the shape of the forecast (daily double-peak, weekend/weekday differences) is preserved.

### Per-zone allocation

ISO-NE's public data feed publishes *system-level* demand at 5-minute granularity. We split that total into 8 zones using fixed proportions estimated from 2022 historical zonal load reports. Per-zone real-time data requires an authenticated ISO Express account.

### What this is for

This is a **technical demonstration** of the trained model's input/output pipeline, not a production forecasting service. The full pipeline (live HRRR weather + per-zone real-time demand) is documented in the report and tracked as future work in the GitHub repo.

### Links

- 📄 [Final report (PDF)](https://github.com/jeffliulab/real-time-power-predict/blob/main/report/final_report.pdf)
- 💻 [GitHub repository](https://github.com/jeffliulab/real-time-power-predict)
- 👤 Author: **Pang Liu** · `pliu07` · Tufts CS-137
