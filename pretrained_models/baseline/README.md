---
license: mit
tags:
  - energy-demand-forecasting
  - cnn-transformer
  - iso-new-england
  - time-series-forecasting
  - multi-modal
language: en
library_name: pytorch
pipeline_tag: tabular-regression
---

# `baseline/best.pt` — Multi-Modal CNN-Transformer (Part 1 baseline)

**Headline:** 5.24 % test MAPE on the last 2 days of 2022 (8 ISO New England load zones, 24-hour day-ahead horizon).

## Model summary

A hybrid CNN-Transformer that fuses HRRR-style weather rasters with per-zone demand history and 44-d calendar features into a unified token sequence, then decodes 24 hourly per-zone demand values for all 8 ISO-NE zones.

| Field | Value |
|---|---|
| Architecture          | CNN-Transformer (joint encoder over unified sequence) |
| Parameters            | **1,753,200** (1.75 M) |
| Spatial token grid    | 8 × 8 (P = 64 spatial tokens per timestep) |
| Sequence length       | (S+24) · (P+1) = 48 · 65 = **3,120** tokens |
| Embedding dim D       | 128 |
| Transformer layers    | 4 encoder, 4 heads, MLP ratio 4, pre-norm |
| Total `epoch` at best | 13 (continuous training, no chained resume) |
| Best val MAPE         | 6.92 % on val 2021 |
| **Test MAPE (2022-12-30/31)** | **5.24 %** |
| File size             | 21 MB |
| SHA256                | `91069db5bc8f93f832aa0a4e4fb600f075ef382617049225d828003c99ae05c0` |

## Per-zone test MAPE (last 2 days of 2022)

| Zone | MAPE |
|---|---|
| ME        | **2.31 %** ⭐ |
| NH        | 3.69 % |
| VT        | 5.95 % |
| CT        | 7.28 % |
| RI        | 5.27 % |
| SEMA      | 5.44 % |
| WCMA      | 5.87 % |
| NEMA_BOST | 6.09 % |
| **Overall** | **5.24 %** |

## Inputs

- **Weather rasters** `X ∈ ℝ^{(S+24) × 7 × 450 × 449}` — HRRR-style 7-channel hourly snapshots (S = 24 history hours, 24 future hours)
- **Per-zone demand** `Y ∈ ℝ^{S × 8}` — historical MWh demand for the 8 ISO-NE zones
- **Calendar features** `C ∈ ℝ^{(S+24) × 44}` — one-hot hour (24) + day-of-week (7) + month (12) + US-holiday flag (1)

## Outputs

- 24-hour day-ahead per-zone demand forecast `Ŷ ∈ ℝ^{24 × 8}` in MWh

## Loading

```python
import torch
from models.cnn_transformer_baseline import CNNTransformerBaselineForecaster

ckpt = torch.load("pretrained_models/baseline/best.pt",
                  map_location="cpu", weights_only=False)
args = ckpt["args"]
model = CNNTransformerBaselineForecaster(
    n_weather_channels=7, n_zones=8, cal_dim=44,
    history_len=args["history_len"],     # 24
    embed_dim=args["embed_dim"],         # 128
    grid_size=args["grid_size"],         # 8
    n_layers=args["n_layers"],           # 4
    n_heads=args["n_heads"],             # 4
    dropout=args["dropout"],             # 0.1
)
model.load_state_dict(ckpt["model"])
model.eval()
norm_stats = ckpt["norm_stats"]   # {weather_mean, weather_std, energy_mean, energy_std}
```

## Training

- Optimizer: AdamW, base LR 1e-3, weight decay 1e-4
- LR schedule: CosineAnnealingLR (T_max = 14 epochs, no chained resume)
- Loss: MSE in z-score space (per the four-step normalization chain)
- Validation: MAPE in physical MWh space, per-zone + overall
- Batch size: 4 per A100 GPU
- Hardware: A100 40 GB
- Wall time: ~22 hours
- Train years: 2019–2020
- Validation year: 2021
- Self-eval test slice: 2022-12-30 to 2022-12-31

## Limitations

1. Test numbers are on a 2-window slice; small-sample variance non-negligible.
2. The CNN trunk is a fixed 5-stage residual stack; spatial-encoder design space not explored.
3. Random seeds (`torch.manual_seed` / `np.random.seed`) are NOT set in the training pipeline — headline MAPE is not bit-reproducible across re-training runs. Empirical claims are pinned to **this specific** checkpoint.

## Citation

```
Liu, Pang. "Multi-Modal Deep Learning for Energy Demand Forecasting"
(real-time-power-predict v1.5, 2026).
GitHub: https://github.com/jeffliulab/real-time-power-predict
```

## License

MIT (see top-level LICENSE file in the repo).
