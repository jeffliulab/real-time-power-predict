---
license: mit
tags:
  - energy-demand-forecasting
  - encoder-decoder
  - cross-attention
  - iso-new-england
  - time-series-forecasting
language: en
library_name: pytorch
pipeline_tag: tabular-regression
---

# `v1_encoder_decoder/best.pt` — Encoder-Decoder, history-spatial cross-attention only

**Headline:** 6.82 % test MAPE on the last 2 days of 2022. **An honest negative result** — lost to the Part 1 baseline by 1.58 pp despite a 30 % larger parameter count. See *Why it lost* below.

## Model summary

Inspired by TFT (Lim et al. 2021) and PatchTST (Nie et al. 2023): a clean encoder/decoder split. The encoder is the same joint sequence encoder as the Part 1 baseline; the decoder issues 24 future-hour query tokens and cross-attends only to **history-spatial** keys in the encoder memory.

| Field | Value |
|---|---|
| Architecture          | Encoder-decoder, history-spatial cross-attention only |
| Parameters            | **2,286,192** (2.29 M) |
| Encoder layers        | 4, 4 heads, pre-norm |
| Decoder layers        | 2, 4 heads, self-attn + cross-attn (history-spatial only) |
| Cross-attention reads | `history-spatial` keys (no future-spatial, no tabular) |
| `use_future_weather_xattn` | **False** |
| Total `epoch` at best | 6 (chained 3-job SLURM, each 24 h, all TIMEOUT'd) |
| Best val MAPE         | 8.63 % on val 2022 |
| **Test MAPE (2022-12-30/31)** | **6.82 %** |
| File size             | 28 MB |

## Per-zone test MAPE (last 2 days of 2022)

| Zone | v1 | baseline | Δ |
|---|---|---|---|
| ME        | 3.22 % | 2.31 % | +0.91 |
| NH        | 5.67 % | 3.69 % | +1.98 |
| VT        | **5.85 %** ⭐ | 5.95 % | **−0.10** (only zone where v1 wins) |
| CT        | 9.56 % | 7.28 % | +2.28 |
| RI        | 7.45 % | 5.27 % | +2.18 |
| SEMA      | 7.22 % | 5.44 % | +1.78 |
| WCMA      | 7.38 % | 5.87 % | +1.51 |
| NEMA_BOST | 8.24 % | 6.09 % | +2.15 |
| **Overall** | **6.82 %** | **5.24 %** | **+1.58 pp** |

The gap is widest on the urban-coastal triplet **CT, NEMA_BOST, RI** — exactly the zones whose demand is most sensitive to fine-grained future weather timing.

## Why it lost (three diagnosed causes)

1. **Information disadvantage** (initial hypothesis) — the v1 decoder is by default disconnected from future-weather spatial tokens. The Part 1 baseline's joint encoder, in contrast, sees future weather directly. **Note**: the `v2_encoder_decoder_xattn` ablation gives the decoder full future-weather access via cross-attention but does **not** improve over v1 — see that model card.
2. **LR-scheduler reset bug** (verified) — `training/train.py` saves `model` and `optimizer` state but **not** `scheduler.state_dict()`. The 3-job chained `--resume` infrastructure resets the cosine LR to 1 × 10⁻³ at every job boundary (epoch 7, 14, …), so v1's loss flat-lines at the epoch where the baseline started its big drop. v1 never reached the small-LR sweet spot.
3. **Validation-set mismatch** — v1 validates on 2022 (harder, more weather extremes); baseline validates on 2021 (milder). Per-epoch val numbers (8.63 % vs 6.92 %) are not directly comparable; test-set MAPEs (the headline 6.82 % vs 5.24 %) **are** comparable because they're on the same slice.

## Loading

```python
import torch
from models.cnn_encoder_decoder import CNNEncoderDecoderForecaster

ckpt = torch.load("pretrained_models/v1_encoder_decoder/best.pt",
                  map_location="cpu", weights_only=False)
args = ckpt["args"]
model = CNNEncoderDecoderForecaster(
    n_weather_channels=7, n_zones=8, cal_dim=44,
    history_len=args["history_len"],
    embed_dim=args["embed_dim"],
    grid_size=args["grid_size"],
    n_encoder_layers=args["n_encoder_layers"],
    n_decoder_layers=args["n_decoder_layers"],
    n_heads=args["n_heads"],
    dropout=args["dropout"],
    use_future_weather_xattn=False,    # v1 default
)
model.load_state_dict(ckpt["model"])
model.eval()
```

## Training

- Optimizer: AdamW, base LR 1e-3, weight decay 1e-4
- LR schedule: CosineAnnealingLR — but **affected by the reset bug** (the cosine restarts on every chained `--resume`)
- Train years: 2019–2021
- Validation year: 2022
- Wall time: ~45 h across 3 chained 24-h SLURM jobs
- Hardware: A100 / P100 (SLURM-decided)
- Best checkpoint selected at **epoch 6** (val MAPE 8.63 %)

## Limitations

Same as `baseline/` plus:

- The "negative result" framing is empirically clean but **infrastructure-confounded** (LR-reset bug). A re-run in a single uninterrupted SLURM window with `scheduler.state_dict()` saved would test whether v1 fundamentally underperforms or whether the bug eats its potential.
- Training was infrastructure-limited; v1 (and v2) never reach the small-LR fine-tuning regime that gives baseline its big drop.

## Citation + License

Same as the project README.
