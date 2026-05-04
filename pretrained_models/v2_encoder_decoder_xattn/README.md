---
license: mit
tags:
  - energy-demand-forecasting
  - encoder-decoder
  - cross-attention
  - ablation-study
  - iso-new-england
language: en
library_name: pytorch
pipeline_tag: tabular-regression
---

# `v2_encoder_decoder_xattn/best.pt` — Encoder-Decoder, **history + future** spatial cross-attention (ablation)

**Headline:** 9.27 % test MAPE — the worst of the three. Useful as a **negative-result ablation** documenting that the v1 architecture's loss to baseline is dominated by the LR-scheduler reset bug, not by missing future-weather information.

## Model summary

Identical to `v1_encoder_decoder` except `use_future_weather_xattn=True` is enabled at construction time. This adds a second cross-attention path per decoder layer that reads from `future-spatial` encoder memory (i.e., the model now sees future weather just like the joint-encoder Part 1 baseline does).

| Field | Value |
|---|---|
| Architecture                | Encoder-decoder, **history-spatial AND future-spatial** cross-attention |
| Parameters                  | **2,419,312** (2.42 M) — adds ~133 K vs v1 from the second cross-attn block + LayerNorm |
| `use_future_weather_xattn`  | **True** |
| Total `epoch` at best       | 13 (chained 3-job SLURM, all TIMEOUT'd at 24 h each) |
| Best val MAPE               | 8.72 % on val 2022 (≈ same as v1's 8.63 %) |
| **Test MAPE (2022-12-30/31)** | **9.27 %** |
| File size                   | 29 MB |

## Per-zone test MAPE (last 2 days of 2022)

| Zone | v2 | v1 | baseline |
|---|---|---|---|
| ME        | 5.91 % | 3.22 % | 2.31 % |
| NH        | 8.51 % | 5.67 % | 3.69 % |
| VT        | 7.78 % | **5.85 %** | 5.95 % |
| CT        | 12.53 % | 9.56 % | 7.28 % |
| RI        | 8.25 % | 7.45 % | 5.27 % |
| SEMA      | 7.38 % | 7.22 % | 5.44 % |
| WCMA      | 11.60 % | 7.38 % | 5.87 % |
| NEMA_BOST | 12.20 % | 8.24 % | 6.09 % |
| **Overall** | **9.27 %** | 6.82 % | **5.24 %** |

## What this ablation shows

- **Information access alone does not close the gap.** v2 has full future-weather access via the new cross-attention path, yet its val MAPE on val 2022 (8.72 %) is essentially the same as v1's (8.63 %). On the test slice v2 is *worse* than v1 by 2.45 pp.
- **The dominant cause of v1's loss to baseline is the LR-scheduler reset bug**, not "information disadvantage." v2 inherits the same bug from the same chained-resume training infrastructure.
- This is the kind of ablation result that flips a hypothesis: it disconfirms our initial reading of v1's defeat.

## Why v2 is "even worse" than v1 — undertrained, not architecturally bad

v2's `best.pt` was selected at **epoch 13** (val MAPE 8.72 %). The training_log shows v2 was still in mid-cosine LR territory (lr ≈ 9 × 10⁻⁴ at epoch 13) — far from the small-LR sweet spot that gives the baseline its big drop (baseline reaches val 6.92 % at epoch 13 with lr ≈ 3 × 10⁻⁴).

Layered onto that:
- v2 has more parameters (+5.8 % over v1) but no extra training budget.
- The 3-job SLURM chain TIMEOUT'd 3 times (5/1, 5/2, 5/3); only one cosine cycle ran cleanly between resets.
- The visible training_log only contains epochs 8–13; earlier epochs were overwritten by chained resumes.

So v2's 9.27 % test MAPE is **not strong evidence that future-weather cross-attention hurts** — it's evidence that v2 was infrastructure-limited. A clean re-run (single uninterrupted SLURM window, `scheduler.state_dict()` saved on each save) would be needed to fairly test whether v2 beats v1 or beats baseline. Future work.

## Loading

```python
import torch
from models.cnn_encoder_decoder import CNNEncoderDecoderForecaster

ckpt = torch.load("pretrained_models/v2_encoder_decoder_xattn/best.pt",
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
    use_future_weather_xattn=True,    # v2 toggle
)
model.load_state_dict(ckpt["model"])
model.eval()
```

## Training

- Optimizer: AdamW, base LR 1e-3, weight decay 1e-4
- LR schedule: CosineAnnealingLR (T_max = 24); **same reset bug** as v1
- Train years: 2019–2021
- Validation year: 2022
- Wall time: ~72 h across 3 chained 24-h SLURM jobs (all TIMEOUT)
- SLURM job IDs: 36804839 (5/1 TIMEOUT), 36804840 (5/2 TIMEOUT), 36804841 (5/3 TIMEOUT)

## Limitations

- **Undertrained.** Best at epoch 13 with lr ≈ 9 × 10⁻⁴; cosine cycle didn't reach small-LR territory. Comparison vs v1 and baseline is fair on the same artifacts but not on the same compute budget.
- LR-scheduler reset bug applies (same as v1).
- Random seeds not set; not bit-reproducible.

## Citation + License

Same as the project README.
