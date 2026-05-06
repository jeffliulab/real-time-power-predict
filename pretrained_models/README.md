# Pretrained models

Trained checkpoints for ISO-NE day-ahead demand forecasting. The shipped
checkpoint is the 1.75 M-parameter CNN-Transformer baseline. Trained
against the 2019–2023 NOAA HRRR weather + ISO-NE per-zone demand archive
on NVIDIA A100-80 GB / 40 GB GPUs.

## Also published on Hugging Face Hub

| Local path        | HuggingFace repo                                                                                   |
|-------------------|----------------------------------------------------------------------------------------------------|
| `baseline/`       | [`jeffliulab/predict-power-baseline`](https://huggingface.co/jeffliulab/predict-power-baseline)    |

The HF repos carry the same `best.pt` + `norm_stats.pt` + model-card README. They're convenient if you want to load via `huggingface_hub.hf_hub_download` rather than cloning this whole repo.

## Quick comparison

| Model | Architecture | Params | val (full year) | test (2022-12-30/31) | Verdict |
|---|---|---|---|---|---|
| `baseline/`              | CNN-Transformer (joint encoder over unified sequence)            | 1,753,200 | 6.92 % @ epoch 13 (val 2021) | **5.24 %** | ⭐ Best |
| `v1_encoder_decoder/`    | Encoder-decoder, history-spatial cross-attention only            | 2,286,192 | 8.63 % @ epoch 6  (val 2022) | 6.82 %     | Lost to baseline by 1.58 pp |
| `v2_encoder_decoder_xattn/` | Encoder-decoder, **history + future** spatial cross-attention | 2,419,312 | 8.72 % @ epoch 13 (val 2022) | 9.27 %     | Lost to v1 by 2.45 pp; undertrained (see notes) |

## Layout (HuggingFace-Model-style)

```
pretrained_models/
├── README.md                          # this file
├── baseline/
│   ├── best.pt                        # 21 MB — model + optimizer + args + norm_stats bundle
│   ├── norm_stats.pt                  # 2 KB  — z-score statistics (dup of best.pt['norm_stats'])
│   └── README.md                      # model card
├── v1_encoder_decoder/
│   ├── best.pt                        # 28 MB
│   ├── norm_stats.pt
│   └── README.md
└── v2_encoder_decoder_xattn/
    ├── best.pt                        # 29 MB
    ├── norm_stats.pt
    └── README.md
```

## Loading a checkpoint

```python
import torch
from models.cnn_transformer_baseline import CNNTransformerBaselineForecaster
# (or CNNEncoderDecoderForecaster for v1/v2)

ckpt = torch.load("pretrained_models/baseline/best.pt",
                  map_location="cpu", weights_only=False)
model = CNNTransformerBaselineForecaster(**{
    k: ckpt["args"][k] for k in
    ["history_len", "embed_dim", "grid_size", "n_layers", "n_heads", "dropout"]
})
model.load_state_dict(ckpt["model"])
model.eval()

norm_stats = ckpt["norm_stats"]   # also bundled separately as norm_stats.pt
# {weather_mean, weather_std, energy_mean, energy_std}
```

For v1 / v2 the model class is `CNNEncoderDecoderForecaster` and you
also need `n_encoder_layers`, `n_decoder_layers`, and (for v2) the
flag `use_future_weather_xattn=True`.

End-to-end evaluation: `bash inference/test_run.sh` (calls
`scripts/self_eval.py` against `runs/cnn_transformer_baseline/checkpoints/best.pt`
and prints the per-zone MAPE table).

## Training environment

- Python 3.10, PyTorch 2.3.1+cu118, CUDA 11.8, cuDNN 8.9
- GPU: NVIDIA A100-40 GB / A100-80 GB
- Loss: MSE in z-score space; metric: MAPE in physical MWh
- Optimizer: AdamW, base LR 3e-4, weight decay 1e-2
- LR schedule: cosine warm-restarts (SGDR)

## Test slice

The "test (2022-12-30/31)" numbers come from a self-evaluation harness on
the last 2 days of 2022 — the same 2-day slice used as the headline test
in the workshop paper at `report/arxiv/paper.pdf`.

## Authorship

**Pang Liu**, Independent Researcher, [`jeff.pang.liu@gmail.com`](mailto:jeff.pang.liu@gmail.com).
Source code: <https://github.com/jeffliulab/real-time-power-predict>
Live demo:   <https://huggingface.co/spaces/jeffliulab/predict-power>
