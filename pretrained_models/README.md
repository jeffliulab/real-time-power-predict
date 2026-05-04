# Pretrained models

Three trained checkpoints for ISO-NE day-ahead demand forecasting.
All trained on Tufts HPC `c26sp1cs0137` cluster against the
2019-2023 weather + demand slice at
`/cluster/tufts/c26sp1cs0137/data/assignment3_data/`.

## Also published on Hugging Face Hub

| Local path                            | HuggingFace repo                                                                              |
|---------------------------------------|-----------------------------------------------------------------------------------------------|
| `baseline/`                           | [`jeffliulab/predict-power-baseline`](https://huggingface.co/jeffliulab/predict-power-baseline) |
| `v1_encoder_decoder/`                 | [`jeffliulab/predict-power-v1`](https://huggingface.co/jeffliulab/predict-power-v1)             |
| `v2_encoder_decoder_xattn/`           | [`jeffliulab/predict-power-v2-undertrained`](https://huggingface.co/jeffliulab/predict-power-v2-undertrained) |

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
- Conda env on Tufts HPC: `/cluster/tufts/c26sp1cs0137/pliu07/conda_envs/cs137`
- GPU: A100 40 GB (baseline), A100/P100 mix (v1, v2)
- Loss: MSE in z-score space; metric: MAPE in physical MWh
- Optimizer: AdamW, base LR 1e-3, weight decay 1e-4
- LR schedule: CosineAnnealingLR (see Reproducibility caveat in
  the report — `scheduler.state_dict()` is NOT saved on resume,
  causing the cosine to reset on every chained `--resume`).

## Test-set caveat

The "test (2022-12-30/31)" column is our self-evaluation harness on the
last 2 days of 2022. The TA grader uses a held-out 2024 slice; absolute
numbers will differ but per-zone gaps and architectural trends should
carry over.

## Authorship + provenance

Solo submission by **Pang Liu** (UTLN: `pliu07`), Tufts CS-137 Spring 2026.
Source code at: <https://github.com/jeffliulab/real-time-power-predict>
Live demo:    <https://huggingface.co/spaces/jeffliulab/predict-power>
