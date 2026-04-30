---
name: Contribution statement (solo)
description: Solo-author declaration for CS-137 Assignment 3.
---

# Contribution Statement

This is a **solo submission**.

**Pang (Jeff) Liu** (UTLN: pliu07) is the sole author and is responsible for every component of this project:

- **Literature survey** — ~27 papers across Transformer time-series forecasting (Vaswani et al. 2017; Lim et al. TFT 2021; Nie et al. PatchTST 2023; Liu et al. iTransformer 2024), spatiotemporal fusion (Gao et al. Earthformer 2022; Bertasius et al. TimeSformer 2021), and short-term load forecasting.
- **Architecture design + implementation** — Part 1 baseline ([models/cnn_transformer_baseline.py](../models/cnn_transformer_baseline.py)) and Part 2 encoder-decoder variant with optional future-weather cross-attention ([models/cnn_encoder_decoder.py](../models/cnn_encoder_decoder.py)).
- **Training pipeline** — ([training/train.py](../training/train.py)) including dataset ([training/data_preparation/dataset.py](../training/data_preparation/dataset.py)), the four-step normalization chain (z-score from 500 train samples → train in z-space → MAPE denormalized to MWh → eval-wrapper closes the loop), and the SLURM chain submission scripts.
- **HPC environment setup** — built persistent `cs137` conda env at `/cluster/tufts/c26sp1cs0137/pliu07/conda_envs/cs137/` with torch 2.3.1+cu118 and bundled cuDNN 8.9, after the course's `class/default` and `cs137/2026spring` modules were removed in late April. All later training and evaluation jobs use this env.
- **Self-evaluation harness** — ([scripts/self_eval.py](../scripts/self_eval.py)) and independent verification of the canonical TA evaluator output ([inference/test_run.sh](../inference/test_run.sh)).
- **Part 3 attention-map analysis** — ([scripts/attention_maps.py](../scripts/attention_maps.py)) and ([scripts/attention_maps.slurm](../scripts/attention_maps.slurm)), including the four geographic-orientation correctness checks (direction of attention rows/cols, row-major reshape, compass orientation via `origin='upper'`, NEMA_BOST east-vs-west sanity assert).
- **Final report and presentation slides** — ([final_report.md](final_report.md)), ([slides.md](slides.md)), and the [report folder index](README.md).

There were no co-authors, no shared code from other students, and no use of external repositories beyond the standard PyTorch / NumPy / matplotlib stack listed in [requirements.txt](../requirements.txt). AI tools (Claude) were used as a coding assistant during development under direct supervision; all design decisions, model choices, and the report content reflect the author's own judgment and responsibility.
