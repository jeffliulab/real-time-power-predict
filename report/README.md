---
name: Report folder guide
description: One-page index of deliverables for the CS-137 Assignment 3 grader.
---

# CS-137 Assignment 3 — Report Folder Guide

This folder contains the final submission deliverables.

## Files

| File | Purpose |
|---|---|
| [final_report.md](final_report.md) | Source markdown of the 8-section report (Parts 1–3 + contribution + references). |
| `final_report.pdf` | Compiled PDF of the report (open this for grading). |
| [slides.md](slides.md) | Source markdown for the 9-slide Marp deck. |
| `slides.pdf` | Compiled PDF of the presentation (matches the 5/5 in-class talk). |
| [contribution_statement.md](contribution_statement.md) | Solo-author statement (also embedded in the report §7). |
| [figures/](figures/) | All PNGs referenced by the report and slides. |

## Test-set caveat (important)

All MAPE numbers in the report and slides (baseline 5.24 %, Part 2 v1 6.82 %, etc.) are computed by our own self-evaluation harness on the **last 2 days of 2022** (the cluster's `test_run.sh` default). The TA's official grading harness will run our submitted `model.py` against a held-out **2024** test set; numbers will differ slightly but the model is data-year-agnostic — trends and per-zone gaps should carry over. This is also explicitly noted in §1 of `final_report.md`.

## Where the rest of the project lives

The `report/` folder is intentionally a thin presentation layer over the rest of the repo:

- Model code → [`models/`](../models/)
- Training pipeline → [`training/`](../training/)
- Inference / eval scripts → [`inference/`](../inference/) and [`scripts/`](../scripts/)
- HPC SLURM scripts → [`scripts/*.slurm`](../scripts/)
- Tests → [`tests/`](../tests/)
- HF Space demo skeleton → [`space/`](../space/)

Top-level [README.md](../README.md) has full architecture details, run instructions, and the v1/v2 SLURM chain status. [progress.md](../progress.md) has a chronological log of decisions and incidents.

## How to reproduce the numbers

```bash
# Self-evaluation against the cluster default (last 2 days of 2022)
bash inference/test_run.sh    # baseline → ~5.24 %
# v1 / v2 numbers come from runs/cnn_encoder_decoder*/eval/*.json
```

Phase 3 attention figures are generated on HPC via `scripts/attention_maps.slurm`; the local script `scripts/attention_maps.py` is verified and includes geographic-orientation sanity checks (see §5 of the report).

## Authorship

Solo submission — Pang (Jeff) Liu (UTLN: pliu07). See [contribution_statement.md](contribution_statement.md).
