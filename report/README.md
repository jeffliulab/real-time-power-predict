---
name: Report folder guide
description: One-page index of deliverables for the CS-137 Assignment 3 grader.
---

# CS-137 Assignment 3 — Report Folder Guide

This folder contains the final submission deliverables.

## Files

| File | Purpose |
|---|---|
| `final_report.pdf` | **Compiled report** (open this for grading; LaTeX source under [tex/](tex/)). |
| `slides.pdf` | **Compiled 7-slide presentation** (matches the spec's 7-page hard cap). |
| [tex/main.tex](tex/main.tex) | LaTeX source: 9 sections + 4 appendices + inline TikZ architecture diagram. |
| [tex/preamble.tex](tex/preamble.tex) | LaTeX preamble (article 11pt, natbib + plainnat, NavyAccent headings, `subcaption`, `booktabs`, `hyperref`, `tikz`). |
| [tex/refs.bib](tex/refs.bib) | BibTeX bibliography with DOIs, arXiv IDs, and stable URLs (15 entries). |
| [tex/arch_only.tex](tex/arch_only.tex) | Standalone TikZ source for [figures/architecture.png](figures/architecture.png) (used in slides). |
| [slides.md](slides.md) | Marp markdown source of the 7-slide deck. |
| [contribution_statement.md](contribution_statement.md) | Solo authorship statement (also embedded as §9 of the report). |
| [figures/](figures/) | All PNGs referenced by the report and slides. |

## Building from source

```bash
# Report (uses xelatex + bibtex via latexmk)
cd tex && latexmk -xelatex -interaction=nonstopmode main.tex
cp main.pdf ../final_report.pdf

# Slides (Marp via npx)
cd .. && npx -y @marp-team/marp-cli@latest --pdf --allow-local-files \
    slides.md -o slides.pdf

# Regenerate matplotlib figures (used by slides 1, 4)
python3 ../scripts/figures/iso_ne_map.py
python3 ../scripts/figures/baseline_per_zone_bar.py

# Regenerate the standalone architecture diagram
cd tex && xelatex arch_only.tex && \
    magick -density 220 arch_only.pdf -alpha remove -alpha off \
    ../figures/architecture.png
```

LaTeX build artifacts (`*.aux`, `*.bbl`, `*.log`, etc.) are gitignored via [tex/.gitignore](tex/.gitignore).

## Test-set caveat (important)

All MAPE numbers in the report and slides (baseline 5.24 %, Part 2 v1 6.82 %, etc.) are computed by our own self-evaluation harness on the **last 2 days of 2022** (the cluster's `inference/test_run.sh` default). The TA's official grading harness will run our submitted `model.py` against a held-out **2024** test set; absolute numbers will differ slightly but the model is data-year agnostic — per-zone gaps and architectural trends should carry over. This caveat is also explicitly noted in §1 of the report.

## Where the rest of the project lives

The `report/` folder is intentionally a thin presentation layer over the rest of the repo:

- Model code → [`models/`](../models/)
- Training pipeline → [`training/`](../training/)
- Inference / eval scripts → [`inference/`](../inference/) and [`scripts/`](../scripts/)
- Slide-figure generators → [`scripts/figures/`](../scripts/figures/)
- HPC SLURM scripts → [`scripts/*.slurm`](../scripts/)
- Tests → [`tests/`](../tests/)
- HF Space demo skeleton → [`space/`](../space/)

Top-level [README.md](../README.md) has full architecture details, run instructions, and the v1/v2 SLURM chain status. [progress.md](../progress.md) has a chronological log of decisions and incidents.

## How to reproduce the headline numbers

```bash
# Self-evaluation against the cluster default (last 2 days of 2022)
bash inference/test_run.sh    # baseline → ~5.24 %
# v1 / v2 numbers are in runs/cnn_encoder_decoder*/eval/*.json
```

Phase 3 attention figures are generated on HPC via `scripts/attention_maps.slurm`; the local script `scripts/attention_maps.py` is verified and includes geographic-orientation sanity checks (see §5 and Appendix C of the report).

## Authorship

Solo submission — Pang Liu (UTLN: pliu07). See [contribution_statement.md](contribution_statement.md) and §9 of `final_report.pdf`.
