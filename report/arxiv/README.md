# arXiv preprint source

A 16–20 page journal-style technical report covering both the **design + training** of a multi-modal CNN-Transformer + Chronos-Bolt zero-shot ensemble for ISO New England day-ahead per-zone demand forecasting (Part I), and the **public live deployment** that surfaced a real, quantified 3-year training-→-2026 distribution drift correlated with state-level rooftop-solar buildout (Part II).

This is the arXiv version. The CS-137 final-submission version sits at [`../submission/submission_report.pdf`](../submission/submission_report.pdf) and is preserved verbatim for course-grading provenance.

## Build

```bash
cd report/arxiv
latexmk -xelatex -interaction=nonstopmode -halt-on-error paper.tex
# → paper.pdf
```

To clean intermediate files: `latexmk -C`.

## Layout

```
report/arxiv/
├── README.md                       (this file)
├── paper.tex                       (orchestrator: \input{...} the sections below)
├── preamble.tex                    (preamble + macros, adapted from CS-137 version)
├── refs.bib                        (bibliography)
├── sec_intro.tex                   §1 Introduction (incl. Abstract)
├── sec_relwork.tex                 §2 Related work
├── sec_data.tex                    §3 Problem setting & data
├── sec_part1_design.tex            §4–§7  PART I: design & training
├── sec_part2_deployment.tex        §8–§10 PART II: deployment & findings
├── sec_discussion.tex              §11 Discussion + §12 Conclusion
├── appendix_a_hrrr_fetcher.tex     A.  HRRR fetcher implementation
├── appendix_b_iso_ne_endpoint.tex  B.  ISO-NE cookie-prime endpoint
├── appendix_c_datasheet.tex        C.  Dataset documentation
├── appendix_d_repro.tex            D.  Reproducibility (full hparams, seeds, env)
├── appendix_e_attention.tex        E.  Attention extraction procedure
├── appendix_f_ai_disclosure.tex    F.  AI-tool disclosure
└── figures/
    ├── architecture.png            (reused from ../figures/, baseline arch diagram)
    ├── attn_*.png                  (4 attention figures from Part 3)
    ├── iso_ne_map.png
    ├── supplementary_*.png
    ├── architecture_3repo.tex      (NEW: TikZ source for the 3-repo deployment diagram)
    ├── deployment_per_zone_mape.png             (NEW)
    ├── training_vs_deployment_per_zone_drift.png (NEW)
    ├── validation_dec30_2022_overlay.png         (NEW)
    └── state_solar_density_vs_mape_drift.png     (NEW)
```

## Reproducibility

Every numeric claim in the paper has a script that produces it:

| Claim | Script |
|---|---|
| Cluster's stored Dec 30 2022 baseline MAPE = 6.54 % | `pretrained_models/baseline/dump/baseline_preds_test_2022_last2d.json` (precomputed) |
| Live pipeline rerun MAPE = 6.41 % on Dec 30 2022 | `scripts/validation/reproduce_dec30_2022.py` |
| Live 7-day rolling backtest MAPE numbers | `https://raw.githubusercontent.com/jeffliulab/new-england-real-time-power-predict-data/main/data/backtest_rolling_7d.json` (cron-refreshed daily) |
| Per-zone deployment MAPE bar chart | `scripts/figures/render_deployment_per_zone_mape.py` |
| Training-vs-deployment per-zone drift figure | `scripts/figures/render_training_vs_deployment_drift.py` |
| Pipeline validation overlay figure | `scripts/figures/render_validation_dec30.py` |
| BTM solar density correlation | `scripts/figures/render_solar_correlation.py` |

The full live demo (Gradio Space + auxiliary cron data repo) is documented at:
- Code:    https://github.com/jeffliulab/real-time-power-predict
- Data:    https://github.com/jeffliulab/new-england-real-time-power-predict-data
- Demo:    https://huggingface.co/spaces/jeffliulab/predict-power
