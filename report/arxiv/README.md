# arXiv preprint source — workshop format

A 4-page main + supplementary appendix, NeurIPS/ICLR-workshop-style
report on a publicly-deployed multimodal forecaster for ISO New
England day-ahead per-zone demand, focused on a quantitative drift
case study (May 2025 vs May 2026) aligned with state-level BTM solar
buildout.

This is the arXiv version, pinned at the **v1.5** release tag.

## Build

```bash
cd report/arxiv
latexmk -xelatex -interaction=nonstopmode -halt-on-error paper.tex
# → paper.pdf
```

If the placeholder TBD markers in the PDF look red, run the fill
script to swap them for actual experimental numbers:

```bash
python scripts/figures/fill_paper_placeholders.py
```

This reads `report/arxiv/data/*.json` and rewrites the placeholder
macros in `preamble.tex`.

## Layout

```
report/arxiv/
├── README.md                     (this file)
├── paper.tex                     (orchestrator)
├── preamble.tex                  (preamble + \PH placeholder macros)
├── refs.bib                      (bibliography)
├── sec_intro.tex                 §1 Introduction (incl. abstract)
├── sec_setup.tex                 §2 Setup — task / data / model / deployment
├── sec_validation.tex            §3 Pipeline validation (Dec 30 2022 rerun)
├── sec_drift.tex                 §4 Two-window deployment-drift quantification
├── sec_btm.tex                   §5 State-level BTM solar alignment
├── sec_discussion.tex            §6 Discussion + Conclusion + Climate impact
├── appendix_a_model.tex          A. Model architecture
├── appendix_b_hparams.tex        B. Hyperparameters + training trajectory
├── appendix_c_bootstrap.tex      C. Bootstrap procedure
├── appendix_d_baselines.tex      D. Naive baseline implementations
├── appendix_e_forecast_shift.tex E. Future-weather forecast vs analyses shift
├── appendix_f_full_tables.tex    F. Full per-window per-zone tables
├── appendix_g_repro.tex          G. Reproducibility
├── appendix_h_attention.tex      H. Attention diagnostics (compressed)
├── data/
│   ├── validation_dec30_2022.json
│   ├── multi_window_results.json
│   ├── btm_correlation.json
│   └── future_weather_shift.json
└── figures/
    ├── architecture.png             (CNN-Transformer arch)
    ├── architecture_3repo.tex       (TikZ 3-repo deployment diagram)
    ├── attn_*.png                   (4 attention diagnostic figures)
    ├── iso_ne_map.png               (ISO-NE 8-zone map)
    ├── validation_dec30_2022_overlay.png
    ├── multi_window_overall_mape.png
    ├── multi_window_per_zone_mape.png
    ├── per_day_mape_timeline.png
    └── btm_solar_correlation.png
```

## Reproducibility

Every numeric claim in the paper traces back to a script + a JSON file.

| Claim | Script | Data |
|---|---|---|
| §3 Validation: 6.41% live vs 6.54% cluster | `scripts/validation/reproduce_dec30_2022.py` + `augment_validation_with_ci.py` | `data/validation_dec30_2022.json` |
| §4 Multi-window MAPE table + per-day + per-zone figures | `scripts/experiments/historical_drift_sweep.py` | `data/multi_window_results.json` |
| §5 BTM solar correlation + Spearman/permutation | `scripts/figures/render_btm_solar_correlation.py` | `data/btm_correlation.json` |
| App. E forecast-vs-analyses shift | `scripts/validation/future_weather_shift_quantify.py` | `data/future_weather_shift.json` |

Public artifacts:
- Code:    https://github.com/jeffliulab/real-time-power-predict
- Data:    https://github.com/jeffliulab/new-england-real-time-power-predict-data
- Demo:    https://huggingface.co/spaces/jeffliulab/predict-power
- Model:   https://huggingface.co/jeffliulab/predict-power-baseline
