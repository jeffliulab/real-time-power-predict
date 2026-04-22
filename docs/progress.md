# Progress & Work Plan

Snapshot: 2026-04-22 (Part 2 due today EOD; 9 days before final due 2026-05-01)

## Part 1 — Baseline CNN-Transformer (40 pts)

**Status: ✅ Done, submitted, independently verified.**

- Architecture: hybrid CNN-Transformer per assignment Figure 2. 1.75 M params.
- Training: 15 epochs planned, 14 completed within 24 h SLURM time limit on A100-40GB (job 453913 on pax052, 2026-04-14 → 2026-04-15).
- Best epoch: 12. Val MAPE 6.92 % on 2021.
- **Independently verified test MAPE: 5.24 %** on last 2 days of 2022.
  - TA evaluator (job 462686, 2026-04-17, old `evaluation/pliu07/` location).
  - Our own `scripts/self_eval.py` (job 36590769, 2026-04-21, against canonical `evaluation/part1-models/pangliu/` submission). Numbers match byte-for-byte.
- Canonical submission: `/cluster/tufts/c26sp1cs0137/data/assignment3_data/evaluation/part1-models/pangliu/`. Old `evaluation/pliu07/` cleaned up.

### Artifacts
- `models/cnn_transformer.py`, `models/__init__.py`
- `training/train.py`, `training/data_preparation/dataset.py`
- `evaluation/pangliu/model.py` (Part 1 eval wrapper)
- `runs/cnn_transformer/{config.json, norm_stats.pt, logs/training_log.csv, figures/training_curves.png}`
- `runs/cnn_transformer/checkpoints/{best.pt, latest.pt}` (21 MB each, gitignored; present locally)
- `scripts/train.slurm` (original trainer), `scripts/self_eval.py` + `scripts/self_eval.slurm` (our own evaluator, model-agnostic)

---

## Part 2 — Architecture Search (30 pts, due 2026-04-22 EOD)

**Status: 🔨 Code ready, pending HPC training.**

### Decision

After surveying ~27 papers across Transformer time-series forecasting, spatiotemporal multi-modal fusion, and 2022–2025 short-term load forecasting, we picked **encoder-decoder with cross-attention** (new model name: `cnn_encoder_decoder`). Rationale:

- Matches the day-ahead inductive bias: known future covariates (weather + calendar) act as queries, historical observations act as memory. Single joint encoder in baseline mixes these roles.
- Encoder attention cost drops from `3120²` to `1560²` (~4× cheaper per layer); reinvested into more epochs (18 vs baseline 14) and an optional future-weather cross-attention branch.
- Stronger report narrative for the "architectural comparison" requirement.

Full technical report: [docs/part2_report.md](part2_report.md).

### Tasks

- [x] Literature survey (~27 papers, recorded in part2_report.md references)
- [x] Implement `models/cnn_encoder_decoder.py` — `CNNEncoderDecoderForecaster` + `CrossAttentionBlock`, reuses baseline `ResBlock2d`/`WeatherCNN`/`EnergyTransformerBlock`
- [x] Register in `models/__init__.py` as `cnn_encoder_decoder`
- [x] Extend `training/train.py` with `--n_encoder_layers`, `--n_decoder_layers`, `--use_future_weather_xattn`, `--warmup_steps` flags + 500-step linear LR warmup logic
- [x] Write `scripts/train_cnn_encoder_decoder.slurm` (18 epoch, A100-gpu partition, 24 h)
- [x] Write `evaluation/part2-models/pangliu/model.py` (eval wrapper, mirrors Part 1)
- [x] Draft `docs/part2_report.md` (architecture, rationale, references, result table skeleton)
- [x] AST syntax-check all modified Python files
- [ ] **HPC smoke test** — confirm param count (estimated ~2.28 M for default 4+2 layers; tune down to 3+1 ≈ 1.82 M if we want parameter parity with baseline 1.75 M)
- [ ] **HPC training** — `sbatch scripts/train_cnn_encoder_decoder.slurm` once HPC is reachable
- [ ] Rsync results back, regenerate training curve
- [ ] Copy `best.pt` + `config.json` to canonical `evaluation/part2-models/pangliu/`
- [ ] Self-eval: `sbatch scripts/self_eval.slurm part2-models/pangliu 2`; confirm MAPE < 5.24 %
- [ ] Fill in result numbers in `docs/part2_report.md`
- [ ] Git commit + push new artifacts (gitignored: checkpoints)

### Open questions / decision points (resolve tomorrow on HPC)

1. **Parameter budget.** 4 encoder + 2 decoder ≈ 2.28 M (30 % heavier than baseline). If we want strict parity, drop to 3 enc + 1 dec ≈ 1.82 M. Current SLURM defaults to 4+2; switching is 2 CLI flags. Recommendation: run 4+2 first, then narrate honestly in report.
2. **Future-weather cross-attention.** Off by default; turn on via `--use_future_weather_xattn` if v1 headroom suggests CT / NEMA_BOST need more signal.
3. **Resume vs rerun** if SLURM cuts off before 18 epochs — `--resume` flag already present from Part 1.

---

## Part 3 — Diagnosis or Independent Study (30 pts, due 2026-05-01)

**Status: ⏳ Not started. Choose between Track A and Track B.**

### Track A — Geographic Attention Maps (recommended)

Extract attention weights between the 24 future tabular tokens (queries) and the `S+24 × P` spatial tokens (keys). Reshape back to the 8×8 spatial grid and map to an ISO-NE geographic overlay. Questions:
- Do different load zones attend to different regions of New England?
- Does the attention track incoming weather fronts across time?
- Do extreme demand days show sharper spatial attention than mild days?

Tight coupling to Part 1 / Part 2 models — both use `nn.MultiheadAttention` which exposes weights via `need_weights=True`.

### Track B — Independent Study (candidates)

- Performance during extreme temperatures (top/bottom decile of daily mean temperature) vs average.
- Effect of excluding holidays / shoulder days on MAPE.
- Transfer from a larger weather encoder (pretrained SimCLR / SatMAE on similar raster data).

### Preliminary recommendation

**Track A** for the main Part 3 deliverable (cheaper, visually compelling, assignment hint nudges toward it). If time allows, a mini Track B on extreme-demand performance as a complement.

### Concrete tasks (Track A)

- [ ] Modify `EnergyTransformerBlock.forward` (and new `CrossAttentionBlock.forward`) to optionally return attention weights
- [ ] Add `scripts/attention_maps.py` — load `best.pt`, run inference on a few validation samples, average attention weights per future-hour × zone, reshape to 2D, overlay on ISO-NE map (reuse the Figure 1 base map from the assignment PDF)
- [ ] Produce 3–5 representative figures for the report
- [ ] 2–3 paragraphs of analysis

---

## Report & Presentation (due 2026-05-01 / 2026-05-04)

- [ ] Report (PDF, ≤ 7 pages recommended): problem statement → Part 1 baseline → Part 2 design & results → Part 3 findings → contributions
- [ ] Presentation slides (≤ 7 pages): same flow, figure-heavy
- [ ] Group member contribution statement (solo project → single paragraph)

## Calendar

| Date | Target |
|---|---|
| 2026-04-21 | ✅ Recovery complete. ✅ Part 1 independently verified. ✅ Part 2 code written. |
| 2026-04-22 | ⏰ **Part 2 deadline EOD.** Morning: HPC smoke test + sbatch training. Afternoon/evening: eval, report numbers, git push, submit. |
| 2026-04-23 – 2026-04-27 | Part 3 Track A implementation + figures. Refine Part 2 if needed. |
| 2026-04-28 – 2026-04-30 | Write final report + slides. |
| 2026-05-01 | ⏰ Final code + report due EOD. |
| 2026-05-04 | ⏰ Presentation slides due EOD. |
| 2026-05-05 | 🎤 Presentations 12:00 – 14:00 EST. |

## Known risks / blockers

- **24 h SLURM wall time.** Baseline did 14 epochs; encoder-decoder's encoder is ~4× cheaper per layer, so 18 epochs should fit. `--patience 5` set as a safety net. `--resume` available if needed.
- **Course modules (`class/default`, `cs137/2026spring`) missing cluster-wide as of 2026-04-21.** Breaks TA's `test_run.sh` on batch partition. Workaround: we have our own `scripts/self_eval.py` / `scripts/self_eval.slurm` that uses the `anaconda/2023.07.tuftsai` module on gpu partition with `--device cpu` (CUDA library mismatch on node). Part 1 verified through this path; Part 2 will use the same.
- **HPC connectivity intermittent tonight.** Deferring all HPC operations to tomorrow morning. All local code is ready and pushed to GitHub.
- **Solo submission, no peer review.** Budget 1 day at the end to re-read spec before final submit.
