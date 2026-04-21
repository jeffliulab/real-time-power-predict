# Progress & Work Plan

Snapshot: 2026-04-21 (7 days before final due 2026-05-01)

## Part 1 — Baseline CNN-Transformer (40 pts)

**Status: ✅ Done, submitted.**

- Architecture: hybrid CNN-Transformer per assignment Figure 2. 1.75 M params.
- Training: 15 epochs planned, 14 completed within 24 h SLURM time limit on A100-40GB (job 453913 on pax052, 2026-04-14 → 2026-04-15).
- Best epoch: 12. Val MAPE 6.92 % on 2021.
- Evaluator self-test: MAPE **5.24 %** on last 2 days of 2022 via `test_run.sh` (job 462686 on 2026-04-17).
- Canonical submission location (after Piazza 4/16 clarification): `/cluster/tufts/c26sp1cs0137/data/assignment3_data/evaluation/part1-models/pangliu/`.

### Artifacts recovered from HPC
- `models/cnn_transformer.py`, `models/__init__.py`
- `training/train.py`, `training/data_preparation/dataset.py`
- `evaluation/pangliu/model.py`
- `runs/cnn_transformer/{config.json,norm_stats.pt,logs/training_log.csv,figures/training_curves.png}`
- `runs/cnn_transformer/checkpoints/{best.pt,latest.pt}` (21 MB each, gitignored; present locally)
- 5 SLURM `.out/.err` logs (gitignored)
- `scripts/train.slurm`

---

## Part 2 — Architecture Search (30 pts, due 2026-04-22 EOD)

**Status: 🔨 Planned — no code yet.**

The assignment asks for an architecture that beats the baseline, with a comparative analysis. Four suggested directions (one is enough):

| Direction | Effort | Expected payoff | Risk |
|---|---|---|---|
| Encoder-Decoder w/ cross-attention | ~200 LOC model + reuse existing trainer | High — future timesteps can cross-attend directly to historical context without paying the full `(S+24)²` self-attention cost | Low; architecturally close to baseline |
| Hierarchical attention (spatial-then-temporal) | ~300 LOC | High for long S; may reduce quadratic cost | Medium, more refactoring |
| Recurrent / GRU decoder over Transformer encoder | ~150 LOC | Medium | Medium, fewer wins historically in forecasting |
| Feature engineering + pretrained CNN backbone | Moderate | Depends heavily on data | High, extra data-engineering time |

**Recommended path: Encoder-Decoder.** Historical weather + demand + calendar go into a Transformer encoder; future calendar (+ masked future weather) become decoder queries that cross-attend over the encoded history. Advantages:
- Reuses the existing dataset, normalization stats, and training loop one-for-one.
- Sequence length is `S·(P+1)` in the encoder and `24·(P+1)` in the decoder instead of `(S+24)·(P+1)` self-attention — lower memory headroom means we can try `G=10` (100 spatial tokens) for free.
- Straightforward to ablate (cross-attention on/off, decoder-only on future weather, shared vs. split CNNs).

### Concrete tasks for Part 2

- [ ] Implement `models/cnn_transformer_ed.py` (encoder-decoder variant)
- [ ] Register in `models/__init__.py` as `cnn_transformer_ed`
- [ ] Add `scripts/train_ed.slurm` (copy from `train.slurm` with `--model cnn_transformer_ed`)
- [ ] Sync to HPC and `sbatch` (24 h A100 budget, 15 epochs, same train/val split)
- [ ] Pull results back and regenerate training curve
- [ ] Run self-test via evaluator with `-J part2-models/pangliu` (wait for Piazza on the exact canonical folder name; if not announced, use `part1-models/pangliu_ed` or similar)
- [ ] Side-by-side comparison table vs baseline (overall + per-zone MAPE)
- [ ] Short discussion (1–2 paragraphs) on why the new arch differs, for the final report

---

## Part 3 — Diagnosis or Independent Study (30 pts, due 2026-05-01)

**Status: ⏳ Not started. Choose between Track A and Track B.**

### Track A — Geographic Attention Maps
Extract attention weights between the 24 future tabular tokens (queries) and the `S+24 × P` spatial tokens (keys). Reshape back to the 8×8 spatial grid and map to a geographic overlay. Questions to answer:
- Do different load zones attend to different regions of New England?
- Does the attention track incoming weather fronts across time?
- Do extreme demand days show sharper spatial attention than mild days?

This track is tightly coupled to Part 1/2: the model must expose attention weights (currently `nn.MultiheadAttention` is used which can return them with `need_weights=True`).

### Track B — Independent Study
Candidate hypotheses:
- Performance during extreme temperatures (e.g. top/bottom decile of daily mean temperature) vs average.
- Effect of excluding holidays / shoulder days on MAPE.
- Transfer from a larger weather encoder (pretrained SimCLR / SatMAE on similar raster data).

**Preliminary recommendation: Track A.** Cheaper to implement (we already have all the weights), has a very publishable-looking visualization for the presentation, and the assignment's hint text nudges toward it. If time allows after Part 2, do a mini Track B on extreme-demand performance.

### Concrete tasks for Part 3 (Track A assumed)

- [ ] Modify `EnergyTransformerBlock.forward` to optionally return attention weights
- [ ] Add `scripts/attention_maps.py` that loads `best.pt`, runs inference on a few validation samples, averages attention weights per future-timestep × zone, reshapes to 2D, overlays on the ISO-NE map (reuse the Figure 1 base map from the assignment PDF).
- [ ] Produce 3–5 representative figures for the report.
- [ ] 2–3 paragraphs of analysis.

---

## Report & Presentation (due 2026-05-01 / 2026-05-04)

- [ ] Report (PDF, ≤ 7 pages recommended): problem statement → Part 1 baseline → Part 2 design & results → Part 3 findings → contributions.
- [ ] Presentation slides (≤ 7 pages): same flow, figure-heavy.
- [ ] Group member contribution statement (solo project → single paragraph).

## Calendar

| Date | Target |
|---|---|
| 2026-04-21 | ✅ Recovery complete (this commit). Start Part 2 encoder-decoder. |
| 2026-04-22 | ⏰ Part 2 milestone: submit EOD. |
| 2026-04-23 – 2026-04-27 | Part 3 Track A + refine Part 2 if needed. |
| 2026-04-28 – 2026-04-30 | Write report and slides. |
| 2026-05-01 | ⏰ Final code + report due EOD. |
| 2026-05-04 | ⏰ Presentation slides due EOD. |
| 2026-05-05 | 🎤 Presentations 12:00 – 14:00 EST. |

## Known risks / blockers

- **24 h SLURM wall time.** Baseline barely finished 14 epochs; encoder-decoder has similar cost. If Part 2 needs more than 15 epochs, either early-stop with `--patience 3` or split training into two 24 h jobs via `--resume`.
- **Batch partition modules broken as of 2026-04-21.** TA's `test_run.sh` uses `-p batch` and `module load class/default cs137/2026spring` — module load currently fails there. Workaround: run evaluator on a gpu-partition srun'd shell with modules loaded. The TA's grading pipeline may run on a different node; not blocking submission placement.
- **No peer review yet.** Solo submission. Budget 1 day at the end to re-read spec before final submission.
