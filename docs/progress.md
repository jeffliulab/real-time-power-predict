# Progress & Work Plan

Snapshot: 2026-04-27 (5 days past Part 2 deadline; 4 days before final due 2026-05-01)

## Part 1 — Baseline CNN-Transformer (40 pts)

**Status: ✅ Done, submitted, independently verified.**

- Architecture: hybrid CNN-Transformer per assignment Figure 2. 1.75 M params.
- Training: 15 epochs planned, 14 completed within 24 h SLURM time limit on A100-40GB (job 453913 on pax052, 2026-04-14 → 2026-04-15).
- Best epoch: 12. Val MAPE 6.92 % on 2021.
- **Independently verified test MAPE: 5.24 %** on last 2 days of 2022.
  - TA evaluator (job 462686, 2026-04-17, old `evaluation/pliu07/` location).
  - Our own `scripts/self_eval.py` (job 36590769, 2026-04-21, against canonical `evaluation/part1-models/pangliu/` submission). Numbers match byte-for-byte.
- Canonical submission: `/cluster/tufts/c26sp1cs0137/data/assignment3_data/evaluation/part1-models/pangliu/`. Old `evaluation/pliu07/` cleaned up.

### Per-zone test MAPE (2022 last 2 days)

| Zone | MAPE |
|---|---|
| ME | 2.31 % |
| NH | 3.69 % |
| VT | 5.95 % |
| **CT** | **7.28 %** ← hardest |
| RI | 5.27 % |
| SEMA | 5.44 % |
| WCMA | 5.87 % |
| **NEMA_BOST** | **6.09 %** ← second hardest |

---

## Part 2 — Architecture Search (30 pts)

**Status: 🚂 Training in progress (chain: v1 → v2 ablation).**

### Decision summary

After surveying ~27 papers across Transformer time-series forecasting, spatiotemporal multi-modal fusion, and 2022–2025 short-term load forecasting: **encoder-decoder with cross-attention** (`cnn_encoder_decoder`).

- Day-ahead forecasting is a translation-style task: known future covariates = queries, past observations = memory. Single joint encoder in baseline mixes these roles.
- Encoder attention cost drops from `3120²` to `1560²` (~4× cheaper per layer).
- Stronger report narrative for the "architectural comparison" requirement.

Full report: [docs/part2_report.md](part2_report.md).

### Best result so far (epoch-6 snapshot, 2026-04-23 frozen)

- Val MAPE on 2022: **8.63 %** (baseline at same epoch 6 was 8.76 %)
- Test MAPE on 2022 last 2 days: **6.82 %** (baseline at convergence: 5.24 %)
- Best at epoch 6 of an aborted 24h SLURM run; needs more training to converge.

### Training chain in flight (submitted 2026-04-27)

**v1 chain** — `cnn_encoder_decoder` with `use_future_weather_xattn=False`:
- `36804770` (RUNNING on P100, started 2026-04-27 ~13:00 EDT)
- `36804771` (PD, depends on 770)
- `36804772` (PD, depends on 771)

**v2 ablation chain** — `cnn_encoder_decoder_xattn` with `use_future_weather_xattn=True`:
- `36804839` (PD, depends on 772, fresh start because state_dict shape differs)
- `36804840` (PD, depends on 839, resumes from v2 latest.pt)
- `36804841` (PD, depends on 840)

72h v1 + 72h v2 = up to **144 h** of compute, no manual intervention required between jobs (`--dependency=afterany`).

### Tasks

- [x] Literature survey (~27 papers)
- [x] Implement `models/cnn_encoder_decoder.py`
- [x] Register in `models/__init__.py`
- [x] Extend `training/train.py` with encdec flags + LR warmup
- [x] Write `scripts/train_cnn_encoder_decoder.slurm`
- [x] Write `evaluation/part2-encoder-decoder/model.py`
- [x] Draft `docs/part2_report.md` (architecture, normalization, references, result skeleton)
- [x] Build persistent conda env at `/cluster/.../conda_envs/cs137/` (replaces broken cs137/2026spring module)
- [x] First 24h training (job 36623639, 2026-04-22 → 2026-04-23): 7 epochs, val MAPE 10.08% → 8.63%
- [x] Snapshot epoch-6 best.pt + run self-eval (job 36804139, 2026-04-27): test MAPE 6.82%
- [x] Submit chain of 6 jobs (any-GPU, --resume) for v1 + v2 ablation (2026-04-27)
- [ ] **Wait for chain to complete** (next ~3-6 days, autonomous)
- [ ] Rsync final best.pt + curves back; copy to canonical HPC `evaluation/part2-models/pangliu/`
- [ ] Self-eval: `sbatch scripts/self_eval.slurm part2-models/pangliu 2`; confirm MAPE
- [ ] Fill in result numbers in `docs/part2_report.md` §5 + `runs/model_registry.json`
- [ ] Update README results table

### Open questions / decisions made

1. **Parameter budget**: Running 4 enc + 2 dec (~2.29 M, +30% vs baseline). Architecture search legitimately allows param changes.
2. **Future-weather cross-attention**: v2 ablation chain submitted to test the hypothesis that information parity with baseline boosts performance on weather-sensitive zones (CT, NEMA_BOST).
3. **GPU constraint**: Removed `--gres=gpu:a100:1` after one job sat in PD for 8h. Now any GPU acceptable; conda env validated on A100 + P100.

---

## Part 3 — Diagnosis or Independent Study (30 pts, due 2026-05-01)

**Status: 📋 Planned. Track A primary; minor real-time demo as bonus.**

### Track A — Geographic Attention Maps (recommended primary)

Extract attention weights between the 24 future tabular tokens (queries) and the spatial tokens (keys). Reshape back to the 8×8 spatial grid and map to an ISO-NE geographic overlay.

Tight coupling to Part 1 / Part 2 models — both use `nn.MultiheadAttention` which exposes weights via `need_weights=True`.

### Bonus — Real-time HF Spaces demo

Scaffold already in `space/` (Gradio + ISO Express data fetch placeholder). After training completes:
1. Copy `best.pt` to `space/checkpoints/`
2. Implement `space/iso_ne_fetch.py` (ISO Express API)
3. Implement `space/weather_fetch.py` (HRRR via Herbie, port from sibling project)
4. Deploy to HF Spaces

Detailed reading list: [docs/part3_references.md](part3_references.md).

### Concrete tasks (Track A)

- [ ] Modify `EnergyTransformerBlock.forward` (and `CrossAttentionBlock.forward`) to optionally return attention weights
- [ ] Add `scripts/attention_maps.py` — load `best.pt`, run inference on a few validation samples, average attention weights per future-hour × zone, reshape to 2D, overlay on ISO-NE map
- [ ] Produce 3–5 representative figures for the report
- [ ] 2–3 paragraphs of analysis

---

## Report & Presentation (due 2026-05-01 / 2026-05-04)

- [ ] Report (PDF, ≤ 7 pages): problem statement → Part 1 baseline → Part 2 design & results → Part 3 findings → contributions
- [ ] Presentation slides (≤ 7 pages): same flow, figure-heavy
- [ ] Group member contribution statement (solo project → single paragraph)

## Calendar

| Date | Target | Status |
|---|---|---|
| 2026-04-14 → 04-15 | Part 1 training (job 453913) | ✅ Done |
| 2026-04-17 | Part 1 TA self-test (job 462686) — MAPE 5.24% | ✅ Done |
| 2026-04-21 | Recovery, Part 2 code written, Part 1 independently re-verified | ✅ Done |
| 2026-04-22 | Smoke tests, CUDA probes, env build, first training (job 36623639 → epoch 6) | ✅ Done |
| 2026-04-22 EOD | ⏰ Part 2 nominal deadline | ⚠️ Late submission acceptable per Piazza |
| 2026-04-23 – 2026-04-26 | Idle while training stuck in PD queue (A100 contention) | — |
| 2026-04-27 | SSH key rotation, DUO investigation, project structure upgrade, chain submission | ✅ Done |
| 2026-04-27 → 2026-04-30 | Chain training (v1 + v2 ablation) on HPC, ~144h budget | 🚂 In progress |
| 2026-04-30 (target) | Final eval + Part 2 report numbers + submission | — |
| 2026-05-01 | ⏰ Part 3 + final report + final submission due EOD | — |
| 2026-05-04 | ⏰ Presentation slides due EOD | — |
| 2026-05-05 | 🎤 Presentations 12:00 – 14:00 EST | — |

---

## Development log

### 2026-04-21 — Project recovery + Part 1 verification

- Recovered all Part 1 code from Tufts HPC after laptop replacement.
- Set up SSH key auth (initial `tufts_hpc` ed25519 key) + ControlMaster mux config.
- Independently verified Part 1's MAPE 5.24 % on canonical submission (job 36590769) — bytes match TA's 4/17 result exactly.
- Cleaned up old `evaluation/pliu07/` folder.
- Wrote `docs/progress.md`, `docs/hpc-evaluation-structure.md`, README files.
- Initial commit + push to https://github.com/jeffliulab/real-time-power-predict.

### 2026-04-22 — Part 2 first attempt (rocky)

- Course modules `class/default` + `cs137/2026spring` discovered missing cluster-wide.
- Tried multiple module combinations (`anaconda/2023.07.tuftsai`, `cuda/11.7`, `cuda/12.2`, `cudnn/8.9.5-11.x`, `cudnn/8.9.7-12.x`) — every combination had some issue (libcublasLt.so.11 missing, cuDNN engine errors, symbol version skew).
- **Built persistent conda env** at `/cluster/.../conda_envs/cs137/` with python 3.10 + torch 2.3.1+cu118 + bundled cuDNN 8.9. Self-validated forward+backward including MultiheadAttention on both A100 and P100.
- Switched all training/eval SLURM scripts to use the conda env directly (no module load).
- First successful training (job 36623639): 7 epochs over 24h before SLURM timeout.
- Result at epoch 6: **val MAPE 8.63 %** (baseline at same epoch: 8.76 %).

### 2026-04-23 → 2026-04-26 — Idle, training stuck in queue

- Resume job 36803989 with `--gres=gpu:a100:1` constraint sat in PD for 8 hours, never started.
- A100 contention high during finals season.

### 2026-04-27 — Major productive day

#### SSH key rotation + DUO investigation

- Rotated `~/.ssh/tufts_hpc` → fresh `tufts_hpc_v2` ed25519 key.
- HPC `~/.ssh/authorized_keys` cleaned: removed both old `pliu07@tufts-hpc` and unknown `jeffl@Workstation` keys; only new `pliu07@JeffMac 2026-04-27` + system NIS key remain.
- **Discovered Tufts fires DUO per SSH session channel**, not per connection. Verified by running `ssh -O check` showing master alive but commands through it still triggering DUO. ControlMaster mux is much less effective than originally hoped.
- Confirmed Kerberos / GSSAPI is not configured on Tufts HPC (no realm, no kinit). Confirmed campus-IP "trusted network" bypass is not configured for ESAI app.
- Tested VSCode Remote-SSH — fails because Tufts login node has glibc 2.17 (CentOS 7), too old for current vscode-server. So we lose the "VSCode Remote-SSH = persistent daemon = no repeated DUO" workaround that's available on most other clusters.
- Findings recorded in memory + CLAUDE.md (hard rules for HPC access: no polling, bundle all queries into one ssh, never silent retry).

#### Training chain submission

- Cancelled stuck job 36803989.
- Removed `--gres=gpu:a100:1` constraint — any GPU acceptable.
- Submitted **v1 chain** (3 × 24h, --resume from latest.pt): jobs 36804770, 36804771, 36804772.
- Submitted **v2 ablation chain** (3 × 24h, fresh + 2 resumes, separate `runs/cnn_encoder_decoder_xattn/` dir): jobs 36804839, 36804840, 36804841.
- Job 36804770 entered RUNNING immediately on P100-PCIE-16GB (p1cmp073).

#### Project structure upgrade (modeled after sibling `real_time_weather_forecasting`)

- Added `inference/predict.py` — CLI inference entry point.
- Added `tests/smoke_test.py` — param count + forward+backward sanity check.
- Added `space/` — HF Spaces real-time demo skeleton (Gradio app, model loader, ISO-NE/HRRR fetch placeholders).
- Added `scripts/hf_upload.py` — push trained ckpt to Hugging Face Hub.
- Added `runs/model_registry.json` — central record of all trained models.
- Added `docs/part3_references.md` — Part 3 reading list and track plan.
- Added `CLAUDE.md` — repo-specific operating rules (HPC consent, no polling, mux preservation).
- Major polish to `README.md` / `README_zh.md` — badges, ToC, architecture diagrams, results table skeleton, status table.

---

## Known risks / blockers

- **Tufts fires DUO per SSH channel.** Every `ssh tufts-login` command costs one DUO push to the user's phone, even with ControlMaster mux alive. Mitigation: bundle all queries into one ssh (e.g. `rsync + sacct + tail logs + commit + grep` in a single heredoc). No silent retries.
- **Part 2 nominal deadline (4/22) passed.** Late submission acceptable but adds pressure. Plan: train chain through 4/30, finalize by 5/1 (final due date).
- **Course modules still missing.** Our conda env workaround is the only way; if the cs137/2026spring module returns, we can switch back.
- **A100 GPU contention.** Removed the constraint; any GPU works. P100 is ~2x slower per epoch but reliable.
- **Solo submission, no peer review.** Budget 1 day at the end to re-read spec before final submit.
