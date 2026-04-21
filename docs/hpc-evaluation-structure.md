# HPC Evaluation Directory — `/cluster/tufts/c26sp1cs0137/data/assignment3_data/evaluation/`

Reference map of the course cluster's shared evaluation area, used by the graders and by each team to self-test their submissions. Snapshot as of 2026-04-21.

## Top-level layout

```
/cluster/tufts/c26sp1cs0137/data/
├── assignment3_data/
│   ├── energy_demand_data/      # owner: plympe01 — raw energy CSVs
│   ├── weather_data/            # owner: lliu10   — raw weather tensors (~278 GB total with energy)
│   └── evaluation/              # ← all grading + submission lives here
│       ├── evaluate.py          # grader script (run with `evaluate.py <MODEL_NAME> [N_DAYS]`)
│       ├── test_run.sh          # SLURM wrapper; uses $SLURM_JOB_NAME as MODEL_NAME
│       ├── example_model/       # reference implementation of the model.py interface
│       ├── part1-models/        # canonical Part 1 submission location (per Piazza 2026-04-16)
│       └── <team_or_netid>/     # older/personal dev folders (pre-part1-models announcement)
```

`evaluate.py` treats **any folder under `evaluation/`** as a submission: it resolves `model.py` at `evaluation/<MODEL_NAME>/model.py`, inserts that folder onto `sys.path`, then calls `module.get_model(metadata)`. The folder name can contain a slash, so `part1-models/pangliu` works as a MODEL_NAME.

## Expected submission interface

Every `model.py` must expose:

```python
def get_model(metadata: dict) -> nn.Module: ...
```

The returned module needs two methods:

| Method | Signature | Purpose |
|---|---|---|
| `adapt_inputs(history_weather, history_energy, future_weather, future_time)` | returns a tuple that gets unpacked into `forward` | preprocessing (history subsampling, calendar feature extraction, normalization) |
| `forward(*adapt_inputs(...))` | returns `(B, 24, n_zones)` **float32 in raw MWh** | actual prediction |

Raw evaluator inputs (shapes from `evaluate.py`):
- `history_weather : (B, 168, 450, 449, 7)` float32 (last 7 days hourly)
- `history_energy  : (B, 168, n_zones)` float32
- `future_weather  : (B, 24, 450, 449, 7)` float32
- `future_time     : (B, 24)` int64 — hours since Unix epoch

`metadata` has keys: `zone_names`, `n_zones`, `history_len=168`, `future_len=24`, `n_weather_vars=7`.

## Part 1 submission location (canonical)

Piazza "Part 1 Model submissions" post (2026-04-16) instructed students to put Part 1 deliverables under:

```
/cluster/tufts/c26sp1cs0137/data/assignment3_data/evaluation/part1-models/<team>/
```

Current state of `part1-models/`:

| Team folder | Submitter (NetID) | Included files |
|---|---|---|
| `as_nl/` | ashen05 | best_model.pt, model.py, norm_stats.pt, train_eval log |
| `egsw/` | swebbe01 | best_weights.pt, config.json, model.py, our_tests/, README.md |
| `eric_hunter/` | ezhao05 | best_model.pt, model.py (self-contained) |
| `logan_odalys_matthew/` | lyuan04 | best_model.pth, model.py |
| `nsp/` | nchang06 | checkpoints/, model.py |
| `pangliu/` | **pliu07 (this repo)** | best.pt, config.json, model.py, models/__init__.py, models/cnn_transformer.py |
| `smna/` | smalla01 | best.pt, model.py, test_run log |

Submission content conventions vary — weights are named `best.pt` / `best_model.pt` / `best_weights.pt` / `best_model.pth`, but `model.py` is always present at the top of the folder. The `model.py` can either bundle its architecture in-line (most teams) or import from a sibling `models/` package inside the submission folder (our approach: [../evaluation/pangliu/model.py](../evaluation/pangliu/model.py)).

## Other folders at the top of `evaluation/`

These are pre-`part1-models` personal or team dev folders that were the original submission convention (same format as Assignment 2). Most remain for historical reference; the graded Part 1 submissions are those inside `part1-models/`.

| Folder | Purpose / status |
|---|---|
| `example_model/` | TA reference implementation (look here if rebuilding `model.py` from scratch) |
| `evaluate.py` | The grader script. Loads model.py via `importlib`, feeds 168 h of history + 24 h of future weather, records MAPE per zone. Top of file documents the expected interface. |
| `evaluate.py.save` | Older snapshot saved by another student; ignore. |
| `test_run.sh` | SLURM submission helper. `sbatch -J <MODEL_NAME> test_run.sh [N_DAYS]`. `N_DAYS` defaults to 2. Note: uses `#SBATCH --partition=batch` + `module load class/default cs137/2026spring`; as of 2026-04-21 this combination fails module loading on batch — submit to `gpu` partition or run from a login node with the modules already loaded. |
| `pliu07/` | **Our older Part 1 submission (2026-04-17)**, placed before the Piazza `part1-models/` instruction. Identical content to `part1-models/pangliu/`. Kept as a backup; should be deleted once TA grades the canonical copy. |
| Personal dev folders (`cbradn01_dev/`, `cgolem01/`, `ezhao05_part1/`, `jnaran01_part1/`, `jocylyn_weiwei_part1/`, `Liam_and_Chris/`, `ohaber02/`, `pherre_ckpt5/`, etc.) | Other students' work-in-progress or pre-canonical submissions. |
| `*_part2/` folders | Some teams have staged Part 2 work in their own folders. Awaiting similar canonical announcement for Part 2 submission path. |

## Running a self-test

```bash
# On HPC
cd /cluster/tufts/c26sp1cs0137/data/assignment3_data/evaluation
sbatch -J part1-models/pangliu test_run.sh 2  # 2 days of inference
```

Logs land inside the submission folder: `part1-models/pangliu/test_run_<job>.out`.

If the batch partition's module load fails, fall back to an srun'd GPU node:

```bash
srun -p gpu --gres=gpu:1 --time=1:00:00 --pty bash
module load class/default cs137/2026spring
cd /cluster/tufts/c26sp1cs0137/data/assignment3_data/evaluation
python evaluate.py part1-models/pangliu 2
```

## Past submission results (ours)

| Date | Location | Test slice | MAPE |
|---|---|---|---|
| 2026-04-17 | `evaluation/pliu07/` (old location) | last 2 days of 2022 | **5.24 %** |
| 2026-04-21 | `evaluation/part1-models/pangliu/` (canonical) | pending re-verification | — |

Per-zone breakdown from the 4/17 run: ME 2.31%, NH 3.69%, VT 5.95%, CT 7.28%, RI 5.27%, SEMA 5.44%, WCMA 5.87%, NEMA_BOST 6.09%.
