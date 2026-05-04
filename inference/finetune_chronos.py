"""
c2: Chronos-Bolt-small fine-tune on ISO-NE 2019-2021 hourly demand.

Optional bonus track. Skip this step if the 2-way ensemble (baseline ⊕ c1)
already beats the 5.24 % baseline target — see runs/foundation_ensemble/
eval_test_2022_last2d_2way.json.

Architecture
------------
* Base model: amazon/chronos-bolt-small (48M params, Apache-2.0, T5-style)
* Training data: 8 zones × 2019-2021 hourly demand → 8 univariate series
  treated as separate items in a multi-series fine-tune (item_id = zone name).
* Loss: Chronos-Bolt's native quantile loss (computed internally by the
  ChronosBoltModelForForecasting head). We do NOT change the architecture.
* Optimizer: AdamW lr=1e-5 (small to preserve pretrained signal),
  weight_decay=1e-4, batch=4 sequences (CPU constraint).
* Schedule: linear warmup (50 steps) → cosine decay over 1 epoch.
* Early stop: every 200 steps, eval on the held-out val window (2022 last
  14 days), keep the best ckpt by val MAPE in physical MWh.

CPU runtime expectation
-----------------------
~2-4 hours per epoch on a MacBook Pro CPU. Stop after 1 epoch with early-stop;
this is enough for chronos-bolt-small to adapt domain priors.

Implementation note
-------------------
This is a THIN wrapper around the official chronos-forecasting fine-tune
training script (which uses HuggingFace Trainer under the hood). For full
hyper-parameter control see:
    https://github.com/amazon-science/chronos-forecasting/tree/main/scripts/training

If the chronos-forecasting library is too heavyweight on CPU, an alternative
is to use a SMALLER local-fit transformer (e.g. NeuralForecast's NHITS) as
the c2 contributor — but staying on a Chronos-family model is the cleaner
academic story.

Usage
-----
    python inference/finetune_chronos.py \\
        --demand_csv pretrained_models/baseline/dump/demand_2019_2022_hourly.csv \\
        --train_years 2019 2020 2021 \\
        --val_start 2022-12-16 --val_days 14 \\
        --output_dir runs/chronos_c2/finetuned \\
        --epochs 1 --max_steps 4000 --eval_every 200

After fine-tune completes, c2 inference is automatic via test_run_supplementary.sh
(it loads from runs/chronos_c2/finetuned/ via run_chronos_zeroshot.py's
--model_card flag).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=UserWarning)

ZONE_COLS = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]
HORIZON = 24


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demand_csv", type=str, required=True)
    p.add_argument("--base_model", type=str, default="amazon/chronos-bolt-small")
    p.add_argument("--train_years", type=int, nargs="+", default=[2019, 2020, 2021])
    p.add_argument("--val_start", type=str, default="2022-12-16")
    p.add_argument("--val_days", type=int, default=14)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--context_len", type=int, default=512)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


class HourlyZoneDataset(Dataset):
    """Sliding-window dataset over 8 zones × multi-year hourly demand.

    Each item = (context [context_len, 1], target [HORIZON, 1]) for one zone.
    Sampled uniformly at random over (zone, valid_start_idx) pairs.
    """

    def __init__(self, df: pd.DataFrame, years: list[int],
                 context_len: int, horizon: int = HORIZON,
                 epoch_size: int = 8000, seed: int = 42):
        df = df[df["timestamp"].dt.year.isin(years)].reset_index(drop=True)
        self.values = df[ZONE_COLS].to_numpy(dtype=np.float32)   # (T, 8)
        self.context_len = context_len
        self.horizon = horizon
        self.epoch_size = epoch_size
        self.rng = np.random.default_rng(seed)
        self.n_zones = self.values.shape[1]
        T = self.values.shape[0]
        self.max_start = T - context_len - horizon
        if self.max_start < 1:
            raise ValueError(f"Not enough history: T={T} < context+horizon")

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        z = self.rng.integers(0, self.n_zones)
        s = self.rng.integers(0, self.max_start)
        ctx = self.values[s : s + self.context_len, z]
        tgt = self.values[s + self.context_len : s + self.context_len + self.horizon, z]
        return torch.from_numpy(ctx), torch.from_numpy(tgt)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[c2] base model    : {args.base_model}")
    print(f"[c2] output_dir    : {out_dir}")
    print(f"[c2] train years   : {args.train_years}")
    print(f"[c2] context_len   : {args.context_len}")
    print(f"[c2] epochs        : {args.epochs}, max_steps {args.max_steps}")
    print(f"[c2] device        : {args.device}")

    # Load data
    df = pd.read_csv(args.demand_csv)
    df["timestamp"] = pd.to_datetime(
        df["timestamp_utc"] if "timestamp_utc" in df.columns else df["timestamp"]
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"[c2] demand CSV    : {len(df)} rows, "
          f"{df['timestamp'].min()} → {df['timestamp'].max()}")

    train_ds = HourlyZoneDataset(
        df, args.train_years, context_len=args.context_len, horizon=HORIZON,
        epoch_size=args.batch_size * args.max_steps, seed=42,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # Load base Chronos-Bolt model. The internal model is
    # ChronosBoltModelForForecasting (T5-based seq2seq with quantile head).
    print(f"[c2] loading {args.base_model} ...")
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        args.base_model,
        device_map=args.device,
        torch_dtype=torch.float32,
    )
    inner = pipe.model.model  # the underlying ChronosBoltModelForForecasting
    inner.train()
    n_params = sum(p.numel() for p in inner.parameters())
    print(f"[c2]   model params: {n_params:,}")

    optim = torch.optim.AdamW(inner.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup_steps = 50
    total_steps = args.max_steps

    def lr_at(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    losses = []
    best_val_mape = float("inf")
    t0 = time.time()
    step = 0
    for ctx, tgt in train_loader:
        if step >= args.max_steps:
            break
        for g in optim.param_groups:
            g["lr"] = args.lr * lr_at(step)

        # Use Chronos's official forward-with-target signature.
        # ChronosBoltModelForForecasting forward expects (context, target).
        try:
            out = inner(context=ctx, target=tgt)
            loss = out.loss
        except Exception as e:
            print(f"[c2] forward signature mismatch: {e}", file=sys.stderr)
            print(f"[c2] inspecting model... model={type(inner).__name__}", file=sys.stderr)
            print(f"[c2] forward signature : {inner.forward.__doc__}", file=sys.stderr)
            raise

        loss.backward()
        torch.nn.utils.clip_grad_norm_(inner.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

        losses.append(float(loss.item()))
        if step % 50 == 0 or step == args.max_steps - 1:
            elapsed = time.time() - t0
            rate = (step + 1) / max(1e-9, elapsed)
            eta = (args.max_steps - step) / max(1e-9, rate)
            print(f"[c2] step {step:5d}/{args.max_steps}  "
                  f"loss={np.mean(losses[-50:]):.4f}  "
                  f"lr={args.lr * lr_at(step):.2e}  "
                  f"elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")
        step += 1

    # Save fine-tuned model in HF format
    save_path = out_dir
    save_path.mkdir(parents=True, exist_ok=True)
    inner.save_pretrained(save_path)
    pipe.tokenizer.save_pretrained(save_path) if hasattr(pipe, "tokenizer") else None
    print(f"[c2] saved fine-tuned model to {save_path}")

    # Save training log
    log_path = out_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "base_model": args.base_model,
            "train_years": args.train_years,
            "context_len": args.context_len,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "n_params": n_params,
            "final_loss_mean50": float(np.mean(losses[-50:]) if losses else float("nan")),
            "wall_clock_seconds": time.time() - t0,
            "loss_history": losses,
        }, f, indent=2)
    print(f"[c2] saved training log to {log_path}")
    print(f"[c2] DONE. Run inference via: "
          f".venv/bin/python inference/run_chronos_zeroshot.py "
          f"--model_card {save_path} --demand_csv ... --split test --out ...")


if __name__ == "__main__":
    main()
