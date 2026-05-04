"""
c1: Chronos-Bolt zero-shot inference for ISO-NE day-ahead demand.

Loads a per-zone hourly demand CSV, runs Chronos-Bolt (zero-shot, no fine-tune)
on each of the 8 zones independently, and writes a JSON with predictions for a
configurable list of forecast start times (each 24h horizon).

Used as the "c1" leg of the supplementary three-way ensemble:
    final = alpha * baseline + beta * c1 + gamma * c2

Design notes
------------
* Univariate per-zone forecast (Chronos-Bolt does not natively take exogenous
  covariates without the AutoGluon wrapper). The decorrelation from baseline
  comes from Chronos NOT using weather, while baseline IS weather-driven.
* Default model: amazon/chronos-bolt-base (205M, Apache-2.0). On CPU laptop
  this runs in seconds for the test slice.
* Default context: last 672 hours (4 weeks) before each forecast start.
  Chronos-Bolt's effective context is ~512; we feed 672 and let the model
  truncate internally — this is the recipe Amazon ships in its tutorials.

Outputs
-------
JSON file at the path given by --out, with shape:
    {
        "model": "chronos-bolt-base",
        "model_card": "amazon/chronos-bolt-base",
        "split": "test"  | "val",
        "forecast_starts": [ISO timestamps],
        "horizon": 24,
        "zones": [ME, NH, VT, CT, RI, SEMA, WCMA, NEMA_BOST],
        "preds":  [[[float MWh, ...]]],   # shape (n_forecasts, 24, 8)
        "context_len": 672,
        "demand_csv": "<path>",
    }

Usage
-----
    python inference/run_chronos_zeroshot.py \\
        --demand_csv pretrained_models/baseline/dump/demand_2022_hourly.csv \\
        --split test \\
        --out runs/chronos_c1/preds_test_2022_last2d.json

    python inference/run_chronos_zeroshot.py \\
        --demand_csv pretrained_models/baseline/dump/demand_2022_hourly.csv \\
        --split val \\
        --val_start 2022-12-16 --val_days 14 \\
        --out runs/chronos_c1/preds_val_2022_last14d.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=UserWarning)

ZONE_COLS = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]
HORIZON = 24
DEFAULT_MODEL = "amazon/chronos-bolt-base"
DEFAULT_CONTEXT = 672


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demand_csv", type=str, required=True,
                   help="CSV with columns: timestamp_utc, ME, NH, ..., NEMA_BOST")
    p.add_argument("--model_card", type=str, default=DEFAULT_MODEL)
    p.add_argument("--context_len", type=int, default=DEFAULT_CONTEXT)
    p.add_argument("--split", type=str, choices=["test", "val"], required=True)
    p.add_argument("--year", type=int, default=2022)
    p.add_argument("--n_test_days", type=int, default=2,
                   help="For split=test: how many final days of the year")
    p.add_argument("--val_start", type=str, default=None,
                   help="For split=val: ISO date YYYY-MM-DD (start of val window)")
    p.add_argument("--val_days", type=int, default=14,
                   help="For split=val: how many consecutive days starting at val_start")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--torch_dtype", type=str, default="float32",
                   choices=["float32", "bfloat16"])
    return p.parse_args()


def load_demand(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "timestamp_utc" in df.columns:
        ts_col = "timestamp_utc"
    elif "timestamp" in df.columns:
        ts_col = "timestamp"
    else:
        raise ValueError(f"No timestamp column in {csv_path}; got {list(df.columns)}")
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True)
    df = df.rename(columns={ts_col: "timestamp"})
    missing = [z for z in ZONE_COLS if z not in df.columns]
    if missing:
        raise ValueError(f"Missing zone columns: {missing}")
    return df[["timestamp"] + ZONE_COLS]


def get_forecast_starts(df: pd.DataFrame, args) -> list[pd.Timestamp]:
    """Return list of forecast start timestamps (each at 00:00 UTC)."""
    if args.split == "test":
        target_dates = pd.date_range(
            end=pd.Timestamp(f"{args.year}-12-31"),
            periods=args.n_test_days,
            freq="D",
        )
        return [pd.Timestamp(d).normalize() for d in target_dates]
    else:
        start = pd.Timestamp(args.val_start).normalize()
        return [start + pd.Timedelta(days=k) for k in range(args.val_days)]


def slice_context(df: pd.DataFrame, start_ts: pd.Timestamp, ctx_len: int) -> np.ndarray:
    """Return (ctx_len, 8) array of demand history strictly before start_ts."""
    end_idx = df.index[df["timestamp"] == start_ts]
    if len(end_idx) == 0:
        raise ValueError(f"start_ts {start_ts} not found in demand CSV")
    end = int(end_idx[0])
    if end < ctx_len:
        raise ValueError(
            f"Need {ctx_len} hours of history before {start_ts}, only {end} available"
        )
    arr = df.loc[end - ctx_len: end - 1, ZONE_COLS].to_numpy(dtype=np.float32)
    if arr.shape != (ctx_len, 8):
        raise ValueError(f"Context shape {arr.shape}, expected ({ctx_len}, 8)")
    return arr


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[c1] loading demand CSV: {args.demand_csv}")
    df = load_demand(Path(args.demand_csv))
    print(f"[c1]   {len(df)} rows, "
          f"{df['timestamp'].min()} → {df['timestamp'].max()}")

    starts = get_forecast_starts(df, args)
    print(f"[c1] split={args.split}  n_forecasts={len(starts)}")
    print(f"[c1]   first start={starts[0]}  last start={starts[-1]}")

    print(f"[c1] loading {args.model_card} (zero-shot, no training)")
    t0 = time.time()
    from chronos import BaseChronosPipeline
    dtype = torch.float32 if args.torch_dtype == "float32" else torch.bfloat16
    pipeline = BaseChronosPipeline.from_pretrained(
        args.model_card,
        device_map=args.device,
        torch_dtype=dtype,
    )
    print(f"[c1]   loaded in {time.time() - t0:.1f}s")

    n_forecasts = len(starts)
    preds = np.zeros((n_forecasts, HORIZON, 8), dtype=np.float32)

    for i, start in enumerate(starts):
        ctx = slice_context(df, start, args.context_len)        # (T, 8)
        ctx_tensor = torch.from_numpy(ctx.T).to(torch.float32)   # (8, T)

        t1 = time.time()
        # Chronos-Bolt returns quantile predictions (B, num_quantiles, H) and mean (B, H)
        quantiles, mean = pipeline.predict_quantiles(
            context=ctx_tensor,
            prediction_length=HORIZON,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        median = quantiles[:, :, 1]   # (8, 24) — q=0.5
        preds[i] = median.cpu().numpy().T   # (24, 8)

        print(f"[c1]   [{i+1}/{n_forecasts}] start={start.strftime('%Y-%m-%d %H:%M')} "
              f"({time.time() - t1:.1f}s)  "
              f"pred mean={preds[i].mean():.0f} MWh")

    out = {
        "model": args.model_card.split("/")[-1],
        "model_card": args.model_card,
        "split": args.split,
        "forecast_starts": [s.isoformat() for s in starts],
        "horizon": HORIZON,
        "zones": ZONE_COLS,
        "preds": preds.tolist(),
        "context_len": args.context_len,
        "demand_csv": str(args.demand_csv),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[c1] wrote {out_path}  ({os.path.getsize(out_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
