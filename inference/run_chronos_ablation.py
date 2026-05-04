"""
Chronos-Bolt ablation runner: extends run_chronos_zeroshot.py to capture all 9
quantile levels (rather than just q=0.5) and supports easy sweeps over
context_len and model_card.

Outputs JSON with shape:
    {
        ...standard fields from run_chronos_zeroshot.py...
        "quantile_levels": [0.1, 0.2, ..., 0.9],
        "quantile_preds": [[[ [9 floats], ...]]],   # (n_forecasts, 24, 8, 9)
        "preds_median":   [[[float, ...]]],         # (n_forecasts, 24, 8) — q=0.5
        "preds_mean":     [[[float, ...]]],         # mean over 9 quantiles
    }

The "preds" key (used by foundation_ensemble.py) defaults to preds_median; pass
--aggregation mean to switch.
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
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demand_csv", type=str, required=True)
    p.add_argument("--model_card", type=str, default="amazon/chronos-bolt-base")
    p.add_argument("--context_len", type=int, default=672)
    p.add_argument("--split", type=str, choices=["test", "val"], required=True)
    p.add_argument("--year", type=int, default=2022)
    p.add_argument("--n_test_days", type=int, default=2)
    p.add_argument("--val_start", type=str, default=None)
    p.add_argument("--val_days", type=int, default=14)
    p.add_argument("--aggregation", type=str, default="median",
                   choices=["median", "mean"],
                   help="Default 'preds' field: median (q=0.5) or mean over 9 quantiles")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--torch_dtype", type=str, default="float32",
                   choices=["float32", "bfloat16"])
    return p.parse_args()


def load_demand(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    ts_col = "timestamp_utc" if "timestamp_utc" in df.columns else "timestamp"
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True)
    df = df.rename(columns={ts_col: "timestamp"})
    return df[["timestamp"] + ZONE_COLS]


def get_forecast_starts(df, args):
    if args.split == "test":
        target_dates = pd.date_range(
            end=pd.Timestamp(f"{args.year}-12-31"),
            periods=args.n_test_days, freq="D",
        )
        return [pd.Timestamp(d).normalize() for d in target_dates]
    else:
        start = pd.Timestamp(args.val_start).normalize()
        return [start + pd.Timedelta(days=k) for k in range(args.val_days)]


def slice_context(df, start_ts, ctx_len):
    end_idx = df.index[df["timestamp"] == start_ts]
    if len(end_idx) == 0:
        raise ValueError(f"start_ts {start_ts} not found in demand CSV")
    end = int(end_idx[0])
    if end < ctx_len:
        raise ValueError(f"Need {ctx_len} hours of history before {start_ts}, only {end} available")
    arr = df.loc[end - ctx_len: end - 1, ZONE_COLS].to_numpy(dtype=np.float32)
    if arr.shape != (ctx_len, 8):
        raise ValueError(f"Context shape {arr.shape}, expected ({ctx_len}, 8)")
    return arr


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ablation] demand_csv={args.demand_csv} model={args.model_card} ctx={args.context_len}")
    df = load_demand(Path(args.demand_csv))
    starts = get_forecast_starts(df, args)
    print(f"[ablation] split={args.split} n_forecasts={len(starts)}")

    print(f"[ablation] loading {args.model_card}")
    t0 = time.time()
    from chronos import BaseChronosPipeline
    dtype = torch.float32 if args.torch_dtype == "float32" else torch.bfloat16
    pipeline = BaseChronosPipeline.from_pretrained(
        args.model_card, device_map=args.device, torch_dtype=dtype,
    )
    print(f"[ablation]   loaded in {time.time() - t0:.1f}s")

    n_forecasts = len(starts)
    n_q = len(QUANTILE_LEVELS)
    quantile_preds = np.zeros((n_forecasts, HORIZON, 8, n_q), dtype=np.float32)

    for i, start in enumerate(starts):
        ctx = slice_context(df, start, args.context_len)
        ctx_tensor = torch.from_numpy(ctx.T).to(torch.float32)  # (8, T)

        t1 = time.time()
        quantiles, _mean = pipeline.predict_quantiles(
            context=ctx_tensor,
            prediction_length=HORIZON,
            quantile_levels=QUANTILE_LEVELS,
        )
        # quantiles: (8, 24, num_quantiles) → reshape to (24, 8, num_quantiles)
        q = quantiles.cpu().numpy()  # (8, 24, n_q)
        quantile_preds[i] = q.transpose(1, 0, 2)

        median_mwh = quantile_preds[i, :, :, QUANTILE_LEVELS.index(0.5)].mean()
        print(f"[ablation]   [{i+1}/{n_forecasts}] start={start.strftime('%Y-%m-%d')} "
              f"({time.time() - t1:.1f}s)  median mean={median_mwh:.0f} MWh")

    preds_median = quantile_preds[..., QUANTILE_LEVELS.index(0.5)]   # (n, 24, 8)
    preds_mean = quantile_preds.mean(axis=-1)                         # (n, 24, 8)
    if args.aggregation == "median":
        preds_default = preds_median
    else:
        preds_default = preds_mean

    out = {
        "model": args.model_card.split("/")[-1],
        "model_card": args.model_card,
        "split": args.split,
        "forecast_starts": [s.isoformat() for s in starts],
        "horizon": HORIZON,
        "zones": ZONE_COLS,
        "context_len": args.context_len,
        "aggregation": args.aggregation,
        "quantile_levels": QUANTILE_LEVELS,
        "quantile_preds": quantile_preds.tolist(),
        "preds_median": preds_median.tolist(),
        "preds_mean": preds_mean.tolist(),
        "preds": preds_default.tolist(),
        "demand_csv": str(args.demand_csv),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[ablation] wrote {out_path}  ({os.path.getsize(out_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
