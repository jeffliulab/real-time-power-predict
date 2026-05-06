"""
Multi-window deployment-drift sweep for the workshop paper §4.

Runs the deployed pipeline across a configurable set of historical
7-day windows. For each forecast issuance time T in each window:

  - Fetches HRRR f00 analyses for [T-24h, T-1h]                     (history weather)
  - Fetches HRRR f01..f24 from the latest long cycle <= T-2h        (future weather)
  - Fetches ISO-NE per-zone hourly demand for [T-30d, T+23h]        (history + Chronos ctx + truth)
  - Runs:
      * baseline forward (CNN-Transformer + real HRRR + per-zone demand)
      * Chronos-Bolt-mini zero-shot on 672h per-zone history
      * per-zone weighted ensemble
      * 3 naive baselines (persistence-1d, persistence-7d, climatological)

Output: report/arxiv/data/multi_window_results.json with shape:

  {
    "schema_version": "workshop-1.0",
    "windows": [
      {
        "label": "W1",
        "start_date": "2025-05-01",
        "end_date":   "2025-05-07",
        "n_forecasts": 7,
        "forecasts": [
          { "start": "2025-05-01T00:00:00",
            "history_24h":   [[...], ...],   # (24, 8) MWh
            "truth_24h":     [[...], ...],
            "predictions": {
              "baseline":         [[...]],
              "chronos":          [[...]],
              "ensemble":         [[...]],
              "persistence_1d":   [[...]],
              "persistence_7d":   [[...]],
              "climatological":   [[...]],
            },
          },
          ...
        ],
        "summary": {
          "baseline":       { "overall": {point, ci_low, ci_high}, "per_zone": {...} },
          "chronos":        { ... },
          "ensemble":       { ... },
          "persistence_1d": { ... },
          "persistence_7d": { ... },
          "climatological": { ... },
        }
      },
      { ... W2 ... }
    ]
  }

Reused (unchanged) modules:
  space/hrrr_fetch.py:fetch_history, fetch_forecast_for_window
  space/iso_ne_zonal.py:fetch_range
  space/model_utils.py:load_baseline, load_chronos, run_chronos_zeroshot,
                        normalize_demand/weather, denormalize_demand,
                        per_zone_ensemble, ALPHA_PER_ZONE_MINI,
                        HISTORY_LEN (24), FUTURE_LEN (24), CHRONOS_CONTEXT (672)
  space/calendar_features.py:encode_range
  scripts/baselines/naive.py:persistence_1d, persistence_7d, climatological
  scripts/figures/bootstrap_mape.py:mape_with_ci

Estimate: ~75 minutes per window on a 4-core CPU with empty HRRR cache,
mostly dominated by HRRR S3 byte-range fetches (8-way parallel). Two
windows ≈ 2.5 h wall clock. Cache hits make subsequent runs much faster.

Usage:
  python scripts/experiments/historical_drift_sweep.py
  # Or override the window definitions:
  python scripts/experiments/historical_drift_sweep.py \
      --windows W1=2025-05-01,2025-05-07 W2=2026-04-28,2026-05-04
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "space"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HERBIE_CONFIG_PATH", "/tmp/herbie_config")

from calendar_features import encode_range                                # noqa: E402
from iso_ne_zonal import ZONE_COLS, fetch_range                            # noqa: E402
from hrrr_fetch import fetch_history as hrrr_fetch_history                  # noqa: E402
from hrrr_fetch import fetch_forecast_for_window                            # noqa: E402
from model_utils import (                                                   # noqa: E402
    ALPHA_PER_ZONE_MINI,
    CHRONOS_CONTEXT,
    FUTURE_LEN,
    HISTORY_LEN,
    load_baseline,
    load_chronos,
    normalize_demand,
    normalize_weather,
    denormalize_demand,
    per_zone_ensemble,
    run_chronos_zeroshot,
)
from baselines.naive import persistence_1d, persistence_7d, climatological  # noqa: E402
from figures.bootstrap_mape import mape_with_ci                             # noqa: E402


DEFAULT_WINDOWS = [
    ("W1", datetime(2025, 5, 1), datetime(2025, 5, 7)),
    ("W2", datetime(2026, 4, 28), datetime(2026, 5, 4)),
]
DAILY_FORECAST_HOUR = 0     # forecasts issued at 00:00 UTC each day
OUT_PATH = ROOT / "report" / "arxiv" / "data" / "multi_window_results.json"

logger = logging.getLogger(__name__)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _parse_windows(spec: list[str]) -> list[tuple[str, datetime, datetime]]:
    out = []
    for s in spec:
        label, dates = s.split("=")
        start_str, end_str = dates.split(",")
        out.append((label,
                     datetime.fromisoformat(start_str),
                     datetime.fromisoformat(end_str)))
    return out


def run_one_forecast(
    start_ts: datetime,
    *,
    model: torch.nn.Module,
    norm_stats: dict,
    chronos_pipeline,
    demand_full: pd.DataFrame,
    parallel: int = 8,
    device: str = "cpu",
) -> dict:
    """Run all 6 models on a single forecast issuance time."""
    print(f"\n  Forecast at {start_ts.isoformat()} UTC", flush=True)

    # Demand windows
    hist_demand = demand_full.loc[
        start_ts - timedelta(hours=HISTORY_LEN):
        start_ts - timedelta(hours=1), ZONE_COLS
    ].to_numpy(np.float32)
    truth = demand_full.loc[
        start_ts:start_ts + timedelta(hours=FUTURE_LEN - 1), ZONE_COLS
    ].to_numpy(np.float32)
    if hist_demand.shape != (HISTORY_LEN, 8):
        raise RuntimeError(f"hist_demand shape {hist_demand.shape} (need (24,8))")
    if truth.shape != (FUTURE_LEN, 8):
        raise RuntimeError(f"truth shape {truth.shape} (need (24,8))")

    # 30-day demand history for naive climatological + Chronos context
    long_history = demand_full.loc[
        start_ts - timedelta(hours=CHRONOS_CONTEXT):
        start_ts - timedelta(hours=1), ZONE_COLS
    ].to_numpy(np.float32)
    long_history_start = start_ts - timedelta(hours=CHRONOS_CONTEXT)
    if long_history.shape[0] < CHRONOS_CONTEXT:
        # pad by repeating earliest sample
        pad_n = CHRONOS_CONTEXT - long_history.shape[0]
        pad = np.repeat(long_history[:1], pad_n, axis=0)
        long_history = np.concatenate([pad, long_history], axis=0)

    # HRRR history (24 f00 analyses)
    t0 = time.time()
    hist_w = hrrr_fetch_history(start_ts, hours=HISTORY_LEN, parallel=parallel)
    print(f"    HRRR hist  {hist_w.shape} in {time.time()-t0:.1f}s")

    # HRRR future (latest long cycle <= T - 2h)
    t0 = time.time()
    fut_w, cycle_used, fxx_start = fetch_forecast_for_window(
        start_ts, hours=FUTURE_LEN, parallel=parallel)
    print(f"    HRRR fut   {fut_w.shape} in {time.time()-t0:.1f}s "
           f"(cycle={cycle_used} f{fxx_start:02d}+)")

    # Calendar
    hist_cal = encode_range(start_ts - timedelta(hours=HISTORY_LEN), HISTORY_LEN)
    fut_cal = encode_range(start_ts, FUTURE_LEN)

    # Baseline forward
    t0 = time.time()
    with torch.no_grad():
        pred_z = model(
            torch.from_numpy(normalize_weather(hist_w, norm_stats)).unsqueeze(0),
            torch.from_numpy(normalize_demand(hist_demand, norm_stats)).unsqueeze(0),
            torch.from_numpy(hist_cal.astype(np.float32)).unsqueeze(0),
            torch.from_numpy(normalize_weather(fut_w, norm_stats)).unsqueeze(0),
            torch.from_numpy(fut_cal.astype(np.float32)).unsqueeze(0),
        )
    baseline_pred = denormalize_demand(pred_z.squeeze(0).cpu().numpy(), norm_stats)
    print(f"    baseline   {baseline_pred.shape} in {time.time()-t0:.1f}s")

    # Chronos zero-shot
    t0 = time.time()
    chronos_pred = run_chronos_zeroshot(chronos_pipeline, long_history)
    print(f"    chronos    {chronos_pred.shape} in {time.time()-t0:.1f}s")

    # Ensemble
    ens_pred = per_zone_ensemble(baseline_pred, chronos_pred, ALPHA_PER_ZONE_MINI)

    # 3 naive baselines
    p1d = persistence_1d(long_history)
    p7d = persistence_7d(long_history)
    clim = climatological(long_history, start_ts,
                           history_start_ts=long_history_start,
                           weeks_back=4)

    return {
        "start": start_ts.isoformat(),
        "history_24h": hist_demand.tolist(),
        "truth_24h":   truth.tolist(),
        "predictions": {
            "baseline":       baseline_pred.tolist(),
            "chronos":        chronos_pred.tolist(),
            "ensemble":       ens_pred.tolist(),
            "persistence_1d": p1d.tolist(),
            "persistence_7d": p7d.tolist(),
            "climatological": clim.tolist(),
        },
    }


def summarise_window(forecasts: list[dict]) -> dict:
    """Compute per-model overall + per-zone MAPE with bootstrap CIs."""
    n_forecasts = len(forecasts)
    truth = np.asarray(
        [f["truth_24h"] for f in forecasts], dtype=np.float32)   # (N, 24, 8)

    out = {}
    model_keys = ["baseline", "chronos", "ensemble",
                  "persistence_1d", "persistence_7d", "climatological"]
    for k in model_keys:
        preds = np.asarray(
            [f["predictions"][k] for f in forecasts], dtype=np.float32)
        overall_pt, lo, hi, pz = mape_with_ci(preds, truth, n_resamples=1000, seed=42)
        out[k] = {
            "overall": {"point": overall_pt, "ci_low": lo, "ci_high": hi},
            "per_zone": {z: {"point": p, "ci_low": l, "ci_high": h}
                         for z, (p, l, h) in pz.items()},
        }
    return out


def fetch_demand_for_window(start_dt: datetime, end_dt: datetime
                              ) -> pd.DataFrame:
    """Fetch ISO-NE per-zone hourly demand covering chronos context +
    forecast horizon for the given window."""
    earliest = start_dt - timedelta(hours=CHRONOS_CONTEXT + 24)
    latest = end_dt + timedelta(hours=FUTURE_LEN + 1)
    print(f"  Fetching ISO-NE per-zone demand "
           f"{earliest.date()} -> {latest.date()} (~{(latest-earliest).days+1} days)...",
           flush=True)
    t0 = time.time()
    df = fetch_range(earliest, latest, hourly=True)
    df = df[ZONE_COLS]
    if df.isna().any().any():
        n_missing = int(df.isna().any(axis=1).sum())
        if n_missing > 0:
            print(f"    [INFO] interpolating {n_missing} hours with missing zones")
            df = df.interpolate(method="time", limit=6).ffill().bfill()
    print(f"  Demand: {df.shape} in {time.time()-t0:.1f}s")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", nargs="+", default=None,
                        help="Override windows. Format: LABEL=YYYY-MM-DD,YYYY-MM-DD")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "space" / "checkpoints" / "best.pt")
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    if args.windows:
        windows = _parse_windows(args.windows)
    else:
        windows = DEFAULT_WINDOWS
    print(f"Windows: {[(l, s.date(), e.date()) for l,s,e in windows]}")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load model + Chronos once
    print("\nLoading baseline checkpoint...")
    model, ns = load_baseline(args.checkpoint, device=args.device)
    print(f"  ({sum(p.numel() for p in model.parameters()):,} params)")
    print("Loading Chronos-Bolt-mini...")
    chronos_pipeline = load_chronos(device=args.device)

    out_doc = {
        "schema_version": "workshop-1.0",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "code_sha": _git_sha(),
        "alpha_per_zone": ALPHA_PER_ZONE_MINI,
        "horizon_hours": FUTURE_LEN,
        "history_hours": HISTORY_LEN,
        "chronos_context_hours": CHRONOS_CONTEXT,
        "windows": [],
    }

    for label, w_start, w_end in windows:
        print(f"\n=== Window {label}: {w_start.date()} -> {w_end.date()} ===")
        demand_full = fetch_demand_for_window(w_start, w_end)
        forecasts = []
        # one forecast at 00:00 UTC each day
        cur = w_start.replace(hour=DAILY_FORECAST_HOUR, minute=0, second=0,
                                microsecond=0)
        end = w_end.replace(hour=DAILY_FORECAST_HOUR)
        while cur <= end:
            try:
                f = run_one_forecast(
                    cur, model=model, norm_stats=ns,
                    chronos_pipeline=chronos_pipeline,
                    demand_full=demand_full,
                    parallel=args.parallel, device=args.device)
                forecasts.append(f)
            except Exception as e:  # noqa: BLE001
                print(f"  [ERR] forecast at {cur} failed: {e}")
            cur += timedelta(days=1)

        if not forecasts:
            print(f"  [WARN] no forecasts succeeded for {label}; skipping")
            continue

        summary = summarise_window(forecasts)
        out_doc["windows"].append({
            "label": label,
            "start_date": w_start.date().isoformat(),
            "end_date": w_end.date().isoformat(),
            "n_forecasts": len(forecasts),
            "forecasts": forecasts,
            "summary": summary,
        })

        # Compact terminal summary
        print(f"\n  Window {label} summary (n={len(forecasts)}):")
        for k, v in summary.items():
            o = v["overall"]
            print(f"    {k:18s}: {o['point']:5.2f}% [{o['ci_low']:.2f}-{o['ci_high']:.2f}]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_doc, indent=2))
    print(f"\nWrote {args.output} ({args.output.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
