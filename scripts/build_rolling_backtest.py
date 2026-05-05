"""
Build a rolling 7-day backtest of the predict-power model on the most
recent fully-published days of ISO-NE per-zone demand.

This script is intended to run in a cloud-side cron (GitHub Actions in
the data repo `new-england-real-time-power-predict-data`). It does NOT
require Tufts HPC access — all inputs come from public sources:

  - HRRR weather: AWS S3 `noaa-hrrr-bdp-pds` via the Herbie library
  - ISO-NE per-zone demand: the public 5-min `fiveminuteestimatedzonalload`
    CSV endpoint (cookie-prime trick handled in `space/iso_ne_zonal.py`)

Strict backtest semantics: at each forecast time T, the model receives
ONLY information that would have been available at T:

  - hist_weather: 24 HRRR f00 analyses for valid hours [T-24, T-1]
  - fut_weather:  HRRR forecast cycle T-1's f01..f24 (i.e., the most
                  recent forecast issued *before* T, with valid hours
                  [T, T+23])
  - hist_demand:  ISO-NE per-zone for hours [T-24, T-1]
  - chronos_ctx:  720h ISO-NE per-zone history ending at T-1
  - calendar:     deterministic from timestamps

Truth for MAPE comparison: ISO-NE per-zone for [T, T+23].

Outputs (to ``--output-dir``, default ``space/assets/``):

  - ``backtest_rolling_7d.json``  — full per-zone forecasts, MAPE, summary
  - ``iso_ne_30d.csv``            — 30-day per-zone history shipped to the
                                    Space for Live tab Chronos context
  - ``last_built.json``           — metadata: built_at, code_sha, period,
                                    summary MAPE
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

# Allow running as `python scripts/build_rolling_backtest.py` from repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "space"))

from calendar_features import encode_range          # noqa: E402
from iso_ne_zonal import ZONE_COLS, fetch_range, fetch_one_day  # noqa: E402
from hrrr_fetch import fetch_history as hrrr_fetch_history       # noqa: E402
from hrrr_fetch import fetch_forecast_for_window                  # noqa: E402
from model_utils import (                                         # noqa: E402
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


def _git_sha(short: bool = True) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse",
             "--short" if short else "HEAD", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
        return out
    except Exception:  # noqa: BLE001
        return "unknown"


def per_zone_mape(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    out = {}
    for j, z in enumerate(ZONE_COLS):
        mask = truth[:, j] != 0
        if mask.sum() == 0:
            out[z] = float("nan")
            continue
        ape = np.abs(pred[mask, j] - truth[mask, j]) / np.abs(truth[mask, j])
        out[z] = float(ape.mean() * 100)
    return out


def overall_mape(pred: np.ndarray, truth: np.ndarray) -> float:
    pz = per_zone_mape(pred, truth)
    vals = [v for v in pz.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def determine_window(days: int = 7,
                       latest_anchor: Optional[datetime] = None,
                       ) -> list[datetime]:
    """Return the list of `days` forecast start times (00:00 UTC each day),
    ending at the most recent date with full ISO-NE per-zone publication.

    The latest forecast issues at T_latest 00:00 UTC. Truth for that
    forecast covers [T_latest, T_latest + 23]. We must therefore confirm
    ISO-NE has data through `T_latest + 23 = T_latest + 23h`.

    `latest_anchor` overrides the default ("yesterday 00:00 UTC"); useful
    for testing.
    """
    if latest_anchor is None:
        # Default: T_latest = yesterday 00:00 UTC
        # (covers truth window [yesterday 00:00, yesterday 23:00])
        latest_anchor = (datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
            - timedelta(days=1))
    # Forecast starts: T_latest, T_latest - 1d, ..., T_latest - (days-1)d
    starts = [latest_anchor - timedelta(days=i) for i in range(days)]
    starts.reverse()    # chronological order
    return starts


def fetch_demand_window(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Fetch hourly per-zone demand for [start_dt, end_dt] inclusive of
    both endpoints (hour-aligned). Returns DataFrame indexed by UTC
    timestamps with 8 zone columns."""
    df = fetch_range(start_dt - timedelta(hours=1),
                       end_dt + timedelta(hours=1),
                       hourly=True)
    idx = pd.date_range(start=start_dt, end=end_dt, freq="1h")
    df = df.reindex(idx)
    return df[ZONE_COLS]


def build_one_forecast(
    start_ts: datetime,
    model: torch.nn.Module,
    norm_stats: dict,
    chronos_pipeline,
    demand_30d: pd.DataFrame,
    truth_window: pd.DataFrame,
    hist_window: pd.DataFrame,
    parallel: int = 8,
    device: str = "cpu",
) -> dict:
    """Run baseline + Chronos + ensemble for one forecast at start_ts.

    All ISO-NE data is provided pre-fetched (so we don't re-fetch CSVs in
    this hot path). HRRR fetches happen here (parallelized).
    """
    print(f"\n=== Forecast at {start_ts.isoformat()} UTC ===", flush=True)

    # --- HRRR history: 24 f00 analyses for [start_ts - 24h, start_ts - 1h]
    print(f"  HRRR history fetch ({parallel}-way parallel)...", flush=True)
    t0 = time.time()
    hist_w_raw = hrrr_fetch_history(start_ts, hours=HISTORY_LEN,
                                     parallel=parallel)
    print(f"    -> {hist_w_raw.shape} in {time.time()-t0:.1f}s")

    # --- HRRR future: latest long cycle <= start_ts - publication_lag,
    # which guarantees enough forecast hours (long cycles go to f48,
    # short cycles only to f18).
    print(f"  HRRR future fetch (latest long cycle, {parallel}-way)...",
          flush=True)
    t0 = time.time()
    fut_w_raw, cycle_used, fxx_start = fetch_forecast_for_window(
        start_ts, hours=FUTURE_LEN, parallel=parallel)
    print(f"    -> {fut_w_raw.shape} in {time.time()-t0:.1f}s "
          f"(cycle={cycle_used} f{fxx_start:02d}..f{fxx_start + FUTURE_LEN - 1:02d})")

    # --- Demand history (24h before start_ts)
    hist_demand = hist_window.to_numpy(dtype=np.float32)
    if hist_demand.shape != (HISTORY_LEN, 8):
        raise RuntimeError(
            f"hist_window shape {hist_demand.shape} != ({HISTORY_LEN}, 8)")

    # --- Calendar
    hist_cal = encode_range(start_ts - timedelta(hours=HISTORY_LEN),
                              HISTORY_LEN)
    fut_cal = encode_range(start_ts, FUTURE_LEN)

    # --- Run baseline (z-score weather + demand internally via run_forecast)
    print("  Running baseline forward pass...", flush=True)
    t0 = time.time()
    hist_w_z = normalize_weather(hist_w_raw, norm_stats)
    fut_w_z = normalize_weather(fut_w_raw, norm_stats)
    hist_y_z = normalize_demand(hist_demand, norm_stats)

    with torch.no_grad():
        hw = torch.from_numpy(hist_w_z).unsqueeze(0).to(device)
        fw = torch.from_numpy(fut_w_z).unsqueeze(0).to(device)
        hy = torch.from_numpy(hist_y_z).unsqueeze(0).to(device)
        hc = torch.from_numpy(hist_cal.astype(np.float32)).unsqueeze(0).to(device)
        fc = torch.from_numpy(fut_cal.astype(np.float32)).unsqueeze(0).to(device)
        pred_z = model(hw, hy, hc, fw, fc)
    baseline_pred = denormalize_demand(pred_z.squeeze(0).cpu().numpy(), norm_stats)
    print(f"    -> baseline {baseline_pred.shape} in {time.time()-t0:.1f}s")

    # --- Run Chronos zero-shot (720h context ending at start_ts - 1h)
    chronos_ctx_end = start_ts - timedelta(hours=1)
    chronos_ctx_start = chronos_ctx_end - timedelta(hours=CHRONOS_CONTEXT - 1)
    chronos_ctx = demand_30d.loc[chronos_ctx_start:chronos_ctx_end].to_numpy(
        dtype=np.float32)
    if chronos_ctx.shape[0] < CHRONOS_CONTEXT:
        # Pad by repeating earliest sample
        pad = np.repeat(chronos_ctx[:1], CHRONOS_CONTEXT - chronos_ctx.shape[0],
                          axis=0)
        chronos_ctx = np.concatenate([pad, chronos_ctx], axis=0)
        print(f"    [WARN] Chronos context padded by "
              f"{CHRONOS_CONTEXT - chronos_ctx.shape[0]} hours")
    elif chronos_ctx.shape[0] > CHRONOS_CONTEXT:
        chronos_ctx = chronos_ctx[-CHRONOS_CONTEXT:]
    print(f"  Running Chronos zero-shot (context {chronos_ctx.shape})...",
          flush=True)
    t0 = time.time()
    chronos_pred = run_chronos_zeroshot(chronos_pipeline, chronos_ctx)
    print(f"    -> chronos {chronos_pred.shape} in {time.time()-t0:.1f}s")

    # --- Ensemble
    ens_pred = per_zone_ensemble(baseline_pred, chronos_pred, ALPHA_PER_ZONE_MINI)

    # --- Truth + MAPE
    truth = truth_window.to_numpy(dtype=np.float32)
    if truth.shape != (FUTURE_LEN, 8):
        raise RuntimeError(
            f"truth_window shape {truth.shape} != ({FUTURE_LEN}, 8)")

    out = {
        "start": start_ts.isoformat(),
        "history_24h": hist_demand.tolist(),
        "truth_24h":   truth.tolist(),
        "baseline":    baseline_pred.tolist(),
        "chronos":     chronos_pred.tolist(),
        "ensemble":    ens_pred.tolist(),
        "mape": {
            "baseline": {
                "per_zone": per_zone_mape(baseline_pred, truth),
                "overall":  overall_mape(baseline_pred, truth),
            },
            "chronos": {
                "per_zone": per_zone_mape(chronos_pred, truth),
                "overall":  overall_mape(chronos_pred, truth),
            },
            "ensemble": {
                "per_zone": per_zone_mape(ens_pred, truth),
                "overall":  overall_mape(ens_pred, truth),
            },
        },
    }
    print(f"  MAPE: baseline {out['mape']['baseline']['overall']:.2f}% | "
          f"chronos {out['mape']['chronos']['overall']:.2f}% | "
          f"ensemble {out['mape']['ensemble']['overall']:.2f}%")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "space" / "assets",
                        help="where to write the 3 output files")
    parser.add_argument("--days", type=int, default=7,
                        help="number of daily forecasts (default 7)")
    parser.add_argument("--parallel", type=int, default=8,
                        help="parallel HRRR fetch workers (default 8)")
    parser.add_argument("--anchor", type=str, default=None,
                        help="override latest forecast date "
                             "(YYYY-MM-DD UTC); default = yesterday")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "space" / "checkpoints" / "best.pt")
    parser.add_argument("--skip-30d-csv", action="store_true",
                        help="don't write iso_ne_30d.csv (faster smoke test)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # ---- Determine forecast windows + total demand range needed
    if args.anchor:
        latest_anchor = datetime.fromisoformat(args.anchor).replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    else:
        latest_anchor = None
    starts = determine_window(args.days, latest_anchor=latest_anchor)
    print(f"Forecast starts ({len(starts)}): "
          f"{starts[0].isoformat()} -> {starts[-1].isoformat()}")

    earliest_chronos_start = starts[0] - timedelta(hours=CHRONOS_CONTEXT)
    latest_truth_end = starts[-1] + timedelta(hours=FUTURE_LEN - 1)
    print(f"Demand range needed: {earliest_chronos_start.isoformat()} -> "
          f"{latest_truth_end.isoformat()} "
          f"({int((latest_truth_end - earliest_chronos_start).total_seconds() / 3600) + 1} hours)")

    # ---- Pre-fetch all ISO-NE per-zone demand in one batch
    print("\nFetching ISO-NE per-zone demand (this may take 30-60s)...",
          flush=True)
    t0 = time.time()
    demand_full = fetch_range(earliest_chronos_start - timedelta(hours=1),
                                latest_truth_end + timedelta(hours=1),
                                hourly=True)
    print(f"  -> {demand_full.shape} in {time.time()-t0:.1f}s")
    # The demand_full df may have NaN rows for unpublished hours; fill with
    # forward+back fill only inside our needed range, and fail if any zone
    # is fully missing.
    demand_full = demand_full[ZONE_COLS]

    # Quick gap report
    n_missing = demand_full.isna().any(axis=1).sum()
    if n_missing > 0:
        print(f"  [INFO] {n_missing} hours have at least one missing zone; "
              f"will be linear-interpolated for Chronos context only")
        demand_full = demand_full.interpolate(method="time", limit=6)
        # Drop any leading NaNs that interpolation can't fill
        demand_full = demand_full.dropna(how="any")

    # ---- Load model + Chronos
    print("\nLoading baseline checkpoint...", flush=True)
    model, norm_stats = load_baseline(args.checkpoint, device=args.device)
    print(f"  ({sum(p.numel() for p in model.parameters()):,} params)")

    print("Loading Chronos-Bolt-mini...", flush=True)
    chronos_pipeline = load_chronos(device=args.device)

    # ---- Run forecasts
    forecasts = []
    for start_ts in starts:
        # Slice demand for this forecast
        hist_window = demand_full.loc[
            start_ts - timedelta(hours=HISTORY_LEN):
            start_ts - timedelta(hours=1)]
        truth_window = demand_full.loc[
            start_ts:
            start_ts + timedelta(hours=FUTURE_LEN - 1)]
        if len(hist_window) != HISTORY_LEN:
            print(f"  [SKIP] {start_ts}: history window has "
                  f"{len(hist_window)} of {HISTORY_LEN} hours")
            continue
        if len(truth_window) != FUTURE_LEN:
            print(f"  [SKIP] {start_ts}: truth window has "
                  f"{len(truth_window)} of {FUTURE_LEN} hours")
            continue

        f = build_one_forecast(
            start_ts=start_ts,
            model=model, norm_stats=norm_stats,
            chronos_pipeline=chronos_pipeline,
            demand_30d=demand_full,
            truth_window=truth_window,
            hist_window=hist_window,
            parallel=args.parallel,
            device=args.device,
        )
        forecasts.append(f)

    if not forecasts:
        raise RuntimeError("No forecasts succeeded; check logs above")

    # ---- Aggregate
    def agg(model_key: str) -> dict:
        per_zone = {z: [] for z in ZONE_COLS}
        overalls = []
        for f in forecasts:
            for z in ZONE_COLS:
                v = f["mape"][model_key]["per_zone"][z]
                if not np.isnan(v):
                    per_zone[z].append(v)
            overalls.append(f["mape"][model_key]["overall"])
        return {
            "per_zone": {z: float(np.mean(vs)) if vs else float("nan")
                         for z, vs in per_zone.items()},
            "overall":  float(np.mean(overalls)),
        }
    summary = {
        "baseline": agg("baseline"),
        "chronos":  agg("chronos"),
        "ensemble": agg("ensemble"),
    }
    print()
    print(f"=== {args.days}-day average ===")
    for k in ("baseline", "chronos", "ensemble"):
        print(f"  {k:<10} overall MAPE: {summary[k]['overall']:.2f}%")

    # ---- Write outputs
    built_at = datetime.now(timezone.utc).isoformat()
    code_sha = _git_sha(short=False)

    backtest_obj = {
        "schema_version": "2.0",
        "type": "rolling_7d",
        "built_at": built_at,
        "code_sha": code_sha,
        "data_period": {
            "first_forecast_start": starts[0].isoformat(),
            "last_forecast_start":  starts[-1].isoformat(),
            "last_truth_hour":      latest_truth_end.isoformat(),
        },
        "zones": ZONE_COLS,
        "horizon_hours": FUTURE_LEN,
        "n_forecasts": len(forecasts),
        "alpha_per_zone": ALPHA_PER_ZONE_MINI,
        "forecasts": forecasts,
        "summary": summary,
        "note": (
            "Strict backtest: at each forecast time T, history weather is "
            f"{HISTORY_LEN} HRRR f00 analyses for cycles [T-24h, T-1h], "
            "future weather is the f01..f24 forecast hours from cycle T-1 "
            "(i.e., the most recent cycle issued before T). No future "
            "analyses are used. Per-zone demand is the public 5-min "
            "fiveminuteestimatedzonalload feed, rolled up to hourly mean."
        ),
    }
    out_json = args.output_dir / "backtest_rolling_7d.json"
    out_json.write_text(json.dumps(backtest_obj, indent=2))
    size_kb = out_json.stat().st_size / 1024
    print(f"\nWrote {out_json} ({size_kb:.1f} KB)")

    # 30-day CSV for Live tab Chronos context
    if not args.skip_30d_csv:
        csv_end = starts[-1] + timedelta(hours=FUTURE_LEN - 1)
        csv_start = csv_end - timedelta(hours=720 - 1)
        csv_df = demand_full.loc[csv_start:csv_end].copy()
        csv_df.index.name = "timestamp_utc"
        out_csv = args.output_dir / "iso_ne_30d.csv"
        csv_df.to_csv(out_csv)
        print(f"Wrote {out_csv} ({out_csv.stat().st_size / 1024:.1f} KB, "
              f"{len(csv_df)} rows)")

    out_meta = {
        "built_at": built_at,
        "code_sha": code_sha,
        "data_period": backtest_obj["data_period"],
        "n_forecasts": len(forecasts),
        "summary_mape_pct": {
            "baseline": summary["baseline"]["overall"],
            "chronos":  summary["chronos"]["overall"],
            "ensemble": summary["ensemble"]["overall"],
        },
    }
    out_meta_path = args.output_dir / "last_built.json"
    out_meta_path.write_text(json.dumps(out_meta, indent=2))
    print(f"Wrote {out_meta_path}")


if __name__ == "__main__":
    main()
