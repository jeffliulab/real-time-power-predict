"""
Quantify the MAPE penalty of substituting HRRR forecasts for HRRR
analyses in the model's future-weather input window (Appendix E of the
workshop paper).

The trained baseline saw HRRR f00 analyses for both history AND future
windows during training (the canonical 'privileged future weather'
training-time setup). At deployment the future is unknown, so we feed
HRRR forecasts (f01..f24) from the most recent long cycle issued before
T. This script measures the size of the resulting input-distribution
shift on the model's output, isolated from the larger drift mechanisms
in the main paper.

Procedure:
  - Pick 5 in-training-window dates that have full HRRR forecast cycle
    coverage in the AWS S3 archive.
  - For each date T, run the baseline twice:
      Setting A (training-distribution): future weather = 24 HRRR f00
        analyses for valid hours [T, T+23] (the "true future weather").
      Setting B (deployable):             future weather = 24 HRRR f01..f24
        from cycle T-1's most recent long-cycle (00/06/12/18 UTC).
  - Compute MAPE_A and MAPE_B against ground-truth demand from the
    training CSV.
  - Report mean (MAPE_B - MAPE_A) with bootstrap CI.

Output: report/arxiv/data/future_weather_shift.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "space"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HERBIE_CONFIG_PATH", "/tmp/herbie_config")

from calendar_features import encode_range                     # noqa: E402
from iso_ne_zonal import ZONE_COLS                              # noqa: E402
from hrrr_fetch import fetch_history, fetch_forecast_for_window  # noqa: E402
from model_utils import (                                        # noqa: E402
    HISTORY_LEN, FUTURE_LEN, load_baseline,
    normalize_demand, normalize_weather, denormalize_demand,
)
from figures.bootstrap_mape import mape_with_ci                  # noqa: E402

DEMAND_CSV = ROOT / "pretrained_models" / "baseline" / "dump" / "demand_2019_2022_hourly.csv"
CHECKPOINT = ROOT / "space" / "checkpoints" / "best.pt"
OUT_PATH = ROOT / "report" / "arxiv" / "data" / "future_weather_shift.json"

# 5 in-training-window dates spread over Dec 2022. Each has full HRRR
# forecast cycle coverage and unambiguous demand truth.
DATES = [
    datetime(2022, 12, 14, 0, 0, 0),
    datetime(2022, 12, 19, 0, 0, 0),
    datetime(2022, 12, 22, 0, 0, 0),
    datetime(2022, 12, 28, 0, 0, 0),
    datetime(2022, 12, 30, 0, 0, 0),
]


def per_zone_mape(pred: np.ndarray, truth: np.ndarray) -> dict:
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
    return float(np.mean(vals))


def run_one(start_ts: datetime, model, norm_stats, df: pd.DataFrame,
             parallel: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Run baseline twice for the same date: future=analyses vs future=forecast.

    Returns (truth, pred_analyses, pred_forecast, hrrr_meta).
    """
    print(f"  Date {start_ts.date()}", flush=True)
    hist_demand = df.loc[
        start_ts - timedelta(hours=HISTORY_LEN):
        start_ts - timedelta(hours=1), ZONE_COLS].to_numpy(np.float32)
    truth = df.loc[
        start_ts:start_ts + timedelta(hours=FUTURE_LEN - 1),
        ZONE_COLS].to_numpy(np.float32)

    # HRRR history (shared by both settings)
    t0 = time.time()
    hist_w = fetch_history(start_ts, hours=HISTORY_LEN, parallel=parallel)
    print(f"    HRRR hist  in {time.time()-t0:.1f}s")

    # Setting A: future = analyses (24 f00 analyses, valid [T, T+23])
    t0 = time.time()
    fut_w_analyses = fetch_history(
        start_ts + timedelta(hours=FUTURE_LEN), hours=FUTURE_LEN, parallel=parallel)
    print(f"    HRRR analyses (future) in {time.time()-t0:.1f}s")

    # Setting B: future = forecast from latest long cycle <= T-2h
    t0 = time.time()
    fut_w_forecast, cycle_used, fxx_start = fetch_forecast_for_window(
        start_ts, hours=FUTURE_LEN, parallel=parallel)
    print(f"    HRRR forecast in {time.time()-t0:.1f}s "
           f"(cycle={cycle_used} f{fxx_start:02d}+)")

    hist_cal = encode_range(start_ts - timedelta(hours=HISTORY_LEN), HISTORY_LEN)
    fut_cal = encode_range(start_ts, FUTURE_LEN)

    def _forward(fut_w):
        with torch.no_grad():
            pred_z = model(
                torch.from_numpy(normalize_weather(hist_w, norm_stats)).unsqueeze(0),
                torch.from_numpy(normalize_demand(hist_demand, norm_stats)).unsqueeze(0),
                torch.from_numpy(hist_cal.astype(np.float32)).unsqueeze(0),
                torch.from_numpy(normalize_weather(fut_w, norm_stats)).unsqueeze(0),
                torch.from_numpy(fut_cal.astype(np.float32)).unsqueeze(0),
            )
        return denormalize_demand(pred_z.squeeze(0).cpu().numpy(), norm_stats)

    pred_a = _forward(fut_w_analyses)
    pred_b = _forward(fut_w_forecast)
    return truth, pred_a, pred_b, {
        "future_forecast_cycle": cycle_used.isoformat(),
        "future_forecast_fxx_start": fxx_start,
    }


def main():
    print("Loading baseline checkpoint...")
    model, ns = load_baseline(CHECKPOINT, device="cpu")
    print(f"  ({sum(p.numel() for p in model.parameters()):,} params)")

    print("Loading training-CSV demand...")
    df = pd.read_csv(DEMAND_CSV, parse_dates=["timestamp_utc"]).set_index("timestamp_utc")

    truths, preds_a, preds_b, per_date = [], [], [], []
    for d in DATES:
        truth, pa, pb, meta = run_one(d, model, ns, df)
        truths.append(truth)
        preds_a.append(pa)
        preds_b.append(pb)
        per_date.append({
            "date": d.isoformat(),
            "mape_analyses": overall_mape(pa, truth),
            "mape_forecast": overall_mape(pb, truth),
            "diff_pp":       overall_mape(pb, truth) - overall_mape(pa, truth),
            "per_zone_analyses": per_zone_mape(pa, truth),
            "per_zone_forecast": per_zone_mape(pb, truth),
            "future_forecast_cycle": meta["future_forecast_cycle"],
            "future_forecast_fxx_start": meta["future_forecast_fxx_start"],
        })
        print(f"    -> analyses {per_date[-1]['mape_analyses']:.2f}% / "
              f"forecast {per_date[-1]['mape_forecast']:.2f}% / "
              f"diff {per_date[-1]['diff_pp']:+.2f} pp")

    # Aggregate with bootstrap CIs over (date, zone) pairs
    truths_arr = np.stack(truths, axis=0)        # (n_dates, 24, 8)
    preds_a_arr = np.stack(preds_a, axis=0)
    preds_b_arr = np.stack(preds_b, axis=0)

    overall_a, lo_a, hi_a, _ = mape_with_ci(preds_a_arr, truths_arr)
    overall_b, lo_b, hi_b, _ = mape_with_ci(preds_b_arr, truths_arr)
    diff_per_pair = (
        np.abs(preds_b_arr - truths_arr) / np.abs(truths_arr)
        - np.abs(preds_a_arr - truths_arr) / np.abs(truths_arr)
    ).mean(axis=1) * 100   # (n_dates, 8) — per (date, zone) MAPE difference
    rng = np.random.default_rng(42)
    flat = diff_per_pair.reshape(-1)
    n_resamples = 1000
    boot = rng.choice(flat, size=(n_resamples, flat.size), replace=True).mean(axis=1)
    diff_point = float(flat.mean())
    diff_lo = float(np.quantile(boot, 0.025))
    diff_hi = float(np.quantile(boot, 0.975))

    out = {
        "schema_version": "workshop-1.0",
        "n_dates": len(DATES),
        "dates": [d.isoformat() for d in DATES],
        "per_date": per_date,
        "aggregate": {
            "analyses_mape":  {"point": overall_a, "ci_low": lo_a, "ci_high": hi_a},
            "forecast_mape":  {"point": overall_b, "ci_low": lo_b, "ci_high": hi_b},
            "diff_forecast_minus_analyses_pp": {
                "point": diff_point, "ci_low": diff_lo, "ci_high": diff_hi},
        },
        "interpretation": (
            f"Substituting HRRR forecasts (f01..f24) for HRRR analyses in the "
            f"model's future-weather input window adds an MAPE penalty of "
            f"{diff_point:+.2f} pp [95 % CI: {diff_lo:+.2f} to {diff_hi:+.2f}] "
            f"averaged over {len(DATES)} in-training-window dates. The CI's "
            f"position relative to zero indicates whether the shift is "
            f"meaningfully non-zero."
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_PATH}")
    print(f"Aggregate: analyses {overall_a:.2f}% [{lo_a:.2f}-{hi_a:.2f}], "
           f"forecast {overall_b:.2f}% [{lo_b:.2f}-{hi_b:.2f}], "
           f"diff {diff_point:+.2f} pp [{diff_lo:+.2f}, {diff_hi:+.2f}]")


if __name__ == "__main__":
    main()
