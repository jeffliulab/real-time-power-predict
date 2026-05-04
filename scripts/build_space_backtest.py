"""
Build the past-week backtest cache for the HF Space demo.

Outputs ``space/assets/backtest_2022_last7d.json``, which holds 7 daily
forecasts (Dec 25-31, 2022 at 00:00 UTC), each containing:
  - 24-hour history before the forecast start (per zone, MWh)
  - 24-hour ground truth after the start (per zone, MWh)
  - Baseline forecast (from cluster runs with REAL HRRR weather; loaded
    from pretrained_models/baseline/dump/*.json)
  - Chronos-Bolt-mini zero-shot forecast (computed locally on demand
    history alone)
  - Per-zone-weighted ensemble forecast
  - MAPE numbers per zone per forecast

The cache is small (~80 KB) and ships with the Space so the Backtest tab
renders instantly without any HPC dependency at request time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

ROOT = Path("/Users/macbookpro/Local_Root/predict-power")
sys.path.insert(0, str(ROOT / "space"))

from model_utils import (
    ALPHA_PER_ZONE_MINI,
    CHRONOS_CONTEXT,
    FUTURE_LEN,
    ZONE_COLS,
    load_chronos,
    per_zone_ensemble,
)

DEMAND_CSV = ROOT / "pretrained_models/baseline/dump/demand_2019_2022_hourly.csv"
BASELINE_TEST = ROOT / "pretrained_models/baseline/dump/baseline_preds_test_2022_last2d.json"
BASELINE_VAL  = ROOT / "pretrained_models/baseline/dump/baseline_preds_val_2022_last14d.json"
OUT = ROOT / "space/assets/backtest_2022_last7d.json"

TARGET_STARTS = [
    "2022-12-25T00:00:00",
    "2022-12-26T00:00:00",
    "2022-12-27T00:00:00",
    "2022-12-28T00:00:00",
    "2022-12-29T00:00:00",
    "2022-12-30T00:00:00",
    "2022-12-31T00:00:00",
]


def per_zone_mape(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """pred / truth: (24, 8) -> {zone: mape%}.  Skip cells where truth==0."""
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


def main():
    print("Loading demand CSV...")
    df = pd.read_csv(DEMAND_CSV, parse_dates=["timestamp_utc"])
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    print("Loading baseline preds (cluster, real HRRR weather)...")
    base_test = json.loads(BASELINE_TEST.read_text())
    base_val  = json.loads(BASELINE_VAL.read_text())

    def find_baseline(start_iso: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (pred (24,8), truth (24,8)) by looking up in test then val."""
        for src in (base_test, base_val):
            if start_iso in src["forecast_starts"]:
                idx = src["forecast_starts"].index(start_iso)
                return (np.asarray(src["preds"][idx], dtype=np.float32),
                        np.asarray(src["truth"][idx], dtype=np.float32))
        raise KeyError(f"forecast start {start_iso} not found in baseline preds")

    print("Loading Chronos-Bolt-mini...")
    chronos = load_chronos(device="cpu")

    forecasts = []
    for start_iso in TARGET_STARTS:
        start_ts = pd.Timestamp(start_iso)

        # 24-h history before the start (per zone, MWh)
        end_idx = df.index[df["timestamp_utc"] == start_ts]
        if len(end_idx) == 0:
            raise RuntimeError(f"start {start_iso} not in demand CSV")
        i0 = int(end_idx[0])
        hist = df.loc[i0 - 24: i0 - 1, ZONE_COLS].to_numpy(dtype=np.float32)

        # Chronos context: 672 h ending at start - 1
        if i0 < CHRONOS_CONTEXT:
            raise RuntimeError(f"need {CHRONOS_CONTEXT}h before {start_iso}, have {i0}")
        ctx = df.loc[i0 - CHRONOS_CONTEXT: i0 - 1, ZONE_COLS].to_numpy(dtype=np.float32)

        # Baseline pred + truth from cluster JSON (real-weather predictions)
        baseline_pred, truth = find_baseline(start_iso)

        # Run Chronos-Bolt-mini
        ctx_tensor = torch.from_numpy(ctx.T).to(torch.float32)   # (8, 672)
        with torch.no_grad():
            quantiles, _ = chronos.predict_quantiles(
                context=ctx_tensor,
                prediction_length=FUTURE_LEN,
                quantile_levels=[0.5],
            )
        chronos_pred = quantiles[:, :, 0].cpu().numpy().T.astype(np.float32)  # (24, 8)

        # Per-zone-weighted ensemble
        ens_pred = per_zone_ensemble(baseline_pred, chronos_pred, ALPHA_PER_ZONE_MINI)

        forecasts.append({
            "start": start_iso,
            "history_24h": hist.tolist(),     # (24, 8)
            "truth_24h":   truth.tolist(),     # (24, 8)
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
        })
        print(f"  {start_iso}: baseline {forecasts[-1]['mape']['baseline']['overall']:.2f}% | "
              f"chronos {forecasts[-1]['mape']['chronos']['overall']:.2f}% | "
              f"ensemble {forecasts[-1]['mape']['ensemble']['overall']:.2f}%")

    # Aggregate over the 7 forecasts
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
            "per_zone": {z: float(np.mean(vs)) if vs else float("nan") for z, vs in per_zone.items()},
            "overall":  float(np.mean(overalls)),
        }

    overall_summary = {
        "baseline": agg("baseline"),
        "chronos":  agg("chronos"),
        "ensemble": agg("ensemble"),
    }
    print()
    print("=== 7-day average (Dec 25-31, 2022) ===")
    for k in ("baseline", "chronos", "ensemble"):
        print(f"  {k:<10} overall MAPE: {overall_summary[k]['overall']:.2f}%")

    out = {
        "schema_version": "1.0",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "zones": ZONE_COLS,
        "horizon_hours": FUTURE_LEN,
        "n_forecasts": len(forecasts),
        "alpha_per_zone": ALPHA_PER_ZONE_MINI,
        "forecasts": forecasts,
        "summary": overall_summary,
        "note": (
            "Baseline forecasts use real HRRR weather (computed on the cluster). "
            "Chronos-Bolt-mini forecasts are zero-shot, computed locally from "
            "672 hours of per-zone demand history. Ensemble is the per-zone "
            "weighted blend with the alpha values shown."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    size_kb = OUT.stat().st_size / 1024
    print(f"\nWrote {OUT} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
