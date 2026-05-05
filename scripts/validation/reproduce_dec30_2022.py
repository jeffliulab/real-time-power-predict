"""
Pipeline-equivalence experiment for Section 9 of the arXiv paper.

Re-runs the *deployed* pipeline (space/hrrr_fetch.py + space/model_utils.py +
space/calendar_features.py) on a date *inside the training window*
(2022-12-30) and compares the result to the cluster's stored prediction
in pretrained_models/baseline/dump/baseline_preds_test_2022_last2d.json.

The deployed pipeline is bit-exact with the live HF Space; the cluster's
stored prediction is what the original training run produced. Equivalence
to within MAPE noise certifies the deployed pipeline as bug-free.

Output:
  - prints overall + per-zone MAPE for both pipelines
  - prints element-wise diff stats
  - saves predictions to data/validation_dec30_2022.json (for the figure
    renderer in render_validation_dec30.py)

Reuses existing modules (no re-implementation):
  space/hrrr_fetch.py:fetch_history          -- HRRR f00 analyses
  space/model_utils.py:load_baseline,
                       normalize_demand,
                       normalize_weather,
                       denormalize_demand    -- model + norm
  space/calendar_features.py:encode_range    -- 44-d calendar one-hot
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "space"))
os.environ.setdefault("HERBIE_CONFIG_PATH", "/tmp/herbie_config")

from calendar_features import encode_range                          # noqa: E402
from model_utils import (HISTORY_LEN, FUTURE_LEN, load_baseline,    # noqa: E402
                         normalize_demand, normalize_weather,
                         denormalize_demand)
from hrrr_fetch import fetch_history                                # noqa: E402

ZONES = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]

DEMAND_CSV = ROOT / "pretrained_models" / "baseline" / "dump" / "demand_2019_2022_hourly.csv"
CLUSTER_DUMP = ROOT / "pretrained_models" / "baseline" / "dump" / "baseline_preds_test_2022_last2d.json"
CHECKPOINT = ROOT / "space" / "checkpoints" / "best.pt"
OUT_PATH = ROOT / "report" / "arxiv" / "data" / "validation_dec30_2022.json"


def _per_zone_mape(pred, truth):
    return [float(
        np.mean(
            np.abs(pred[:, j] - truth[:, j]) / np.abs(truth[:, j])
        ) * 100
    ) for j in range(8)]


def main():
    start_ts = datetime(2022, 12, 30, 0, 0, 0)
    print(f"=== Pipeline-equivalence check at {start_ts} ===")

    # --- Demand
    print("Loading training-CSV demand...")
    df = pd.read_csv(DEMAND_CSV, parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
    hist = df.loc[
        start_ts - timedelta(hours=HISTORY_LEN):
        start_ts - timedelta(hours=1), ZONES].to_numpy(np.float32)
    truth = df.loc[
        start_ts:start_ts + timedelta(hours=FUTURE_LEN - 1), ZONES
    ].to_numpy(np.float32)

    # --- HRRR (use future analyses to match training distribution; this is
    # what the cluster's stored prediction was generated against)
    print("Fetching HRRR history (24 cycles, 8-way parallel)...")
    hist_w = fetch_history(start_ts, hours=HISTORY_LEN, parallel=8)
    print("Fetching HRRR future window (24 cycles, 8-way parallel)...")
    fut_w = fetch_history(start_ts + timedelta(hours=FUTURE_LEN),
                           hours=FUTURE_LEN, parallel=8)

    # --- Calendar
    hist_cal = encode_range(start_ts - timedelta(hours=HISTORY_LEN), HISTORY_LEN)
    fut_cal = encode_range(start_ts, FUTURE_LEN)

    # --- Model forward
    print(f"Loading baseline checkpoint from {CHECKPOINT}...")
    model, ns = load_baseline(CHECKPOINT, device="cpu")
    with torch.no_grad():
        pred_z = model(
            torch.from_numpy(normalize_weather(hist_w, ns)).unsqueeze(0),
            torch.from_numpy(normalize_demand(hist, ns)).unsqueeze(0),
            torch.from_numpy(hist_cal.astype(np.float32)).unsqueeze(0),
            torch.from_numpy(normalize_weather(fut_w, ns)).unsqueeze(0),
            torch.from_numpy(fut_cal.astype(np.float32)).unsqueeze(0),
        )
    live_pred = denormalize_demand(pred_z.squeeze(0).cpu().numpy(), ns)

    # --- Cluster's stored prediction
    cluster = json.loads(CLUSTER_DUMP.read_text())
    idx = cluster["forecast_starts"].index(start_ts.isoformat())
    cluster_pred = np.asarray(cluster["preds"][idx], dtype=np.float32)
    cluster_truth = np.asarray(cluster["truth"][idx], dtype=np.float32)

    # --- Sanity check that truth aligns
    truth_diff = np.abs(truth - cluster_truth).max()
    print(f"\nTruth alignment check: max diff = {truth_diff:.4f} MW (should be ~0)")

    # --- MAPE comparison
    live_overall = float(
        np.mean(np.abs(live_pred - truth) / np.abs(truth)) * 100
    )
    cluster_overall = float(
        np.mean(np.abs(cluster_pred - truth) / np.abs(truth)) * 100
    )
    live_pz = _per_zone_mape(live_pred, truth)
    cluster_pz = _per_zone_mape(cluster_pred, truth)

    print(f"\n=== Results ===")
    print(f"Live pipeline overall MAPE      : {live_overall:.2f} %")
    print(f"Cluster stored overall MAPE     : {cluster_overall:.2f} %")
    print(f"Difference                      : {abs(live_overall - cluster_overall):.2f} pp")

    # --- Element-wise diff
    diff = live_pred - cluster_pred
    print(f"\nElement-wise diff (live - cluster):")
    print(f"  mean        : {diff.mean():.2f} MW")
    print(f"  max abs     : {np.abs(diff).max():.2f} MW")
    rel = np.abs(diff) / np.abs(cluster_pred)
    print(f"  max abs as %% of cluster pred: {rel.max() * 100:.2f} %")

    # --- Per-zone
    print(f"\nPer-zone MAPE (live | cluster | diff):")
    for j, z in enumerate(ZONES):
        print(f"  {z:<11}: {live_pz[j]:5.2f} | {cluster_pz[j]:5.2f} | "
               f"{abs(live_pz[j] - cluster_pz[j]):.2f}")

    # --- Save for figure
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "forecast_start": start_ts.isoformat(),
        "zones": ZONES,
        "truth": truth.tolist(),
        "cluster_pred": cluster_pred.tolist(),
        "live_pred": live_pred.tolist(),
        "live_overall_mape": live_overall,
        "cluster_overall_mape": cluster_overall,
        "live_per_zone_mape": live_pz,
        "cluster_per_zone_mape": cluster_pz,
        "max_abs_diff_mw": float(np.abs(diff).max()),
        "max_rel_diff_pct": float(rel.max() * 100),
    }, indent=2))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
