"""
Drift-Weighted Ensemble Re-weighting — the novel inference-time method
introduced in §6 of v1.6.

The deployed system blends two predictions per zone:

    y_ens[h, z] = alpha_z * y_baseline[h, z]  +  (1 - alpha_z) * y_chronos[h, z]

In production today the per-zone weights alpha_z are *frozen* — they were
fit once on a 14-day 2022 validation slice and have not been refreshed.
As §4 shows, the trained baseline drifts year-over-year in high-BTM zones,
which means the OPTIMAL alpha is also drifting. Frozen-alpha leaves
performance on the table.

This method refits alpha_z monthly on a rolling validation slice. For
each forecast date T:

    validation_slice = [T - 16 days, T - 2 days]   # 14 days, with 2d label-buffer
    For each zone z:
        alpha_z^new = argmin_{a in 0..1, step 0.05}  zone_MAPE_z on validation slice
                     given the convex combination a*baseline + (1-a)*chronos

    y_ens_T = drift_weighted_alpha applied to baseline + chronos forecasts for T

No retraining of either component model is required. The grid-search is
deterministic and uses no additional hyperparameters (no SGD, no L2 reg).

We benchmark three modes across W0=2022-05, W1=2025-05, W2=2026-04/05:

    frozen        — ALPHA_PER_ZONE_MINI (current production)
    drift_weighted — re-fit on each forecast's rolling validation slice
    oracle        — best alpha on the test window itself (UPPER BOUND; not deployable)

We expect drift_weighted to close a meaningful fraction of the gap between
frozen and oracle, especially in the later windows where drift has
accumulated.

Inputs
------
    本地文件/report/arxiv/data/multi_year_drift.json   — sweep output from
        scripts/experiments/historical_drift_sweep.py (in-window forecasts)
    + live baseline + Chronos runs on each window's validation slice

Outputs
-------
    本地文件/report/arxiv/data/drift_weighted_benchmark.json
    本地文件/report/arxiv/data/validation_slice_preds.json   (intermediate)

Determinism: all numbers are reproducible. The alpha grid search is
deterministic; baseline forward is deterministic given fixed checkpoint
weights and fixed inputs; Chronos zero-shot is deterministic at quantile
levels we use (0.5 median).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "space"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from iso_ne_zonal import ZONE_COLS  # noqa: E402
from model_utils import (  # noqa: E402
    ALPHA_PER_ZONE_MINI,
    CHRONOS_CONTEXT,
    FUTURE_LEN,
    HISTORY_LEN,
    load_baseline,
    load_chronos,
)
from historical_drift_sweep import (  # noqa: E402
    fetch_demand_for_window,
    run_one_forecast,
)


N_VAL_DAYS = 14
LABEL_BUFFER_DAYS = 2          # forecast at T uses validation ending T-2d
ALPHA_GRID = np.arange(0.0, 1.0 + 1e-9, 0.05).round(2)

INPUT_SWEEP = ROOT / "report" / "arxiv" / "data" / "multi_year_drift.json"
OUT_VALID = ROOT / "report" / "arxiv" / "data" / "validation_slice_preds.json"
OUT_BENCH = ROOT / "report" / "arxiv" / "data" / "drift_weighted_benchmark.json"


def _per_day_per_zone_mape(preds_24x8: np.ndarray,
                            truth_24x8: np.ndarray) -> np.ndarray:
    """Return (8,) per-zone MAPE in % for one (24, 8) forecast."""
    abs_pct = np.abs(preds_24x8 - truth_24x8) / np.abs(truth_24x8)
    return 100.0 * abs_pct.mean(axis=0)


def _grid_search_alpha_per_zone(baseline_stack: np.ndarray,
                                 chronos_stack: np.ndarray,
                                 truth_stack: np.ndarray) -> np.ndarray:
    """Per-zone grid search over alpha in [0, 1] step 0.05.

    Args:
        baseline_stack, chronos_stack, truth_stack: (N, 24, 8) arrays.

    Returns:
        (8,) array of optimal alpha per zone.
    """
    n_z = truth_stack.shape[2]
    best_alpha = np.zeros(n_z, dtype=np.float32)
    for z in range(n_z):
        b = baseline_stack[..., z]       # (N, 24)
        c = chronos_stack[..., z]
        t = truth_stack[..., z]
        best_mape = np.inf
        best_a = 0.5
        for a in ALPHA_GRID:
            mix = a * b + (1 - a) * c
            mape = float(100.0 * (np.abs(mix - t) / np.abs(t)).mean())
            if mape < best_mape:
                best_mape = mape
                best_a = float(a)
        best_alpha[z] = best_a
    return best_alpha


def _apply_alpha(alpha_per_zone: np.ndarray,
                  baseline_24x8: np.ndarray,
                  chronos_24x8: np.ndarray) -> np.ndarray:
    return (alpha_per_zone[None, :] * baseline_24x8
            + (1 - alpha_per_zone[None, :]) * chronos_24x8)


def _stack_window(window: dict, key: str) -> np.ndarray:
    """Stack one model's predictions across forecasts in a window.

    Returns (N_forecasts, 24, 8) array.
    """
    if key == "truth_24h":
        return np.stack([np.array(f["truth_24h"]) for f in window["forecasts"]],
                         axis=0)
    return np.stack([np.array(f["predictions"][key])
                       for f in window["forecasts"]], axis=0)


def run_validation_slice(window_label: str,
                          window_start: datetime,
                          *,
                          model, norm_stats, chronos_pipeline,
                          parallel: int = 8,
                          device: str = "cpu",
                          ) -> list[dict]:
    """Run baseline+Chronos forecasts on the 14-day validation slice
    ending 2 days before `window_start`. Demand is fetched via the
    multi-source fetcher (bundled CSV for 2019-2022, live for recent)."""
    slice_end = window_start - timedelta(days=LABEL_BUFFER_DAYS)
    slice_start = slice_end - timedelta(days=N_VAL_DAYS)
    # Need additional history for Chronos context (~28d before slice start).
    demand_window_start = slice_start - timedelta(hours=CHRONOS_CONTEXT)
    demand_window_end = slice_end + timedelta(hours=FUTURE_LEN + 2)
    print(f"\n--- Validation slice for {window_label}: "
           f"{slice_start.date()} -> {slice_end.date()} ({N_VAL_DAYS} days) ---")
    print(f"  Demand window: {demand_window_start.date()} -> {demand_window_end.date()}")

    demand_full = fetch_demand_for_window(demand_window_start,
                                            demand_window_end)
    forecasts: list[dict] = []
    cur = slice_start
    while cur < slice_end:
        try:
            f = run_one_forecast(
                cur.replace(hour=0, minute=0, second=0, microsecond=0),
                model=model, norm_stats=norm_stats,
                chronos_pipeline=chronos_pipeline,
                demand_full=demand_full,
                parallel=parallel, device=device)
            forecasts.append(f)
        except Exception as e:  # noqa: BLE001
            print(f"  [SKIP] validation date {cur.date()} failed: {e}")
        cur += timedelta(days=1)
    print(f"  Completed {len(forecasts)} / {N_VAL_DAYS} validation forecasts "
           f"for {window_label}.")
    return forecasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=Path,
                         default=ROOT / "space" / "checkpoints" / "best.pt")
    parser.add_argument("--cache-only", action="store_true",
                         help="Skip live data fetch; use cached "
                               "validation_slice_preds.json if present.")
    args = parser.parse_args()

    # Load sweep output for in-window forecasts.
    if not INPUT_SWEEP.exists():
        raise SystemExit(f"Expected sweep output at {INPUT_SWEEP}; run "
                          f"scripts/experiments/historical_drift_sweep.py first.")
    sweep = json.loads(INPUT_SWEEP.read_text())
    windows = {w["label"]: w for w in sweep["windows"]}
    print(f"Loaded sweep with {len(windows)} windows: {list(windows.keys())}")

    # Validation slices: load from cache or recompute.
    if args.cache_only and OUT_VALID.exists():
        valid = json.loads(OUT_VALID.read_text())
        print(f"Loaded cached validation slice preds from {OUT_VALID}.")
    else:
        print("\nLoading baseline checkpoint...")
        model, ns = load_baseline(args.checkpoint, device=args.device)
        print(f"  ({sum(p.numel() for p in model.parameters()):,} params)")
        print("Loading Chronos-Bolt-mini...")
        chronos_pipeline = load_chronos(device=args.device)

        valid = {"schema_version": "v1.6-validation-slices",
                  "n_val_days": N_VAL_DAYS,
                  "label_buffer_days": LABEL_BUFFER_DAYS,
                  "windows": {}}
        for label, window in windows.items():
            w_start = datetime.fromisoformat(window["start_date"])
            slice_fcs = run_validation_slice(
                label, w_start,
                model=model, norm_stats=ns,
                chronos_pipeline=chronos_pipeline,
                parallel=args.parallel, device=args.device)
            valid["windows"][label] = {"slice_forecasts": slice_fcs}
        OUT_VALID.parent.mkdir(parents=True, exist_ok=True)
        OUT_VALID.write_text(json.dumps(valid, indent=2))
        print(f"\nWrote {OUT_VALID}")

    # ---- Benchmark: three modes ----
    print("\n=== Drift-weighted ensemble benchmark ===")
    out: dict = {
        "schema_version": "v1.6-drift-weighted",
        "alpha_grid": ALPHA_GRID.tolist(),
        "n_val_days": N_VAL_DAYS,
        "label_buffer_days": LABEL_BUFFER_DAYS,
        "frozen_alpha": ALPHA_PER_ZONE_MINI,
        "windows": {},
    }

    frozen_alpha_arr = np.array(
        [ALPHA_PER_ZONE_MINI[z] for z in ZONE_COLS], dtype=np.float32)

    for label, window in windows.items():
        baseline_test = _stack_window(window, "baseline")    # (N, 24, 8)
        chronos_test = _stack_window(window, "chronos")
        truth_test = _stack_window(window, "truth_24h")
        n_fcs = truth_test.shape[0]

        slice_fcs = valid["windows"].get(label, {}).get("slice_forecasts", [])
        if not slice_fcs:
            print(f"  [SKIP] no validation slice forecasts for {label}; "
                   f"drift-weighted mode unavailable.")
            continue

        baseline_val = np.stack([np.array(f["predictions"]["baseline"])
                                   for f in slice_fcs], axis=0)
        chronos_val = np.stack([np.array(f["predictions"]["chronos"])
                                  for f in slice_fcs], axis=0)
        truth_val = np.stack([np.array(f["truth_24h"]) for f in slice_fcs],
                               axis=0)
        n_val = truth_val.shape[0]

        # FROZEN: apply ALPHA_PER_ZONE_MINI to every forecast in window
        frozen_preds = np.stack(
            [_apply_alpha(frozen_alpha_arr, baseline_test[i], chronos_test[i])
             for i in range(n_fcs)], axis=0)

        # DRIFT-WEIGHTED: single refit on validation slice, applied to all
        # forecasts in window (matches a monthly-refit deployment cadence).
        # (Per-day rolling refit would require longer validation history
        # than we currently have for W0's 14-day slice; we report this
        # window-level refit as a conservative approximation.)
        dw_alpha = _grid_search_alpha_per_zone(
            baseline_val, chronos_val, truth_val)
        dw_preds = np.stack(
            [_apply_alpha(dw_alpha, baseline_test[i], chronos_test[i])
             for i in range(n_fcs)], axis=0)

        # ORACLE: alpha minimizing MAPE on the test window itself (UPPER BOUND).
        oracle_alpha = _grid_search_alpha_per_zone(
            baseline_test, chronos_test, truth_test)
        oracle_preds = np.stack(
            [_apply_alpha(oracle_alpha, baseline_test[i], chronos_test[i])
             for i in range(n_fcs)], axis=0)

        # Per-zone + overall MAPE for each mode
        def _summary(preds):
            per_zone = (100.0 *
                         (np.abs(preds - truth_test) / np.abs(truth_test))
                         .mean(axis=(0, 1)))      # (8,)
            overall = float(per_zone.mean())
            return {"overall_mape_pct": overall,
                     "per_zone_mape_pct":
                         {z: float(v) for z, v in zip(ZONE_COLS, per_zone)}}

        out["windows"][label] = {
            "n_forecasts_in_window": n_fcs,
            "n_validation_days": n_val,
            "frozen_alpha": {z: float(a) for z, a in zip(ZONE_COLS, frozen_alpha_arr)},
            "drift_weighted_alpha": {z: float(a) for z, a in zip(ZONE_COLS, dw_alpha)},
            "oracle_alpha": {z: float(a) for z, a in zip(ZONE_COLS, oracle_alpha)},
            "frozen": _summary(frozen_preds),
            "drift_weighted": _summary(dw_preds),
            "oracle": _summary(oracle_preds),
        }

        print(f"\n  Window {label}:  frozen / drift-weighted / oracle "
               f"(overall MAPE pct)")
        for mode in ["frozen", "drift_weighted", "oracle"]:
            o = out["windows"][label][mode]["overall_mape_pct"]
            print(f"    {mode:16s}: {o:6.2f} %")
        gap = (out["windows"][label]["frozen"]["overall_mape_pct"]
               - out["windows"][label]["drift_weighted"]["overall_mape_pct"])
        oracle_gap = (out["windows"][label]["frozen"]["overall_mape_pct"]
                       - out["windows"][label]["oracle"]["overall_mape_pct"])
        if oracle_gap > 1e-6:
            pct_closed = 100.0 * gap / oracle_gap
            print(f"    drift-weighted closes {pct_closed:.1f} % of the "
                   f"frozen-to-oracle gap.")

    OUT_BENCH.parent.mkdir(parents=True, exist_ok=True)
    OUT_BENCH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_BENCH}")


if __name__ == "__main__":
    main()
