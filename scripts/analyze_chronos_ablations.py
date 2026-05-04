"""
Analyze the chronos ablation sweep results:

    runs/chronos_ablation/ctx/ctx{168,336,672,1344}_{test,val}.json
    runs/chronos_ablation/size/size_{tiny,mini,small,base}_{test,val}.json

For each config, we report:
  - c1-only MAPE on test (median aggregation; mean aggregation for the C2 axis)
  - per-zone-weighted ensemble MAPE on test (alpha grid-searched on val)
  - per-zone alpha values

Outputs JSON + a printable summary table.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path("/Users/macbookpro/Local_Root/predict-power")
ABL = ROOT / "runs" / "chronos_ablation"
ZONES = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]


def load_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def truth_array(d: dict) -> np.ndarray:
    """Reconstruct the truth array (n_forecasts, 24, 8) by reading the demand CSV
    referenced in the JSON and slicing 24 hours starting from each forecast_starts."""
    import pandas as pd
    df = pd.read_csv(d["demand_csv"])
    ts_col = "timestamp_utc" if "timestamp_utc" in df.columns else "timestamp"
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True)
    df = df.rename(columns={ts_col: "timestamp"})
    starts = [pd.Timestamp(s) for s in d["forecast_starts"]]
    truth = np.zeros((len(starts), 24, 8), dtype=np.float32)
    for i, s in enumerate(starts):
        idx = df.index[df["timestamp"] == s]
        if len(idx) == 0:
            raise RuntimeError(f"forecast start {s} not in demand CSV")
        i0 = int(idx[0])
        truth[i] = df.loc[i0:i0 + 23, ZONES].to_numpy(dtype=np.float32)
    return truth


def per_zone_mape(preds: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """preds, truth: (n_forecasts, 24, 8). Returns per-zone MAPE in percent."""
    out = np.zeros(8, dtype=np.float32)
    for z in range(8):
        mask = truth[:, :, z] != 0
        ape = np.abs(preds[:, :, z] - truth[:, :, z]) / np.abs(truth[:, :, z])
        out[z] = ape[mask].mean() * 100
    return out


def overall_mape(preds: np.ndarray, truth: np.ndarray) -> float:
    return float(per_zone_mape(preds, truth).mean())


def grid_search_per_zone_alpha(
    base_val: np.ndarray, c1_val: np.ndarray, truth_val: np.ndarray,
    step: float = 0.05,
) -> np.ndarray:
    """For each zone, pick alpha in [0, 1] (step=0.05) minimising val MAPE."""
    alphas = np.arange(0.0, 1.0 + 1e-9, step)
    best = np.zeros(8, dtype=np.float32)
    for z in range(8):
        best_alpha = 0.0
        best_mape = float("inf")
        for a in alphas:
            blend = a * base_val[:, :, z] + (1 - a) * c1_val[:, :, z]
            mask = truth_val[:, :, z] != 0
            ape = np.abs(blend - truth_val[:, :, z]) / np.abs(truth_val[:, :, z])
            mape = float(ape[mask].mean() * 100)
            if mape < best_mape:
                best_mape = mape
                best_alpha = float(a)
        best[z] = best_alpha
    return best


def grid_search_global_alpha(
    base_val: np.ndarray, c1_val: np.ndarray, truth_val: np.ndarray,
    step: float = 0.05,
) -> float:
    alphas = np.arange(0.0, 1.0 + 1e-9, step)
    best_alpha, best_mape = 0.0, float("inf")
    for a in alphas:
        blend = a * base_val + (1 - a) * c1_val
        mape = overall_mape(blend, truth_val)
        if mape < best_mape:
            best_mape, best_alpha = mape, float(a)
    return best_alpha


def evaluate_config(
    test_json: Path, val_json: Path,
    baseline_test: dict, baseline_val: dict,
    aggregation: str = "median",
) -> dict:
    """Returns {c1_test, ensemble_test_perzone, ensemble_test_global, per_zone_alpha}."""
    c1_test = load_json(test_json)
    c1_val = load_json(val_json)

    # Pick the right preds key
    if aggregation == "median":
        c1_test_preds = np.array(c1_test.get("preds_median", c1_test["preds"]), dtype=np.float32)
        c1_val_preds = np.array(c1_val.get("preds_median", c1_val["preds"]), dtype=np.float32)
    elif aggregation == "mean":
        c1_test_preds = np.array(c1_test["preds_mean"], dtype=np.float32)
        c1_val_preds = np.array(c1_val["preds_mean"], dtype=np.float32)
    else:
        raise ValueError(aggregation)

    base_test_preds = np.array(baseline_test["preds"], dtype=np.float32)
    base_val_preds = np.array(baseline_val["preds"], dtype=np.float32)
    truth_test = np.array(baseline_test["truth"], dtype=np.float32)
    truth_val = np.array(baseline_val["truth"], dtype=np.float32)

    # Sanity: shapes must match
    assert c1_test_preds.shape == base_test_preds.shape == truth_test.shape, \
        (c1_test_preds.shape, base_test_preds.shape, truth_test.shape)

    # MAPE — c1 alone on test
    c1_mape = overall_mape(c1_test_preds, truth_test)
    c1_per_zone = per_zone_mape(c1_test_preds, truth_test).tolist()

    # Per-zone alpha — grid-searched on val, applied to test
    pz_alpha = grid_search_per_zone_alpha(base_val_preds, c1_val_preds, truth_val)
    blend_pz = pz_alpha[None, None, :] * base_test_preds + (1 - pz_alpha[None, None, :]) * c1_test_preds
    ens_pz_mape = overall_mape(blend_pz, truth_test)
    ens_pz_per_zone = per_zone_mape(blend_pz, truth_test).tolist()

    # Global alpha
    global_alpha = grid_search_global_alpha(base_val_preds, c1_val_preds, truth_val)
    blend_g = global_alpha * base_test_preds + (1 - global_alpha) * c1_test_preds
    ens_g_mape = overall_mape(blend_g, truth_test)

    return {
        "c1_overall_mape": c1_mape,
        "c1_per_zone_mape": c1_per_zone,
        "ens_perzone_mape": ens_pz_mape,
        "ens_perzone_per_zone": ens_pz_per_zone,
        "ens_global_mape": ens_g_mape,
        "global_alpha": global_alpha,
        "per_zone_alpha": pz_alpha.tolist(),
    }


def main():
    # Load baseline preds + truth (these are produced by HPC dump earlier)
    base_test_path = ROOT / "pretrained_models/baseline/dump/baseline_preds_test_2022_last2d.json"
    base_val_path  = ROOT / "pretrained_models/baseline/dump/baseline_preds_val_2022_last14d.json"
    baseline_test = load_json(base_test_path)
    baseline_val = load_json(base_val_path)
    base_test_preds = np.array(baseline_test["preds"], dtype=np.float32)
    truth_test = np.array(baseline_test["truth"], dtype=np.float32)
    baseline_overall = overall_mape(base_test_preds, truth_test)
    baseline_pz = per_zone_mape(base_test_preds, truth_test).tolist()
    print(f"BASELINE (Part 1): overall test MAPE = {baseline_overall:.2f} %")
    print()

    results = {
        "baseline_overall": baseline_overall,
        "baseline_per_zone": baseline_pz,
        "ctx": {},
        "size": {},
        "aggregation": {},
    }

    # ---- C1: context length sweep (model = base) ----
    print("=" * 70)
    print("C1 — context length sweep (model = chronos-bolt-base)")
    print("=" * 70)
    print(f"{'ctx':>6} | {'c1-only':>9} | {'ens (global)':>13} | {'ens (per-zone)':>15} | global alpha")
    print("-" * 70)
    for ctx in [168, 336, 672, 1344]:
        test_p = ABL / "ctx" / f"ctx{ctx}_test.json"
        val_p  = ABL / "ctx" / f"ctx{ctx}_val.json"
        r = evaluate_config(test_p, val_p, baseline_test, baseline_val)
        results["ctx"][str(ctx)] = r
        print(f"{ctx:>6} | {r['c1_overall_mape']:>8.2f}% | {r['ens_global_mape']:>12.2f}% | "
              f"{r['ens_perzone_mape']:>14.2f}% | {r['global_alpha']:.2f}")
    print()

    # ---- C3: model size sweep (ctx = 672) ----
    print("=" * 70)
    print("C3 — model size sweep (context = 672)")
    print("=" * 70)
    print(f"{'size':>6} | {'c1-only':>9} | {'ens (global)':>13} | {'ens (per-zone)':>15} | global alpha")
    print("-" * 70)
    for sz in ["tiny", "mini", "small", "base"]:
        test_p = ABL / "size" / f"size_{sz}_test.json"
        val_p  = ABL / "size" / f"size_{sz}_val.json"
        # base is symlink → same as ctx672
        r = evaluate_config(test_p, val_p, baseline_test, baseline_val)
        results["size"][sz] = r
        print(f"{sz:>6} | {r['c1_overall_mape']:>8.2f}% | {r['ens_global_mape']:>12.2f}% | "
              f"{r['ens_perzone_mape']:>14.2f}% | {r['global_alpha']:.2f}")
    print()

    # ---- C2: quantile aggregation (using ctx=672, base) ----
    print("=" * 70)
    print("C2 — quantile aggregation (ctx=672, base)")
    print("=" * 70)
    print(f"{'agg':>6} | {'c1-only':>9} | {'ens (global)':>13} | {'ens (per-zone)':>15}")
    print("-" * 70)
    test_p = ABL / "ctx" / "ctx672_test.json"
    val_p  = ABL / "ctx" / "ctx672_val.json"
    for agg in ["median", "mean"]:
        r = evaluate_config(test_p, val_p, baseline_test, baseline_val, aggregation=agg)
        results["aggregation"][agg] = r
        print(f"{agg:>6} | {r['c1_overall_mape']:>8.2f}% | {r['ens_global_mape']:>12.2f}% | "
              f"{r['ens_perzone_mape']:>14.2f}%")
    print()

    # Save full results
    out_path = ROOT / "runs" / "foundation_ensemble" / "ablation_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
