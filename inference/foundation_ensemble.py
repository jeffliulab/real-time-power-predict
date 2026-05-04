"""
Foundation-model ensemble: linear blend of baseline + c1 (Chronos zero-shot)
+ c2 (Chronos fine-tuned), with weights fit on a validation window.

Algorithm
---------
1. Load three sets of predictions (baseline, c1, c2) for both val and test
   windows, plus ground truth (from the baseline dump JSON).
2. On the val window, grid-search (alpha, beta) ∈ [0, 1]² with alpha+beta ≤ 1
   to minimise overall MAPE of
       y_hat = alpha * baseline + beta * c1 + (1 - alpha - beta) * c2
3. (Optional ablation) per-zone weight search: 8 independent (alpha_z, beta_z).
4. Apply best weights to the test window; report per-zone and overall MAPE.

If c2 is not provided (`--c2_*` omitted), runs the 2-way ensemble:
    y_hat = alpha * baseline + (1 - alpha) * c1, alpha ∈ [0, 1] grid

All computations in physical MWh. MAPE = mean(|pred-truth|/|truth|) per zone,
average across zones.

Usage
-----
Two-way (after step 1 + step 2):
    python inference/foundation_ensemble.py \\
        --baseline_test pretrained_models/baseline/dump/baseline_preds_test_2022_last2d.json \\
        --baseline_val  pretrained_models/baseline/dump/baseline_preds_val_2022_last14d.json \\
        --c1_test runs/chronos_c1/preds_test_2022_last2d.json \\
        --c1_val  runs/chronos_c1/preds_val_2022_last14d.json \\
        --out runs/foundation_ensemble/eval_test_2022_last2d.json

Three-way (after c2 also trained):
    python inference/foundation_ensemble.py \\
        --baseline_test ... --baseline_val ... \\
        --c1_test ...      --c1_val ... \\
        --c2_test ...      --c2_val ... \\
        --out runs/foundation_ensemble/eval_test_2022_last2d_3way.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ZONE_COLS = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_test", type=str, required=True)
    p.add_argument("--baseline_val",  type=str, required=True)
    p.add_argument("--c1_test", type=str, required=True)
    p.add_argument("--c1_val",  type=str, required=True)
    p.add_argument("--c2_test", type=str, default=None)
    p.add_argument("--c2_val",  type=str, default=None)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--per_zone", action="store_true",
                   help="Also report per-zone-independent weights as an ablation")
    p.add_argument("--step", type=float, default=0.05,
                   help="Grid step size for alpha/beta search")
    return p.parse_args()


def load_preds(path):
    with open(path) as f:
        return json.load(f)


def to_array(d, key="preds"):
    """Convert JSON nested list to np.ndarray of shape (N, 24, 8)."""
    arr = np.asarray(d[key], dtype=np.float32)
    assert arr.ndim == 3 and arr.shape[1] == 24 and arr.shape[2] == 8, \
        f"unexpected shape {arr.shape} from {d.get('split', '?')}"
    return arr


def align(baseline, c1, c2=None):
    """Sanity-check that all sources cover the same forecast starts."""
    starts_b = baseline["forecast_starts"]
    starts_1 = c1["forecast_starts"]
    if starts_b != starts_1:
        raise ValueError(f"forecast_starts mismatch: baseline {starts_b[:3]}... "
                         f"vs c1 {starts_1[:3]}...")
    if c2 is not None:
        starts_2 = c2["forecast_starts"]
        if starts_b != starts_2:
            raise ValueError(f"forecast_starts mismatch: baseline vs c2")
    return starts_b


def mape(pred, truth):
    """Mean absolute percentage error in %, computed in physical MWh.

    Per-zone first, then averaged across zones (matches scripts/self_eval.py).
    """
    per_zone = []
    for j in range(8):
        mask = truth[:, :, j] != 0
        if mask.sum() == 0:
            per_zone.append(float("nan"))
            continue
        ape = np.abs(pred[:, :, j] - truth[:, :, j]) / np.abs(truth[:, :, j])
        per_zone.append(float(ape[mask].mean() * 100))
    overall = float(np.nanmean(per_zone))
    return overall, per_zone


def grid_search_2way(baseline, c1, truth, step=0.05):
    best = (1.0, mape(baseline, truth)[0])
    alphas = np.arange(0.0, 1.0 + 1e-9, step)
    history = []
    for a in alphas:
        blend = a * baseline + (1 - a) * c1
        m, _ = mape(blend, truth)
        history.append((float(a), m))
        if m < best[1]:
            best = (float(a), m)
    return best, history


def grid_search_3way(baseline, c1, c2, truth, step=0.05):
    best = (1.0, 0.0, mape(baseline, truth)[0])
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    history = []
    for a in grid:
        for b in grid:
            if a + b > 1.0 + 1e-9:
                continue
            g = 1.0 - a - b
            blend = a * baseline + b * c1 + g * c2
            m, _ = mape(blend, truth)
            history.append((float(a), float(b), m))
            if m < best[2]:
                best = (float(a), float(b), m)
    return best, history


def per_zone_search(baseline, c1, c2, truth, step=0.05):
    """Independent weights per zone. Returns (alpha[8], beta[8]) and best MAPE."""
    n_z = 8
    alpha = np.ones(n_z, dtype=np.float32)
    beta = np.zeros(n_z, dtype=np.float32)
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    for j in range(n_z):
        best_m = float("inf")
        best_ab = (1.0, 0.0)
        for a in grid:
            for b in grid:
                if a + b > 1.0 + 1e-9:
                    continue
                g = 1.0 - a - b
                if c2 is not None:
                    blend_zone = a * baseline[:, :, j] + b * c1[:, :, j] + g * c2[:, :, j]
                else:
                    blend_zone = a * baseline[:, :, j] + (1 - a) * c1[:, :, j]
                mask = truth[:, :, j] != 0
                if mask.sum() == 0:
                    continue
                ape = np.abs(blend_zone - truth[:, :, j]) / np.abs(truth[:, :, j])
                m = float(ape[mask].mean() * 100)
                if m < best_m:
                    best_m = m
                    best_ab = (float(a), float(b))
        alpha[j] = best_ab[0]
        beta[j] = best_ab[1]
    return alpha, beta


def apply_per_zone(baseline, c1, c2, alpha, beta):
    blend = np.zeros_like(baseline)
    for j in range(8):
        a = alpha[j]
        b = beta[j]
        if c2 is not None:
            g = 1.0 - a - b
            blend[:, :, j] = a * baseline[:, :, j] + b * c1[:, :, j] + g * c2[:, :, j]
        else:
            blend[:, :, j] = a * baseline[:, :, j] + (1 - a) * c1[:, :, j]
    return blend


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bt = load_preds(args.baseline_test)
    bv = load_preds(args.baseline_val)
    c1t = load_preds(args.c1_test)
    c1v = load_preds(args.c1_val)
    c2t = load_preds(args.c2_test) if args.c2_test else None
    c2v = load_preds(args.c2_val)  if args.c2_val  else None

    align(bt, c1t, c2t)
    align(bv, c1v, c2v)

    # Pull arrays
    base_test = to_array(bt, "preds")
    base_val  = to_array(bv, "preds")
    truth_test = to_array(bt, "truth")
    truth_val  = to_array(bv, "truth")
    c1_test = to_array(c1t, "preds")
    c1_val  = to_array(c1v, "preds")
    c2_test = to_array(c2t, "preds") if c2t else None
    c2_val  = to_array(c2v, "preds") if c2v else None

    n_way = 3 if c2_test is not None else 2
    print(f"=== {n_way}-way ensemble ===")
    print(f"Test slice: {base_test.shape[0]} forecasts × 24h × 8 zones")
    print(f"Val  slice: {base_val.shape[0]}  forecasts × 24h × 8 zones")

    # Per-model standalone MAPE on test
    print("\n--- Standalone MAPE on test slice ---")
    base_overall, base_pz = mape(base_test, truth_test)
    c1_overall, c1_pz = mape(c1_test, truth_test)
    print(f"  baseline : {base_overall:.2f} %")
    print(f"  c1       : {c1_overall:.2f} %")
    if c2_test is not None:
        c2_overall, c2_pz = mape(c2_test, truth_test)
        print(f"  c2       : {c2_overall:.2f} %")

    # Standalone MAPE on val
    print("\n--- Standalone MAPE on val slice ---")
    bv_overall, _ = mape(base_val, truth_val)
    c1v_overall, _ = mape(c1_val, truth_val)
    print(f"  baseline : {bv_overall:.2f} %")
    print(f"  c1       : {c1v_overall:.2f} %")
    if c2_val is not None:
        c2v_overall, _ = mape(c2_val, truth_val)
        print(f"  c2       : {c2v_overall:.2f} %")

    # Grid search on val
    print(f"\n--- Grid search on val (step={args.step}) ---")
    if c2_test is None:
        (alpha_star, val_best), hist = grid_search_2way(
            base_val, c1_val, truth_val, step=args.step
        )
        beta_star = 1.0 - alpha_star
        print(f"  best α* = {alpha_star:.2f}  (β* = {beta_star:.2f})  "
              f"val MAPE = {val_best:.2f} %")
        # apply
        blend_test = alpha_star * base_test + (1 - alpha_star) * c1_test
        blend_overall, blend_pz = mape(blend_test, truth_test)
        weights = {"alpha_baseline": alpha_star, "beta_c1": beta_star}
    else:
        (alpha_star, beta_star, val_best), hist = grid_search_3way(
            base_val, c1_val, c2_val, truth_val, step=args.step
        )
        gamma_star = 1.0 - alpha_star - beta_star
        print(f"  best α* = {alpha_star:.2f}  β* = {beta_star:.2f}  "
              f"γ* = {gamma_star:.2f}  val MAPE = {val_best:.2f} %")
        blend_test = (alpha_star * base_test
                      + beta_star * c1_test
                      + gamma_star * c2_test)
        blend_overall, blend_pz = mape(blend_test, truth_test)
        weights = {
            "alpha_baseline": alpha_star,
            "beta_c1": beta_star,
            "gamma_c2": gamma_star,
        }

    print(f"\n--- Test slice (final) ---")
    print(f"  ENSEMBLE OVERALL MAPE: {blend_overall:.2f} %")
    print(f"  baseline alone      : {base_overall:.2f} %")
    delta = base_overall - blend_overall
    sign = "✓ improved" if delta > 0 else "✗ regressed"
    print(f"  Δ = {delta:+.2f} pp  {sign}")
    print(f"  Per-zone MAPE:")
    for j, z in enumerate(ZONE_COLS):
        print(f"    {z:10s}  baseline {base_pz[j]:5.2f}%  "
              f"ensemble {blend_pz[j]:5.2f}%  "
              f"Δ {base_pz[j] - blend_pz[j]:+5.2f}")

    out = {
        "n_way": n_way,
        "weights_global": weights,
        "val_mape_at_best_weights": val_best,
        "test_mape_ensemble": blend_overall,
        "test_mape_baseline_alone": base_overall,
        "test_mape_c1_alone": c1_overall,
        "test_mape_c2_alone": (c2_overall if c2_test is not None else None),
        "test_per_zone_baseline": dict(zip(ZONE_COLS, base_pz)),
        "test_per_zone_ensemble": dict(zip(ZONE_COLS, blend_pz)),
        "delta_pp": delta,
        "ensemble_test_preds": blend_test.tolist(),
        "test_forecast_starts": bt["forecast_starts"],
        "step": args.step,
    }

    if args.per_zone:
        print("\n--- Per-zone independent weights (ablation) ---")
        alpha_pz, beta_pz = per_zone_search(
            base_val, c1_val, c2_val, truth_val, step=args.step
        )
        blend_pz_test = apply_per_zone(
            base_test, c1_test, c2_test, alpha_pz, beta_pz
        )
        pz_overall, pz_pz = mape(blend_pz_test, truth_test)
        print(f"  Per-zone-weighted test MAPE: {pz_overall:.2f} %")
        print(f"  α per zone : "
              + " ".join(f"{ZONE_COLS[j]}={alpha_pz[j]:.2f}" for j in range(8)))
        print(f"  β per zone : "
              + " ".join(f"{ZONE_COLS[j]}={beta_pz[j]:.2f}" for j in range(8)))
        out["per_zone_weights"] = {
            "alpha": [float(x) for x in alpha_pz],
            "beta": [float(x) for x in beta_pz],
            "test_mape": pz_overall,
            "test_per_zone_mape": dict(zip(ZONE_COLS, pz_pz)),
        }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
