"""
Bootstrap MAPE confidence intervals for the workshop paper.

Resampling unit: (forecast_day, zone) pairs. The ISO-NE forecast windows
in the paper are 24-hour blocks, and within a 24-hour block consecutive-
hour MAPE residuals are highly autocorrelated, so bootstrapping at the
hour level would dramatically underestimate variance. Resampling
(forecast_day, zone) pairs respects the natural unit of variability:
each pair is one trial of "the model on a random day-zone combination".

Public API:

  mape_with_ci(preds, truth, n_resamples=1000, ci=0.95, seed=42)
      → (overall_mape, ci_low, ci_high, per_zone_dict)

where per_zone_dict[zone] = (point, ci_low, ci_high) for each of the 8
zones. The per-zone CIs are computed by bootstrapping within that zone's
column (n_forecasts samples, with replacement).

  diff_with_ci(preds_a, preds_b, truth, n_resamples=1000, ci=0.95)
      → (mean_diff, ci_low, ci_high)

paired test for MAPE_a - MAPE_b on the same (day, zone) grid.

Imported by render_*.py figure scripts and by historical_drift_sweep.py.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

ZONES = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]


def _per_pair_mape(preds: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Compute MAPE per (day, zone) pair, averaged over the 24-hour horizon.

    Args:
        preds, truth: ``(n_forecasts, 24, 8)`` float arrays in MWh.
    Returns:
        ``(n_forecasts, 8)`` float array; entry [d, z] is the mean MAPE
        across the 24 forecast hours for forecast day d, zone z.
    """
    if preds.shape != truth.shape:
        raise ValueError(f"preds {preds.shape} != truth {truth.shape}")
    abs_pct_err = np.abs(preds - truth) / np.abs(truth)
    return abs_pct_err.mean(axis=1) * 100   # mean over hour axis


def mape_with_ci(preds: np.ndarray,
                  truth: np.ndarray,
                  n_resamples: int = 1000,
                  ci: float = 0.95,
                  seed: int = 42,
                  ) -> Tuple[float, float, float, dict]:
    """Overall MAPE + bootstrap CI, plus per-zone MAPEs + per-zone CIs.

    Args:
        preds, truth: ``(n_forecasts, 24, 8)``.
        n_resamples: bootstrap draws (default 1000).
        ci: confidence level (default 0.95 → 2.5/97.5 percentiles).
        seed: numpy RNG seed.
    Returns:
        (overall_point, overall_ci_low, overall_ci_high, per_zone_dict)
        where per_zone_dict[zone] = (point, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    pair = _per_pair_mape(preds, truth)             # (n_forecasts, 8)
    n_forecasts, n_zones = pair.shape
    if n_zones != len(ZONES):
        raise ValueError(f"expected 8 zones, got {n_zones}")

    overall_point = float(pair.mean())

    # Overall CI: resample over (day, zone) pairs, n_forecasts*n_zones picks
    # per resample.
    flat = pair.reshape(-1)
    n_pairs = flat.size
    boot = rng.choice(flat, size=(n_resamples, n_pairs), replace=True)
    boot_overall = boot.mean(axis=1)
    alpha = (1 - ci) / 2
    overall_ci_low = float(np.quantile(boot_overall, alpha))
    overall_ci_high = float(np.quantile(boot_overall, 1 - alpha))

    # Per-zone CI: bootstrap within each zone's column independently.
    per_zone = {}
    for j, z in enumerate(ZONES):
        col = pair[:, j]
        boot_z = rng.choice(col, size=(n_resamples, n_forecasts), replace=True).mean(axis=1)
        per_zone[z] = (
            float(col.mean()),
            float(np.quantile(boot_z, alpha)),
            float(np.quantile(boot_z, 1 - alpha)),
        )

    return overall_point, overall_ci_low, overall_ci_high, per_zone


def diff_with_ci(preds_a: np.ndarray,
                  preds_b: np.ndarray,
                  truth: np.ndarray,
                  n_resamples: int = 1000,
                  ci: float = 0.95,
                  seed: int = 42,
                  ) -> Tuple[float, float, float]:
    """Paired MAPE difference (a - b) with bootstrap CI.

    Useful for "is model A meaningfully better than model B on the same
    test slice?" The CI tells you whether 0 is plausible.
    """
    rng = np.random.default_rng(seed)
    pair_a = _per_pair_mape(preds_a, truth).reshape(-1)
    pair_b = _per_pair_mape(preds_b, truth).reshape(-1)
    diff = pair_a - pair_b
    point = float(diff.mean())
    boot = rng.choice(diff, size=(n_resamples, diff.size), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    return point, float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))


if __name__ == "__main__":
    # Smoke test: synthesize 7 forecast days with a known mean MAPE
    rng = np.random.default_rng(0)
    truth = rng.uniform(800, 2000, size=(7, 24, 8)).astype(np.float32)
    preds = truth * (1 + rng.normal(0, 0.10, size=(7, 24, 8))).astype(np.float32)
    overall, lo, hi, pz = mape_with_ci(preds, truth)
    print(f"Overall MAPE: {overall:.2f}% [95% CI: {lo:.2f}-{hi:.2f}]")
    for z, (p, l, h) in pz.items():
        print(f"  {z:11s}: {p:.2f}% [{l:.2f}-{h:.2f}]")
