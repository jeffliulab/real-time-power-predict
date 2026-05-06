"""
Augment report/arxiv/data/validation_dec30_2022.json with bootstrap CIs.

The original validation experiment (reproduce_dec30_2022.py) reports
overall and per-zone point-estimate MAPE for both the live pipeline and
the cluster's stored prediction on 2022-12-30. For the workshop paper §3
we want bootstrap 95 % CIs on those numbers to make the equivalence
claim ("within 0.13 pp") falsifiable.

This script reads the existing JSON, computes bootstrap CIs from the
saved (preds, truth) arrays without re-fetching HRRR, and writes the
augmented JSON back. Idempotent.

Note: the validation experiment uses a single forecast day (n_forecasts
= 1), so per-zone bootstrap CIs degenerate to (point, point, point) —
there's no within-zone variability to bootstrap. For overall CI we have
8 zones x 1 day = 8 (day, zone) pairs, which is small but usable. The
paper text discloses this n=1 caveat.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = ROOT / "report" / "arxiv" / "data" / "validation_dec30_2022.json"
sys.path.insert(0, str(ROOT / "scripts" / "figures"))

from bootstrap_mape import mape_with_ci, diff_with_ci   # noqa: E402


def main():
    if not JSON_PATH.exists():
        sys.exit(f"missing {JSON_PATH}; run reproduce_dec30_2022.py first")
    d = json.loads(JSON_PATH.read_text())

    truth = np.asarray(d["truth"], dtype=np.float32)
    cluster_pred = np.asarray(d["cluster_pred"], dtype=np.float32)
    live_pred = np.asarray(d["live_pred"], dtype=np.float32)

    # mape_with_ci expects shape (n_forecasts, 24, 8); we have (24, 8) for
    # a single forecast → unsqueeze.
    truth_b = truth[None, :, :]
    cluster_b = cluster_pred[None, :, :]
    live_b = live_pred[None, :, :]

    cluster_overall, cl_lo, cl_hi, cluster_pz = mape_with_ci(cluster_b, truth_b)
    live_overall, li_lo, li_hi, live_pz = mape_with_ci(live_b, truth_b)
    diff_point, diff_lo, diff_hi = diff_with_ci(live_b, cluster_b, truth_b)

    print(f"Cluster MAPE : {cluster_overall:.2f}% [95% CI: {cl_lo:.2f}-{cl_hi:.2f}]")
    print(f"Live MAPE    : {live_overall:.2f}% [95% CI: {li_lo:.2f}-{li_hi:.2f}]")
    print(f"Diff (l-c)   : {diff_point:+.3f} pp [95% CI: {diff_lo:+.3f} to {diff_hi:+.3f}]")

    d["bootstrap_ci"] = {
        "n_resamples": 1000,
        "ci_level": 0.95,
        "seed": 42,
        "resample_unit": "(forecast_day, zone) pair",
        "n_pairs": int(truth_b.shape[0] * truth_b.shape[2]),
        "cluster_overall_mape": {
            "point": cluster_overall, "ci_low": cl_lo, "ci_high": cl_hi},
        "live_overall_mape": {
            "point": live_overall, "ci_low": li_lo, "ci_high": li_hi},
        "diff_live_minus_cluster": {
            "point": diff_point, "ci_low": diff_lo, "ci_high": diff_hi},
        "cluster_per_zone_mape": {
            z: {"point": p, "ci_low": l, "ci_high": h}
            for z, (p, l, h) in cluster_pz.items()},
        "live_per_zone_mape": {
            z: {"point": p, "ci_low": l, "ci_high": h}
            for z, (p, l, h) in live_pz.items()},
    }

    JSON_PATH.write_text(json.dumps(d, indent=2))
    print(f"\nAugmented {JSON_PATH}")


if __name__ == "__main__":
    main()
