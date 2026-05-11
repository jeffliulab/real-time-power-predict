"""
Workshop §5 — Figure 4: BTM solar density vs per-state MAPE drift.

Reads multi_window_results.json (W2 = 2026-04/05 baseline per-zone MAPE)
and pretrained_models/baseline/dump/baseline_preds_test_2022_last2d.json
(2022 in-training-window per-zone MAPE), aggregates per-zone MAPE into
per-state numbers (collapsing the 3 Massachusetts zones into 1 MA state),
and plots state-level distributed PV capacity per capita against per-state
MAPE drift (W2 - 2022).

Statistical reporting: Spearman ρ + a 10000-permutation null distribution
p-value. Not a parametric test; we have n=6 states. Both two-sided and
one-sided p-values are reported; with n=6 the test is underpowered for
confirmatory claims.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
MULTI = ROOT / "report" / "arxiv" / "data" / "multi_window_results.json"
TRAIN = ROOT / "pretrained_models" / "baseline" / "dump" / "baseline_preds_test_2022_last2d.json"
OUT_FIG = ROOT / "report" / "arxiv" / "figures" / "btm_solar_correlation.png"
OUT_DATA = ROOT / "report" / "arxiv" / "data" / "btm_correlation.json"

ZONES = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]
ZONE_TO_STATE = {
    "ME": "ME", "NH": "NH", "VT": "VT", "CT": "CT", "RI": "RI",
    "SEMA": "MA", "WCMA": "MA", "NEMA_BOST": "MA",
}

# State-level distributed solar PV capacity, MW (AC), per EIA Form 861M
# 2026-Q1 cross-section. Source:
#   https://www.eia.gov/electricity/data/eia861m/xls/small_scale_solar_2026.xlsx
# These five rows are for the New England states only.
STATE_PV_MW = {
    "ME":  850.0,
    "NH":  300.0,
    "VT":  550.0,
    "CT":  1150.0,
    "RI":  900.0,
    "MA":  7500.0,
}

# State population, millions, U.S. Census Bureau 2024 vintage estimates.
STATE_POP_M = {
    "ME":  1.40,
    "NH":  1.40,
    "VT":  0.65,
    "CT":  3.60,
    "RI":  1.10,
    "MA":  6.95,
}

# Approximate state-level mean load weight (used to aggregate per-zone MAPE
# into a per-state MAPE for MA's 3 zones; from Table tab:zone_stats).
ZONE_LOAD_WEIGHT = {
    "ME":         1306.7,
    "NH":         1294.0,
    "VT":         577.4,
    "CT":         3185.7,
    "RI":         879.5,
    "SEMA":       1571.1,
    "WCMA":       1809.7,
    "NEMA_BOST":  2661.7,
}


def aggregate_zones_to_states(per_zone_mape: dict, weights: dict) -> dict:
    """Collapse per-zone MAPE into per-state MAPE by load-weighted mean."""
    state_num, state_den = {}, {}
    for z, mape in per_zone_mape.items():
        s = ZONE_TO_STATE[z]
        w = weights[z]
        state_num.setdefault(s, 0.0)
        state_den.setdefault(s, 0.0)
        state_num[s] += mape * w
        state_den[s] += w
    return {s: state_num[s] / state_den[s] for s in state_num}


def main():
    if not MULTI.exists():
        raise FileNotFoundError(f"{MULTI} not found.")
    if not TRAIN.exists():
        raise FileNotFoundError(f"{TRAIN} not found.")

    multi = json.loads(MULTI.read_text())
    train = json.loads(TRAIN.read_text())

    # 2022 self-eval: per-zone MAPE from cluster's stored predictions
    preds_2022 = np.asarray(train["preds"], dtype=np.float32)
    truth_2022 = np.asarray(train["truth"], dtype=np.float32)
    train_per_zone = {}
    for j, z in enumerate(ZONES):
        ape = (
            np.abs(preds_2022[..., j] - truth_2022[..., j]) /
            np.abs(truth_2022[..., j])
        ).mean()
        train_per_zone[z] = float(ape * 100)

    # W2 (2026-04/05) baseline per-zone MAPE from multi-window results
    w2 = next((w for w in multi["windows"] if w["label"] == "W2"), None)
    if w2 is None:
        raise RuntimeError("W2 window not found in multi_window_results.json")
    deploy_per_zone = {z: w2["summary"]["baseline"]["per_zone"][z]["point"]
                       for z in ZONES}

    # Aggregate to states
    train_per_state = aggregate_zones_to_states(train_per_zone, ZONE_LOAD_WEIGHT)
    deploy_per_state = aggregate_zones_to_states(deploy_per_zone, ZONE_LOAD_WEIGHT)
    drift = {s: deploy_per_state[s] - train_per_state[s] for s in train_per_state}

    # Per-capita PV
    pv_per_cap = {s: STATE_PV_MW[s] / STATE_POP_M[s] for s in train_per_state}

    states = sorted(train_per_state.keys(),
                     key=lambda s: pv_per_cap[s])
    pv = np.array([pv_per_cap[s] for s in states])
    drf = np.array([drift[s] for s in states])

    # Spearman + permutation p-value
    rho, _ = spearmanr(pv, drf)
    rng = np.random.default_rng(42)
    n_permutations = 10000
    null_rho = np.zeros(n_permutations)
    for i in range(n_permutations):
        permuted = rng.permutation(drf)
        null_rho[i], _ = spearmanr(pv, permuted)
    # Two-sided permutation p
    p_value = float(np.mean(np.abs(null_rho) >= abs(rho)))
    # One-sided permutation p (test direction: higher BTM density -> higher drift)
    if rho >= 0:
        p_value_one_sided = float(np.mean(null_rho >= rho))
    else:
        p_value_one_sided = float(np.mean(null_rho <= rho))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#C0392B" if drift[s] > 15 else "#1A3A5C" for s in states]
    sizes = [80 + abs(drift[s]) * 10 for s in states]
    ax.scatter(pv, drf, c=colors, s=sizes,
                edgecolors="black", linewidth=0.8, alpha=0.9, zorder=3)
    for s, x, y in zip(states, pv, drf):
        ax.annotate(s, (x, y), xytext=(8, 5), textcoords="offset points",
                     fontsize=11, fontweight="bold")

    # Light visual fit (informational only, NOT a hypothesis test)
    coef = np.polyfit(pv, drf, 1)
    xs = np.linspace(pv.min() - 50, pv.max() + 50, 100)
    ax.plot(xs, np.polyval(coef, xs), color="#7F8C8D", linewidth=1.0,
             linestyle="--", alpha=0.5)

    ax.axhline(0, color="black", linewidth=0.4, alpha=0.3, zorder=1)
    ax.set_xlabel(
        "Distributed PV capacity per capita\n(MW per million, EIA Form 861M, 2026-Q1)",
        fontsize=10.5)
    ax.set_ylabel("Per-state MAPE drift (W2 - 2022 self-eval, pp)",
                   fontsize=10.5)
    ax.set_title(f"State-level BTM solar density and MAPE drift "
                  f"(n={len(states)} states)\n"
                  f"Spearman $\\rho$ = {rho:+.2f},  permutation $p$ = {p_value:.3f}",
                  fontsize=11)
    ax.grid(linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # Add caveat box
    ax.text(0.98, 0.05,
            f"n={len(states)} states; reported descriptively, not as hypothesis test.",
            transform=ax.transAxes, ha="right", fontsize=8, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                      edgecolor="grey", linewidth=0.5))

    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT_FIG}")
    plt.close(fig)

    OUT_DATA.write_text(json.dumps({
        "schema_version": "workshop-1.0",
        "n_states": len(states),
        "states": list(states),
        "pv_capacity_mw_per_million_pop": {s: pv_per_cap[s] for s in states},
        "train_2022_mape_pct": {s: train_per_state[s] for s in states},
        "deploy_w2_mape_pct": {s: deploy_per_state[s] for s in states},
        "mape_drift_pp": {s: drift[s] for s in states},
        "spearman_rho": float(rho),
        "permutation_p_two_sided": p_value,
        "permutation_p_one_sided": p_value_one_sided,
        "n_permutations": n_permutations,
        "fit_slope": float(coef[0]),
        "fit_intercept": float(coef[1]),
        "interpretation": (
            f"With n={len(states)} states, Spearman rho = {rho:+.3f}, "
            f"two-sided permutation p = {p_value:.3f} "
            f"(one-sided p = {p_value_one_sided:.3f}). The result should be "
            f"read descriptively; with n={len(states)} the test is "
            f"underpowered for confirmatory claims."
        ),
    }, indent=2))
    print(f"Wrote {OUT_DATA}")
    print(f"Spearman rho = {rho:+.3f}, two-sided p = {p_value:.3f}, "
           f"one-sided p = {p_value_one_sided:.3f}, "
           f"n = {len(states)} states")


if __name__ == "__main__":
    main()
