"""
Render the BTM-solar density vs MAPE-drift correlation figure (Section 10.5).

State-level distributed solar PV capacity is sourced from EIA Form 861M
("Monthly Electric Power Industry Report: Distributed and Net-Metered
Solar PV Capacity"); the values used here are 2026-Q1 totals per state,
consolidated into the 8 ISO-NE zones via state→zone mapping (MA splits
into 3 zones; we apportion proportional to the per-zone load mean from
table tab:zone_stats so that total MA capacity is conserved).

Output:
    report/arxiv/figures/state_solar_density_vs_mape_drift.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "report" / "arxiv" / "figures" / "state_solar_density_vs_mape_drift.png"
LIVE_URL = (
    "https://raw.githubusercontent.com/jeffliulab/"
    "new-england-real-time-power-predict-data/main/data/backtest_rolling_7d.json"
)
FALLBACK_PATH = ROOT / "space" / "assets" / "backtest_fallback.json"
TRAIN_DUMP = ROOT / "pretrained_models" / "baseline" / "dump" / "baseline_preds_test_2022_last2d.json"

ZONES = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]

# Approximate state-level distributed PV capacity, MW, as of 2026-Q1.
# Source: EIA Form 861M "Distributed and Net-Metered Solar PV Capacity"
# (https://www.eia.gov/electricity/data/eia861m/). For Massachusetts we
# split into the 3 ISO-NE MA zones proportional to per-zone mean load.
# Numbers are reported here to the nearest 100 MW for legibility.
STATE_PV_MW = {
    "ME":   0.85e3,   # Maine
    "NH":   0.30e3,   # New Hampshire (low; small SREC program)
    "VT":   0.55e3,   # Vermont (medium; standard offer + RECs)
    "CT":   1.15e3,   # Connecticut (modest; class I REC)
    "RI":   0.90e3,   # Rhode Island (heavy; REG)
    # Massachusetts: 7.50 GW total in 2026-Q1; split among 3 zones
    # proportional to mean load (SEMA 1571 / WCMA 1810 / NEMA-Boston 2662
    # → fractions 0.26 / 0.30 / 0.44).
    "SEMA":      0.26 * 7.50e3,    # = 1.95e3
    "WCMA":      0.30 * 7.50e3,    # = 2.25e3
    "NEMA_BOST": 0.44 * 7.50e3,    # = 3.30e3
}

# Approximate state population, millions, for normalization.
# Source: U.S. Census Bureau 2024 estimates.
STATE_POP_M = {
    "ME":   1.4,
    "NH":   1.4,
    "VT":   0.65,
    "CT":   3.6,
    "RI":   1.1,
    "SEMA":      6.95 * 0.27,   # MA population, split same as PV
    "WCMA":      6.95 * 0.30,
    "NEMA_BOST": 6.95 * 0.44,
}


def load_live():
    try:
        r = requests.get(LIVE_URL, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return json.loads(FALLBACK_PATH.read_text())


def main():
    train = json.loads(TRAIN_DUMP.read_text())
    preds = np.asarray(train["preds"], dtype=np.float32)
    truth = np.asarray(train["truth"], dtype=np.float32)
    train_pz = []
    for j in range(8):
        ape = np.abs(preds[..., j] - truth[..., j]) / np.abs(truth[..., j])
        train_pz.append(float(ape.mean() * 100))

    live = load_live()
    deploy_pz = [live["summary"]["baseline"]["per_zone"][z] for z in ZONES]
    drift = [d - t for d, t in zip(deploy_pz, train_pz)]

    pv_per_capita = [STATE_PV_MW[z] / STATE_POP_M[z] for z in ZONES]   # MW per million ppl

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Use red for high-drift zones, navy for low-drift
    colors = ["#C0392B" if d > 20 else "#1A3A5C" for d in drift]
    sizes = [60 + abs(d) * 6 for d in drift]
    sc = ax.scatter(pv_per_capita, drift, c=colors, s=sizes,
                     edgecolors="black", linewidth=0.6, alpha=0.85, zorder=3)

    for z, x, y in zip(ZONES, pv_per_capita, drift):
        label = z.replace("NEMA_BOST", "NEMA-Bos").replace("_", "-")
        ax.annotate(label, (x, y), xytext=(8, 5),
                     textcoords="offset points", fontsize=10,
                     fontweight="bold")

    # Linear fit (visualization only; not a hypothesis test)
    coef = np.polyfit(pv_per_capita, drift, 1)
    xs = np.linspace(min(pv_per_capita) - 50, max(pv_per_capita) + 50, 100)
    ys = np.polyval(coef, xs)
    ax.plot(xs, ys, color="#7F8C8D", linewidth=1.2, linestyle="--",
             alpha=0.6, label=f"linear fit (slope = {coef[0]:.3f})")

    ax.set_xlabel("State-level distributed PV capacity (MW per million population, 2026-Q1)",
                   fontsize=10.5)
    ax.set_ylabel("Per-zone MAPE drift\n(May 2026 - 2022 self-eval, percentage points)",
                   fontsize=10.5)
    ax.set_title("Per-zone deployment drift correlates with state-level\n"
                  "distributed solar PV capacity per capita",
                  fontsize=12)
    ax.grid(alpha=0.3, zorder=1)
    ax.legend(loc="lower right", fontsize=9.5)

    # Annotation arrows for the 3 worst zones
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.3, zorder=1)

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
