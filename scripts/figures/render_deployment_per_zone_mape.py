"""
Render the per-zone MAPE bar chart for the live deployment window.

Reads the live JSON from the auxiliary data repo at:
    https://raw.githubusercontent.com/jeffliulab/new-england-real-time-power-predict-data/main/data/backtest_rolling_7d.json

If the network is unavailable, falls back to the bundled snapshot at
    space/assets/backtest_fallback.json

Output:
    report/arxiv/figures/deployment_per_zone_mape.png

Usage:
    python scripts/figures/render_deployment_per_zone_mape.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "report" / "arxiv" / "figures" / "deployment_per_zone_mape.png"
LIVE_URL = (
    "https://raw.githubusercontent.com/jeffliulab/"
    "new-england-real-time-power-predict-data/main/data/backtest_rolling_7d.json"
)
FALLBACK_PATH = ROOT / "space" / "assets" / "backtest_fallback.json"

ZONES = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]
ZONE_LABELS = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA-Boston"]
NAVY = "#1A3A5C"
TEAL = "#16A085"
ACCENT = "#2E86DE"


def load_backtest():
    try:
        r = requests.get(LIVE_URL, timeout=15)
        r.raise_for_status()
        print(f"Loaded live data from {LIVE_URL}")
        return r.json()
    except Exception as e:
        print(f"Live URL failed ({e}); falling back to bundled snapshot")
        return json.loads(FALLBACK_PATH.read_text())


def main():
    data = load_backtest()
    s = data["summary"]
    period = data.get("data_period", {})

    baseline_pz = [s["baseline"]["per_zone"][z] for z in ZONES]
    chronos_pz = [s["chronos"]["per_zone"][z] for z in ZONES]
    ensemble_pz = [s["ensemble"]["per_zone"][z] for z in ZONES]

    x = np.arange(len(ZONES))
    width = 0.27

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, baseline_pz, width, label="Baseline (real HRRR)",
            color=NAVY, edgecolor="black", linewidth=0.5)
    ax.bar(x, chronos_pz, width, label="Chronos-Bolt-mini (zero-shot)",
            color=TEAL, edgecolor="black", linewidth=0.5)
    ax.bar(x + width, ensemble_pz, width, label="Ensemble (per-zone $\\alpha$)",
            color=ACCENT, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(ZONE_LABELS)
    ax.set_ylabel("MAPE (%)", fontsize=11)
    title = (f"Live deployment per-zone MAPE\n"
             f"({period.get('first_forecast_start', '?')[:10]} to "
             f"{period.get('last_forecast_start', '?')[:10]}, "
             f"refreshed {data.get('built_at', '?')[:10]})")
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)

    # Annotate the catastrophic per-zone bars
    for i, val in enumerate(baseline_pz):
        if val > 30:
            ax.text(x[i] - width, val + 1, f"{val:.0f}%",
                    ha="center", va="bottom", fontsize=9, color="darkred",
                    fontweight="bold")

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
