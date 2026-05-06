"""
Workshop §4 — Figure 1: Overall MAPE by model variant across windows.

Reads report/arxiv/data/multi_window_results.json and produces a grouped
bar chart showing 6 model variants × 2 windows (W1=2025-05, W2=2026-04/05),
with bootstrap 95 % CI error bars.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "report" / "arxiv" / "data" / "multi_window_results.json"
OUT = ROOT / "report" / "arxiv" / "figures" / "multi_window_overall_mape.png"

MODEL_LABELS = {
    "baseline": "Baseline\n(CNN-Transformer)",
    "chronos": "Chronos-Bolt-mini\n(zero-shot)",
    "ensemble": "Ensemble\n(per-zone $\\alpha$)",
    "persistence_1d": "Persistence\n(lag-1d)",
    "persistence_7d": "Persistence\n(lag-7d)",
    "climatological": "Climatological\nmean (4 weeks)",
}
MODEL_ORDER = ["persistence_1d", "persistence_7d", "climatological",
                "baseline", "chronos", "ensemble"]
COLORS = {
    "W1": "#7F8C8D",
    "W2": "#C0392B",
}


def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"{DATA} not found. Run scripts/experiments/historical_drift_sweep.py first.")
    d = json.loads(DATA.read_text())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(MODEL_ORDER))
    width = 0.4

    for i, w in enumerate(d["windows"]):
        label = w["label"]
        offset = (i - 0.5) * width
        s = w["summary"]
        points = [s[m]["overall"]["point"] for m in MODEL_ORDER]
        lo = [s[m]["overall"]["ci_low"] for m in MODEL_ORDER]
        hi = [s[m]["overall"]["ci_high"] for m in MODEL_ORDER]
        yerr = [
            np.array(points) - np.array(lo),
            np.array(hi) - np.array(points),
        ]
        ax.bar(x + offset, points, width,
                color=COLORS.get(label, "#1A3A5C"),
                edgecolor="black", linewidth=0.5,
                label=f"{label} ({w['start_date']} to {w['end_date']})")
        ax.errorbar(x + offset, points, yerr=yerr,
                     fmt="none", ecolor="black", capsize=3, linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=9)
    ax.set_ylabel("Overall MAPE (%) [95 % CI]", fontsize=10)
    ax.set_title("Two-window deployment MAPE: trained models vs naive baselines\n"
                  "Bars = bootstrap point estimate; error bars = 95 % CI over (day, zone) pairs",
                  fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
