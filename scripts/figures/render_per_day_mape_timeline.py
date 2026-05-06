"""
Workshop §4 — Figure 3: Per-day MAPE timeline across both windows.

Computes daily MAPE for each model variant and plots them as a line
chart. Visualizes whether model-vs-model gaps survive day-to-day noise
and shows the W1 → W2 jump along a fictive shared time axis (since W1
and W2 are 1 year apart, they share calendar position only).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "report" / "arxiv" / "data" / "multi_window_results.json"
OUT = ROOT / "report" / "arxiv" / "figures" / "per_day_mape_timeline.png"

ZONES = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]
MODEL_ORDER = ["baseline", "chronos", "ensemble",
                "persistence_1d", "persistence_7d", "climatological"]
MODEL_LABELS = {
    "baseline": "Baseline",
    "chronos": "Chronos-Bolt-mini",
    "ensemble": "Ensemble",
    "persistence_1d": "Persistence-1d",
    "persistence_7d": "Persistence-7d",
    "climatological": "Climatological",
}
COLORS = {
    "baseline":       "#1A3A5C",
    "chronos":        "#16A085",
    "ensemble":       "#2E86DE",
    "persistence_1d": "#7F8C8D",
    "persistence_7d": "#95A5A6",
    "climatological": "#BDC3C7",
}


def daily_mape(forecasts: list[dict], model_key: str) -> tuple[list[str], list[float]]:
    """Per-day MAPE for one model across the window's forecasts."""
    days, mapes = [], []
    for f in forecasts:
        truth = np.asarray(f["truth_24h"], dtype=np.float32)
        pred = np.asarray(f["predictions"][model_key], dtype=np.float32)
        # mean over (24, 8) cells
        mask = truth != 0
        mape = float((np.abs(pred[mask] - truth[mask]) / np.abs(truth[mask])).mean() * 100)
        days.append(f["start"][:10])
        mapes.append(mape)
    return days, mapes


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"{DATA} not found.")
    d = json.loads(DATA.read_text())

    fig, axes = plt.subplots(1, len(d["windows"]), figsize=(13, 5),
                                sharey=True)
    if len(d["windows"]) == 1:
        axes = [axes]

    for ax, w in zip(axes, d["windows"]):
        for m in MODEL_ORDER:
            days, mapes = daily_mape(w["forecasts"], m)
            x = np.arange(len(days))
            lw = 2.2 if m in ("baseline", "chronos", "ensemble") else 1.0
            ls = "-" if m in ("baseline", "chronos", "ensemble") else "--"
            ax.plot(x, mapes, marker="o", linewidth=lw, linestyle=ls,
                    color=COLORS[m], label=MODEL_LABELS[m], alpha=0.95)
        ax.set_title(f"{w['label']}: {w['start_date']} - {w['end_date']}",
                      fontsize=11)
        ax.set_xticks(range(len(w["forecasts"])))
        ax.set_xticklabels([f["start"][5:10] for f in w["forecasts"]],
                            fontsize=9, rotation=30)
        ax.grid(linestyle="--", alpha=0.35)
        ax.set_xlabel("Forecast issue date (UTC)", fontsize=10)

    axes[0].set_ylabel("Daily MAPE (%)", fontsize=10)
    axes[-1].legend(loc="upper right", fontsize=8.5, framealpha=0.95,
                     bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
