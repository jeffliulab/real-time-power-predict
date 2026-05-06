"""
Workshop §4 — Figure 2: Per-zone MAPE for the 3 trained-model variants
across 2 windows.

8 zones × 3 model variants (baseline / Chronos / ensemble) × 2 windows
= 48 bars, grouped by zone with model variant as inner-hue and window as
outer-hue. The "ME / NEMA-Boston stay flat, RI / SEMA / WCMA explode"
story renders here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "report" / "arxiv" / "data" / "multi_window_results.json"
OUT = ROOT / "report" / "arxiv" / "figures" / "multi_window_per_zone_mape.png"

ZONES = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]
ZONE_LABELS = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA-Bos"]
MODELS = ["baseline", "chronos", "ensemble"]
MODEL_LABELS = ["Baseline", "Chronos", "Ensemble"]
WINDOW_HATCHES = {"W1": "", "W2": "//"}


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"{DATA} not found.")
    d = json.loads(DATA.read_text())

    n_zones = len(ZONES)
    n_models = len(MODELS)
    n_windows = len(d["windows"])
    width = 0.12
    inner = np.arange(n_models * n_windows) - (n_models * n_windows - 1) / 2
    inner = inner * width

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x_zones = np.arange(n_zones)

    palette = {
        "baseline": "#1A3A5C",
        "chronos":  "#16A085",
        "ensemble": "#2E86DE",
    }

    for w_idx, w in enumerate(d["windows"]):
        s = w["summary"]
        for m_idx, m in enumerate(MODELS):
            pos = w_idx * n_models + m_idx
            offset = inner[pos]
            points = [s[m]["per_zone"][z]["point"] for z in ZONES]
            lo = [s[m]["per_zone"][z]["ci_low"] for z in ZONES]
            hi = [s[m]["per_zone"][z]["ci_high"] for z in ZONES]
            yerr = [
                np.array(points) - np.array(lo),
                np.array(hi) - np.array(points),
            ]
            label = f"{MODEL_LABELS[m_idx]} ({w['label']})"
            ax.bar(x_zones + offset, points, width,
                    color=palette[m], edgecolor="black", linewidth=0.4,
                    hatch=WINDOW_HATCHES.get(w["label"], ""),
                    alpha=0.95 if w_idx == n_windows - 1 else 0.55,
                    label=label)
            ax.errorbar(x_zones + offset, points, yerr=yerr,
                         fmt="none", ecolor="black", capsize=2, linewidth=0.5)

    ax.set_xticks(x_zones)
    ax.set_xticklabels(ZONE_LABELS, fontsize=10)
    ax.set_ylabel("Per-zone MAPE (%) [95 % CI]", fontsize=10)
    ax.set_title("Per-zone MAPE across two deployment windows.  "
                  "Hatched bars = W2 (2026); solid = W1 (2025).",
                  fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8.5, ncol=2, framealpha=0.95)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
