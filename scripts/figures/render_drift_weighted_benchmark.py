"""
Figure 6: drift-weighted ensemble benchmark — frozen vs drift-weighted
vs oracle MAPE across W0, W1, W2.

Reads drift_weighted_benchmark.json (produced by
scripts/experiments/drift_weighted_ensemble.py) and renders two panels:

- Left: overall MAPE per window, grouped by mode (frozen / drift-weighted /
  oracle). Shows that drift-weighted closes most of the gap between
  frozen-production and an in-window oracle upper bound.
- Right: per-zone MAPE in W2 for the three modes, sorted by frozen-alpha
  MAPE descending. Shows that the gain is concentrated in the zones where
  the trained baseline is most degraded.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "report" / "arxiv" / "data" / "drift_weighted_benchmark.json"
OUT_FIG = ROOT / "report" / "arxiv" / "figures" / "drift_weighted_benchmark.png"

ZONES = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]
MODES = ["frozen", "drift_weighted", "oracle"]
MODE_LABEL = {"frozen": "frozen (production)",
              "drift_weighted": "drift-weighted (ours)",
              "oracle": "oracle (upper bound)"}
MODE_COLORS = {"frozen": "#C0392B",
               "drift_weighted": "#27AE60",
               "oracle": "#7F8C8D"}


def main() -> None:
    if not INPUT.exists():
        raise SystemExit(f"Expected {INPUT}; run drift_weighted_ensemble.py first.")
    b = json.loads(INPUT.read_text())
    windows = b["windows"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                              gridspec_kw={"width_ratios": [1.3, 2]},
                              constrained_layout=True)

    # LEFT panel: overall MAPE per window x mode
    labels = list(windows.keys())  # W0, W1, W2
    x = np.arange(len(labels))
    bar_w = 0.27
    for i, mode in enumerate(MODES):
        vals = [windows[lbl][mode]["overall_mape_pct"] for lbl in labels]
        offset = (i - 1) * bar_w
        bars = axes[0].bar(x + offset, vals, width=bar_w,
                            color=MODE_COLORS[mode], label=MODE_LABEL[mode],
                            edgecolor="black", linewidth=0.5)
        for bx, v in zip(bars, vals):
            axes[0].text(bx.get_x() + bx.get_width() / 2,
                          bx.get_height() + 0.3,
                          f"{v:.1f}", ha="center", fontsize=8.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=10.5)
    axes[0].set_ylabel("Overall MAPE (%)", fontsize=10.5)
    axes[0].set_title("Overall MAPE per window × mode", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.92)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)
    axes[0].set_axisbelow(True)

    # RIGHT panel: per-zone MAPE in W2 for the three modes
    if "W2" in windows:
        w2 = windows["W2"]
        # Sort by frozen-mode MAPE descending
        per_zone_frozen = w2["frozen"]["per_zone_mape_pct"]
        sorted_zones = sorted(ZONES, key=lambda z: per_zone_frozen[z],
                                reverse=True)
        xz = np.arange(len(sorted_zones))
        bar_w2 = 0.27
        for i, mode in enumerate(MODES):
            pz = w2[mode]["per_zone_mape_pct"]
            vals = [pz[z] for z in sorted_zones]
            offset = (i - 1) * bar_w2
            axes[1].bar(xz + offset, vals, width=bar_w2,
                         color=MODE_COLORS[mode], label=MODE_LABEL[mode],
                         edgecolor="black", linewidth=0.4)
        axes[1].set_xticks(xz)
        axes[1].set_xticklabels([z.replace("_", "\n") for z in sorted_zones],
                                  fontsize=9)
        axes[1].set_ylabel("MAPE in W2 (%)", fontsize=10.5)
        axes[1].set_title("W2 per-zone MAPE (sorted by frozen, descending)",
                           fontsize=11)
        axes[1].legend(loc="upper right", fontsize=9, framealpha=0.92)
        axes[1].grid(axis="y", linestyle="--", alpha=0.3)
        axes[1].set_axisbelow(True)
    else:
        axes[1].text(0.5, 0.5, "W2 not present in benchmark JSON",
                      transform=axes[1].transAxes, ha="center")

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT_FIG}")
    plt.close(fig)


if __name__ == "__main__":
    main()
