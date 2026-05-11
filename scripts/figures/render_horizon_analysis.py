"""
Forecast-horizon analysis: does drift hit hour-1 (near-term) the same as
hour-24 (far-term)?

Hypothesis: a foundation model conditioned on recent history (Chronos)
should be relatively immune to drift at very short horizons (hour-1 sees
the most recent observation, so any persistence-like signal it picks up
is by definition current). The bespoke baseline, with no history-window
adaptation, should drift more uniformly across horizons. If the drift
gap (W2 - W1) shrinks at the short end for Chronos but stays flat for
baseline, that's another piece of evidence that the deployed bespoke
model is the one losing ground.

Inputs : report/arxiv/data/multi_window_results.json
Outputs: report/arxiv/figures/horizon_analysis.png
         report/arxiv/data/horizon_analysis.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "report" / "arxiv" / "data" / "multi_window_results.json"
OUT_FIG = ROOT / "report" / "arxiv" / "figures" / "horizon_analysis.png"
OUT_DATA = ROOT / "report" / "arxiv" / "data" / "horizon_analysis.json"

MODELS = ["baseline", "chronos", "ensemble"]
MODEL_DISPLAY = {"baseline": "Baseline", "chronos": "Chronos",
                  "ensemble": "Ensemble"}
COLORS = {"baseline": "#C0392B", "chronos": "#2C7BB6", "ensemble": "#27AE60"}
LINESTYLES = {"W1": "-", "W2": "--"}


def _per_horizon_mape(forecasts: list, model: str) -> np.ndarray:
    """Return shape (24,) — MAPE per forecast horizon, averaged over days and zones."""
    truths = np.stack([np.array(f["truth_24h"]) for f in forecasts], axis=0)   # (n,24,8)
    preds = np.stack([np.array(f["predictions"][model]) for f in forecasts], axis=0)
    ape = np.abs(preds - truths) / np.abs(truths)
    return 100.0 * ape.mean(axis=(0, 2))   # average over forecasts and zones


def main() -> None:
    multi = json.loads(INPUT.read_text())
    windows = {w["label"]: w for w in multi["windows"]}

    out = {"schema_version": "v1.6-horizon-analysis",
            "windows": {}}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                              constrained_layout=True, sharey=True)
    horizons = np.arange(1, 25)

    for i, m in enumerate(MODELS):
        for wlabel in ["W1", "W2"]:
            mape = _per_horizon_mape(windows[wlabel]["forecasts"], m)
            out["windows"].setdefault(wlabel, {})[m] = mape.tolist()
            axes[0].plot(horizons, mape,
                          color=COLORS[m], linestyle=LINESTYLES[wlabel],
                          linewidth=1.6, marker="o", markersize=3,
                          label=f"{MODEL_DISPLAY[m]} ({wlabel})")

        # Right panel: drift = W2 - W1 by horizon
        diff = np.array(out["windows"]["W2"][m]) - np.array(out["windows"]["W1"][m])
        axes[1].plot(horizons, diff,
                      color=COLORS[m], linewidth=1.8, marker="o", markersize=3,
                      label=MODEL_DISPLAY[m])

    axes[0].set_xlabel("Forecast horizon (hours ahead)", fontsize=10.5)
    axes[0].set_ylabel("Mean MAPE across zones (%)", fontsize=10.5)
    axes[0].set_title("Per-horizon MAPE, both windows", fontsize=11)
    axes[0].grid(linestyle="--", alpha=0.3)
    axes[0].set_axisbelow(True)
    axes[0].legend(loc="upper left", fontsize=8.5, framealpha=0.92, ncol=2)
    axes[0].set_xticks([1, 6, 12, 18, 24])

    axes[1].axhline(0, color="black", linewidth=0.6, alpha=0.4)
    axes[1].set_xlabel("Forecast horizon (hours ahead)", fontsize=10.5)
    axes[1].set_ylabel("W2 - W1 MAPE drift (pp)", fontsize=10.5)
    axes[1].set_title("Drift (W2 - W1) by horizon", fontsize=11)
    axes[1].grid(linestyle="--", alpha=0.3)
    axes[1].set_axisbelow(True)
    axes[1].legend(loc="upper left", fontsize=9, framealpha=0.92)
    axes[1].set_xticks([1, 6, 12, 18, 24])

    fig.suptitle(
        "Forecast-horizon MAPE and drift across both windows",
        fontsize=11.5, y=1.04)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT_FIG}")
    plt.close(fig)

    # Headline numbers
    summary = {}
    for m in MODELS:
        h1 = np.array(out["windows"]["W2"][m])[0] - np.array(out["windows"]["W1"][m])[0]
        h12 = np.array(out["windows"]["W2"][m])[11] - np.array(out["windows"]["W1"][m])[11]
        h24 = np.array(out["windows"]["W2"][m])[23] - np.array(out["windows"]["W1"][m])[23]
        summary[m] = {
            "drift_h1_pp": float(h1),
            "drift_h12_pp": float(h12),
            "drift_h24_pp": float(h24),
            "drift_mean_pp": float((np.array(out["windows"]["W2"][m]) -
                                     np.array(out["windows"]["W1"][m])).mean()),
        }
    out["summary"] = summary

    OUT_DATA.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_DATA}")
    print()
    print("Drift (W2 - W1) at selected horizons (pp):")
    for m, r in summary.items():
        print(f"  {m:9s}: h+1 = {r['drift_h1_pp']:+5.2f},  "
               f"h+12 = {r['drift_h12_pp']:+5.2f},  "
               f"h+24 = {r['drift_h24_pp']:+5.2f},  "
               f"mean = {r['drift_mean_pp']:+5.2f}")


if __name__ == "__main__":
    main()
