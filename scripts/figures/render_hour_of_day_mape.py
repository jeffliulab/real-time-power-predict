"""
Hour-of-day MAPE decomposition: directly shows where BTM solar bites.

Reads multi_window_results.json. For each (window, model, zone, hour-of-day),
computes MAPE across the 7 forecast days in that window, then renders a heatmap.

Why this figure: the BTM-solar hypothesis predicts that the trained baseline
(weather-aware, learned the 2019-2020 weather→demand mapping) should fail
preferentially during midday hours when behind-the-meter PV is generating
and depressing the net load that the meter reads. In contrast, the
weather-agnostic Chronos foundation model just sees the demand series and
should NOT exhibit a midday-specific failure.

If the hypothesis is right we expect:
- W1 (May 2025) baseline: small, roughly flat error across hours.
- W2 (May 2026) baseline: large error concentrated in hours 14-22 UTC
  (10-18 ET), the BTM-solar generation window.
- W2 Chronos: error magnitude grows but distributed across hours.
- High-BTM zones (SEMA, WCMA, RI) show this pattern; low-BTM (ME, NH, VT) do not.

Forecasts start at 00:00 UTC, so hour-index h directly maps to UTC hour-of-day.
Eastern Time (EDT in late spring) = UTC - 4.

Outputs:
    report/arxiv/figures/hour_of_day_mape.png   — 4-panel heatmap (W1/W2 × baseline/Chronos)
    report/arxiv/data/hour_of_day_mape.json     — raw per-(window, model, zone, hour) MAPE numbers
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "report" / "arxiv" / "data" / "multi_window_results.json"
OUT_FIG = ROOT / "report" / "arxiv" / "figures" / "hour_of_day_mape.png"
OUT_DATA = ROOT / "report" / "arxiv" / "data" / "hour_of_day_mape.json"

ZONES = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]
ZONE_BTM_RANK = {"ME": "low", "NH": "low", "VT": "low",
                 "CT": "mid", "NEMA_BOST": "mid",
                 "RI": "high", "SEMA": "high", "WCMA": "high"}
MODELS_TO_PLOT = ["baseline", "chronos", "ensemble"]
MODEL_DISPLAY = {"baseline": "Baseline (CNN-Transformer + HRRR)",
                  "chronos": "Chronos-Bolt-mini (zero-shot)",
                  "ensemble": "Ensemble (per-zone $\\alpha$)"}


def per_hour_zone_mape(forecasts: list, model: str) -> np.ndarray:
    """For a window and a model, return a (24, 8) array of MAPE.

    MAPE[h, z] = 100 * mean over forecasts of |pred[h,z] - truth[h,z]| / |truth[h,z]|.
    """
    truths = np.stack([np.array(f["truth_24h"]) for f in forecasts], axis=0)
    preds = np.stack([np.array(f["predictions"][model]) for f in forecasts], axis=0)
    ape = np.abs(preds - truths) / np.abs(truths)
    return 100.0 * ape.mean(axis=0)


def main() -> None:
    multi = json.loads(INPUT.read_text())
    windows = {w["label"]: w for w in multi["windows"]}

    # Compute per-(window, model, zone, hour) MAPE.
    out: dict = {"schema_version": "v1.6-hour-of-day", "windows": {}}
    panels: dict = {}
    for wlabel, w in windows.items():
        out["windows"][wlabel] = {"models": {}}
        panels[wlabel] = {}
        for m in MODELS_TO_PLOT:
            mape = per_hour_zone_mape(w["forecasts"], m)
            panels[wlabel][m] = mape
            out["windows"][wlabel]["models"][m] = {
                "shape": "(hour=24, zone=8)",
                "zones": ZONES,
                "mape_pct": mape.tolist(),
                "overall_mape_pct": float(mape.mean()),
            }

    # Pick a shared colour scale so panels compare visually.
    all_vals = np.stack([panels[w][m] for w in panels for m in MODELS_TO_PLOT])
    vmax = float(np.percentile(all_vals, 95))
    vmin = 0.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = "magma_r"

    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5),
                              constrained_layout=True, sharex=True, sharey=True)
    for i, wlabel in enumerate(["W1", "W2"]):
        for j, m in enumerate(MODELS_TO_PLOT):
            ax = axes[i, j]
            mape = panels[wlabel][m]
            im = ax.imshow(mape.T, aspect="auto", origin="lower",
                            cmap=cmap, norm=norm,
                            extent=[-0.5, 23.5, -0.5, 7.5])
            ax.set_yticks(range(8))
            ax.set_yticklabels([f"{z}" for z in ZONES], fontsize=9)
            ax.set_xticks([0, 6, 12, 18, 23])
            ax.set_xticklabels(["00", "06", "12", "18", "23"], fontsize=9)
            if i == 0:
                ax.set_title(MODEL_DISPLAY[m], fontsize=10.5)
            if j == 0:
                ax.set_ylabel(f"{wlabel}\nzone", fontsize=10.5)
            if i == 1:
                ax.set_xlabel("Hour of day (UTC)", fontsize=10)
            ax.tick_params(axis="both", which="both", length=2)

            # Annotate hottest cells.
            zmax_idx = np.unravel_index(np.argmax(mape), mape.shape)
            ax.scatter(zmax_idx[0], zmax_idx[1],
                        marker="x", c="white", s=22, linewidths=1.0)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                         shrink=0.85, pad=0.02, fraction=0.025)
    cbar.set_label("MAPE (%)", fontsize=10)

    fig.suptitle(
        "Hour-of-day MAPE by zone, W1 (May 2025) vs W2 (Apr-May 2026)\n"
        "Baseline error in W2 concentrates at midday in the high-BTM zones "
        "(SEMA, WCMA, RI); Chronos drift is more diffuse across hours",
        fontsize=11.5)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT_FIG}")
    plt.close(fig)

    # Compute headline numbers for the paper text.
    # Midday-vs-rest baseline degradation in high-BTM zones, W2.
    high_btm_idxs = [i for i, z in enumerate(ZONES) if ZONE_BTM_RANK[z] == "high"]
    midday_hours = list(range(14, 22))   # 14-21 UTC = 10-17 ET
    nonmidday_hours = [h for h in range(24) if h not in midday_hours]

    def _slice_mean(mape_2d, hours, zones):
        sub = mape_2d[np.ix_(hours, zones)]
        return float(sub.mean())

    summary = {
        "midday_hours_utc": midday_hours,
        "high_btm_zones": [ZONES[i] for i in high_btm_idxs],
        "results": {}
    }
    for wlabel in ["W1", "W2"]:
        summary["results"][wlabel] = {}
        for m in MODELS_TO_PLOT:
            mape = panels[wlabel][m]
            summary["results"][wlabel][m] = {
                "midday_high_btm_mape_pct":
                    _slice_mean(mape, midday_hours, high_btm_idxs),
                "nonmidday_high_btm_mape_pct":
                    _slice_mean(mape, nonmidday_hours, high_btm_idxs),
                "midday_low_btm_mape_pct":
                    _slice_mean(mape,
                                 midday_hours,
                                 [i for i, z in enumerate(ZONES)
                                  if ZONE_BTM_RANK[z] == "low"]),
                "nonmidday_low_btm_mape_pct":
                    _slice_mean(mape,
                                 nonmidday_hours,
                                 [i for i, z in enumerate(ZONES)
                                  if ZONE_BTM_RANK[z] == "low"]),
            }
    out["summary"] = summary

    OUT_DATA.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_DATA}")
    print()
    print("Headline summary (high-BTM zones, W2):")
    r = summary["results"]["W2"]["baseline"]
    print(f"  Baseline midday MAPE   : {r['midday_high_btm_mape_pct']:.1f}%")
    print(f"  Baseline non-midday MAPE: {r['nonmidday_high_btm_mape_pct']:.1f}%")
    print(f"  midday/nonmidday ratio : "
           f"{r['midday_high_btm_mape_pct']/r['nonmidday_high_btm_mape_pct']:.2f}")
    r = summary["results"]["W2"]["chronos"]
    print(f"  Chronos midday MAPE    : {r['midday_high_btm_mape_pct']:.1f}%")
    print(f"  Chronos non-midday MAPE: {r['nonmidday_high_btm_mape_pct']:.1f}%")
    print(f"  midday/nonmidday ratio : "
           f"{r['midday_high_btm_mape_pct']/r['nonmidday_high_btm_mape_pct']:.2f}")


if __name__ == "__main__":
    main()
