"""
Daily load curve overlay: high-BTM zone (WCMA) vs low-BTM zone (ME), W1 vs W2.

The hour-of-day MAPE figure shows the baseline fails preferentially during
midday in high-BTM zones in W2. This figure shows the *physical reason*: the
midday demand trough deepens dramatically year-over-year in WCMA but stays
flat in ME, because behind-the-meter PV is suppressing the meter-read load
in WCMA but not in ME.

We plot the mean diurnal demand profile (averaged over the 7 forecast days
in each window) for two zones × two windows, with the baseline + Chronos
predictions overlaid for the W2 panels so the reader can see both:
  (a) the duck curve has formed in WCMA between W1 and W2,
  (b) the trained baseline did NOT capture this; Chronos fits the overall
      shape better but still over-predicts the midday trough.

Inputs : report/arxiv/data/multi_window_results.json
Outputs: report/arxiv/figures/load_curves_btm_signature.png
         report/arxiv/data/load_curves_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "report" / "arxiv" / "data" / "multi_window_results.json"
OUT_FIG = ROOT / "report" / "arxiv" / "figures" / "load_curves_btm_signature.png"
OUT_DATA = ROOT / "report" / "arxiv" / "data" / "load_curves_summary.json"

ZONES = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]
ZONE_TO_IDX = {z: i for i, z in enumerate(ZONES)}

HIGH_BTM = "WCMA"   # Massachusetts, dense rooftop PV adoption
LOW_BTM = "ME"      # Maine, very low BTM density


def _stack_window(forecasts: list, key: str) -> np.ndarray:
    return np.stack([np.array(f[key]) if key == "truth_24h"
                      else np.array(f["predictions"][key])
                      for f in forecasts], axis=0)


def main() -> None:
    multi = json.loads(INPUT.read_text())
    windows = {w["label"]: w for w in multi["windows"]}

    # Per (window, zone): mean diurnal truth + baseline-pred + chronos-pred
    rows = {}
    for wlabel, w in windows.items():
        truths = _stack_window(w["forecasts"], "truth_24h")
        baselines = _stack_window(w["forecasts"], "baseline")
        chronos = _stack_window(w["forecasts"], "chronos")
        for zone in [HIGH_BTM, LOW_BTM]:
            zi = ZONE_TO_IDX[zone]
            rows[(wlabel, zone)] = {
                "truth_mean": truths[:, :, zi].mean(axis=0),
                "truth_std": truths[:, :, zi].std(axis=0),
                "baseline_mean": baselines[:, :, zi].mean(axis=0),
                "chronos_mean": chronos[:, :, zi].mean(axis=0),
            }

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.0),
                              constrained_layout=True, sharex=True)
    hours = np.arange(24)

    for i, zone in enumerate([HIGH_BTM, LOW_BTM]):
        for j, wlabel in enumerate(["W1", "W2"]):
            ax = axes[i, j]
            r = rows[(wlabel, zone)]
            ax.fill_between(hours, r["truth_mean"] - r["truth_std"],
                              r["truth_mean"] + r["truth_std"],
                              color="#1A3A5C", alpha=0.18,
                              label="actual demand ±1σ" if (i == 0 and j == 0) else None)
            ax.plot(hours, r["truth_mean"],
                     color="#1A3A5C", linewidth=2.4, marker="o",
                     markersize=3, label="actual demand")
            ax.plot(hours, r["baseline_mean"],
                     color="#C0392B", linewidth=1.6, linestyle="--",
                     label="baseline pred")
            ax.plot(hours, r["chronos_mean"],
                     color="#2C7BB6", linewidth=1.6, linestyle=":",
                     label="Chronos pred")
            zone_name = zone.replace("_", "\\_") if "_" in zone else zone
            btm_label = "high BTM" if zone == HIGH_BTM else "low BTM"
            ax.set_title(f"{zone} ({btm_label}) — {wlabel}", fontsize=10.5)
            ax.set_ylabel("Demand (MWh/h)", fontsize=10)
            if i == 1:
                ax.set_xlabel("Hour of day (UTC)", fontsize=10)
            ax.set_xticks([0, 6, 12, 18, 23])
            ax.set_xticklabels(["00", "06", "12", "18", "23"], fontsize=9)
            ax.grid(linestyle="--", alpha=0.3)
            ax.set_axisbelow(True)
            if i == 0 and j == 0:
                ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    fig.suptitle(
        "Mean diurnal demand: high-BTM zone WCMA shows a deepening midday trough "
        "between W1 (May 2025) and W2 (Apr-May 2026), absent in low-BTM ME. "
        "The baseline (red dashed) is calibrated to the 2019-2020 weather→demand "
        "mapping and over-predicts the midday W2 WCMA load.",
        fontsize=10.5, y=1.02)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT_FIG}")
    plt.close(fig)

    # Quantify the duck-curve depth: (peak - midday_trough) / peak.
    summary = {
        "schema_version": "v1.6-load-curves",
        "high_btm_zone": HIGH_BTM,
        "low_btm_zone": LOW_BTM,
        "midday_hours_utc": list(range(14, 22)),  # 10-17 ET
        "evening_peak_hours_utc": list(range(22, 24)) + list(range(0, 4)),
    }
    duck_results = {}
    for (wlabel, zone), r in rows.items():
        truth = r["truth_mean"]
        midday = float(truth[14:22].mean())
        peak = float(truth.max())
        morning = float(truth[10:14].mean())
        depth = (peak - midday) / peak * 100.0
        duck_results[f"{wlabel}_{zone}"] = {
            "peak_mwh": peak,
            "midday_trough_mwh": midday,
            "morning_mwh": morning,
            "duck_curve_depth_pct": depth,
        }
    summary["duck_curve_depth"] = duck_results

    OUT_DATA.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_DATA}")
    print()
    print("Duck-curve depth (peak-trough)/peak:")
    for k, v in duck_results.items():
        print(f"  {k}: {v['duck_curve_depth_pct']:.1f}% "
               f"(peak {v['peak_mwh']:.0f} MWh, midday {v['midday_trough_mwh']:.0f} MWh)")


if __name__ == "__main__":
    main()
