"""
Render the pipeline-equivalence overlay figure (Section 9).

Reads data/validation_dec30_2022.json (produced by
scripts/validation/reproduce_dec30_2022.py) and produces an 8-panel
figure (one per zone) with three traces: ground truth, cluster's stored
prediction, and the live-pipeline rerun.

Output:
    report/arxiv/figures/validation_dec30_2022_overlay.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "report" / "arxiv" / "data" / "validation_dec30_2022.json"
OUT_PATH = ROOT / "report" / "arxiv" / "figures" / "validation_dec30_2022_overlay.png"

NAVY = "#1A3A5C"
GREY = "#7F8C8D"
ACCENT = "#2E86DE"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run "
            f"scripts/validation/reproduce_dec30_2022.py first.")
    d = json.loads(DATA_PATH.read_text())
    zones = d["zones"]
    truth = np.asarray(d["truth"])
    cluster_pred = np.asarray(d["cluster_pred"])
    live_pred = np.asarray(d["live_pred"])

    fig, axs = plt.subplots(4, 2, figsize=(11, 9), sharex=True)
    hours = np.arange(24)

    for j, z in enumerate(zones):
        r, c = j // 2, j % 2
        ax = axs[r, c]
        ax.plot(hours, truth[:, j], color=NAVY, linewidth=2.0,
                label="ground truth")
        ax.plot(hours, cluster_pred[:, j], color=GREY, linewidth=1.4,
                linestyle="--", label="cluster pred (6.54 % MAPE)")
        ax.plot(hours, live_pred[:, j], color=ACCENT, linewidth=1.6,
                linestyle=":", label="live pipeline (6.41 % MAPE)")
        ax.set_title(z.replace("_", "-"), fontsize=10)
        ax.grid(alpha=0.3)
        if r == 3:
            ax.set_xlabel("Forecast hour", fontsize=9)
        if c == 0:
            ax.set_ylabel("MWh", fontsize=9)

    axs[0, 0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Pipeline-equivalence check on 2022-12-30: cluster's stored "
                  "prediction vs live deployment rerun\n"
                  f"Overall MAPE: cluster {d['cluster_overall_mape']:.2f}\\% vs "
                  f"live {d['live_overall_mape']:.2f}\\%, max element diff "
                  f"{d['max_abs_diff_mw']:.0f} MW ({d['max_rel_diff_pct']:.1f}\\%)",
                  fontsize=11)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
