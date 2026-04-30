"""
Per-zone test MAPE bar chart for the Part 1 baseline.
"""

from pathlib import Path

import matplotlib.pyplot as plt

NAVY = "#1A3A5C"
ACCENT = "#2E86DE"
GOOD = "#1B8E4F"
BAD = "#C0392B"
INK = "#1F2937"

# Per-zone test MAPE (last 2 days of 2022) — Part 1 baseline.
ZONES = [
    ("ME",        2.31),
    ("NH",        3.69),
    ("VT",        5.95),
    ("CT",        7.28),
    ("RI",        5.27),
    ("SEMA",      5.44),
    ("WCMA",      5.87),
    ("NEMA_BOST", 6.09),
]
OVERALL = 5.24


def main(out_path: Path) -> None:
    sorted_zones = sorted(ZONES, key=lambda r: r[1])
    labels = [r[0] for r in sorted_zones]
    values = [r[1] for r in sorted_zones]
    colors = [
        GOOD if v < OVERALL else (BAD if v > OVERALL + 1.0 else NAVY)
        for v in values
    ]

    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=180)
    bars = ax.barh(labels, values, color=colors, edgecolor=INK, linewidth=0.6)

    ax.axvline(OVERALL, color=ACCENT, linestyle="--", linewidth=1.2,
               label=f"Overall = {OVERALL:.2f}%")
    ax.set_xlabel("Test MAPE (%)", color=INK, fontsize=11)
    ax.set_title("Part 1 baseline — per-zone test MAPE\n(self-eval on 2022-12-30/31)",
                 color=NAVY, fontsize=12, fontweight="bold", pad=10)

    for bar, v in zip(bars, values):
        ax.text(v + 0.08, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=9.5, color=INK)

    ax.set_xlim(0, max(values) * 1.18)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", colors=INK)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    out = (Path(__file__).resolve().parents[2]
           / "report" / "figures" / "baseline_per_zone.png")
    main(out)
