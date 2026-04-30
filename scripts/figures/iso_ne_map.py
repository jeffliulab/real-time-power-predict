"""
Stylized ISO-NE 8-zone map for the title slide of the final presentation.

Approximates each load zone as a labeled rectangle in roughly its real
geographic position; intent is illustrative, not GIS-accurate.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

NAVY = "#1A3A5C"
ACCENT = "#2E86DE"
INK = "#1F2937"

# (zone_code, center_x, center_y, width, height, fill_color)
ZONES = [
    ("ME",        0.62, 0.82, 0.34, 0.24, "#dde7f0"),
    ("NH",        0.55, 0.62, 0.18, 0.18, "#dde7f0"),
    ("VT",        0.32, 0.62, 0.18, 0.18, "#dde7f0"),
    ("WCMA",      0.32, 0.40, 0.20, 0.16, "#cfe1ee"),
    ("NEMA_BOST", 0.55, 0.40, 0.20, 0.16, "#bcd2e3"),
    ("SEMA",      0.62, 0.24, 0.16, 0.14, "#bcd2e3"),
    ("RI",        0.45, 0.20, 0.10, 0.12, "#bcd2e3"),
    ("CT",        0.28, 0.20, 0.20, 0.14, "#cfe1ee"),
]


def main(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=180)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_axis_off()

    for code, cx, cy, w, h, fill in ZONES:
        rect = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.005,rounding_size=0.012",
            linewidth=1.2, edgecolor=NAVY, facecolor=fill,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, code, ha="center", va="center",
                fontsize=11, fontweight="bold", color=INK)

    # Compass rose (top-right)
    ax.annotate("N", xy=(0.94, 0.95), ha="center", fontsize=10,
                color=NAVY, fontweight="bold")
    ax.annotate("", xy=(0.94, 0.92), xytext=(0.94, 0.86),
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.0))

    # Title
    ax.set_title("ISO New England — 8 load zones",
                 color=NAVY, fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "report" / "figures" / "iso_ne_map.png"
    main(out)
