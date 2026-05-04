"""
Re-render the 4 attention map figures with a real New England geographic
basemap (coastline + state boundaries) instead of the bare 8x8 grid that
the original `scripts/attention_maps.py` produces on HPC.

Reads .npz arrays from
    runs/cnn_transformer_baseline/figures/attention/
        aggregate.npz
        per_hour.npz
        per_zone.npz
        extreme_vs_mild.npz
        metadata.json
and writes
    report/figures/attn_aggregate.png
    report/figures/attn_per_hour.png
    report/figures/attn_per_zone.png
    report/figures/attn_extreme_vs_mild.png

No HPC dependency: once the .npz files are pulled back from one HPC run,
this renderer can be re-run locally with style tweaks at zero DUO cost.

Usage
-----
    .venv/bin/python scripts/render_attention_basemap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parent.parent
ATTN_DIR = ROOT / "runs/cnn_transformer_baseline/figures/attention"
OUT_DIR = ROOT / "report/figures"
ZONE_COLS = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]
GRID = 8
DPI = 200

# --- Style: match the report's TikZ palette ---------------------------
NAVY = "#1A3A5C"
ACCENT = "#2E86DE"
SUBINK = "#4B5563"
AMBER = "#C97B12"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.titleweight": "bold",
    "axes.titlecolor": NAVY,
})

# --- Geographic anchors -----------------------------------------------
# Approximate geographic centroid of each ISO-NE load zone.
# Used to label zones on the basemap.
ZONE_CENTROIDS = {
    "ME":         (45.4, -69.0),
    "NH":         (43.7, -71.6),
    "VT":         (44.0, -72.7),
    "MA-WCMA":    (42.4, -72.5),
    "MA-NEMA":    (42.4, -71.1),     # metro Boston
    "MA-SEMA":    (41.7, -70.7),     # SE Mass / Cape area
    "CT":         (41.6, -72.7),
    "RI":         (41.7, -71.5),
}
# Map ISO-NE zone code → label location for the basemap (skip overlapping ones)
ZONE_LABEL_POS = {
    "ME":        (45.4, -69.0),
    "NH":        (43.8, -71.6),
    "VT":        (44.0, -72.7),
    "WCMA":      (42.4, -72.5),
    "NEMA_BOST": (42.4, -71.1),
    "SEMA":      (41.65, -70.7),
    "CT":        (41.55, -72.7),
    "RI":        (41.7, -71.5),
}
# Major cities to anchor the reader's geographic intuition.
CITIES = [
    ("Boston",     42.36, -71.06),
    ("Hartford",   41.76, -72.68),
    ("Providence", 41.82, -71.41),
    ("Portland",   43.66, -70.26),
    ("Burlington", 44.48, -73.21),
]


def load_metadata():
    return json.loads((ATTN_DIR / "metadata.json").read_text())


def make_basemap_axes(figsize=(7.0, 6.4)):
    """Create a figure + Cartopy axes covering the New England bbox."""
    fig = plt.figure(figsize=figsize)
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([-74.0, -66.0, 40.5, 47.5], crs=proj)

    # Coastline + states
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"),
                   linewidth=0.8, edgecolor=NAVY, alpha=0.85)
    ax.add_feature(cfeature.STATES.with_scale("10m"),
                   linewidth=0.5, edgecolor=NAVY, alpha=0.55)
    ax.add_feature(cfeature.LAND.with_scale("10m"),
                   facecolor="#F7F4EC", alpha=0.4, zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"),
                   facecolor="#DCE8F0", alpha=0.5, zorder=0)

    # Lat/lon gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color=SUBINK,
                      alpha=0.3, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False
    return fig, ax


def overlay_attention(ax, attn_2d, vmin=None, vmax=None, alpha=0.62, cmap="magma"):
    """
    Plot the 8x8 attention as a coloured mesh on top of the basemap.
    The 8x8 grid spans (40.5, -74) to (47.5, -66): each cell is
    7°/8 = 0.875° latitude × 8°/8 = 1° longitude.
    Row 0 = north, col 0 = west.
    """
    lat_edges = np.linspace(47.5, 40.5, GRID + 1)   # row 0 at top (north)
    lon_edges = np.linspace(-74.0, -66.0, GRID + 1)
    lon_grid, lat_grid = np.meshgrid(lon_edges, lat_edges)

    if vmin is None:
        vmin = float(attn_2d.min())
    if vmax is None:
        vmax = float(attn_2d.max())

    mesh = ax.pcolormesh(lon_grid, lat_grid, attn_2d,
                         transform=ccrs.PlateCarree(),
                         cmap=cmap, vmin=vmin, vmax=vmax,
                         shading="auto", alpha=alpha, zorder=2)
    return mesh


def overlay_cities(ax):
    for name, lat, lon in CITIES:
        ax.plot(lon, lat, marker="o", markersize=3.5,
                color="#222", markerfacecolor="white",
                markeredgewidth=0.8, zorder=5,
                transform=ccrs.PlateCarree())
        ax.annotate(name, xy=(lon, lat), xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=7, color="#111",
                    bbox=dict(boxstyle="round,pad=0.18",
                              facecolor="white", edgecolor="none",
                              alpha=0.7),
                    zorder=6,
                    transform=None)


def overlay_zone_labels(ax, alpha=0.65):
    for zone, (lat, lon) in ZONE_LABEL_POS.items():
        ax.text(lon, lat, zone, ha="center", va="center",
                fontsize=8, fontweight="bold", color=NAVY, alpha=alpha,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor=NAVY, linewidth=0.4,
                          alpha=0.65),
                transform=ccrs.PlateCarree(), zorder=4)


def mark_peak_geographic(ax, attn_2d, color=AMBER):
    """Place a star at the geographic centroid of the highest-attention cell."""
    r, c = np.unravel_index(np.argmax(attn_2d), attn_2d.shape)
    # cell (r, c) spans lat [47.5 - r*0.875, 47.5 - (r+1)*0.875] × lon [-74 + c, -74 + (c+1)]
    cell_lat = 47.5 - (r + 0.5) * 0.875
    cell_lon = -74.0 + (c + 0.5) * 1.0
    ax.plot(cell_lon, cell_lat, marker="*", markersize=18,
            color=color, markeredgecolor="black", markeredgewidth=0.9,
            zorder=7, transform=ccrs.PlateCarree())
    return r, c, cell_lat, cell_lon


def render_aggregate():
    data = np.load(ATTN_DIR / "aggregate.npz")
    attn = data["attn"]                 # (8, 8)
    n_samples = int(data["n_samples"])

    fig, ax = make_basemap_axes(figsize=(7.5, 6.6))
    mesh = overlay_attention(ax, attn)
    overlay_zone_labels(ax)
    overlay_cities(ax)
    r, c, lat, lon = mark_peak_geographic(ax, attn)

    cb = fig.colorbar(mesh, ax=ax, fraction=0.038, pad=0.04, label="attention weight")
    east_mass = float(attn[:, 4:].sum())
    west_mass = float(attn[:, :4].sum())
    ax.set_title(
        f"Aggregate attention — where the model looks ({n_samples} sample days)\n"
        f"mean over future hours × history hours × samples",
        loc="left",
    )
    fig.text(0.5, 0.02,
             f"peak cell ({r},{c}) ≈ ({lat:.1f}°N, {lon:+.1f}°)   |   "
             f"east-half mass {east_mass:.4f} > west-half mass {west_mass:.4f}",
             ha="center", fontsize=8, color=SUBINK, style="italic")
    fig.tight_layout()
    out = OUT_DIR / "attn_aggregate.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


def render_per_hour():
    data = np.load(ATTN_DIR / "per_hour.npz")
    attn_F = data["attn"]                              # (24, 8, 8)
    indices = list(data["indices_displayed"])          # [0, 5, 11, 17, 23]
    labels = list(data["forecast_hours_displayed"])    # [1, 6, 12, 18, 24]

    selected = [attn_F[i] for i in indices]
    vmin = float(min(a.min() for a in selected))
    vmax = float(max(a.max() for a in selected))

    fig = plt.figure(figsize=(4.4 * len(selected), 5.2))
    proj = ccrs.PlateCarree()
    axes = []
    for k, (a, lab) in enumerate(zip(selected, labels)):
        ax = fig.add_subplot(1, len(selected), k + 1, projection=proj)
        ax.set_extent([-74.0, -66.0, 40.5, 47.5], crs=proj)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"),
                       linewidth=0.6, edgecolor=NAVY, alpha=0.8)
        ax.add_feature(cfeature.STATES.with_scale("10m"),
                       linewidth=0.4, edgecolor=NAVY, alpha=0.5)
        ax.add_feature(cfeature.OCEAN.with_scale("10m"),
                       facecolor="#DCE8F0", alpha=0.5, zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("10m"),
                       facecolor="#F7F4EC", alpha=0.4, zorder=0)
        mesh = overlay_attention(ax, a, vmin=vmin, vmax=vmax, alpha=0.62)
        mark_peak_geographic(ax, a)
        ax.set_title(f"forecast t+{lab}", color=NAVY)
        axes.append(ax)

    cb = fig.colorbar(mesh, ax=axes, fraction=0.018, pad=0.02,
                      label="attention weight")
    fig.suptitle("Attention by forecast hour  (history-aggregated, sample-averaged)",
                 color=NAVY, fontweight="bold")
    out = OUT_DIR / "attn_per_hour.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


def render_per_zone():
    data = np.load(ATTN_DIR / "per_zone.npz")
    attn = data["attn"]                  # (n_zones=8, 8, 8)
    zones_in_data = list(data["zones"]) if "zones" in data.files else ZONE_COLS
    # zones may be saved as bytes/np.array
    zones_in_data = [str(z) for z in zones_in_data]
    vmin = float(attn.min())
    vmax = float(attn.max())

    fig = plt.figure(figsize=(16, 8.5))
    proj = ccrs.PlateCarree()
    axes = []
    for k, zone in enumerate(zones_in_data[:8]):
        ax = fig.add_subplot(2, 4, k + 1, projection=proj)
        ax.set_extent([-74.0, -66.0, 40.5, 47.5], crs=proj)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"),
                       linewidth=0.5, edgecolor=NAVY, alpha=0.8)
        ax.add_feature(cfeature.STATES.with_scale("10m"),
                       linewidth=0.3, edgecolor=NAVY, alpha=0.5)
        ax.add_feature(cfeature.OCEAN.with_scale("10m"),
                       facecolor="#DCE8F0", alpha=0.45, zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("10m"),
                       facecolor="#F7F4EC", alpha=0.35, zorder=0)
        mesh = overlay_attention(ax, attn[k], vmin=vmin, vmax=vmax, alpha=0.62)
        mark_peak_geographic(ax, attn[k])
        # Highlight this zone's centroid
        if zone in ZONE_LABEL_POS:
            zlat, zlon = ZONE_LABEL_POS[zone]
            ax.plot(zlon, zlat, marker="o", markersize=4,
                    color="white", markeredgecolor=NAVY, markeredgewidth=0.9,
                    zorder=5, transform=ccrs.PlateCarree())
        ax.set_title(zone, color=NAVY)
        axes.append(ax)

    cb = fig.colorbar(mesh, ax=axes, fraction=0.014, pad=0.02,
                      label="attention weight")
    fig.suptitle("Per-zone attention pattern", color=NAVY, fontweight="bold")
    out = OUT_DIR / "attn_per_zone.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


def render_extreme_vs_mild():
    data = np.load(ATTN_DIR / "extreme_vs_mild.npz")
    mild = data["mild"]
    extreme = data["extreme"]
    diff = data["diff"]
    vmin = float(min(mild.min(), extreme.min()))
    vmax = float(max(mild.max(), extreme.max()))
    dlim = float(max(abs(diff.min()), abs(diff.max())))

    fig = plt.figure(figsize=(15, 5.3))
    proj = ccrs.PlateCarree()
    panels = [
        (mild,    "Mild day (Apr 15)",      "magma",  vmin, vmax),
        (extreme, "Extreme heat (Jul 21)",  "magma",  vmin, vmax),
        (diff,    "Difference (extreme − mild)", "RdBu_r", -dlim, dlim),
    ]
    axes = []
    last_mesh_left = None
    last_mesh_right = None
    for k, (a, ttl, cmap, vlo, vhi) in enumerate(panels):
        ax = fig.add_subplot(1, 3, k + 1, projection=proj)
        ax.set_extent([-74.0, -66.0, 40.5, 47.5], crs=proj)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"),
                       linewidth=0.5, edgecolor=NAVY, alpha=0.8)
        ax.add_feature(cfeature.STATES.with_scale("10m"),
                       linewidth=0.3, edgecolor=NAVY, alpha=0.5)
        ax.add_feature(cfeature.OCEAN.with_scale("10m"),
                       facecolor="#DCE8F0", alpha=0.45, zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("10m"),
                       facecolor="#F7F4EC", alpha=0.35, zorder=0)
        mesh = overlay_attention(ax, a, vmin=vlo, vmax=vhi, alpha=0.62, cmap=cmap)
        if cmap == "magma":
            last_mesh_left = mesh
            mark_peak_geographic(ax, a)
        else:
            last_mesh_right = mesh
        ax.set_title(ttl, color=NAVY)
        axes.append(ax)

    fig.colorbar(last_mesh_left, ax=axes[:2], fraction=0.024, pad=0.02,
                 label="attention weight")
    fig.colorbar(last_mesh_right, ax=axes[2], fraction=0.046, pad=0.04,
                 label="Δ attention (extreme − mild)")
    fig.suptitle("Attention shifts on extreme vs. mild days",
                 color=NAVY, fontweight="bold")
    out = OUT_DIR / "attn_extreme_vs_mild.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    if not ATTN_DIR.exists():
        print(f"ERROR: {ATTN_DIR} does not exist. Run scripts/attention_maps.py on HPC first, then rsync.", file=sys.stderr)
        sys.exit(1)
    if not (ATTN_DIR / "aggregate.npz").exists():
        print(f"ERROR: {ATTN_DIR / 'aggregate.npz'} missing. Make sure HPC ran the updated attention_maps.py with the .npz dump.", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading from: {ATTN_DIR}")
    print(f"Writing to:   {OUT_DIR}")
    print()

    md = load_metadata()
    print(f"Sampled days: {[s['label'] for s in md['samples']]}")
    print(f"Bbox: {md['bbox']}")
    print()

    print("Rendering attn_aggregate.png ...")
    render_aggregate()
    print("Rendering attn_per_hour.png ...")
    render_per_hour()
    print("Rendering attn_per_zone.png ...")
    render_per_zone()
    print("Rendering attn_extreme_vs_mild.png ...")
    render_extreme_vs_mild()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
