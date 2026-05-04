"""Render two figures for the supplementary section:

  1. report/figures/supplementary_per_zone.png — per-zone MAPE 4-bar chart
     (baseline / c1 / ensemble-global / ensemble-per-zone)

  2. report/figures/supplementary_alpha_search.png — global alpha grid-search
     curve on val + per-zone alpha bar chart
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
ZONES = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]


def load(p):
    with open(p) as f:
        return json.load(f)


def per_zone_mape(preds, truth):
    preds = np.asarray(preds, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    out = []
    for j in range(8):
        mask = truth[:, :, j] != 0
        ape = np.abs(preds[:, :, j] - truth[:, :, j]) / np.abs(truth[:, :, j])
        out.append(float(ape[mask].mean() * 100))
    return out


def main():
    bt = load(ROOT / "pretrained_models/baseline/dump/baseline_preds_test_2022_last2d.json")
    c1t = load(ROOT / "runs/chronos_c1/preds_test_2022_last2d.json")
    ens = load(ROOT / "runs/foundation_ensemble/eval_test_2022_last2d_2way_perzone.json")

    truth = np.asarray(bt["truth"], dtype=np.float32)
    base_pz = per_zone_mape(bt["preds"], truth)
    c1_pz = per_zone_mape(c1t["preds"], truth)
    glob_pz = [ens["test_per_zone_ensemble"][z] for z in ZONES]
    pz_pz = [ens["per_zone_weights"]["test_per_zone_mape"][z] for z in ZONES]

    base_overall = float(np.mean(base_pz))
    c1_overall = float(np.mean(c1_pz))
    glob_overall = ens["test_mape_ensemble"]
    pz_overall = ens["per_zone_weights"]["test_mape"]

    print(f"baseline overall: {base_overall:.2f}%")
    print(f"c1 overall      : {c1_overall:.2f}%")
    print(f"ensemble global : {glob_overall:.2f}%")
    print(f"ensemble per-zone: {pz_overall:.2f}%")

    figs_dir = ROOT / "report/figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # -------- Figure 1: per-zone MAPE 4-bar chart --------
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(ZONES))
    w = 0.20
    ax.bar(x - 1.5 * w, base_pz, w, label=f"baseline ({base_overall:.2f}%)",
           color="#666666", edgecolor="black", linewidth=0.5)
    ax.bar(x - 0.5 * w, c1_pz, w, label=f"c1 zero-shot ({c1_overall:.2f}%)",
           color="#4c9be8", edgecolor="black", linewidth=0.5)
    ax.bar(x + 0.5 * w, glob_pz, w,
           label=f"ensemble (global α=0.15, {glob_overall:.2f}%)",
           color="#f0a45b", edgecolor="black", linewidth=0.5)
    ax.bar(x + 1.5 * w, pz_pz, w,
           label=f"ensemble (per-zone α, {pz_overall:.2f}%)",
           color="#c54040", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ZONES)
    ax.set_ylabel("Test MAPE (%)")
    ax.set_title(
        "Per-zone test MAPE on last 2 days of 2022\n"
        f"baseline 5.24% → ensemble per-zone 4.33% "
        f"(Δ = -0.91 pp, ~17% relative)"
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.set_ylim(0, max(max(base_pz), max(c1_pz), max(glob_pz)) * 1.20)
    fig.tight_layout()
    out1 = figs_dir / "supplementary_per_zone.png"
    fig.savefig(out1, dpi=160)
    print(f"wrote {out1}")
    plt.close(fig)

    # -------- Figure 2: global alpha curve + per-zone alpha bar --------
    # Reconstruct alpha grid from ensemble JSON (we ran with step=0.05)
    base_val = np.asarray(load(ROOT / "pretrained_models/baseline/dump/baseline_preds_val_2022_last14d.json")["preds"], dtype=np.float32)
    truth_val = np.asarray(load(ROOT / "pretrained_models/baseline/dump/baseline_preds_val_2022_last14d.json")["truth"], dtype=np.float32)
    c1_val = np.asarray(load(ROOT / "runs/chronos_c1/preds_val_2022_last14d.json")["preds"], dtype=np.float32)

    def overall_mape(preds, truth):
        per = []
        for j in range(8):
            mask = truth[:, :, j] != 0
            ape = np.abs(preds[:, :, j] - truth[:, :, j]) / np.abs(truth[:, :, j])
            per.append(float(ape[mask].mean() * 100))
        return float(np.mean(per))

    alphas = np.arange(0.0, 1.0 + 1e-9, 0.05)
    val_mapes = []
    for a in alphas:
        blend = a * base_val + (1 - a) * c1_val
        val_mapes.append(overall_mape(blend, truth_val))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1.plot(alphas, val_mapes, marker="o", color="#444", linewidth=1.5, markersize=5)
    best_a = alphas[int(np.argmin(val_mapes))]
    ax1.axvline(best_a, linestyle="--", color="#c54040",
                label=f"α* = {best_a:.2f} (val MAPE {min(val_mapes):.2f}%)")
    ax1.set_xlabel("α (weight on baseline)")
    ax1.set_ylabel("Val MAPE (%)")
    ax1.set_title("Global α grid search on val (last 14 days of 2022)")
    ax1.grid(linestyle=":", alpha=0.5)
    ax1.legend()

    pz_alphas = np.asarray(ens["per_zone_weights"]["alpha"])
    bars = ax2.bar(ZONES, pz_alphas, color="#4c9be8", edgecolor="black", linewidth=0.5)
    for b, val in zip(bars, pz_alphas):
        ax2.text(b.get_x() + b.get_width() / 2, val + 0.02,
                 f"{val:.2f}", ha="center", fontsize=9)
    ax2.set_ylabel("α (weight on baseline)")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Per-zone optimal α (1−α goes to c1)")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    out2 = figs_dir / "supplementary_alpha_search.png"
    fig.savefig(out2, dpi=160)
    print(f"wrote {out2}")
    plt.close(fig)


if __name__ == "__main__":
    main()
