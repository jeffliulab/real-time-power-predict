"""
Render the training-time-vs-deployment per-zone MAPE drift figure.

Combines:
  - Training-time test MAPE (2022-12-30/31) from
    pretrained_models/baseline/dump/baseline_preds_test_2022_last2d.json
  - Deployment-period MAPE (May 2026) from the live data repo (or fallback).

Output:
    report/arxiv/figures/training_vs_deployment_per_zone_drift.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "report" / "arxiv" / "figures" / "training_vs_deployment_per_zone_drift.png"

LIVE_URL = (
    "https://raw.githubusercontent.com/jeffliulab/"
    "new-england-real-time-power-predict-data/main/data/backtest_rolling_7d.json"
)
FALLBACK_PATH = ROOT / "space" / "assets" / "backtest_fallback.json"
TRAIN_DUMP = ROOT / "pretrained_models" / "baseline" / "dump" / "baseline_preds_test_2022_last2d.json"

ZONES = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]
ZONE_LABELS = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA-Boston"]
NAVY = "#1A3A5C"
ACCENT = "#2E86DE"
RED = "#C0392B"


def load_live():
    try:
        r = requests.get(LIVE_URL, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return json.loads(FALLBACK_PATH.read_text())


def main():
    train = json.loads(TRAIN_DUMP.read_text())
    # train["preds"] / train["truth"] are (2 forecasts, 24 hours, 8 zones)
    preds = np.asarray(train["preds"], dtype=np.float32)
    truth = np.asarray(train["truth"], dtype=np.float32)
    train_pz = []
    for j in range(8):
        ape = np.abs(preds[..., j] - truth[..., j]) / np.abs(truth[..., j])
        train_pz.append(float(ape.mean() * 100))

    live = load_live()
    deploy_pz = [live["summary"]["baseline"]["per_zone"][z] for z in ZONES]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    x = np.arange(len(ZONES))
    width = 0.38

    bars1 = ax.bar(x - width / 2, train_pz, width,
                    label="2022 self-eval (Dec 30-31)",
                    color=NAVY, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, deploy_pz, width,
                    label="May 2026 live deployment (7-day rolling)",
                    color=RED, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(ZONE_LABELS)
    ax.set_ylabel("Baseline MAPE (%)", fontsize=11)
    ax.set_title("Per-zone MAPE: training-window self-eval vs live deployment "
                  "(both: real HRRR + per-zone demand)",
                  fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)

    for bar, val in zip(bars2, deploy_pz):
        if val > 30:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                    f"{val:.0f}%", ha="center", va="bottom",
                    fontsize=9, color=RED, fontweight="bold")

    # Drift annotations between bar pairs
    for i, (t, d) in enumerate(zip(train_pz, deploy_pz)):
        delta = d - t
        if delta > 5:
            ax.annotate(f"+{delta:.0f} pp",
                         xy=(x[i], max(t, d) + 6),
                         ha="center", fontsize=8.5, color=RED, style="italic")

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
