"""Gradio Space: ISO-NE day-ahead demand forecasting (real-time + backtest).

Always-now real-time forecast on truly real inputs:
  - HRRR f00 weather analyses for the past 24 h (NOAA AWS S3, public)
  - HRRR forecast cycle T-1's f01..f24 for the future 24 h (no future
    analyses are used — strict deployable forecast)
  - Per-zone ISO-NE 5-minute estimated zonal load, rolled up to hourly
  - Calendar features (deterministic from timestamps)
  - Chronos-Bolt-mini zero-shot foundation-model ensemble

Backtest tab loads a 7-day rolling cache from the auxiliary data repo
(``new-england-real-time-power-predict-data``), refreshed daily by a
GitHub Actions cron. Cache is fetched once at Space startup; falls back
to a bundled snapshot if the data repo is unreachable.

Disclosure (also in about.md): the trained baseline saw f00 ANALYSES
for both history AND future windows during training (a form of data
leakage). At deployment we substitute HRRR f01..f24 forecasts for the
future window — there is no future-data leak, but the model sees a
slightly out-of-distribution input. Live MAPE will therefore be a bit
worse than the offline 5.24 % headline.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots

from calendar_features import encode_range
from iso_ne_fetch import ZONE_COLS, fetch_recent_demand_mwh, fetch_long_history_mwh
from model_utils import (
    load_baseline,
    load_chronos,
    run_chronos_zeroshot,
    run_forecast,
    per_zone_ensemble,
    ALPHA_PER_ZONE_MINI,
    HISTORY_LEN,
    FUTURE_LEN,
)
from hrrr_fetch import (
    fetch_history as hrrr_fetch_history,
    fetch_forecast_for_window,
)

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
ABOUT = (ROOT / "about.md").read_text()

DATA_REPO_BASE = "https://raw.githubusercontent.com/jeffliulab/new-england-real-time-power-predict-data/main"
BACKTEST_URL = f"{DATA_REPO_BASE}/data/backtest_rolling_7d.json"
THIRTY_DAY_CSV_URL = f"{DATA_REPO_BASE}/data/iso_ne_30d.csv"
LAST_BUILT_URL = f"{DATA_REPO_BASE}/data/last_built.json"
THIRTY_DAY_CACHE_PATH = Path("/tmp/iso_ne_30d.csv")

NAVY = "#1A3A5C"
ACCENT = "#2E86DE"
AMBER = "#C97B12"
TEAL = "#16A085"
GREY = "#7F8C8D"

print("Loading baseline checkpoint...")
MODEL, NORM_STATS = load_baseline(ROOT / "checkpoints" / "best.pt", device="cpu")
print(f"Loaded baseline ({sum(p.numel() for p in MODEL.parameters()):,} params)")

# Lazy-loaded Chronos pipeline (loaded on first Live forecast click)
_CHRONOS = {"pipeline": None}


def _get_chronos():
    if _CHRONOS["pipeline"] is None:
        print("Loading Chronos-Bolt-mini (zero-shot, ~80 MB) — first call only...")
        _CHRONOS["pipeline"] = load_chronos(device="cpu")
        print("Chronos-Bolt-mini loaded.")
    return _CHRONOS["pipeline"]


def _bootstrap_data_repo():
    """At startup, fetch the latest backtest JSON + 30-day CSV from the
    auxiliary data repo. Saves the CSV to /tmp so iso_ne_fetch can find it.
    Returns (backtest_dict, last_built_dict) or (None, None) if data repo
    unreachable (Space falls back to bundled snapshot)."""
    backtest = None
    last_built = None
    try:
        r = requests.get(BACKTEST_URL, timeout=15)
        r.raise_for_status()
        backtest = r.json()
        print(f"Loaded backtest JSON from data repo: "
              f"{backtest.get('n_forecasts')} forecasts, "
              f"built_at={backtest.get('built_at')}")
    except Exception as e:  # noqa: BLE001
        print(f"WARN: failed to fetch backtest JSON ({e}); will use bundled fallback")

    try:
        r = requests.get(LAST_BUILT_URL, timeout=10)
        r.raise_for_status()
        last_built = r.json()
    except Exception as e:  # noqa: BLE001
        print(f"WARN: failed to fetch last_built metadata ({e})")

    try:
        r = requests.get(THIRTY_DAY_CSV_URL, timeout=20)
        r.raise_for_status()
        THIRTY_DAY_CACHE_PATH.write_bytes(r.content)
        print(f"Cached 30d CSV at {THIRTY_DAY_CACHE_PATH} "
              f"({len(r.content) / 1024:.1f} KB)")
    except Exception as e:  # noqa: BLE001
        print(f"WARN: failed to fetch 30d CSV ({e}); Chronos context will use bundled sample")

    return backtest, last_built


BACKTEST, LAST_BUILT = _bootstrap_data_repo()
if BACKTEST is None:
    # Fallback to bundled snapshot if it exists (shipped with the Space)
    fallback = ASSETS / "backtest_fallback.json"
    if fallback.exists():
        try:
            BACKTEST = json.loads(fallback.read_text())
            print("Using bundled backtest_fallback.json")
        except Exception as e:  # noqa: BLE001
            print(f"WARN: bundled fallback also failed: {e}")


def _now_utc_hour() -> datetime:
    return datetime.now(timezone.utc).replace(
        minute=0, second=0, microsecond=0, tzinfo=None)


# =====================================================================
#  Live forecast (real-time)
# =====================================================================

def live_forecast(progress: Optional[gr.Progress] = None):
    """Pull real HRRR + real ISO-NE per-zone, run baseline + Chronos
    ensemble, and return plots + summary markdown.

    Uses Gradio's Progress widget for the slow HRRR fetch step.
    """
    progress = progress or gr.Progress()

    target = _now_utc_hour()

    progress(0.05, desc="Fetching ISO-NE per-zone demand...")
    try:
        hist_demand, demand_src = fetch_recent_demand_mwh(target)
    except Exception as e:  # noqa: BLE001
        return _error_panel(f"ISO-NE demand fetch failed: {e}")

    progress(0.10, desc="Fetching HRRR weather history (24 cycles)...")
    fetched = {"count": 0}
    def _hist_progress(done, total, label):
        fetched["count"] = done
        progress(0.10 + 0.40 * done / total,
                 desc=f"HRRR history {done}/{total} — {label}")
    try:
        hist_w_raw = hrrr_fetch_history(target, hours=HISTORY_LEN,
                                          parallel=4,
                                          progress=_hist_progress)
    except Exception as e:  # noqa: BLE001
        return _error_panel(f"HRRR history fetch failed: {e}")

    progress(0.55, desc="Fetching HRRR weather forecast (latest long cycle)...")
    def _fut_progress(done, total, label):
        progress(0.55 + 0.20 * done / total,
                 desc=f"HRRR forecast {done}/{total} — {label}")
    try:
        fut_w_raw, cycle_for_future, fxx_start = fetch_forecast_for_window(
            target, hours=FUTURE_LEN, parallel=4,
            progress=_fut_progress)
    except Exception as e:  # noqa: BLE001
        return _error_panel(f"HRRR forecast fetch failed: {e}")

    progress(0.80, desc="Running baseline forward pass...")
    hist_cal = encode_range(target - timedelta(hours=HISTORY_LEN), HISTORY_LEN)
    fut_cal = encode_range(target, FUTURE_LEN)
    try:
        baseline_pred = run_forecast(
            MODEL, hist_demand, hist_cal, fut_cal, NORM_STATS,
            hist_weather_raw=hist_w_raw, future_weather_raw=fut_w_raw,
            device="cpu")
    except Exception as e:  # noqa: BLE001
        return _error_panel(f"Baseline forward failed: {e}")

    progress(0.88, desc="Running Chronos-Bolt-mini zero-shot...")
    try:
        long_history, long_src = fetch_long_history_mwh(target, hours=720)
        chronos_pipeline = _get_chronos()
        chronos_pred = run_chronos_zeroshot(chronos_pipeline, long_history)
    except Exception as e:  # noqa: BLE001
        return _error_panel(f"Chronos forecast failed: {e}")

    progress(0.95, desc="Computing ensemble + plotting...")
    ens_pred = per_zone_ensemble(baseline_pred, chronos_pred, ALPHA_PER_ZONE_MINI)

    line = _live_line_plot(target, hist_demand, ens_pred,
                            overlay={"Baseline (with HRRR)": baseline_pred,
                                     "Chronos zero-shot": chronos_pred})
    bar = _live_bar_plot(target, ens_pred[0])

    sys_total = ens_pred.sum(axis=1)
    summary = (
        f"### Forecast issued at **{target.strftime('%Y-%m-%d %H:00')} UTC**\n\n"
        f"**Inputs**\n"
        f"- Demand history: `{demand_src}`\n"
        f"- Chronos context: `{long_src}`\n"
        f"- Weather history: real HRRR f00 analyses, "
        f"24 cycles {(target - timedelta(hours=24)).strftime('%Y-%m-%d %H:00')} → "
        f"{(target - timedelta(hours=1)).strftime('%H:00')} UTC\n"
        f"- Weather forecast: real HRRR cycle "
        f"{cycle_for_future.strftime('%Y-%m-%d %H:00')} UTC, "
        f"f{fxx_start:02d}..f{fxx_start + FUTURE_LEN - 1:02d}\n\n"
        f"**Output**: 24-hour ensemble forecast covering "
        f"**{target.strftime('%H:00')} → {(target + timedelta(hours=24)).strftime('%H:00')} UTC** · "
        f"system-level peak: **{sys_total.max():,.0f} MW**"
    )
    progress(1.0, desc="Done")
    return line, bar, summary


def _error_panel(msg: str):
    return go.Figure(), go.Figure(), f"### ⚠ Forecast failed\n\n{msg}"


def _live_line_plot(target: datetime, hist: np.ndarray, pred: np.ndarray,
                    overlay: dict[str, np.ndarray]):
    """8 panels (4×2). History solid + ensemble dashed; overlays as dotted."""
    fig = make_subplots(rows=4, cols=2, shared_xaxes=False,
                          subplot_titles=ZONE_COLS,
                          vertical_spacing=0.10, horizontal_spacing=0.07)
    hist_t = [target - timedelta(hours=24 - i) for i in range(24)]
    fut_t = [target + timedelta(hours=i + 1) for i in range(24)]
    overlay_palette = [GREY, TEAL, AMBER]
    for i, zone in enumerate(ZONE_COLS):
        r, c = i // 2 + 1, i % 2 + 1
        fig.add_trace(go.Scatter(
            x=hist_t, y=hist[:, i], mode="lines",
            line=dict(color=NAVY, width=2),
            name="history", showlegend=(i == 0),
        ), row=r, col=c)
        fig.add_trace(go.Scatter(
            x=fut_t, y=pred[:, i], mode="lines",
            line=dict(color=ACCENT, width=2.5, dash="dash"),
            name="forecast (ensemble)", showlegend=(i == 0),
        ), row=r, col=c)
        for k, (label, arr) in enumerate(overlay.items()):
            fig.add_trace(go.Scatter(
                x=fut_t, y=arr[:, i], mode="lines",
                line=dict(color=overlay_palette[k], width=1.2, dash="dot"),
                name=label, showlegend=(i == 0),
                opacity=0.85,
            ), row=r, col=c)
        fig.add_vline(x=target, line=dict(color="grey", width=1, dash="dot"),
                       row=r, col=c)
    fig.update_layout(
        title="Per-zone demand: history (solid) and 24-h forecast (dashed)",
        height=820, plot_bgcolor="white",
        margin=dict(l=40, r=20, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="MW", title_standoff=4)
    return fig


def _live_bar_plot(target: datetime, next_hour_pred: np.ndarray):
    order = np.argsort(next_hour_pred)
    fig = go.Figure(go.Bar(
        x=next_hour_pred[order], y=[ZONE_COLS[i] for i in order],
        orientation="h", marker_color=NAVY,
        text=[f"{v:,.0f}" for v in next_hour_pred[order]],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Predicted demand at t+1h "
              f"({(target + timedelta(hours=1)).strftime('%Y-%m-%d %H:00')} UTC)",
        xaxis_title="MW", height=350, plot_bgcolor="white",
        margin=dict(l=80, r=40, t=60, b=40),
    )
    return fig


def _alpha_table_md() -> str:
    rows = " | ".join(f"{z}: {ALPHA_PER_ZONE_MINI[z]:.2f}" for z in ZONE_COLS)
    return f"**Per-zone α (weight on Baseline; 1−α goes to Chronos):** {rows}"


# =====================================================================
#  Backtest tab (rolling 7-day, loaded at startup from data repo)
# =====================================================================

def _backtest_overview_plot():
    if BACKTEST is None:
        return go.Figure()
    forecasts = BACKTEST["forecasts"]
    fig = make_subplots(rows=4, cols=2, shared_xaxes=False,
                          subplot_titles=ZONE_COLS,
                          vertical_spacing=0.10, horizontal_spacing=0.07)
    for i, zone in enumerate(ZONE_COLS):
        r, c = i // 2 + 1, i % 2 + 1
        for f in forecasts:
            start = datetime.fromisoformat(f["start"])
            t = [start + timedelta(hours=h) for h in range(24)]
            truth = np.asarray(f["truth_24h"])[:, i]
            base = np.asarray(f["baseline"])[:, i]
            chron = np.asarray(f["chronos"])[:, i]
            ens = np.asarray(f["ensemble"])[:, i]
            show = (i == 0 and f is forecasts[0])
            fig.add_trace(go.Scatter(
                x=t, y=truth, mode="lines",
                line=dict(color=NAVY, width=2),
                name="actual demand", showlegend=show,
            ), row=r, col=c)
            fig.add_trace(go.Scatter(
                x=t, y=base, mode="lines",
                line=dict(color=GREY, width=1, dash="dot"),
                name="baseline (real HRRR)", showlegend=show,
                opacity=0.85,
            ), row=r, col=c)
            fig.add_trace(go.Scatter(
                x=t, y=chron, mode="lines",
                line=dict(color=TEAL, width=1, dash="dot"),
                name="chronos zero-shot", showlegend=show,
                opacity=0.85,
            ), row=r, col=c)
            fig.add_trace(go.Scatter(
                x=t, y=ens, mode="lines",
                line=dict(color=ACCENT, width=2, dash="dash"),
                name="ensemble", showlegend=show,
            ), row=r, col=c)
    period = BACKTEST.get("data_period", {})
    fig.update_layout(
        title=(f"7-day rolling backtest "
               f"({period.get('first_forecast_start', '?')[:10]} → "
               f"{period.get('last_forecast_start', '?')[:10]}) "
               f"— actual vs 3 model variants"),
        height=900, plot_bgcolor="white",
        margin=dict(l=40, r=20, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                     xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="MW", title_standoff=4)
    return fig


def _backtest_summary_md() -> str:
    if BACKTEST is None:
        return ("_Rolling backtest unavailable — auxiliary data repo unreachable_\n\n"
                "The Backtest tab loads its data from "
                "[`new-england-real-time-power-predict-data`]"
                "(https://github.com/jeffliulab/new-england-real-time-power-predict-data) "
                "which a GitHub Actions cron refreshes every day.")
    s = BACKTEST["summary"]
    period = BACKTEST.get("data_period", {})
    rows = []
    rows.append("| Model | " + " | ".join(ZONE_COLS) + " | **Overall** |")
    rows.append("|---|" + "|".join(["---"] * (len(ZONE_COLS) + 1)) + "|")
    for key, label in (("baseline", "Baseline (real HRRR)"),
                        ("chronos",  "Chronos-Bolt-mini (zero-shot)"),
                        ("ensemble", "Ensemble (per-zone α)")):
        per_zone = " | ".join(f"{s[key]['per_zone'][z]:.2f}" for z in ZONE_COLS)
        rows.append(f"| {label} | {per_zone} | **{s[key]['overall']:.2f}** |")
    table = "\n".join(rows)
    built_at = BACKTEST.get("built_at", "?")
    return (
        f"### Last 7 days of forecasts — per-zone & overall MAPE (%)\n\n"
        f"_Window: {period.get('first_forecast_start', '?')[:16]} UTC → "
        f"{period.get('last_forecast_start', '?')[:16]} UTC · "
        f"refreshed {built_at[:16]} UTC_\n\n"
        f"{table}\n\n"
        f"_Each forecast issues a 24-hour prediction at 00:00 UTC. The baseline uses "
        f"real HRRR f00 analyses for the history window (24 cycles) and HRRR f01..f24 "
        f"forecasts from the cycle issued at T-1 for the future window — strict deployable "
        f"backtest with no future-data leak. See **About** for the disclosure on the "
        f"training-time future_weather mismatch._"
    )


def _backtest_bars():
    if BACKTEST is None:
        return go.Figure()
    s = BACKTEST["summary"]
    labels = ["Baseline\n(real HRRR)", "Chronos-Bolt-mini\n(zero-shot)",
              "Ensemble\n(per-zone α)"]
    values = [s["baseline"]["overall"], s["chronos"]["overall"],
              s["ensemble"]["overall"]]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=[GREY, TEAL, ACCENT],
        text=[f"{v:.2f}%" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title="Overall MAPE on the 7-day backtest (lower is better)",
        yaxis_title="MAPE (%)", height=350, plot_bgcolor="white",
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def _live_performance_md() -> str:
    """Dynamic 'live performance' block for the About tab. Reads the
    current backtest summary from BACKTEST (loaded at startup from the
    auxiliary data repo) so numbers always reflect the latest cron run."""
    if BACKTEST is None:
        return ("_Live performance numbers unavailable — the auxiliary data "
                "repo is unreachable. The Backtest tab will show a bundled "
                "fallback snapshot._\n\n---\n")
    s = BACKTEST["summary"]
    period = BACKTEST.get("data_period", {})
    built_at = BACKTEST.get("built_at", "?")[:16]
    first_day = period.get("first_forecast_start", "?")[:10]
    last_day = period.get("last_forecast_start", "?")[:10]
    return (
        f"### 📊 Live performance — refreshed daily at 04:00 UTC\n\n"
        f"_Window: {first_day} → {last_day} · refreshed {built_at} UTC_\n\n"
        f"| Model | Overall MAPE on the last 7 days |\n|---|---|\n"
        f"| Baseline (real HRRR) | {s['baseline']['overall']:.2f} % |\n"
        f"| Chronos-Bolt-mini (zero-shot) | {s['chronos']['overall']:.2f} % |\n"
        f"| **Ensemble (per-zone α)** | **{s['ensemble']['overall']:.2f} %** |\n\n"
        f"_See the **Backtest** tab for the full per-zone breakdown. "
        f"Numbers above are computed end-to-end on real ISO-NE per-zone "
        f"load + real HRRR f00 analyses + HRRR f01..f24 forecasts (no "
        f"future leakage); read the prose below for why these are higher "
        f"than the offline 4.21 % headline._\n\n---\n"
    )


# =====================================================================
#  Gradio UI
# =====================================================================

with gr.Blocks(title="ISO-NE Energy Demand Forecast",
                 theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown(
        "# ⚡ Multi-Modal Deep Learning for Energy Demand Forecasting\n"
        "**Author:** Pang Liu · Independent Researcher · "
        "[GitHub](https://github.com/jeffliulab/real-time-power-predict)\n\n"
        "Live tab pulls real ISO-NE per-zone demand + real HRRR weather "
        "(history analyses + forecast-cycle predictions) and runs the trained "
        "CNN-Transformer baseline blended with Chronos-Bolt-mini in a per-zone "
        "weighted ensemble. The Backtest tab shows the same model on the most "
        "recent 7 fully-published days, refreshed daily by GitHub Actions cron "
        "in the auxiliary data repo."
    )
    with gr.Row():
        run_btn = gr.Button("🔮 Forecast next 24 h (now)",
                              variant="primary", scale=1, size="lg")
    summary_md = gr.Markdown()
    with gr.Tabs():
        with gr.Tab("Real-time forecast"):
            line_plot = gr.Plot(label="Per-zone history + forecast")
            bar_plot = gr.Plot(label="Predicted next-hour demand")
            gr.Markdown(_alpha_table_md())
        with gr.Tab("Backtest (last 7 days)"):
            gr.Markdown(
                "_Strict-discipline backtest_ — at each forecast time T the "
                "model sees only data available before T. History weather: "
                "24 HRRR f00 analyses; future weather: f01..f24 from cycle "
                "T-1 (the most recent cycle issued before T)."
            )
            backtest_plot = gr.Plot(value=_backtest_overview_plot(),
                                      label="7-day per-zone comparison")
            backtest_bars = gr.Plot(value=_backtest_bars(),
                                      label="Overall MAPE")
            gr.Markdown(_backtest_summary_md())
        with gr.Tab("About"):
            gr.Markdown(_live_performance_md())
            gr.Markdown(ABOUT)
            gr.Markdown("### Figures from the report")
            with gr.Row():
                gr.Image(str(ASSETS / "iso_ne_map.png"),
                         label="ISO-NE 8 load zones",
                         show_label=True, height=320)
                gr.Image(str(ASSETS / "baseline_per_zone.png"),
                         label="Per-zone test MAPE (Part 1 baseline, "
                               "real HRRR cluster runs)",
                         show_label=True, height=320)
            with gr.Row():
                gr.Image(str(ASSETS / "comparison_table.png"),
                         label="All model variants compared on the "
                               "2-day 2022 self-eval slice",
                         show_label=True, height=320)
                gr.Image(str(ASSETS / "ensemble_bars.png"),
                         label="Foundation-model ensemble: per-zone α "
                               "ablation (Chronos-Bolt-base)",
                         show_label=True, height=320)
            gr.Image(str(ASSETS / "architecture.png"),
                     label="Baseline CNN-Transformer architecture",
                     show_label=True)
            gr.Image(str(ASSETS / "attn_aggregate.png"),
                     label="Attention diagnostic: aggregate attention map "
                           "on the New England basemap (Part 3)",
                     show_label=True)

    run_btn.click(live_forecast,
                  outputs=[line_plot, bar_plot, summary_md])


if __name__ == "__main__":
    demo.launch()
