"""Gradio Space: Multi-Modal Deep Learning for Energy Demand Forecasting.

Real-time mode (always-now, no user-supplied datetime):
  - Pulls the most recent 24 h of ISO-NE system demand from the EIA Open
    Data API (free key, exposed to this Space as the `EIA_API_KEY`
    secret), splits it into the 8 ISO-NE zones via fixed proportions,
    and runs the chosen model on it.
  - Falls back to the bundled 2022 sample window when the live API is
    unreachable.

Two model modes:
  - Baseline only:  Part 1 CNN-Transformer (1.75 M params) on synthetic
                    weather + real demand history.
  - Ensemble (Baseline + Chronos-Bolt-mini): weather-aware baseline
                    blended per-zone with the 21 M-param foundation
                    model used zero-shot on demand history. Per Table 10
                    of the report, mini gives the best per-zone ensemble
                    (4.21 % test MAPE) and is small enough to run on the
                    HF Spaces free CPU tier.

Backtest tab:
  - Pre-computed 7-day backtest (Dec 25-31, 2022) showing all three
    models' forecasts vs. ground truth, with per-zone and overall MAPE.
  - The baseline forecasts in this cache use REAL HRRR weather (computed
    on the cluster), so this tab demonstrates the headline accuracy
    that the live tab can't reach without weather inputs.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go
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
)

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
ABOUT = (ROOT / "about.md").read_text()
BACKTEST_JSON = ASSETS / "backtest_2022_last7d.json"

NAVY = "#1A3A5C"
ACCENT = "#2E86DE"
AMBER = "#C97B12"
TEAL = "#16A085"
GREY = "#7F8C8D"

print("Loading baseline checkpoint...")
MODEL, NORM_STATS = load_baseline(ROOT / "checkpoints" / "best.pt", device="cpu")
print(f"Loaded baseline ({sum(p.numel() for p in MODEL.parameters()):,} params)")

# Lazy-loaded Chronos pipeline (only when the user picks the ensemble model).
_CHRONOS = {"pipeline": None}


def _get_chronos():
    if _CHRONOS["pipeline"] is None:
        print("Loading Chronos-Bolt-mini (zero-shot, ~80 MB) — first call only...")
        _CHRONOS["pipeline"] = load_chronos(device="cpu")
        print("Chronos-Bolt-mini loaded.")
    return _CHRONOS["pipeline"]


def _now_utc_hour() -> datetime:
    return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)


# =====================================================================
#  Real-time forecast
# =====================================================================

def forecast(model_choice: str):
    """Always-now real-time forecast: pulls live demand, runs the chosen model."""
    target = _now_utc_hour()
    hist_start = target - timedelta(hours=24)

    hist_demand, source = fetch_recent_demand_mwh(target)
    hist_cal = encode_range(hist_start, 24)
    fut_cal = encode_range(target, 24)

    baseline_pred = run_forecast(MODEL, hist_demand, hist_cal, fut_cal,
                                  NORM_STATS, device="cpu")

    if model_choice == "Ensemble (Baseline + Chronos-Bolt-mini)":
        long_history, long_source = fetch_long_history_mwh(target, hours=720)
        pipeline = _get_chronos()
        chronos_pred = run_chronos_zeroshot(pipeline, long_history)
        pred_mwh = per_zone_ensemble(baseline_pred, chronos_pred,
                                      ALPHA_PER_ZONE_MINI)
        line = _line_plot(target, hist_demand, pred_mwh,
                           overlay={"Baseline only": baseline_pred,
                                    "Chronos-Bolt-mini only": chronos_pred})
        active_label = (
            f"**Ensemble** (Baseline ⊕ Chronos-Bolt-mini, per-zone α). "
            f"Demand source for Chronos: `{long_source}`."
        )
    else:
        pred_mwh = baseline_pred
        line = _line_plot(target, hist_demand, pred_mwh)
        active_label = "**Baseline only** (Part 1 CNN-Transformer, synthetic weather)."

    bar = _bar_plot(target, pred_mwh[0])
    sys_total = pred_mwh.sum(axis=1)
    summary = (
        f"{active_label}  \n"
        f"Demand history source: `{source}`  ·  "
        f"forecast issued at **{target.strftime('%Y-%m-%d %H:00')} UTC**  ·  "
        f"covers next 24 h to **{(target + timedelta(hours=24)).strftime('%Y-%m-%d %H:00')} UTC**  ·  "
        f"system-level peak: **{sys_total.max():,.0f} MW**."
    )
    return line, bar, summary


def _line_plot(target: datetime, hist: np.ndarray, pred: np.ndarray,
                overlay: dict[str, np.ndarray] | None = None):
    """4 subplots * 2 zones each, each showing history + forecast (+ optional overlays)."""
    fig = make_subplots(rows=4, cols=2, shared_xaxes=False,
                        subplot_titles=ZONE_COLS,
                        vertical_spacing=0.10, horizontal_spacing=0.07)
    hist_t = [target - timedelta(hours=24 - i) for i in range(24)]
    fut_t = [target + timedelta(hours=i + 1) for i in range(24)]
    overlay = overlay or {}
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
            name="forecast (active)", showlegend=(i == 0),
        ), row=r, col=c)
        for k, (label, arr) in enumerate(overlay.items()):
            colour = overlay_palette[k % len(overlay_palette)]
            fig.add_trace(go.Scatter(
                x=fut_t, y=arr[:, i], mode="lines",
                line=dict(color=colour, width=1.2, dash="dot"),
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


def _bar_plot(target: datetime, next_hour_pred: np.ndarray):
    """Horizontal bar: predicted demand at target+1h, sorted."""
    order = np.argsort(next_hour_pred)
    fig = go.Figure(go.Bar(
        x=next_hour_pred[order], y=[ZONE_COLS[i] for i in order],
        orientation="h", marker_color=NAVY,
        text=[f"{v:,.0f}" for v in next_hour_pred[order]],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Predicted demand at t+1h ({(target + timedelta(hours=1)).strftime('%Y-%m-%d %H:00')} UTC)",
        xaxis_title="MW", height=350, plot_bgcolor="white",
        margin=dict(l=80, r=40, t=60, b=40),
    )
    return fig


def _alpha_table_md() -> str:
    rows = " | ".join(f"{z}: {ALPHA_PER_ZONE_MINI[z]:.2f}" for z in ZONE_COLS)
    return f"**Per-zone α (weight on Baseline; 1−α goes to Chronos):** {rows}"


# =====================================================================
#  Backtest tab (cached: 7 forecasts, Dec 25-31, 2022, real HRRR weather)
# =====================================================================

if BACKTEST_JSON.exists():
    BACKTEST = json.loads(BACKTEST_JSON.read_text())
else:
    BACKTEST = None
    print(f"WARNING: backtest cache not found at {BACKTEST_JSON}")


def _backtest_overview_plot():
    """One row per zone, showing 7-day truth vs. each model's forecast."""
    if BACKTEST is None:
        return go.Figure()
    forecasts = BACKTEST["forecasts"]
    fig = make_subplots(rows=4, cols=2, shared_xaxes=False,
                        subplot_titles=ZONE_COLS,
                        vertical_spacing=0.10, horizontal_spacing=0.07)
    for i, zone in enumerate(ZONE_COLS):
        r, c = i // 2 + 1, i % 2 + 1
        for f in forecasts:
            start = datetime.fromisoformat(f["start"]).replace(tzinfo=timezone.utc)
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
    fig.update_layout(
        title="7-day backtest, Dec 25-31 2022 — actual demand vs. 3 model variants",
        height=900, plot_bgcolor="white",
        margin=dict(l=40, r=20, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="MW", title_standoff=4)
    return fig


def _backtest_summary_md() -> str:
    if BACKTEST is None:
        return "_Backtest cache missing — re-run `scripts/build_space_backtest.py`._"
    s = BACKTEST["summary"]
    rows = []
    rows.append("| Model | " + " | ".join(ZONE_COLS) + " | **Overall** |")
    rows.append("|---|" + "|".join(["---"] * (len(ZONE_COLS) + 1)) + "|")
    for key, label in (("baseline", "Baseline (real HRRR)"),
                       ("chronos",  "Chronos-Bolt-mini (zero-shot)"),
                       ("ensemble", "Ensemble (per-zone α)")):
        per_zone = " | ".join(f"{s[key]['per_zone'][z]:.2f}" for z in ZONE_COLS)
        rows.append(f"| {label} | {per_zone} | **{s[key]['overall']:.2f}** |")
    table = "\n".join(rows)
    return (
        f"### 7-day average MAPE (%) over {BACKTEST['n_forecasts']} forecasts (Dec 25–31, 2022)\n\n"
        f"{table}\n\n"
        f"_Each forecast is a 24-hour prediction starting at 00:00 UTC. The "
        f"baseline numbers in this table use **real HRRR weather** (computed "
        f"on the cluster), so they reflect the headline 5.24 % test MAPE setup. "
        f"The live tab above uses synthetic weather, so its accuracy is lower; "
        f"the **Ensemble** path closes most of that gap because Chronos-Bolt-mini "
        f"doesn't need weather at all._"
    )


def _backtest_bars():
    """Bar chart: overall MAPE per model (averaged over 7 forecasts)."""
    if BACKTEST is None:
        return go.Figure()
    s = BACKTEST["summary"]
    labels = ["Baseline\n(real HRRR)", "Chronos-Bolt-mini\n(zero-shot)", "Ensemble\n(per-zone α)"]
    values = [s["baseline"]["overall"], s["chronos"]["overall"], s["ensemble"]["overall"]]
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


# =====================================================================
#  Gradio layout
# =====================================================================

with gr.Blocks(title="ISO-NE Energy Demand Forecast",
               theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown(
        "# ⚡ Multi-Modal Deep Learning for Energy Demand Forecasting\n"
        "**Author:** Pang Liu · Tufts CS-137 · "
        "[GitHub](https://github.com/jeffliulab/real-time-power-predict)\n\n"
        "> 🔴 **Real-time mode**: every click pulls the most recent ISO-NE system "
        "demand from the EIA Open Data API and forecasts the next 24 h.\n"
        "> ⚠ **Demo limitation**: weather inputs are synthetic (training-mean "
        "zeros) since real-time HRRR rasters aren't available in this Space. "
        "The cluster runs reach **5.24 % MAPE** with real HRRR weather; the "
        "**Ensemble** path adds Chronos-Bolt-mini (zero-shot on demand history "
        "only — no weather) and reaches **4.21 % MAPE** in our offline "
        "evaluation. See the **Backtest** tab for a 7-day side-by-side "
        "comparison and the **About** tab for full details."
    )
    with gr.Row():
        model_choice = gr.Radio(
            choices=["Baseline only",
                     "Ensemble (Baseline + Chronos-Bolt-mini)"],
            value="Ensemble (Baseline + Chronos-Bolt-mini)",
            label="Model",
            scale=2,
        )
        run_btn = gr.Button("Forecast next 24 h (now)",
                             variant="primary", scale=1)
    summary_md = gr.Markdown()
    with gr.Tabs():
        with gr.Tab("Real-time forecast"):
            line_plot = gr.Plot(label="Per-zone history + forecast")
            bar_plot = gr.Plot(label="Predicted next-hour demand")
            gr.Markdown(_alpha_table_md())
        with gr.Tab("Backtest (last 7 days of 2022)"):
            gr.Markdown(
                "These are 7 daily forecasts on the held-out 2022-12-25 → "
                "12-31 window, each issued at 00:00 UTC for the next 24 h. "
                "The **baseline** column uses real HRRR weather (computed "
                "offline on the cluster); **Chronos-Bolt-mini** is zero-shot; "
                "the **ensemble** is the per-zone weighted blend reported in "
                "the paper."
            )
            backtest_plot = gr.Plot(value=_backtest_overview_plot(),
                                     label="7-day per-zone comparison")
            backtest_bars = gr.Plot(value=_backtest_bars(),
                                     label="Overall MAPE")
            gr.Markdown(_backtest_summary_md())
        with gr.Tab("About"):
            gr.Markdown(ABOUT)
            with gr.Row():
                gr.Image(str(ASSETS / "iso_ne_map.png"),
                         label="ISO-NE 8 load zones",
                         show_label=True, height=320)
                gr.Image(str(ASSETS / "baseline_per_zone.png"),
                         label="Per-zone test MAPE (cluster, real weather)",
                         show_label=True, height=320)
            gr.Image(str(ASSETS / "architecture.png"),
                     label="Baseline CNN-Transformer architecture",
                     show_label=True)

    run_btn.click(forecast, inputs=[model_choice],
                  outputs=[line_plot, bar_plot, summary_md])
    demo.load(forecast, inputs=[model_choice],
              outputs=[line_plot, bar_plot, summary_md])


if __name__ == "__main__":
    demo.launch()
