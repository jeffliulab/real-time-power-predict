"""Gradio Space: Multi-Modal Deep Learning for Energy Demand Forecasting.

Pipeline at request time:
  1. Fetch recent 24 h ISO-NE demand (live API, or 2022 CSV fallback).
  2. Build calendar features for past 24 h + future 24 h.
  3. Use synthetic (zero) weather in z-score space.
  4. Run the trained baseline forward; denormalize predictions.
  5. Plot per-zone history + forecast.

Banner is explicit about the synthetic-weather caveat.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from calendar_features import encode_range
from iso_ne_fetch import ZONE_COLS, fetch_recent_demand_mwh
from model_utils import load_baseline, run_forecast

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
ABOUT = (ROOT / "about.md").read_text()

NAVY = "#1A3A5C"
ACCENT = "#2E86DE"

print("Loading baseline checkpoint...")
MODEL, NORM_STATS = load_baseline(ROOT / "checkpoints" / "best.pt", device="cpu")
print(f"Loaded ({sum(p.numel() for p in MODEL.parameters()):,} params)")


def _parse_dt(text: str) -> datetime:
    if not text or text.lower().startswith("(leave"):
        dt = datetime.now(timezone.utc)
    else:
        try:
            dt = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
        except ValueError:
            dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.replace(minute=0, second=0, microsecond=0)


def forecast(target_dt_text: str):
    target = _parse_dt(target_dt_text)
    hist_start = target - timedelta(hours=24)

    hist_demand, source = fetch_recent_demand_mwh(target)
    hist_cal = encode_range(hist_start, 24)
    fut_cal = encode_range(target, 24)

    pred_mwh = run_forecast(MODEL, hist_demand, hist_cal, fut_cal,
                            NORM_STATS, device="cpu")

    line = _line_plot(target, hist_demand, pred_mwh)
    bar = _bar_plot(target, pred_mwh[0])
    sys_total = pred_mwh.sum(axis=1)
    summary = (
        f"**Source:** `{source}` · "
        f"forecast horizon: **{target.strftime('%Y-%m-%d %H:00')} UTC** "
        f"to **{(target + timedelta(hours=24)).strftime('%Y-%m-%d %H:00')} UTC** · "
        f"system-level peak in next 24 h: **{sys_total.max():,.0f} MW**."
    )
    return line, bar, summary


def _line_plot(target: datetime, hist: np.ndarray, pred: np.ndarray):
    """4 subplots * 2 zones each, each showing history + forecast."""
    fig = make_subplots(rows=4, cols=2, shared_xaxes=False,
                        subplot_titles=ZONE_COLS,
                        vertical_spacing=0.10, horizontal_spacing=0.07)
    hist_t = [target - timedelta(hours=24 - i) for i in range(24)]
    fut_t = [target + timedelta(hours=i + 1) for i in range(24)]
    for i, zone in enumerate(ZONE_COLS):
        r, c = i // 2 + 1, i % 2 + 1
        fig.add_trace(go.Scatter(
            x=hist_t, y=hist[:, i], mode="lines",
            line=dict(color=NAVY, width=2),
            name="history", showlegend=(i == 0),
        ), row=r, col=c)
        fig.add_trace(go.Scatter(
            x=fut_t, y=pred[:, i], mode="lines",
            line=dict(color=ACCENT, width=2, dash="dash"),
            name="24-h forecast", showlegend=(i == 0),
        ), row=r, col=c)
        fig.add_vline(x=target, line=dict(color="grey", width=1, dash="dot"),
                      row=r, col=c)
    fig.update_layout(
        title="Per-zone demand: history (solid) and 24-h forecast (dashed)",
        height=800, plot_bgcolor="white",
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


with gr.Blocks(title="ISO-NE Energy Demand Forecast",
               theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown(
        "# ⚡ Multi-Modal Deep Learning for Energy Demand Forecasting\n"
        "**Author:** Pang Liu · Tufts CS-137 · "
        "[GitHub](https://github.com/jeffliulab/real-time-power-predict)\n\n"
        "> ⚠ **Demo limitation:** weather inputs are synthetic (training-mean "
        "zeros). The cluster runs reach **5.24 % MAPE** with real HRRR weather; "
        "this Space lets you exercise the model's input/output pipeline on **live "
        "ISO-NE demand history**. See the *About* tab for details."
    )
    with gr.Row():
        dt_input = gr.Textbox(
            label="Target datetime (UTC, ISO-8601 — e.g. 2022-12-30T18:00). Leave empty for now.",
            value="",
            placeholder="(leave empty for now)",
        )
        run_btn = gr.Button("Run forecast", variant="primary", scale=0)
    summary_md = gr.Markdown()
    with gr.Tabs():
        with gr.Tab("Forecast"):
            line_plot = gr.Plot(label="Per-zone history + forecast")
            bar_plot = gr.Plot(label="Predicted next-hour demand")
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

    run_btn.click(forecast, inputs=dt_input,
                  outputs=[line_plot, bar_plot, summary_md])
    demo.load(forecast, inputs=dt_input,
              outputs=[line_plot, bar_plot, summary_md])


if __name__ == "__main__":
    demo.launch()
