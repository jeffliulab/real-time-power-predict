"""
ISO-NE Day-Ahead Demand Forecast — HF Spaces demo (scaffold).

Status: skeleton. Full real-time pipeline (live ISO-NE + HRRR fetching)
is Part 3 work. Currently this app loads a saved sample input and runs
inference to demonstrate the trained model end-to-end.
"""

import logging
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch
import numpy as np

from model_utils import load_model_from_checkpoint, run_inference, ZONE_COLS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Lazy-load the model on first request to keep cold-start fast
_MODEL_CACHE = {}


def get_model():
    if "model" not in _MODEL_CACHE:
        ckpt_path = Path(__file__).parent / "checkpoints" / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"checkpoint not found at {ckpt_path}. "
                "Upload trained best.pt before launching the Space."
            )
        log.info("Loading model from %s", ckpt_path)
        _MODEL_CACHE["model"], _MODEL_CACHE["norm_stats"] = \
            load_model_from_checkpoint(ckpt_path, device="cpu")
    return _MODEL_CACHE["model"], _MODEL_CACHE["norm_stats"]


def run_forecast():
    """Runs an inference and returns a 24-hour per-zone forecast plot."""
    model, norm_stats = get_model()

    # TODO: replace with real-time ISO-NE + HRRR fetching when Part 3 is done.
    # For now, load a saved sample shipped with the Space.
    sample_path = Path(__file__).parent / "checkpoints" / "sample_input.pt"
    if not sample_path.exists():
        return None, "No sample_input.pt available. Real-time fetch pipeline pending."

    sample = torch.load(sample_path, weights_only=True, map_location="cpu")
    pred_mwh = run_inference(model, sample, norm_stats, device="cpu")

    # plot
    import plotly.graph_objects as go
    fig = go.Figure()
    pred_np = pred_mwh.squeeze(0).numpy()
    hours = list(range(1, 25))
    for j, zone in enumerate(ZONE_COLS):
        fig.add_trace(go.Scatter(
            x=hours, y=pred_np[:, j], mode="lines+markers", name=zone,
        ))
    fig.update_layout(
        title="ISO-NE 24h Day-Ahead Demand Forecast (per zone)",
        xaxis_title="Forecast hour (t+h)",
        yaxis_title="MWh",
        template="plotly_white",
        height=500,
    )
    summary = (
        f"Forecast issued: {datetime.utcnow().isoformat()}Z\n"
        f"Total system load (peak hour): "
        f"{int(pred_np.sum(axis=1).max()):,} MWh"
    )
    return fig, summary


with gr.Blocks(title="ISO-NE Day-Ahead Demand Forecast") as demo:
    gr.Markdown(
        "# ISO-NE Day-Ahead Demand Forecast\n"
        "Encoder-Decoder CNN-Transformer trained on 2019-2021 ISO New "
        "England data. Click **Run Forecast** to see a 24-hour "
        "per-zone load forecast.\n\n"
        "*Note: live data ingestion (ISO Express + HRRR) is Part 3 work; "
        "this demo currently shows inference on a saved sample.*"
    )
    btn = gr.Button("Run Forecast", variant="primary")
    plot = gr.Plot(label="Forecast")
    summary = gr.Textbox(label="Summary", lines=3)
    btn.click(run_forecast, outputs=[plot, summary])


if __name__ == "__main__":
    demo.launch()
