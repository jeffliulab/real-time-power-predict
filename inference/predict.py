"""
Day-ahead ISO-NE energy demand forecast — CLI inference.

Loads a trained checkpoint, takes preprocessed history+future inputs,
runs the model, and prints/saves the 24-hour per-zone MWh forecast.

Two modes:

1. **From a saved sample** (testing / repro): point at a `.pt` file
   containing the 5-tensor input bundle the dataset produces.
       python -m inference.predict \\
           --checkpoint runs/cnn_encoder_decoder/checkpoints/best.pt \\
           --sample sample_input.pt

2. **From real-time data** (real demo): see `space/iso_ne_fetch.py`
   for the live pipeline that fetches ISO-NE actual demand + HRRR
   weather. This CLI script is the offline / testing counterpart.

Output: (24, 8) MWh predictions, one row per future hour, columns are
the 8 ISO-NE load zones in canonical order
(ME, NH, VT, CT, RI, SEMA, WCMA, NEMA_BOST).
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models import create_model, get_model_defaults
from training.data_preparation.dataset import ZONE_COLS, N_ZONES, CAL_DIM


def load_model(checkpoint_path, device="cpu"):
    """Load trained model + norm stats from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model_name = args["model"]

    common_kwargs = dict(
        n_weather_channels=7,
        n_zones=N_ZONES,
        cal_dim=CAL_DIM,
        history_len=args.get("history_len", 24),
        embed_dim=args.get("embed_dim", 128),
        grid_size=args.get("grid_size", 8),
        n_heads=args.get("n_heads", 4),
        dropout=args.get("dropout", 0.1),
    )
    if model_name == "cnn_encoder_decoder":
        model = create_model(
            model_name,
            n_encoder_layers=args.get("n_encoder_layers", 4),
            n_decoder_layers=args.get("n_decoder_layers", 2),
            use_future_weather_xattn=args.get("use_future_weather_xattn", False),
            **common_kwargs,
        )
    else:
        model = create_model(
            model_name,
            n_layers=args.get("n_layers", 4),
            **common_kwargs,
        )

    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    norm_stats = ckpt.get("norm_stats")
    if norm_stats is None:
        raise RuntimeError(
            "Checkpoint has no embedded norm_stats; cannot denormalize "
            "predictions. Re-train with newer trainer that stores norm_stats."
        )
    return model, norm_stats, args


def predict(model, hist_weather, hist_energy, hist_cal,
            future_weather, future_cal, norm_stats, device="cpu"):
    """
    Run model + denormalize predictions to physical MWh.

    Inputs (already normalized in z-score space — same convention as the
    dataset's __getitem__):
        hist_weather   : (B, S, 450, 449, 7)
        hist_energy    : (B, S, 8)
        hist_cal       : (B, S, 44)
        future_weather : (B, 24, 450, 449, 7)
        future_cal     : (B, 24, 44)

    Returns: predictions in raw MWh, shape (B, 24, 8) on CPU.
    """
    e_mean = norm_stats["energy_mean"].to(device)
    e_std = norm_stats["energy_std"].to(device)

    with torch.no_grad():
        pred_norm = model(
            hist_weather.to(device),
            hist_energy.to(device),
            hist_cal.to(device),
            future_weather.to(device),
            future_cal.to(device),
        )
        pred_real = pred_norm * e_std + e_mean

    return pred_real.cpu()


def load_sample(sample_path):
    """Load a pre-saved input sample (5-tensor tuple from dataset)."""
    blob = torch.load(sample_path, weights_only=True, map_location="cpu")
    if isinstance(blob, dict):
        return (blob["hist_weather"], blob["hist_energy"], blob["hist_cal"],
                blob["future_weather"], blob["future_cal"])
    return blob  # assume tuple


def format_forecast(pred_mwh, start_time=None):
    """Pretty-print a (24, 8) prediction tensor."""
    if start_time is None:
        start_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    print()
    print("=" * 76)
    print(f"  24h Day-Ahead Demand Forecast — ISO New England")
    print(f"  Forecast issued: {start_time.strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 76)
    header = "Hour" + "".join(f"  {z:>9s}" for z in ZONE_COLS) + "    Total"
    print(header)
    print("-" * 76)
    pred_np = pred_mwh.numpy() if isinstance(pred_mwh, torch.Tensor) else pred_mwh
    for h in range(24):
        row = pred_np[0, h] if pred_np.ndim == 3 else pred_np[h]
        total = float(row.sum())
        line = f"+{h+1:2d}h" + "".join(f"  {v:9.0f}" for v in row) + f"  {total:9.0f}"
        print(line)
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(description="ISO-NE day-ahead demand forecast")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a trained model .pt file")
    parser.add_argument("--sample", type=str,
                        help="Path to a saved 5-tensor input sample (.pt)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str,
                        help="Write JSON output to this file (else stdout only)")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint} ...", flush=True)
    model, norm_stats, model_args = load_model(args.checkpoint, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {model_args['model']} ({n_params:,} params)")

    if args.sample:
        print(f"Loading sample from {args.sample} ...", flush=True)
        hw, he, hc, fw, fc = load_sample(args.sample)
        # add batch dim if missing
        if hw.dim() == 4:
            hw, he, hc, fw, fc = (t.unsqueeze(0) for t in (hw, he, hc, fw, fc))
    else:
        raise SystemExit(
            "No input source given. Pass --sample <path.pt>, or use the\n"
            "real-time pipeline at space/app.py for live ISO-NE data."
        )

    print("Running inference ...", flush=True)
    pred_mwh = predict(model, hw, he, hc, fw, fc, norm_stats, args.device)
    format_forecast(pred_mwh)

    if args.output:
        out = {
            "model": model_args["model"],
            "checkpoint": str(args.checkpoint),
            "issued_at_utc": datetime.utcnow().isoformat(),
            "zones": ZONE_COLS,
            "forecast_mwh": pred_mwh.squeeze(0).tolist(),  # (24, 8)
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nForecast written to {args.output}")


if __name__ == "__main__":
    main()
