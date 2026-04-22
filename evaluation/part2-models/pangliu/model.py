"""
Evaluation wrapper for the Part 2 encoder-decoder CNN-Transformer.

Interface required by evaluate.py:
    get_model(metadata) -> nn.Module with adapt_inputs() and forward()

    adapt_inputs(history_weather, history_energy, future_weather, future_time)
        Raw inputs:
            history_weather : (B, 168, 450, 449, 7)
            history_energy  : (B, 168, n_zones)
            future_weather  : (B, 24, 450, 449, 7)
            future_time     : (B, 24) int64 (hours since epoch)
        Returns a tuple unpacked into forward().

    forward(*adapt_inputs(...)) -> (B, 24, n_zones) float32 in MWh

Mirrors evaluation/pangliu/model.py (Part 1 wrapper); only the model
class and a few config keys differ.
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

EVAL_DIR = Path(__file__).parent
ROOT = EVAL_DIR.parent.parent.parent  # project root on HPC: .../pliu07/assignment3

sys.path.insert(0, str(ROOT))

from models.cnn_encoder_decoder import CNNEncoderDecoderForecaster

CHECKPOINT_PATH = EVAL_DIR / "best.pt"
CONFIG_PATH = EVAL_DIR / "config.json"

US_HOLIDAYS = {
    (1, 1), (1, 20), (2, 17), (5, 26), (6, 19),
    (7, 4), (9, 1), (10, 13), (11, 11), (11, 27), (12, 25),
}

CAL_DIM = 44  # hour(24) + dow(7) + month(12) + holiday(1)


def extract_calendar_features(hours_epoch):
    """Extract calendar features from (B, T) int64 hours-since-epoch."""
    B, T = hours_epoch.shape
    hours_np = hours_epoch.cpu().numpy()

    features = np.zeros((B, T, CAL_DIM), dtype=np.float32)
    for b in range(B):
        dti = pd.DatetimeIndex(hours_np[b].astype("datetime64[h]"))
        hours = dti.hour
        dows = dti.dayofweek
        months = dti.month - 1

        idx = np.arange(T)
        features[b, idx, hours] = 1.0
        features[b, idx, 24 + dows] = 1.0
        features[b, idx, 31 + months] = 1.0

        for i, dt in enumerate(dti):
            if (dt.month, dt.day) in US_HOLIDAYS:
                features[b, i, 43] = 1.0

    return torch.from_numpy(features)


class EvalWrapper(nn.Module):
    """Wraps the trained encoder-decoder model for the evaluation harness."""

    def __init__(self, model, norm_stats, history_len):
        super().__init__()
        self.model = model
        self.history_len = history_len

        if norm_stats is not None:
            self.register_buffer("weather_mean", norm_stats["weather_mean"])
            self.register_buffer("weather_std", norm_stats["weather_std"])
            self.register_buffer("energy_mean", norm_stats["energy_mean"])
            self.register_buffer("energy_std", norm_stats["energy_std"])
        else:
            self.weather_mean = None

    def adapt_inputs(self, history_weather, history_energy,
                     future_weather, future_time):
        S = self.history_len

        hist_weather = history_weather[:, -S:, :, :, :]
        hist_energy = history_energy[:, -S:, :]

        hist_start_hour = future_time[:, 0:1] - S
        hist_hours = hist_start_hour + torch.arange(
            S, device=future_time.device
        ).unsqueeze(0)
        hist_cal = extract_calendar_features(hist_hours)
        future_cal = extract_calendar_features(future_time)

        if self.weather_mean is not None:
            device = hist_weather.device
            wm = self.weather_mean.to(device)
            ws = self.weather_std.to(device)
            em = self.energy_mean.to(device)
            es = self.energy_std.to(device)

            hist_weather = (hist_weather - wm) / (ws + 1e-7)
            future_weather = (future_weather - wm) / (ws + 1e-7)
            hist_energy = (hist_energy - em) / (es + 1e-7)

        hist_cal = hist_cal.to(hist_weather.device)
        future_cal = future_cal.to(hist_weather.device)

        return (hist_weather, hist_energy, hist_cal,
                future_weather, future_cal)

    def forward(self, hist_weather, hist_energy, hist_cal,
                future_weather, future_cal):
        pred_norm = self.model(hist_weather, hist_energy, hist_cal,
                               future_weather, future_cal)

        if self.weather_mean is not None:
            device = pred_norm.device
            em = self.energy_mean.to(device)
            es = self.energy_std.to(device)
            pred = pred_norm * es + em
        else:
            pred = pred_norm

        return pred


def get_model(metadata):
    n_zones = metadata["n_zones"]
    n_weather_vars = metadata["n_weather_vars"]

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        history_len = config.get("history_len", 24)
        embed_dim = config.get("embed_dim", 128)
        grid_size = config.get("grid_size", 8)
        n_encoder_layers = config.get("n_encoder_layers", 4)
        n_decoder_layers = config.get("n_decoder_layers", 2)
        n_heads = config.get("n_heads", 4)
        dropout = config.get("dropout", 0.1)
        use_future_weather_xattn = config.get("use_future_weather_xattn", False)
    else:
        history_len = 24
        embed_dim = 128
        grid_size = 8
        n_encoder_layers = 4
        n_decoder_layers = 2
        n_heads = 4
        dropout = 0.1
        use_future_weather_xattn = False

    model = CNNEncoderDecoderForecaster(
        n_weather_channels=n_weather_vars,
        n_zones=n_zones,
        cal_dim=CAL_DIM,
        history_len=history_len,
        embed_dim=embed_dim,
        grid_size=grid_size,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        n_heads=n_heads,
        dropout=dropout,
        use_future_weather_xattn=use_future_weather_xattn,
    )

    norm_stats = None
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        norm_stats = ckpt.get("norm_stats")
        print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    else:
        print(f"WARNING: No checkpoint at {CHECKPOINT_PATH}, using random weights")

    return EvalWrapper(model, norm_stats, history_len)
