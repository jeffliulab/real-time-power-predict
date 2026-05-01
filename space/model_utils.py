"""Model loading + inference helpers for the HF Spaces app.

Loads the Part 1 CNN-Transformer baseline (1.75 M params, 5.24 % MAPE
on the 2022-12-30/31 self-eval slice) and runs forward on a synthetic
weather tensor + real recent ISO-NE demand history.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from models.cnn_transformer_baseline import CNNTransformerBaselineForecaster  # noqa: E402

ZONE_COLS = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]
N_ZONES = 8
CAL_DIM = 44
HISTORY_LEN = 24
FUTURE_LEN = 24
WEATHER_H, WEATHER_W, WEATHER_C = 450, 449, 7


def load_baseline(ckpt_path, device: str = "cpu"):
    """Load the trained baseline + its norm_stats from a single checkpoint."""
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = CNNTransformerBaselineForecaster(
        n_weather_channels=WEATHER_C,
        n_zones=N_ZONES,
        cal_dim=CAL_DIM,
        history_len=args.get("history_len", HISTORY_LEN),
        embed_dim=args.get("embed_dim", 128),
        grid_size=args.get("grid_size", 8),
        n_layers=args.get("n_layers", 4),
        n_heads=args.get("n_heads", 4),
        dropout=args.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    norm_stats = ckpt.get("norm_stats")
    if norm_stats is None:
        ns_path = ckpt_path.parent / "norm_stats.pt"
        if ns_path.exists():
            norm_stats = torch.load(ns_path, map_location=device, weights_only=False)
        else:
            raise RuntimeError(
                f"checkpoint {ckpt_path} missing norm_stats and no sibling norm_stats.pt"
            )
    return model, norm_stats


def normalize_demand(demand_mwh: np.ndarray, norm_stats: dict) -> np.ndarray:
    """(T, 8) MWh -> (T, 8) z-scored."""
    mean = norm_stats["energy_mean"].cpu().numpy().reshape(-1)
    std = norm_stats["energy_std"].cpu().numpy().reshape(-1)
    return ((demand_mwh - mean) / std).astype(np.float32)


def denormalize_demand(z: np.ndarray, norm_stats: dict) -> np.ndarray:
    mean = norm_stats["energy_mean"].cpu().numpy().reshape(-1)
    std = norm_stats["energy_std"].cpu().numpy().reshape(-1)
    return (z * std + mean).astype(np.float32)


def synthetic_weather_z(history_len: int = HISTORY_LEN,
                        future_len: int = FUTURE_LEN) -> np.ndarray:
    """Return a (S+24, H, W, C) array of zeros (training-mean weather
    in z-score space). The baseline still produces calibrated per-zone
    output because the tabular branch carries demand+calendar info."""
    return np.zeros((history_len + future_len, WEATHER_H, WEATHER_W, WEATHER_C),
                    dtype=np.float32)


@torch.no_grad()
def run_forecast(model: torch.nn.Module,
                 hist_demand_mwh: np.ndarray,
                 hist_cal: np.ndarray,
                 future_cal: np.ndarray,
                 norm_stats: dict,
                 device: str = "cpu") -> np.ndarray:
    """Run the baseline on synthetic weather + real demand history.

    Args:
      hist_demand_mwh: (24, 8) recent ISO-NE per-zone demand in MWh.
      hist_cal:        (24, 44) calendar features for the history window.
      future_cal:      (24, 44) calendar features for the next 24 h.

    Returns:
      (24, 8) forecast in MWh.
    """
    weather = synthetic_weather_z()                           # (48, H, W, C)
    hist_w = torch.from_numpy(weather[:HISTORY_LEN]).unsqueeze(0).to(device)
    fut_w = torch.from_numpy(weather[HISTORY_LEN:]).unsqueeze(0).to(device)

    hist_y_z = normalize_demand(hist_demand_mwh, norm_stats)
    hist_y = torch.from_numpy(hist_y_z).unsqueeze(0).to(device)
    hist_c = torch.from_numpy(hist_cal.astype(np.float32)).unsqueeze(0).to(device)
    fut_c = torch.from_numpy(future_cal.astype(np.float32)).unsqueeze(0).to(device)

    pred_z = model(hist_w, hist_y, hist_c, fut_w, fut_c)      # (1, 24, 8) z-space
    pred_mwh = denormalize_demand(pred_z.squeeze(0).cpu().numpy(), norm_stats)
    return pred_mwh
