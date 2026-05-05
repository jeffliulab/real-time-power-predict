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


def normalize_weather(raster: np.ndarray, norm_stats: dict) -> np.ndarray:
    """(T, H, W, 7) raw HRRR -> (T, H, W, 7) z-scored using training stats.

    norm_stats stores per-channel mean/std as (1, 1, 1, 7) tensors.
    """
    mean = norm_stats["weather_mean"].cpu().numpy().reshape(1, 1, 1, -1)
    std = norm_stats["weather_std"].cpu().numpy().reshape(1, 1, 1, -1)
    return ((raster - mean) / std).astype(np.float32)


def synthetic_weather_z(history_len: int = HISTORY_LEN,
                        future_len: int = FUTURE_LEN) -> np.ndarray:
    """Return a (S+24, H, W, C) array of zeros (training-mean weather
    in z-score space). Kept as a fallback when the live HRRR fetcher
    fails (e.g. no network, S3 outage); the model is degraded but still
    produces calibrated output from demand + calendar."""
    return np.zeros((history_len + future_len, WEATHER_H, WEATHER_W, WEATHER_C),
                    dtype=np.float32)


@torch.no_grad()
def run_forecast(model: torch.nn.Module,
                 hist_demand_mwh: np.ndarray,
                 hist_cal: np.ndarray,
                 future_cal: np.ndarray,
                 norm_stats: dict,
                 hist_weather_raw: np.ndarray,
                 future_weather_raw: np.ndarray,
                 device: str = "cpu") -> np.ndarray:
    """Run the baseline forecast.

    Args:
      hist_demand_mwh:    (24, 8) recent ISO-NE per-zone demand in MWh.
      hist_cal:           (24, 44) calendar features for the history window.
      future_cal:         (24, 44) calendar features for the next 24 h.
      hist_weather_raw:   (24, 450, 449, 7) RAW HRRR f00 analyses for the
                          history window. Will be z-scored internally.
      future_weather_raw: (24, 450, 449, 7) RAW HRRR f01..f24 forecasts
                          (or analyses, if available) for the future
                          window. Will be z-scored internally.

    Returns:
      (24, 8) forecast in MWh.
    """
    if hist_weather_raw.shape != (HISTORY_LEN, WEATHER_H, WEATHER_W, WEATHER_C):
        raise ValueError(
            f"hist_weather_raw shape {hist_weather_raw.shape} != "
            f"({HISTORY_LEN}, {WEATHER_H}, {WEATHER_W}, {WEATHER_C})")
    if future_weather_raw.shape != (FUTURE_LEN, WEATHER_H, WEATHER_W, WEATHER_C):
        raise ValueError(
            f"future_weather_raw shape {future_weather_raw.shape} != "
            f"({FUTURE_LEN}, {WEATHER_H}, {WEATHER_W}, {WEATHER_C})")

    hist_w_z = normalize_weather(hist_weather_raw, norm_stats)
    fut_w_z = normalize_weather(future_weather_raw, norm_stats)

    hist_w = torch.from_numpy(hist_w_z).unsqueeze(0).to(device)
    fut_w = torch.from_numpy(fut_w_z).unsqueeze(0).to(device)

    hist_y_z = normalize_demand(hist_demand_mwh, norm_stats)
    hist_y = torch.from_numpy(hist_y_z).unsqueeze(0).to(device)
    hist_c = torch.from_numpy(hist_cal.astype(np.float32)).unsqueeze(0).to(device)
    fut_c = torch.from_numpy(future_cal.astype(np.float32)).unsqueeze(0).to(device)

    pred_z = model(hist_w, hist_y, hist_c, fut_w, fut_c)      # (1, 24, 8) z-space
    pred_mwh = denormalize_demand(pred_z.squeeze(0).cpu().numpy(), norm_stats)
    return pred_mwh


# =====================================================================
# Foundation-model ensemble (Chronos-Bolt-mini, zero-shot)
# =====================================================================
#
# Per Table 10 of the report, chronos-bolt-mini (21 M params) gives the
# best per-zone ensemble (4.21 % test MAPE) on the 2-day 2022 self-eval
# slice — actually slightly better than chronos-bolt-base (205 M, 4.33 %).
# Smaller weights => faster cold start + lower memory on the HF Spaces
# free tier (16 GB RAM, 2 vCPU). We hard-code the per-zone alpha that the
# offline grid search returned for the mini variant:
#
#   alpha[z] = weight on the BASELINE prediction for zone z;
#   (1 - alpha[z]) goes to the Chronos zero-shot prediction.
#
# Higher alpha => baseline dominates (good for small, weather-driven zones
# like ME / NH / VT). alpha = 0 => baseline is dropped entirely (good for
# the dense urban-coastal zones CT / SEMA / NEMA_BOST that Chronos models
# better from demand history alone).

CHRONOS_MODEL_CARD = "amazon/chronos-bolt-mini"
CHRONOS_CONTEXT = 672            # 4 weeks of hourly history per zone
CHRONOS_QUANTILE = 0.5           # use median for the point forecast

ALPHA_PER_ZONE_MINI = {
    "ME":        0.30,
    "NH":        0.30,
    "VT":        0.80,
    "CT":        0.00,
    "RI":        0.10,
    "SEMA":      0.00,
    "WCMA":      0.05,
    "NEMA_BOST": 0.00,
}


def load_chronos(model_card: str = CHRONOS_MODEL_CARD, device: str = "cpu"):
    """Load Chronos-Bolt pipeline (lazy import so baseline-only path doesn't need
    chronos-forecasting installed at module-load time)."""
    from chronos import BaseChronosPipeline   # noqa: WPS433
    pipeline = BaseChronosPipeline.from_pretrained(
        model_card, device_map=device, torch_dtype=torch.float32,
    )
    return pipeline


@torch.no_grad()
def run_chronos_zeroshot(pipeline,
                         hist_demand_mwh_long: np.ndarray) -> np.ndarray:
    """Run Chronos-Bolt zero-shot for a 24-h forecast on each of the 8 zones
    independently.

    Args:
      hist_demand_mwh_long: (T, 8) per-zone demand history in MWh, with
        T >= CHRONOS_CONTEXT. Only the last CHRONOS_CONTEXT rows are used;
        if shorter, we pad by repeating the earliest available sample
        (same fallback the baseline uses when the live API is short).

    Returns:
      (24, 8) zero-shot median forecast in MWh.
    """
    T, n_zones = hist_demand_mwh_long.shape
    if T < CHRONOS_CONTEXT:
        # Pad by repeating the first available row at the front.
        pad = np.repeat(hist_demand_mwh_long[:1], CHRONOS_CONTEXT - T, axis=0)
        hist_demand_mwh_long = np.concatenate([pad, hist_demand_mwh_long], axis=0)
    ctx = hist_demand_mwh_long[-CHRONOS_CONTEXT:]                   # (672, 8)
    ctx_tensor = torch.from_numpy(ctx.T).to(torch.float32)          # (8, 672)

    quantiles, _mean = pipeline.predict_quantiles(
        context=ctx_tensor,
        prediction_length=FUTURE_LEN,
        quantile_levels=[CHRONOS_QUANTILE],
    )
    # quantiles: (8 zones, 24 hours, 1 quantile) -> (24, 8)
    median = quantiles[:, :, 0].cpu().numpy().T                     # (24, 8)
    return median.astype(np.float32)


def per_zone_ensemble(baseline_mwh: np.ndarray,
                      chronos_mwh: np.ndarray,
                      alpha_per_zone: dict[str, float] = ALPHA_PER_ZONE_MINI) -> np.ndarray:
    """Late-fusion ensemble:
        y_ens[h, z] = alpha[z] * y_baseline[h, z] + (1 - alpha[z]) * y_chronos[h, z]
    """
    alpha = np.array([alpha_per_zone[z] for z in ZONE_COLS], dtype=np.float32)
    return alpha[None, :] * baseline_mwh + (1 - alpha[None, :]) * chronos_mwh
