"""Model loading + inference helpers for the HF Spaces app.

Self-contained — does NOT depend on the project's models/ package
(HF Spaces uploads a flat directory). Imports the local copy of model
classes from `space/models/`.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from models.cnn_encoder_decoder import CNNEncoderDecoderForecaster  # noqa: E402

ZONE_COLS = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]
N_ZONES = 8
CAL_DIM = 44


def load_model_from_checkpoint(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = CNNEncoderDecoderForecaster(
        n_weather_channels=7,
        n_zones=N_ZONES,
        cal_dim=CAL_DIM,
        history_len=args.get("history_len", 24),
        embed_dim=args.get("embed_dim", 128),
        grid_size=args.get("grid_size", 8),
        n_encoder_layers=args.get("n_encoder_layers", 4),
        n_decoder_layers=args.get("n_decoder_layers", 2),
        n_heads=args.get("n_heads", 4),
        dropout=args.get("dropout", 0.1),
        use_future_weather_xattn=args.get("use_future_weather_xattn", False),
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    norm_stats = ckpt.get("norm_stats")
    if norm_stats is None:
        raise RuntimeError("checkpoint missing norm_stats")
    return model, norm_stats


def run_inference(model, sample, norm_stats, device="cpu"):
    """Run model on a 5-tensor input bundle, return (B, 24, 8) MWh."""
    if isinstance(sample, dict):
        keys = ["hist_weather", "hist_energy", "hist_cal",
                "future_weather", "future_cal"]
        tensors = [sample[k] for k in keys]
    else:
        tensors = list(sample)
    # ensure batch dim
    tensors = [t.unsqueeze(0) if t.dim() in (3, 1, 2) and "weather" in str(t.shape) or t.dim() < 4 else t
               for t in tensors]
    # safer: just check first tensor (hist_weather)
    if tensors[0].dim() == 4:  # (S, H, W, C)
        tensors = [t.unsqueeze(0) for t in tensors]

    e_mean = norm_stats["energy_mean"].to(device)
    e_std = norm_stats["energy_std"].to(device)

    with torch.no_grad():
        pred_norm = model(*[t.to(device) for t in tensors])
        pred_mwh = pred_norm * e_std + e_mean

    return pred_mwh.cpu()
