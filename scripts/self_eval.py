"""
Self-contained Part 1 evaluator.

Drop-in replacement for the TA's evaluate.py while the course modules
(class/default, cs137/2026spring) are unavailable on the cluster.

Loads our checkpoint, runs inference on the last N days of 2022 (matching
the TA evaluator default of 2), computes MAPE in physical space using the
same formula train.py uses during validation.

Usage (on a compute node):
    python scripts/self_eval.py \
        --ckpt runs/cnn_transformer/checkpoints/best.pt \
        --n_days 2
"""

import argparse
import json
import sys
from pathlib import Path

print("[self_eval] starting imports", flush=True)
import torch  # must come before numpy/pandas to avoid BLAS conflict in tuftsai env
print(f"[self_eval] torch {torch.__version__} ok (cuda={torch.cuda.is_available()})", flush=True)
import numpy as np
print("[self_eval] numpy ok", flush=True)
import pandas as pd
print("[self_eval] pandas ok", flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.data_preparation.dataset import (
    EnergyForecastDataset, ZONE_COLS, N_ZONES,
)
print("[self_eval] dataset module ok", flush=True)
from models import create_model
print("[self_eval] models module ok", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str,
                   default="runs/cnn_transformer/checkpoints/best.pt")
    p.add_argument("--norm_stats", type=str,
                   default="runs/cnn_transformer/norm_stats.pt")
    p.add_argument("--data_root", type=str,
                   default="/cluster/tufts/c26sp1cs0137/data/assignment3_data")
    p.add_argument("--year", type=int, default=2022)
    p.add_argument("--n_days", type=int, default=2,
                   help="How many final days of the year to test on")
    p.add_argument("--history_len", type=int, default=24)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def select_final_days(dataset, year, n_days):
    """Keep only samples whose prediction-start timestamp is one of the final
    n_days midnights of the given year (00:00 UTC)."""
    target_dates = pd.date_range(
        end=pd.Timestamp(f"{year}-12-31"), periods=n_days, freq="D"
    )
    target_ts = set(pd.Timestamp(d).normalize() for d in target_dates)

    kept = []
    for idx, t_idx in enumerate(dataset.samples):
        ts = pd.Timestamp(dataset.timestamps[t_idx])
        if ts.hour == 0 and ts.normalize() in target_ts:
            kept.append(t_idx)
    dataset.samples = kept
    return [pd.Timestamp(dataset.timestamps[t]) for t in kept]


def main():
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ckpt_path = (ROOT / args.ckpt) if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    norm_path = (ROOT / args.norm_stats) if not Path(args.norm_stats).is_absolute() else Path(args.norm_stats)

    print(f"Device    : {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Norm stats: {norm_path}")
    print(f"Data root : {args.data_root}")
    print(f"Test slice: last {args.n_days} day(s) of {args.year}")
    print("=" * 70)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = ckpt.get("args", {})
    norm_stats = torch.load(norm_path, weights_only=True)

    model = create_model(
        model_args.get("model", "cnn_transformer"),
        n_weather_channels=7,
        n_zones=N_ZONES,
        cal_dim=44,
        history_len=model_args.get("history_len", args.history_len),
        embed_dim=model_args.get("embed_dim", 128),
        grid_size=model_args.get("grid_size", 8),
        n_layers=model_args.get("n_layers", 4),
        n_heads=model_args.get("n_heads", 4),
        dropout=model_args.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model     : {model_args.get('model', 'cnn_transformer')} "
          f"({n_params:,} params)")
    print(f"Best val MAPE at ckpt time: "
          f"{ckpt.get('best_val_mape', float('nan')):.2f}%")

    ds = EnergyForecastDataset(
        args.data_root, [args.year],
        history_len=model_args.get("history_len", args.history_len),
        norm_stats=norm_stats,
    )
    dates = select_final_days(ds, args.year, args.n_days)
    print(f"Prediction starts: {[d.strftime('%Y-%m-%d %H:%M') for d in dates]}")
    print(f"# samples: {len(ds)}")
    print("=" * 70)

    e_mean = norm_stats["energy_mean"].to(device)
    e_std = norm_stats["energy_std"].to(device)

    all_preds_real, all_targets_real = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            if sample is None:
                print(f"  [{i+1}] sample None (weather missing), skipping")
                continue
            hist_w, hist_e, hist_c, fut_w, fut_c, target = sample
            hist_w = hist_w.unsqueeze(0).to(device)
            hist_e = hist_e.unsqueeze(0).to(device)
            hist_c = hist_c.unsqueeze(0).to(device)
            fut_w = fut_w.unsqueeze(0).to(device)
            fut_c = fut_c.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)

            pred = model(hist_w, hist_e, hist_c, fut_w, fut_c)

            pred_real = pred * e_std + e_mean
            target_real = target * e_std + e_mean
            all_preds_real.append(pred_real.cpu())
            all_targets_real.append(target_real.cpu())
            print(f"  [{i+1}/{len(ds)}] {dates[i].strftime('%Y-%m-%d')}  "
                  f"pred mean={pred_real.mean().item():.0f} MWh, "
                  f"true mean={target_real.mean().item():.0f} MWh")

    preds = torch.cat(all_preds_real)
    targets = torch.cat(all_targets_real)

    per_zone = {}
    for j, zone in enumerate(ZONE_COLS):
        mask = targets[:, :, j] != 0
        if mask.sum() == 0:
            per_zone[zone] = float("nan")
            continue
        ape = ((preds[:, :, j] - targets[:, :, j]).abs()
               / targets[:, :, j].abs())
        per_zone[zone] = (ape[mask].mean() * 100).item()

    overall = np.nanmean(list(per_zone.values()))

    print("=" * 70)
    print(f"Overall MAPE : {overall:.2f}%")
    for zone in ZONE_COLS:
        print(f"  {zone:10s} {per_zone[zone]:.2f}%")

    out = {
        "overall_mape": overall,
        "per_zone_mape": per_zone,
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "ckpt": str(ckpt_path),
        "n_samples": len(preds),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
