"""
Dump baseline predictions + ground truth + demand history for the
supplementary three-way ensemble (baseline ⊕ c1 Chronos zero-shot ⊕ c2
Chronos fine-tune).

Designed to run ONCE on HPC. Produces three artifacts under
runs/cnn_transformer_baseline/dump/ that the local laptop pipeline rsyncs
back. Total output size: ~3-5 MB.

Artifacts produced
------------------
1. baseline_preds_test_2022_last2d.json
       Baseline preds + ground truth for the test slice (last 2 days of 2022).
       Shape: preds (2, 24, 8), truth (2, 24, 8) — physical MWh.

2. baseline_preds_val_2022_last14d.json
       Baseline preds + ground truth for the validation window we use to fit
       ensemble weights. Default: 14 days immediately before the test slice
       (2022-12-16 .. 2022-12-29 inclusive), so 14 forecasts each 24h.
       Shape: preds (14, 24, 8), truth (14, 24, 8).

3. demand_2019_2022_hourly.csv
       Concatenated per-zone hourly demand from
       energy_demand_data/target_energy_zonal_*.csv. Used by c1 (Chronos
       zero-shot context) and c2 (Chronos fine-tune training set).
       Columns: timestamp_utc, ME, NH, VT, CT, RI, SEMA, WCMA, NEMA_BOST.

Usage
-----
    python scripts/dump_baseline_preds.py
        # writes to runs/cnn_transformer_baseline/dump/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.data_preparation.dataset import (
    EnergyForecastDataset, ZONE_COLS, N_ZONES,
)
from models import create_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str,
                   default="runs/cnn_transformer_baseline/checkpoints/best.pt")
    p.add_argument("--norm_stats", type=str,
                   default="runs/cnn_transformer_baseline/norm_stats.pt")
    p.add_argument("--data_root", type=str,
                   default="data")
    p.add_argument("--year", type=int, default=2022)
    p.add_argument("--n_test_days", type=int, default=2)
    p.add_argument("--val_days", type=int, default=14,
                   help="Validation window: this many days IMMEDIATELY before "
                        "the test slice (no overlap)")
    p.add_argument("--history_len", type=int, default=24)
    p.add_argument("--out_dir", type=str,
                   default="runs/cnn_transformer_baseline/dump")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def select_starts(dataset, starts: list[pd.Timestamp]):
    """Filter dataset.samples to only those starting at the given timestamps."""
    target_set = set(pd.Timestamp(s).normalize() for s in starts)
    kept = []
    for t_idx in dataset.samples:
        ts = pd.Timestamp(dataset.timestamps[t_idx])
        if ts.hour == 0 and ts.normalize() in target_set:
            kept.append(t_idx)
    dataset.samples = kept
    return [pd.Timestamp(dataset.timestamps[t]) for t in kept]


def run_inference(model, dataset, device, e_mean, e_std):
    """Returns (preds, truth, dates) — all in physical MWh, np.ndarray."""
    preds_all, truth_all, dates = [], [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample is None:
                print(f"  [{i+1}/{len(dataset)}] sample None, skipping")
                continue
            hist_w, hist_e, hist_c, fut_w, fut_c, target = sample
            hist_w = hist_w.unsqueeze(0).to(device)
            hist_e = hist_e.unsqueeze(0).to(device)
            hist_c = hist_c.unsqueeze(0).to(device)
            fut_w = fut_w.unsqueeze(0).to(device)
            fut_c = fut_c.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)

            pred = model(hist_w, hist_e, hist_c, fut_w, fut_c)
            pred_real = (pred * e_std + e_mean).cpu().numpy()[0]
            target_real = (target * e_std + e_mean).cpu().numpy()[0]
            preds_all.append(pred_real.astype(np.float32))
            truth_all.append(target_real.astype(np.float32))

            t_idx = dataset.samples[i]
            dates.append(pd.Timestamp(dataset.timestamps[t_idx]))
            print(f"  [{i+1}/{len(dataset)}] {dates[-1].strftime('%Y-%m-%d')}  "
                  f"pred mean={pred_real.mean():.0f} MWh  "
                  f"true mean={target_real.mean():.0f} MWh")
    return np.stack(preds_all), np.stack(truth_all), dates


def compute_mape(preds, truth):
    per_zone = {}
    for j, zone in enumerate(ZONE_COLS):
        mask = truth[:, :, j] != 0
        if mask.sum() == 0:
            per_zone[zone] = float("nan")
            continue
        ape = np.abs(preds[:, :, j] - truth[:, :, j]) / np.abs(truth[:, :, j])
        per_zone[zone] = float(ape[mask].mean() * 100)
    overall = float(np.nanmean(list(per_zone.values())))
    return overall, per_zone


def dump_split(split, model, dataset_factory, starts, out_path, device,
               e_mean, e_std):
    print(f"\n=== Dumping {split} ({len(starts)} forecast(s)) ===")
    ds = dataset_factory()
    kept = select_starts(ds, starts)
    if len(kept) != len(starts):
        print(f"  WARNING: requested {len(starts)} starts, got {len(kept)} "
              f"after filtering (likely missing weather frames)")
    print(f"  starts: {[k.strftime('%Y-%m-%d') for k in kept]}")
    preds, truth, dates = run_inference(model, ds, device, e_mean, e_std)
    overall, per_zone = compute_mape(preds, truth)
    print(f"  Overall MAPE: {overall:.2f} %")
    for z in ZONE_COLS:
        print(f"    {z:10s} {per_zone[z]:.2f}%")

    out = {
        "split": split,
        "model": "cnn_transformer_baseline",
        "forecast_starts": [d.isoformat() for d in dates],
        "zones": ZONE_COLS,
        "horizon": 24,
        "preds": preds.tolist(),
        "truth": truth.tolist(),
        "overall_mape": overall,
        "per_zone_mape": per_zone,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {out_path}")
    return overall, per_zone


def dump_demand_csv(data_root: Path, out_path: Path):
    print(f"\n=== Dumping demand history → {out_path} ===")
    energy_dir = data_root / "energy_demand_data"
    csvs = sorted(energy_dir.glob("target_energy_zonal_*.csv"))
    print(f"  found {len(csvs)} CSVs in {energy_dir}")
    dfs = [pd.read_csv(p, parse_dates=["timestamp_utc"]) for p in csvs]
    df = pd.concat(dfs).sort_values("timestamp_utc").reset_index(drop=True)
    keep = ["timestamp_utc"] + ZONE_COLS
    df = df[keep]
    df.to_csv(out_path, index=False)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  wrote {len(df)} rows, {size_mb:.2f} MB "
          f"({df['timestamp_utc'].min()} → {df['timestamp_utc'].max()})")


def main():
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ckpt_path = (ROOT / args.ckpt) if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    norm_path = (ROOT / args.norm_stats) if not Path(args.norm_stats).is_absolute() else Path(args.norm_stats)
    out_dir = (ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device    : {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Norm stats: {norm_path}")
    print(f"Data root : {args.data_root}")
    print(f"Out dir   : {out_dir}")
    print("=" * 70)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = ckpt.get("args", {})
    norm_stats = torch.load(norm_path, weights_only=True)

    model_name = model_args.get("model", "cnn_transformer_baseline")
    common = dict(
        n_weather_channels=7,
        n_zones=N_ZONES,
        cal_dim=44,
        history_len=model_args.get("history_len", args.history_len),
        embed_dim=model_args.get("embed_dim", 128),
        grid_size=model_args.get("grid_size", 8),
        n_heads=model_args.get("n_heads", 4),
        dropout=model_args.get("dropout", 0.1),
    )
    if model_name == "cnn_encoder_decoder":
        model = create_model(
            model_name,
            n_encoder_layers=model_args.get("n_encoder_layers", 4),
            n_decoder_layers=model_args.get("n_decoder_layers", 2),
            use_future_weather_xattn=model_args.get("use_future_weather_xattn", False),
            **common,
        ).to(device)
    else:
        model = create_model(
            model_name,
            n_layers=model_args.get("n_layers", 4),
            **common,
        ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}  ({n_params:,} params)")

    e_mean = norm_stats["energy_mean"].to(device)
    e_std = norm_stats["energy_std"].to(device)

    # Forecast start lists
    test_starts = list(pd.date_range(
        end=pd.Timestamp(f"{args.year}-12-31"),
        periods=args.n_test_days,
        freq="D",
    ))
    val_end = test_starts[0] - pd.Timedelta(days=1)
    val_starts = list(pd.date_range(
        end=val_end, periods=args.val_days, freq="D"
    ))
    print(f"Test starts: {[s.strftime('%Y-%m-%d') for s in test_starts]}")
    print(f"Val starts : {[s.strftime('%Y-%m-%d') for s in val_starts[:3]]} "
          f"... {val_starts[-1].strftime('%Y-%m-%d')}  ({len(val_starts)} total)")

    def make_ds():
        return EnergyForecastDataset(
            args.data_root, [args.year],
            history_len=common["history_len"],
            norm_stats=norm_stats,
        )

    test_overall, _ = dump_split(
        "test", model, make_ds, test_starts,
        out_dir / f"baseline_preds_test_{args.year}_last{args.n_test_days}d.json",
        device, e_mean, e_std,
    )
    val_overall, _ = dump_split(
        "val", model, make_ds, val_starts,
        out_dir / f"baseline_preds_val_{args.year}_last{args.val_days}d.json",
        device, e_mean, e_std,
    )

    dump_demand_csv(Path(args.data_root), out_dir / "demand_2019_2022_hourly.csv")

    # Summary file (also useful for quick inspection)
    summary = {
        "year": args.year,
        "test_overall_mape": test_overall,
        "val_overall_mape": val_overall,
        "test_starts": [s.isoformat() for s in test_starts],
        "val_starts":  [s.isoformat() for s in val_starts],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + "=" * 70)
    print(f"DONE. Test MAPE = {test_overall:.2f} %, Val MAPE = {val_overall:.2f} %")
    print(f"Artifacts in {out_dir}")
    print("Pull locally with:")
    print(f"  rsync -av tufts-login:.../predict-power/{out_dir.relative_to(ROOT)}/ "
          f"./pretrained_models/baseline/dump/")


if __name__ == "__main__":
    main()
