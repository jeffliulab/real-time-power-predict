"""
Training script for energy demand forecasting models.

Usage:
    python training/train.py --model cnn_transformer_baseline --epochs 30 --batch_size 4
    python training/train.py --model cnn_transformer_baseline --history_len 48 --grid_size 6
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import os
import torch
# Disable cuDNN when env var set — Tufts HPC's cuDNN (both their modules and
# torch's bundled one) has symbol/engine issues with torch 2.3.1+cu121 on
# A100 nodes, causing loss.backward() to fail. Disabling falls back to
# native CUDA kernels. Slower (~1.5x) but reliable.
if os.environ.get("DISABLE_CUDNN", "0") == "1":
    torch.backends.cudnn.enabled = False
    print("NOTE: cuDNN disabled via DISABLE_CUDNN=1", flush=True)
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training.data_preparation.dataset import (
    get_dataloaders, N_ZONES, CAL_DIM, ZONE_COLS,
)
from models import create_model, get_model_defaults, MODEL_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser(description="Train energy forecasting model")

    parser.add_argument("--model", type=str, default="cnn_transformer_baseline",
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data_root", type=str,
                        default="/cluster/tufts/c26sp1cs0137/data/assignment3_data")

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau", "none"])

    # Model
    parser.add_argument("--history_len", type=int, default=24)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--grid_size", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_encoder_layers", type=int, default=4,
                        help="Encoder layers for cnn_encoder_decoder")
    parser.add_argument("--n_decoder_layers", type=int, default=2,
                        help="Decoder layers for cnn_encoder_decoder")
    parser.add_argument("--use_future_weather_xattn", action="store_true",
                        help="Enable cross-attn to future weather in decoder")
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Linear LR warmup for N optimizer steps")

    # Data
    parser.add_argument("--train_years", type=int, nargs="+",
                        default=[2019, 2020, 2021])
    parser.add_argument("--val_years", type=int, nargs="+",
                        default=[2022])

    # Infrastructure
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def setup_output_dir(args):
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        # Use project directory for runs, not data_root (which may be read-only)
        project_root = Path(__file__).resolve().parent.parent
        out = project_root / "runs" / args.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    return out


def get_device(args):
    if args.device:
        return torch.device(args.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetricLogger:
    """Log training metrics to CSV and generate plots."""

    def __init__(self, log_dir):
        self.log_path = log_dir / "training_log.csv"
        self.history = []
        self.header = [
            "epoch", "train_loss", "val_loss", "val_mape",
        ] + [f"val_mape_{z}" for z in ZONE_COLS] + ["lr", "epoch_time"]

        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(self.header)

    def log(self, metrics):
        self.history.append(metrics)
        row = [metrics.get(h, "") for h in self.header]
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def plot(self, fig_dir):
        if len(self.history) < 2:
            return
        epochs = [m["epoch"] for m in self.history]
        train_loss = [m["train_loss"] for m in self.history]
        val_loss = [m["val_loss"] for m in self.history]
        val_mape = [m["val_mape"] for m in self.history]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(epochs, train_loss, label="Train")
        axes[0].plot(epochs, val_loss, label="Val")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, val_mape, "g-o", markersize=3)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAPE (%)")
        axes[1].set_title("Validation MAPE")
        axes[1].grid(True, alpha=0.3)

        for zone in ZONE_COLS:
            key = f"val_mape_{zone}"
            vals = [m.get(key, float("nan")) for m in self.history]
            axes[2].plot(epochs, vals, label=zone)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("MAPE (%)")
        axes[2].set_title("Validation MAPE by Zone")
        axes[2].legend(fontsize=7)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(fig_dir / "training_curves.png", dpi=150)
        plt.close()


def compute_mape(preds, targets, norm_stats):
    """
    Compute MAPE in physical (denormalized) space.

    Parameters
    ----------
    preds   : (N, 24, n_zones) normalized
    targets : (N, 24, n_zones) normalized
    norm_stats : dict with energy_mean, energy_std

    Returns
    -------
    overall_mape : float
    per_zone_mape : dict {zone_name: float}
    """
    e_mean = norm_stats["energy_mean"]  # (1, n_zones)
    e_std = norm_stats["energy_std"]

    preds_real = preds * e_std + e_mean
    targets_real = targets * e_std + e_mean

    mask = targets_real != 0
    per_zone = {}
    for j, zone in enumerate(ZONE_COLS):
        zone_mask = mask[:, :, j]
        if zone_mask.sum() > 0:
            ape = ((preds_real[:, :, j] - targets_real[:, :, j]).abs()
                   / targets_real[:, :, j].abs())
            per_zone[zone] = (ape[zone_mask].mean() * 100).item()
        else:
            per_zone[zone] = float("nan")

    overall = sum(v for v in per_zone.values() if not np.isnan(v)) / len(per_zone)
    return overall, per_zone


def train_one_epoch(model, loader, optimizer, criterion, device, epoch=0,
                    warmup_steps=0, base_lr=1e-3, step_counter=None):
    model.train()
    total_loss = 0
    n_batches = 0
    n_total = len(loader)

    for batch_idx, batch in enumerate(loader):
        if batch is None:
            continue
        hist_w, hist_e, hist_c, fut_w, fut_c, target = batch
        hist_w = hist_w.to(device)
        hist_e = hist_e.to(device)
        hist_c = hist_c.to(device)
        fut_w = fut_w.to(device)
        fut_c = fut_c.to(device)
        target = target.to(device)

        pred = model(hist_w, hist_e, hist_c, fut_w, fut_c)
        loss = criterion(pred, target)

        if torch.isnan(loss):
            print(f"  [epoch {epoch}] NaN loss at batch {batch_idx}, skipping")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Linear LR warmup (first `warmup_steps` optimizer steps)
        if step_counter is not None:
            step_counter[0] += 1
            if warmup_steps > 0 and step_counter[0] <= warmup_steps:
                lr = base_lr * step_counter[0] / warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        del hist_w, hist_e, hist_c, fut_w, fut_c, target, pred, loss

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n_total:
            avg = total_loss / n_batches if n_batches > 0 else float("nan")
            print(f"  [train] batch {batch_idx+1}/{n_total} loss={avg:.4f}", flush=True)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds, all_targets = [], []
    n_total = len(loader)

    for batch_idx, batch in enumerate(loader):
        if batch is None:
            continue
        hist_w, hist_e, hist_c, fut_w, fut_c, target = batch
        hist_w = hist_w.to(device)
        hist_e = hist_e.to(device)
        hist_c = hist_c.to(device)
        fut_w = fut_w.to(device)
        fut_c = fut_c.to(device)
        target = target.to(device)

        pred = model(hist_w, hist_e, hist_c, fut_w, fut_c)
        loss = criterion(pred, target)

        if not torch.isnan(loss):
            total_loss += loss.item()
            n_batches += 1

        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())

        del hist_w, hist_e, hist_c, fut_w, fut_c, target, pred, loss

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == n_total:
            print(f"  [val] batch {batch_idx+1}/{n_total}", flush=True)

    val_loss = total_loss / max(n_batches, 1)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    return val_loss, preds, targets


def main():
    args = parse_args()
    device = get_device(args)
    out_dir = setup_output_dir(args)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Output: {out_dir}")
    print(f"History length: {args.history_len}")

    # Data
    train_loader, val_loader, norm_stats = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        history_len=args.history_len,
        num_workers=args.num_workers,
        train_years=args.train_years,
        val_years=args.val_years,
        output_dir=str(out_dir),
    )

    # Model
    model_kwargs = {
        "n_weather_channels": 7,
        "n_zones": N_ZONES,
        "cal_dim": CAL_DIM,
        "history_len": args.history_len,
        "embed_dim": args.embed_dim,
        "grid_size": args.grid_size,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
    }
    if args.model == "cnn_encoder_decoder":
        model_kwargs["n_encoder_layers"] = args.n_encoder_layers
        model_kwargs["n_decoder_layers"] = args.n_decoder_layers
        model_kwargs["use_future_weather_xattn"] = args.use_future_weather_xattn
    else:
        model_kwargs["n_layers"] = args.n_layers
    model = create_model(args.model, **model_kwargs)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    seq_len = (args.history_len + 24) * (args.grid_size ** 2 + 1)
    print(f"Sequence length: {seq_len}")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)
    else:
        scheduler = None

    criterion = nn.MSELoss()

    # Resume
    start_epoch = 0
    best_val_mape = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_mape = ckpt.get("best_val_mape", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    logger = MetricLogger(out_dir / "logs")

    norm_stats_cpu = {k: v.cpu() if torch.is_tensor(v) else v
                      for k, v in norm_stats.items()}

    epochs_no_improve = 0
    step_counter = [0]  # mutable box for global optimizer-step count

    print(f"\nTraining for {args.epochs} epochs...")
    if args.warmup_steps > 0:
        print(f"LR warmup: linear for {args.warmup_steps} steps")
    if args.patience > 0:
        print(f"Early stopping enabled: patience={args.patience}")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, device, epoch,
                                     warmup_steps=args.warmup_steps,
                                     base_lr=args.lr,
                                     step_counter=step_counter)
        val_loss, preds, targets = validate(model, val_loader,
                                            criterion, device)

        overall_mape, zone_mapes = compute_mape(preds, targets, norm_stats_cpu)

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mape": overall_mape,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": time.time() - t0,
        }
        for zone, m in zone_mapes.items():
            metrics[f"val_mape_{zone}"] = m

        logger.log(metrics)
        logger.plot(out_dir / "figures")

        # Scheduler step
        if args.scheduler == "cosine" and scheduler:
            scheduler.step()
        elif args.scheduler == "plateau" and scheduler:
            scheduler.step(val_loss)

        # Checkpointing
        ckpt_data = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_mape": min(best_val_mape, overall_mape),
            "args": vars(args),
            "norm_stats": norm_stats_cpu,
        }
        torch.save(ckpt_data, out_dir / "checkpoints" / "latest.pt")

        if overall_mape < best_val_mape:
            best_val_mape = overall_mape
            epochs_no_improve = 0
            torch.save(ckpt_data, out_dir / "checkpoints" / "best.pt")
            marker = " *best*"
        else:
            epochs_no_improve += 1
            marker = ""

        zone_strs = [f"{z[:4]}={zone_mapes[z]:.1f}%" for z in ZONE_COLS]
        print(f"Epoch {epoch:3d} | train={train_loss:.4f} val={val_loss:.4f} | "
              f"MAPE={overall_mape:.2f}% | {' '.join(zone_strs)} | "
              f"lr={optimizer.param_groups[0]['lr']:.1e} "
              f"t={time.time()-t0:.0f}s{marker}")

        # Early stopping
        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs")
            break

    print("=" * 70)
    print(f"Training complete. Best val MAPE: {best_val_mape:.2f}%")
    print(f"Checkpoints: {out_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
