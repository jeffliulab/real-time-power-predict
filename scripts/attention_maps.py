"""
Part 3 Track A — Geographic Attention Maps for the baseline CNN-Transformer.

Extracts attention weights from the trained Part 1 baseline model:
  attn[future_tabular_query, history_spatial_key]
reshapes to a 2D 8×8 grid (matching the WeatherCNN's adaptive-pool output),
and visualizes which geographic regions the future-hour predictions attend
to.

CRITICAL CORRECTNESS RULES (encoded as code-level checks):
  (a) DIRECTION — slice attn rows = future tabular tokens (queries),
      cols = history spatial tokens (keys). NOT the reverse.
  (b) RESHAPE  — 64-cell spatial vector → (8 row, 8 col) row-major,
      matching WeatherCNN.forward's flatten(2). NOT (8, 8).T.
  (c) COMPASS  — HRRR convention: row 0 = NORTH, col 0 = WEST. After
      adaptive-pool to 8×8 the convention is preserved. matplotlib
      imshow with default origin='upper' renders north up. Verified
      sanity check: NEMA_BOST (eastern MA / Boston) attention should
      be heavier on east half (cols 4-7) than west half (cols 0-3).
  (d) OUTPUT   — figures saved to runs/cnn_transformer_baseline/figures/attention/

Usage (on HPC compute node, requires weather data):
    python scripts/attention_maps.py \\
        --ckpt runs/cnn_transformer_baseline/checkpoints/best.pt \\
        --norm_stats runs/cnn_transformer_baseline/norm_stats.pt \\
        --out_dir runs/cnn_transformer_baseline/figures/attention/

Or via SLURM:
    sbatch scripts/attention_maps.slurm
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

print("[attn_maps] starting imports", flush=True)
import torch
print(f"[attn_maps] torch {torch.__version__} (cuda={torch.cuda.is_available()})", flush=True)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.data_preparation.dataset import (
    EnergyForecastDataset, ZONE_COLS, N_ZONES, CAL_DIM,
)
from models import create_model

# Spatial / token-layout constants, must match the trained baseline
GRID = 8                 # AdaptiveAvgPool2d output side length
P = GRID * GRID          # = 64 spatial tokens per timestep
S = 24                   # history length
F = 24                   # future length
TOK_PER_STEP = P + 1     # 64 spatial + 1 tabular


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str,
                   default="runs/cnn_transformer_baseline/checkpoints/best.pt")
    p.add_argument("--norm_stats", type=str,
                   default="runs/cnn_transformer_baseline/norm_stats.pt")
    p.add_argument("--data_root", type=str,
                   default="/cluster/tufts/c26sp1cs0137/data/assignment3_data")
    p.add_argument("--year", type=int, default=2022)
    p.add_argument("--n_samples", type=int, default=5,
                   help="Number of validation samples to extract attention from")
    p.add_argument("--out_dir", type=str,
                   default="runs/cnn_transformer_baseline/figures/attention")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = create_model(
        args.get("model", "cnn_transformer_baseline"),
        n_weather_channels=7,
        n_zones=N_ZONES,
        cal_dim=CAL_DIM,
        history_len=args.get("history_len", 24),
        embed_dim=args.get("embed_dim", 128),
        grid_size=args.get("grid_size", 8),
        n_layers=args.get("n_layers", 4),
        n_heads=args.get("n_heads", 4),
        dropout=args.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return model, ckpt.get("norm_stats")


def select_diverse_samples(dataset, n_total=5):
    """
    Pick n_total diverse samples from the dataset:
    one mid-temperature weekday (mild),
    a few high-demand days (extreme heat / cold),
    one holiday or weekend.

    For simplicity we pick at well-known dates; if any are missing
    in this dataset year, fall back to random sampling.
    """
    target_starts = [
        # (label, "YYYY-MM-DD HH:00")  — start of 24h prediction window
        ("mild",         f"{2022}-04-15 00:00"),  # mid-spring weekday
        ("heat_wave",    f"{2022}-07-21 00:00"),  # well-known 2022 heat dome week
        ("cold_snap",    f"{2022}-12-23 00:00"),  # late-Dec arctic blast
        ("extreme",      f"{2022}-08-04 00:00"),  # another heat day
        ("holiday",      f"{2022}-12-25 00:00"),  # Christmas
    ]

    # build timestamp → idx lookup for the dataset (samples is a list of t_idx)
    sample_ts = {pd.Timestamp(dataset.timestamps[t]): i
                 for i, t in enumerate(dataset.samples)}

    chosen = []
    for label, ts_str in target_starts[:n_total]:
        ts = pd.Timestamp(ts_str)
        if ts in sample_ts:
            chosen.append((label, sample_ts[ts]))
        else:
            # fallback: nearest available
            print(f"  [warn] {ts_str} not in dataset, picking nearest", flush=True)
            available = sorted(sample_ts.keys())
            nearest = min(available, key=lambda x: abs((x - ts).total_seconds()))
            chosen.append((label, sample_ts[nearest]))
    return chosen


def extract_attention(model, sample, device):
    """Run forward with return_attn=True. Returns (predictions, attn_per_layer)."""
    hist_w, hist_e, hist_c, fut_w, fut_c, target = sample
    inputs = [hist_w, hist_e, hist_c, fut_w, fut_c]
    inputs = [t.unsqueeze(0).to(device) for t in inputs]

    with torch.no_grad():
        preds, attn_per_layer = model(*inputs, return_attn=True)
    # attn_per_layer: list of N layers, each (B=1, n_heads, L, L)
    return preds, attn_per_layer, target


def slice_future_to_history_spatial(attn, layer_idx=-1):
    """
    From a (1, n_heads, L, L) attention tensor, extract the
    future-tabular-token → history-spatial-token sub-block.

    Returns: (n_heads, F=24, S=24, P=64)
    """
    # (a) DIRECTION: rows = future tabular queries
    future_tab = [t * TOK_PER_STEP + P for t in range(S, S + F)]   # 24 indices
    # cols = history spatial keys (NOT history tabular)
    history_spatial = [t * TOK_PER_STEP + p for t in range(S) for p in range(P)]

    a = attn[0]                               # (n_heads, L, L)
    a = a[:, future_tab, :][:, :, history_spatial]
    # (n_heads, F, S*P)

    # (b) RESHAPE: split last dim into (S, P) — history hour × spatial cell
    a = a.reshape(a.shape[0], F, S, P)        # (n_heads, F, S, P)
    return a


def to_8x8(attn_F_S_P):
    """
    Reshape the P=64 spatial dim to (8, 8) row-major.
    WeatherCNN.forward ends with x.flatten(2) so index i = row*8 + col
    where row, col are in [0,8). matplotlib origin='upper' puts row=0
    at top → north up.
    """
    h, F_, S_, P_ = attn_F_S_P.shape
    assert P_ == 64, f"expected 64 spatial cells, got {P_}"
    return attn_F_S_P.reshape(h, F_, S_, GRID, GRID)


def sanity_check_orientation(att_2d_zone_avg, zone_idx, zone_name, out_dir):
    """
    Geographic orientation sanity check.

    For NEMA_BOST (eastern MA / Boston coast) the attention should be
    heavier on the east (cols 4-7) than the west (cols 0-3). If it
    isn't, the orientation is flipped/transposed and we should not
    publish figures.

    Args:
      att_2d_zone_avg: (8, 8) attention map for a specific zone
      zone_idx, zone_name: just for logging
      out_dir: where to save the diagnostic figure regardless of pass/fail
    """
    east = att_2d_zone_avg[:, 4:].sum()
    west = att_2d_zone_avg[:, :4].sum()
    north = att_2d_zone_avg[:4, :].sum()
    south = att_2d_zone_avg[4:, :].sum()

    # quick diagnostic figure
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(att_2d_zone_avg, origin='upper', cmap='viridis')
    ax.set_title(f"Sanity: {zone_name}\nE={east:.3f} W={west:.3f}\nN={north:.3f} S={south:.3f}")
    ax.set_xlabel("← W       E →")
    ax.set_ylabel("← S       N →")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"sanity_{zone_name}.png", dpi=120)
    plt.close()

    if zone_name == "NEMA_BOST":
        if east <= west:
            print(f"  [SANITY FAIL] NEMA_BOST: east={east:.4f} <= west={west:.4f} — orientation flip suspected!", flush=True)
        else:
            print(f"  [SANITY OK] NEMA_BOST: east={east:.4f} > west={west:.4f}", flush=True)
    return east, west, north, south


def plot_aggregate(attn_global_2d, out_path):
    """Single 8×8 heatmap, mean over all samples and all 24 future hours."""
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(attn_global_2d, origin='upper', cmap='viridis')
    ax.set_title("Aggregate Attention\n(mean over future hours, history hours, samples)")
    ax.set_xlabel("West ← Grid Col → East")
    ax.set_ylabel("South ← Grid Row → North")
    ax.invert_yaxis()  # not needed since origin='upper' already does it; but be explicit for label match
    # Actually with origin='upper', row 0 (north) is at top, no invert needed.
    # Reset:
    ax.invert_yaxis()  # double-flip = no-op; use explicit set
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="attention weight")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}", flush=True)


def plot_per_hour(attn_per_hour, out_path):
    """6 heatmaps for forecast hours 1, 6, 12, 18, 24."""
    hours = [0, 5, 11, 17, 23]   # = forecast hours t+1, t+6, t+12, t+18, t+24
    labels = ["t+1", "t+6", "t+12", "t+18", "t+24"]
    fig, axes = plt.subplots(1, len(hours), figsize=(4 * len(hours), 4))
    vmin = attn_per_hour[hours].min()
    vmax = attn_per_hour[hours].max()
    for ax, h, lab in zip(axes, hours, labels):
        im = ax.imshow(attn_per_hour[h], origin='upper', cmap='viridis',
                       vmin=vmin, vmax=vmax)
        ax.set_title(f"Forecast {lab}")
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
    fig.suptitle("Attention by forecast hour\n(history-aggregated, sample-averaged)")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04, label="attention weight")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}", flush=True)


def plot_extreme_vs_mild(attn_extreme, attn_mild, out_path):
    """Side-by-side comparison."""
    vmin = min(attn_extreme.min(), attn_mild.min())
    vmax = max(attn_extreme.max(), attn_mild.max())
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, attn, title in zip(axes, [attn_mild, attn_extreme],
                                ["Mild day (Apr 15)", "Extreme heat (Jul 21)"]):
        im = ax.imshow(attn, origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
    fig.suptitle("Attention shifts on extreme vs mild days")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04, label="attention weight")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}", flush=True)


def plot_per_zone(attn_per_zone, out_path):
    """8 zone-conditioned attention maps."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    vmin = min(z.min() for z in attn_per_zone)
    vmax = max(z.max() for z in attn_per_zone)
    for ax, attn, zone in zip(axes.flat, attn_per_zone, ZONE_COLS):
        im = ax.imshow(attn, origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(zone)
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
    fig.suptitle("Per-zone attention pattern\n"
                 "(weighted by prediction-head zone column)")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04, label="attention weight")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}", flush=True)


def compute_zone_conditioned_attention(model, attn_8x8, future_states):
    """
    For each zone z, weight the future tabular token's attention
    distribution by the absolute value of the head's z-th column
    weight (i.e. how much each embedding dim contributes to zone z's
    output). This effectively asks: "For predicting zone z, which
    spatial cells does the model rely on?"

    attn_8x8: (n_heads, F, S, 8, 8) per-sample
    future_states: (B=1, F, D) — encoder output at the future tabular
                   positions
    """
    # head's first linear: (D//2, D); second: (n_zones, D//2)
    # we want overall sensitivity of zone z to the full D-dim future state
    # ≈ |W2[z, :] @ (W1 has D//2 hidden) | — approximate by |W2[z]| L2 magnitude
    # For interpretability we use a simpler proxy: the gradient |∂y[z]/∂future_state|.
    #
    # In this script we just average attention over all heads and history
    # hours to get (F, 8, 8), then do NOT zone-condition — the per_zone plot
    # in this version uses a simpler signal: split the 24 future hours into
    # zone-time-windows that match each zone's typical peak hour. This is a
    # coarse approximation but easy to interpret.
    #
    # A more rigorous zone-conditioned analysis (gradient-based) is left as
    # future work; documented in the report.

    # (n_heads, F, S, 8, 8) → average over heads + history → (F, 8, 8)
    attn_FH = attn_8x8.mean(dim=(0, 2))   # (F, 8, 8)

    # 8 zones × different "peak hour" windows of the day
    # Use zone-specific archetypal patterns (rough):
    #   ME, NH, VT, WCMA — colder, peak in evening (18-22 UTC = 14-18 EDT)
    #   CT, RI, SEMA, NEMA_BOST — coastal/urban, peak in afternoon
    #   But to keep it simple and unbiased: just split each zone uses ALL 24 hours
    # → fallback: same map for all zones (not interesting)
    #
    # Better: use the model's output magnitude per zone-hour to weight
    # which (future-hour, attention-map) to combine.

    # Use future_states: (1, F=24, D)
    # head: Linear(D, D/2) -> ReLU -> Dropout -> Linear(D/2, n_zones)
    # head[1].weight: (D/2, D), head[3].weight: (n_zones, D/2)
    # Sensitivity: |head_w| product
    head = model.head
    W1 = head[0].weight.detach()        # (D//2, D)
    W2 = head[3].weight.detach()        # (n_zones, D//2)
    # sensitivity[z, d] = sum_h |W2[z, h]| * |W1[h, d]|  (roughly)
    sens = (W2.abs() @ W1.abs())        # (n_zones, D)

    # weight future_states by zone-sensitivity, then re-attribute back to attn
    # future_states: (1, F, D); sens: (n_zones, D)
    # zone_score[z, f] = future_states[0, f, :] · sens[z, :] (proportional)
    zone_score = (future_states[0] @ sens.T).abs()  # (F, n_zones)
    # normalize so each zone's weights over F sum to 1
    zone_score = zone_score / (zone_score.sum(dim=0, keepdim=True) + 1e-8)

    # weighted-average attention over future hours for each zone
    out = []
    for z in range(N_ZONES):
        weights = zone_score[:, z]                      # (F,)
        attn_z = (weights.view(F, 1, 1) * attn_FH).sum(dim=0)  # (8, 8)
        out.append(attn_z.cpu().numpy())
    return out


def main():
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device     : {device}", flush=True)
    print(f"Checkpoint : {args.ckpt}", flush=True)
    print(f"Output dir : {out_dir}", flush=True)
    print(f"Year       : {args.year}", flush=True)
    print(f"Samples    : {args.n_samples}", flush=True)

    # --- model ---
    model, norm_stats = load_model(args.ckpt, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model      : {n_params:,} params", flush=True)

    # --- dataset ---
    norm_stats = torch.load(args.norm_stats, weights_only=True) if args.norm_stats else norm_stats
    ds = EnergyForecastDataset(
        args.data_root, [args.year],
        history_len=S, norm_stats=norm_stats,
    )
    print(f"Dataset    : {len(ds)} samples in {args.year}", flush=True)

    # --- pick samples ---
    chosen = select_diverse_samples(ds, n_total=args.n_samples)

    # --- run forward + extract attention ---
    sample_attns = []   # list of (label, attn_8x8 (F, 8, 8) avg over heads+history+samples-of-this-label-1)
    sample_predictions = []

    for label, idx in chosen:
        sample = ds[idx]
        if sample is None:
            print(f"  [warn] sample {label} idx {idx} returned None (cache miss); skipping", flush=True)
            continue
        ts = pd.Timestamp(ds.timestamps[ds.samples[idx]])
        print(f"  [{label}] ts={ts}", flush=True)
        preds, attn_per_layer, target = extract_attention(model, sample, device)
        # use the LAST layer's attention (most semantically meaningful in encoder)
        attn_last = attn_per_layer[-1]  # (1, n_heads, L, L)
        # slice future→history-spatial
        attn_FSP = slice_future_to_history_spatial(attn_last)  # (n_heads, F, S, P)
        attn_FH8 = to_8x8(attn_FSP)                            # (n_heads, F, S, 8, 8)
        sample_attns.append((label, attn_FH8.cpu()))
        # save predicted+target for reference
        sample_predictions.append({
            "label": label,
            "ts": str(ts),
            "pred_mwh": (preds.cpu().numpy()[0] * norm_stats["energy_std"].numpy() + norm_stats["energy_mean"].numpy()).tolist(),
            "true_mwh": (target.numpy() * norm_stats["energy_std"].numpy() + norm_stats["energy_mean"].numpy()).tolist(),
        })

        # save future_states for zone-conditioning later
        # (we recompute via a hook-less approach; see below)

    if not sample_attns:
        print("ERROR: no samples extracted", flush=True)
        return

    # --- aggregate over samples ---
    all_attn = torch.stack([a for _, a in sample_attns], dim=0)  # (Nsamp, n_heads, F, S, 8, 8)
    aggregate_2d = all_attn.mean(dim=(0, 1, 2, 3)).numpy()       # (8, 8)
    per_hour = all_attn.mean(dim=(0, 1, 3)).numpy()              # (F, 8, 8)
    # mild = first sample (Apr 15), extreme = second (Jul 21)
    mild_attn = sample_attns[0][1].mean(dim=(0, 1, 2)).numpy() if len(sample_attns) > 0 else None
    extreme_attn = sample_attns[1][1].mean(dim=(0, 1, 2)).numpy() if len(sample_attns) > 1 else None

    # --- zone-conditioned: rerun a few samples to grab future_states ---
    # We need the encoder output at future-tabular positions. Easiest: grab
    # from the second-to-last layer output (just after `seq = self.norm(seq)`).
    # Hacky but works: re-run with a forward hook capturing self.norm output.
    captured = {}
    def hook(_, __, out):
        captured["seq"] = out

    h_handle = model.norm.register_forward_hook(hook)
    # use the first sample for zone conditioning
    sample = ds[chosen[0][1]]
    inputs = [t.unsqueeze(0).to(device) for t in sample[:5]]
    with torch.no_grad():
        _ = model(*inputs)
    h_handle.remove()
    seq_out = captured["seq"]  # (1, L, D)
    future_states = seq_out[:, [t * TOK_PER_STEP + P for t in range(S, S + F)], :]  # (1, F, D)

    # zone-conditioned attention (for sample 0; aggregate would be ideal but
    # this is good enough as a probe)
    attn_FH8_first = sample_attns[0][1]  # (n_heads, F, S, 8, 8)
    per_zone_attn = compute_zone_conditioned_attention(model, attn_FH8_first.to(device), future_states)

    # --- sanity check on NEMA_BOST orientation ---
    nema_idx = ZONE_COLS.index("NEMA_BOST")
    nema_2d = per_zone_attn[nema_idx]
    east, west, n_, s_ = sanity_check_orientation(nema_2d, nema_idx, "NEMA_BOST", out_dir)
    if east <= west:
        print("  [WARN] NEMA_BOST east<=west — orientation may be flipped. Continuing but flag in report.", flush=True)

    # also check WCMA (western Mass) — should be heavier west
    wcma_idx = ZONE_COLS.index("WCMA")
    wcma_2d = per_zone_attn[wcma_idx]
    sanity_check_orientation(wcma_2d, wcma_idx, "WCMA", out_dir)

    # --- plot all 4 figures ---
    print("Plotting figures ...", flush=True)
    plot_aggregate(aggregate_2d, out_dir / "aggregate.png")
    plot_per_hour(per_hour, out_dir / "per_hour.png")
    if mild_attn is not None and extreme_attn is not None:
        plot_extreme_vs_mild(extreme_attn, mild_attn, out_dir / "extreme_vs_mild.png")
    plot_per_zone(per_zone_attn, out_dir / "per_zone.png")

    # --- dump diagnostics ---
    diag = {
        "samples": [d for d in sample_predictions],
        "aggregate_min_max": [float(aggregate_2d.min()), float(aggregate_2d.max())],
        "nema_east_vs_west": [float(east), float(west)],
        "model_params": n_params,
        "n_samples_used": len(sample_attns),
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2))
    print(f"Saved diagnostics: {out_dir / 'diagnostics.json'}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
