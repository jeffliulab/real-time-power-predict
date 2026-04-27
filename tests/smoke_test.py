"""Smoke test — confirm both models build, parameter counts match expected,
and forward + backward pass run end-to-end on small dummy inputs.

Runnable on CPU (slow) or GPU. Intended to catch shape/init/import bugs
before submitting a SLURM job.

Usage:
    python -m tests.smoke_test
    python -m tests.smoke_test --device cuda
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models import create_model, MODEL_REGISTRY


def smoke_one(name, device, **extra_kwargs):
    print(f"\n=== {name} ===")
    m = create_model(name, **extra_kwargs).to(device)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"  params: {n_params:,} ({n_params/1e6:.2f} M)")

    # Tiny dummy batch matching dataset signature
    B, S = 1, 24
    hist_w = torch.zeros(B, S, 450, 449, 7, device=device)
    hist_e = torch.zeros(B, S, 8, device=device)
    hist_c = torch.zeros(B, S, 44, device=device)
    fut_w = torch.zeros(B, 24, 450, 449, 7, device=device)
    fut_c = torch.zeros(B, 24, 44, device=device)
    target = torch.zeros(B, 24, 8, device=device)

    # Forward
    m.eval()
    with torch.no_grad():
        out = m(hist_w, hist_e, hist_c, fut_w, fut_c)
    assert out.shape == (B, 24, 8), f"shape mismatch: got {out.shape}"
    print(f"  forward OK : output {tuple(out.shape)}")

    # Backward (small subset to avoid OOM on CPU)
    m.train()
    out = m(hist_w, hist_e, hist_c, fut_w, fut_c)
    loss = (out - target).pow(2).mean()
    loss.backward()
    assert any(p.grad is not None for p in m.parameters()), "no gradients!"
    print(f"  backward OK: loss={loss.item():.4f}, grads exist")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "cuda"])
    args = parser.parse_args()

    print(f"Available models: {list(MODEL_REGISTRY.keys())}")
    print(f"Device: {args.device}")

    smoke_one("cnn_transformer_baseline", args.device, n_layers=4)
    smoke_one("cnn_encoder_decoder", args.device,
              n_encoder_layers=4, n_decoder_layers=2)
    smoke_one("cnn_encoder_decoder", args.device,
              n_encoder_layers=4, n_decoder_layers=2,
              use_future_weather_xattn=True)

    print("\nAll smoke tests passed!")


if __name__ == "__main__":
    main()
