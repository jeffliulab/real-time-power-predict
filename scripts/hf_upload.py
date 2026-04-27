"""
Upload trained checkpoints to Hugging Face Hub.

Usage:
    python scripts/hf_upload.py --model cnn_encoder_decoder
    python scripts/hf_upload.py --model cnn_encoder_decoder --tag with_xattn

HF repo: https://huggingface.co/jeffliulab/iso-ne-power-forecast
Token: read from ~/.hf_token (never commit) or HF_TOKEN env var.

What gets uploaded:
    runs/<MODEL>/checkpoints/best.pt    -> <MODEL>[_<tag>]/best.pt
    runs/<MODEL>/checkpoints/latest.pt  -> <MODEL>[_<tag>]/latest.pt   (--include_latest)
    runs/<MODEL>/config.json            -> <MODEL>[_<tag>]/config.json
    runs/<MODEL>/norm_stats.pt          -> <MODEL>[_<tag>]/norm_stats.pt
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

HF_REPO_ID = "jeffliulab/iso-ne-power-forecast"
ROOT = Path(__file__).parent.parent


def get_token():
    token_path = Path.home() / ".hf_token"
    if token_path.exists():
        return token_path.read_text().strip()
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    print("ERROR: No HuggingFace token found.")
    print("  Option 1: echo 'hf_xxx' > ~/.hf_token")
    print("  Option 2: export HF_TOKEN=hf_xxx")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="Upload checkpoint to HuggingFace Hub")
    p.add_argument("--model", required=True,
                   help="Model name (subdir under runs/)")
    p.add_argument("--tag", default=None,
                   help="Optional tag suffix (e.g. 'with_xattn')")
    p.add_argument("--note", default=None,
                   help="Optional note for commit message")
    p.add_argument("--include_latest", action="store_true",
                   help="Also upload latest.pt")
    p.add_argument("--repo", default=HF_REPO_ID,
                   help=f"HF repo id (default: {HF_REPO_ID})")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("Run: pip install huggingface_hub --user")
        sys.exit(1)

    token = get_token()
    api = HfApi()

    run_dir = ROOT / "runs" / args.model
    if not run_dir.exists():
        print(f"ERROR: run dir not found: {run_dir}")
        sys.exit(1)

    tag_suffix = f"_{args.tag}" if args.tag else ""
    dest_dir = f"{args.model}{tag_suffix}"

    files_to_upload = [
        ("checkpoints/best.pt", "best.pt"),
        ("config.json", "config.json"),
        ("norm_stats.pt", "norm_stats.pt"),
    ]
    if args.include_latest:
        files_to_upload.append(("checkpoints/latest.pt", "latest.pt"))

    timestamp = datetime.now().strftime("%Y%m%d")
    note = f" — {args.note}" if args.note else ""
    commit_msg = f"Upload {args.model}{tag_suffix} ({timestamp}){note}"

    for local_rel, dest_name in files_to_upload:
        src = run_dir / local_rel
        if not src.exists():
            print(f"  SKIP {local_rel} (not found)")
            continue
        dest_path = f"{dest_dir}/{dest_name}"
        print(f"  upload  {src} -> {dest_path}")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dest_path,
            repo_id=args.repo,
            token=token,
            commit_message=commit_msg,
        )

    print("\nDone:")
    print(f"  https://huggingface.co/{args.repo}/tree/main/{dest_dir}")


if __name__ == "__main__":
    main()
