#!/usr/bin/env python3
"""Download a Hugging Face dataset and save it to disk.

Default usage downloads NVIDIA HelpSteer into data/raw/HelpSteer:

    python scripts/download.py

The saved dataset can then be exported to JSONL with:

    python scripts/export_helpsteer_jsonl.py
"""

from __future__ import annotations

import argparse
import os

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="nvidia/HelpSteer")
    ap.add_argument("--config", type=str, default="", help="Optional HF dataset config name.")
    ap.add_argument("--cache_dir", type=str, default="data/hf_cache")
    ap.add_argument("--save_dir", type=str, default="data/raw/HelpSteer")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_dir) or ".", exist_ok=True)

    if args.config.strip():
        ds = load_dataset(args.dataset, args.config, cache_dir=args.cache_dir)
    else:
        ds = load_dataset(args.dataset, cache_dir=args.cache_dir)

    ds.save_to_disk(args.save_dir)
    print(ds)
    print(f"[OK] saved dataset to {args.save_dir}")


if __name__ == "__main__":
    main()

