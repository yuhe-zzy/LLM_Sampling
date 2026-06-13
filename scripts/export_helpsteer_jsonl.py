#!/usr/bin/env python3
"""Export a saved HelpSteer dataset split to JSONL.

Default usage:

    python scripts/export_helpsteer_jsonl.py

This reads data/raw/HelpSteer, exports the train split, and writes
data/raw/helpsteer.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os

from datasets import DatasetDict, load_from_disk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default="data/raw/HelpSteer")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--output", type=str, default="data/raw/helpsteer.jsonl")
    args = ap.parse_args()

    ds_obj = load_from_disk(args.input_dir)
    if isinstance(ds_obj, DatasetDict):
        if args.split not in ds_obj:
            raise ValueError(f"Split {args.split!r} not found. Available splits: {list(ds_obj.keys())}")
        ds = ds_obj[args.split]
    else:
        ds = ds_obj

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in ds:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {args.output} rows={len(ds)}")


if __name__ == "__main__":
    main()

