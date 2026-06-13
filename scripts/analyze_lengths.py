#!/usr/bin/env python3
"""Summarize token lengths for raw response or pairwise preference JSONL files."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from typing import Any, Dict, List

import numpy as np
from transformers import AutoTokenizer


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pct(xs, p):
    return float(np.percentile(xs, p))


def summarize(name, xs):
    xs = np.array(xs, dtype=np.int64)
    return {
        "name": name,
        "count": int(xs.size),
        "min": int(xs.min()) if xs.size else None,
        "p50": int(pct(xs, 50)) if xs.size else None,
        "p90": int(pct(xs, 90)) if xs.size else None,
        "p95": int(pct(xs, 95)) if xs.size else None,
        "p99": int(pct(xs, 99)) if xs.size else None,
        "max": int(xs.max()) if xs.size else None,
        "mean": float(xs.mean()) if xs.size else None,
    }


def bucket_counts(xs, buckets):
    counts = Counter()
    for value in xs:
        placed = False
        for bucket in buckets:
            if value <= bucket:
                counts[f"<= {bucket}"] += 1
                placed = True
                break
        if not placed:
            counts[f"> {buckets[-1]}"] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--mode", choices=["response", "pair"], required=True)
    ap.add_argument("--sample", type=int, default=5000, help="0 means all rows.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--buckets", type=str, default="256,512,1024,1537,2048,4096")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    rows = read_jsonl(args.input)
    total_rows = len(rows)
    if args.sample and args.sample > 0 and len(rows) > args.sample:
        random.seed(args.seed)
        rows = random.sample(rows, args.sample)
    print(f"[Loaded] {len(rows)} rows from {total_rows}")

    prompt_lens, resp_lens, total_lens = [], [], []
    if args.mode == "response":
        for row in rows:
            prompt = row["prompt"]
            response = row["response"]
            lp = len(tok(prompt, add_special_tokens=False).input_ids)
            ly = len(tok(response, add_special_tokens=False).input_ids)
            prompt_lens.append(lp)
            resp_lens.append(ly)
            total_lens.append(lp + ly)
    else:
        for row in rows:
            prompt = row["prompt"]
            lp = len(tok(prompt, add_special_tokens=False).input_ids)
            for key in ("chosen", "rejected"):
                response = row[key]
                ly = len(tok(response, add_special_tokens=False).input_ids)
                prompt_lens.append(lp)
                resp_lens.append(ly)
                total_lens.append(lp + ly)

    print("\n=== Summary ===")
    for item in [
        summarize("prompt_tokens", prompt_lens),
        summarize("response_tokens", resp_lens),
        summarize("prompt+response_tokens", total_lens),
    ]:
        print(
            f"{item['name']}: n={item['count']} min={item['min']} p50={item['p50']} "
            f"p90={item['p90']} p95={item['p95']} p99={item['p99']} "
            f"max={item['max']} mean={item['mean']:.1f}"
        )

    buckets = [int(x.strip()) for x in args.buckets.split(",") if x.strip()]
    counts = bucket_counts(total_lens, buckets)
    print("\n=== prompt+response histogram ===")
    denom = max(1, len(total_lens))
    for key in [f"<= {b}" for b in buckets] + [f"> {buckets[-1]}"]:
        count = counts.get(key, 0)
        print(f"{key:>8}: {count:>7} ({100.0 * count / denom:5.1f}%)")


if __name__ == "__main__":
    main()

