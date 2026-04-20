# scripts/analyze_lengths.py
import argparse
import json
import os
import random
from collections import Counter

import numpy as np
from transformers import AutoTokenizer


def read_jsonl(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
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
    # buckets like [256, 512, 1024, 2048, 4096]
    c = Counter()
    for v in xs:
        placed = False
        for b in buckets:
            if v <= b:
                c[f"<= {b}"] += 1
                placed = True
                break
        if not placed:
            c[f"> {buckets[-1]}"] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="Local HF model dir or HF repo id")
    ap.add_argument("--input", type=str, required=True, help="Path to JSONL (helpsteer.jsonl or pairs_train.jsonl)")
    ap.add_argument("--mode", choices=["response", "pair"], required=True,
                    help="response: JSONL has prompt/response. pair: JSONL has prompt/chosen/rejected.")
    ap.add_argument("--sample", type=int, default=5000, help="How many rows to sample (0 means all).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--buckets", type=str, default="256,512,1024,2048,4096",
                    help="Comma-separated bucket edges for histogram")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    # Load rows (optionally sample)
    rows = read_jsonl(args.input, limit=None)
    n = len(rows)
    if args.sample and args.sample > 0 and n > args.sample:
        random.seed(args.seed)
        rows = random.sample(rows, args.sample)
    print(f"[Loaded] {len(rows)} rows (from {n} total)")

    prompt_lens = []
    resp_lens = []
    total_lens = []

    if args.mode == "response":
        for r in rows:
            p = r["prompt"]
            y = r["response"]
            lp = len(tok(p, add_special_tokens=False).input_ids)
            ly = len(tok(y, add_special_tokens=False).input_ids)
            prompt_lens.append(lp)
            resp_lens.append(ly)
            total_lens.append(lp + ly)
    else:  # pair
        # count both chosen and rejected as "responses"
        for r in rows:
            p = r["prompt"]
            c = r["chosen"]
            rej = r["rejected"]
            lp = len(tok(p, add_special_tokens=False).input_ids)
            lc = len(tok(c, add_special_tokens=False).input_ids)
            lr = len(tok(rej, add_special_tokens=False).input_ids)
            # treat chosen and rejected separately
            prompt_lens.append(lp); resp_lens.append(lc); total_lens.append(lp + lc)
            prompt_lens.append(lp); resp_lens.append(lr); total_lens.append(lp + lr)

    # Summaries
    s_prompt = summarize("prompt_tokens", prompt_lens)
    s_resp = summarize("response_tokens", resp_lens)
    s_total = summarize("prompt+response_tokens", total_lens)

    print("\n=== Summary (token lengths) ===")
    for s in [s_prompt, s_resp, s_total]:
        print(
            f"{s['name']}: n={s['count']}  min={s['min']}  p50={s['p50']}  p90={s['p90']}  "
            f"p95={s['p95']}  p99={s['p99']}  max={s['max']}  mean={s['mean']:.1f}"
        )

    # Buckets
    buckets = [int(x.strip()) for x in args.buckets.split(",") if x.strip()]
    bc = bucket_counts(total_lens, buckets)

    print("\n=== Histogram for prompt+response (counts) ===")
    total = len(total_lens)
    for k in [f"<= {b}" for b in buckets] + [f"> {buckets[-1]}"]:
        cnt = bc.get(k, 0)
        print(f"{k:>8}: {cnt:>7}  ({cnt/total*100:>5.1f}%)")


if __name__ == "__main__":
    main()