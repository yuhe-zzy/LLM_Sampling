#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_pairs.py

Build:
1. pairwise training data from response-level or pair-level data
2. prompt-level eval data with fixed candidate responses per prompt

Main new features:
- audit prompt response-count distribution K
- optional exact-K filtering, e.g. keep only prompts with exactly 4 responses
- eval output now stores prompt + fixed responses, instead of prompt only

Typical usage (recommended for current project):
python build_pairs.py \
  --input data/raw/helpsteer.jsonl \
  --out_pairs data/processed/pairs_train.jsonl \
  --out_eval_prompts data/processed/eval_prompt_responses.jsonl \
  --input_format response \
  --dedup_responses \
  --keep_exact_k 4 \
  --pair_mode all \
  --eval_prompts 500 \
  --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# Utilities
# -----------------------------

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error in {path} at line {line_no}: {e}") from e


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_get(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d:
            return d[k]
    return None


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def infer_score_fields(example: Dict[str, Any]) -> List[str]:
    """
    Heuristic: pick numeric fields among common HelpSteer dimensions.
    Override with --score_fields if needed.
    """
    candidates = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
    found = [k for k in candidates if k in example and is_number(example[k])]
    if found:
        return found

    blacklist = {"prompt", "response", "text", "chosen", "rejected", "id", "prompt_id", "response_id"}
    fields = []
    for k, v in example.items():
        if k in blacklist:
            continue
        if is_number(v):
            fields.append(k)
    return fields


def detect_input_format(example: Dict[str, Any]) -> str:
    """
    Returns "response" or "pair".
    """
    if "chosen" in example and "rejected" in example:
        return "pair"
    pair_keys = {"response_a", "response_b", "y_a", "y_b", "answer_a", "answer_b"}
    if any(k in example for k in pair_keys):
        return "pair"
    if "response" in example or "answer" in example or "output" in example:
        return "response"
    if any(k.endswith("_a") for k in example.keys()) and any(k.endswith("_b") for k in example.keys()):
        return "pair"
    return "response"


def compute_u_from_flat_scores(row: Dict[str, Any], score_fields: List[str]) -> Optional[float]:
    vals = []
    for k in score_fields:
        v = row.get(k, None)
        if is_number(v):
            vals.append(float(v))
    if not vals:
        return None
    return sum(vals) / len(vals)


def compute_u_from_nested_scores(scores_obj: Any, score_fields: List[str]) -> Optional[float]:
    if not isinstance(scores_obj, dict):
        return None
    vals = []
    for k in score_fields:
        v = scores_obj.get(k, None)
        if is_number(v):
            vals.append(float(v))
    if not vals:
        return None
    return sum(vals) / len(vals)


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class RespItem:
    response: str
    u: float
    raw: Dict[str, Any]


# -----------------------------
# Pair generation strategies
# -----------------------------

def all_pairs_from_bucket(items: List[RespItem]) -> List[Tuple[int, int]]:
    idx = list(range(len(items)))
    pairs = []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            pairs.append((idx[i], idx[j]))
    return pairs


def sampled_pairs_from_bucket(items: List[RespItem], M: int, rng: random.Random) -> List[Tuple[int, int]]:
    """
    Sample up to M pairs per prompt with a mix of easy + hard comparisons.
    Works for K in [2, 8]. If K=2, returns 1 pair.
    """
    K = len(items)
    if K < 2:
        return []
    if K == 2:
        return [(0, 1)]

    order = sorted(range(K), key=lambda i: items[i].u, reverse=True)
    top = order[0]
    bot = order[-1]

    pairs: List[Tuple[int, int]] = []
    pairs.append((top, bot))

    if K > 2 and len(pairs) < M:
        mid = rng.choice(order[1:-1])
        pairs.append((top, mid))

    if K > 3 and len(pairs) < M:
        j = rng.randint(0, K - 2)
        a, b = order[j], order[j + 1]
        pairs.append((a, b))

    cand = all_pairs_from_bucket(items)
    rng.shuffle(cand)
    for a, b in cand:
        if len(pairs) >= M:
            break
        if (a, b) in pairs or (b, a) in pairs:
            continue
        pairs.append((a, b))

    uniq = []
    seen = set()
    for a, b in pairs:
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            uniq.append((a, b))
    return uniq[:M]


# -----------------------------
# Build helpers
# -----------------------------

def build_from_response_level(
    rows: Iterable[Dict[str, Any]],
    prompt_key: str,
    response_key: str,
    score_fields: List[str],
    min_k: int,
) -> Dict[str, List[RespItem]]:
    buckets: Dict[str, List[RespItem]] = defaultdict(list)
    for r in rows:
        prompt = r.get(prompt_key, None)
        resp = r.get(response_key, None)
        if not isinstance(prompt, str) or not isinstance(resp, str):
            continue
        u = compute_u_from_flat_scores(r, score_fields)
        if u is None:
            continue
        buckets[prompt].append(RespItem(response=resp, u=u, raw=r))

    buckets = {p: items for p, items in buckets.items() if len(items) >= min_k}
    return buckets


def build_pairs_from_pair_level(
    rows: Iterable[Dict[str, Any]],
    prompt_key: str,
    score_fields: List[str],
) -> List[Dict[str, Any]]:
    """
    Directly build (prompt, chosen, rejected, delta) from pair-level rows.
    """
    out = []
    for r in rows:
        prompt = r.get(prompt_key, None)
        if not isinstance(prompt, str):
            continue

        chosen = r.get("chosen", None)
        rejected = r.get("rejected", None)
        if isinstance(chosen, str) and isinstance(rejected, str):
            out.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "delta": None,
                "meta": {"source": "pair_labeled"}
            })
            continue

        ra = safe_get(r, ["response_a", "y_a", "answer_a"])
        rb = safe_get(r, ["response_b", "y_b", "answer_b"])
        if not (isinstance(ra, str) and isinstance(rb, str)):
            continue

        ua = None
        ub = None

        sa = safe_get(r, ["scores_a", "score_a", "rating_a"])
        sb = safe_get(r, ["scores_b", "score_b", "rating_b"])
        if isinstance(sa, dict) and isinstance(sb, dict):
            ua = compute_u_from_nested_scores(sa, score_fields)
            ub = compute_u_from_nested_scores(sb, score_fields)

        if ua is None or ub is None:
            vals_a = {}
            vals_b = {}
            for k in score_fields:
                ka = f"{k}_a"
                kb = f"{k}_b"
                if ka in r:
                    vals_a[k] = r[ka]
                if kb in r:
                    vals_b[k] = r[kb]
            ua = compute_u_from_nested_scores(vals_a, score_fields) if vals_a else ua
            ub = compute_u_from_nested_scores(vals_b, score_fields) if vals_b else ub

        if ua is None or ub is None:
            continue

        if ua >= ub:
            chosen, rejected = ra, rb
            delta = float(ua - ub)
        else:
            chosen, rejected = rb, ra
            delta = float(ub - ua)

        out.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "delta": delta,
            "meta": {"source": "pair_scored", "u_a": ua, "u_b": ub}
        })
    return out


def dedup_prompt_buckets(
    buckets: Dict[str, List[RespItem]],
    strip_text: bool = True,
) -> Dict[str, List[RespItem]]:
    new_buckets: Dict[str, List[RespItem]] = {}
    for p, items in buckets.items():
        seen = set()
        uniq = []
        for it_ in items:
            key = it_.response.strip() if strip_text else it_.response
            if key in seen:
                continue
            seen.add(key)
            uniq.append(it_)
        new_buckets[p] = uniq
    return new_buckets


def filter_buckets_by_k(
    buckets: Dict[str, List[RespItem]],
    min_k: int = 0,
    max_k: int = 0,
    keep_exact_k: int = 0,
) -> Dict[str, List[RespItem]]:
    out = {}
    for p, items in buckets.items():
        k = len(items)
        if keep_exact_k > 0:
            if k == keep_exact_k:
                out[p] = items
            continue

        if min_k > 0 and k < min_k:
            continue
        if max_k > 0 and k > max_k:
            continue
        out[p] = items
    return out


def maybe_cap_bucket_size(
    buckets: Dict[str, List[RespItem]],
    cap_k: int,
    cap_policy: str,
    rng: random.Random,
) -> Dict[str, List[RespItem]]:
    """
    If cap_k > 0 and len(items) > cap_k, keep only cap_k responses.

    cap_policy:
      - topu: keep highest-u responses
      - random: random subset
      - spread: keep top, bottom, then fill from middle
    """
    if cap_k <= 0:
        return buckets

    out = {}
    for p, items in buckets.items():
        if len(items) <= cap_k:
            out[p] = items
            continue

        if cap_policy == "topu":
            chosen = sorted(items, key=lambda z: z.u, reverse=True)[:cap_k]
        elif cap_policy == "random":
            idx = list(range(len(items)))
            rng.shuffle(idx)
            idx = idx[:cap_k]
            chosen = [items[i] for i in idx]
        elif cap_policy == "spread":
            order = sorted(items, key=lambda z: z.u, reverse=True)
            keep = []
            if cap_k >= 1:
                keep.append(order[0])
            if cap_k >= 2:
                keep.append(order[-1])

            middle = order[1:-1]
            if len(keep) < cap_k and middle:
                step = max(1, len(middle) // max(1, cap_k - len(keep)))
                picked = middle[::step]
                for z in picked:
                    if len(keep) >= cap_k:
                        break
                    keep.append(z)

            if len(keep) < cap_k:
                used = {id(z) for z in keep}
                for z in order:
                    if id(z) not in used:
                        keep.append(z)
                    if len(keep) >= cap_k:
                        break
            chosen = keep[:cap_k]
        else:
            raise ValueError(f"Unknown cap_policy: {cap_policy}")

        out[p] = chosen
    return out


def summarize_k_distribution(buckets: Dict[str, List[RespItem]], title: str) -> None:
    ks = [len(v) for v in buckets.values()]
    print(f"\n=== {title} ===")
    if not ks:
        print("[WARN] no prompts")
        return

    cnt = Counter(ks)
    total = len(ks)
    print(f"num_prompts = {total}")
    print(f"minK = {min(ks)}, maxK = {max(ks)}, meanK = {sum(ks)/len(ks):.4f}")
    for k in sorted(cnt):
        print(f"K={k}: {cnt[k]} prompts ({100.0 * cnt[k] / total:.2f}%)")


def build_eval_rows_from_buckets(
    prompts: List[str],
    buckets: Dict[str, List[RespItem]],
) -> List[Dict[str, Any]]:
    rows = []
    for i, p in enumerate(prompts):
        items = buckets[p]
        items_sorted = sorted(items, key=lambda z: z.u, reverse=True)
        rows.append({
            "prompt_id": i,
            "prompt": p,
            "K": len(items_sorted),
            "responses": [
                {
                    "response_id": j,
                    "text": it_.response,
                    "u": it_.u,
                }
                for j, it_ in enumerate(items_sorted)
            ]
        })
    return rows


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Path to raw JSONL.")
    ap.add_argument("--out_pairs", type=str, required=True, help="Output JSONL for (prompt, chosen, rejected, delta).")
    ap.add_argument("--out_eval_prompts", type=str, required=True,
                    help="Output JSONL for eval prompts WITH fixed responses.")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--input_format", type=str, default="auto", choices=["auto", "response", "pair"])
    ap.add_argument("--prompt_key", type=str, default="prompt")
    ap.add_argument("--response_key", type=str, default="response", help="For response-level input.")
    ap.add_argument("--score_fields", type=str, default="", help="Comma-separated score fields. Empty => infer.")

    # bucket filtering
    ap.add_argument("--min_k", type=int, default=2, help="Minimum #responses per prompt before later filtering.")
    ap.add_argument("--max_k", type=int, default=0, help="Keep prompts with K <= max_k. 0 means no max filter.")
    ap.add_argument("--keep_exact_k", type=int, default=0,
                    help="If >0, keep only prompts with exactly this many responses, e.g. 4.")

    # dedup / cap
    ap.add_argument("--dedup_responses", action="store_true", help="Deduplicate identical responses per prompt.")
    ap.add_argument("--cap_k", type=int, default=0,
                    help="If >0, cap each prompt to at most K responses after filtering.")
    ap.add_argument("--cap_policy", type=str, default="topu", choices=["topu", "random", "spread"],
                    help="How to cap prompts with too many responses.")

    # pair generation
    ap.add_argument("--pair_mode", type=str, default="all", choices=["all", "sample"])
    ap.add_argument("--pairs_per_prompt", type=int, default=3, help="Used when pair_mode=sample.")

    # prompt sampling
    ap.add_argument("--max_prompts", type=int, default=0, help="If >0, keep only first N prompts after shuffle.")
    ap.add_argument("--eval_prompts", type=int, default=500)

    args = ap.parse_args()
    rng = random.Random(args.seed)

    # Peek an example to infer format/fields
    it = read_jsonl(args.input)
    try:
        first = next(it)
    except StopIteration:
        raise ValueError("Empty input file.")

    # Prepare score fields
    if args.score_fields.strip():
        score_fields = [s.strip() for s in args.score_fields.split(",") if s.strip()]
    else:
        score_fields = infer_score_fields(first)
        if not score_fields:
            raise ValueError("Could not infer score fields. Please pass --score_fields a,b,c")

    # Detect format
    fmt = args.input_format
    if fmt == "auto":
        fmt = detect_input_format(first)

    def rows_iter():
        yield first
        for r in it:
            yield r

    if fmt == "pair":
        # For pair-level input, we can still build training pairs,
        # but cannot naturally reconstruct fixed response sets per prompt
        # as cleanly as response-level input.
        pairs = build_pairs_from_pair_level(
            rows=rows_iter(),
            prompt_key=args.prompt_key,
            score_fields=score_fields,
        )

        if args.max_prompts and args.max_prompts > 0:
            prompts = list({p["prompt"] for p in pairs})
            rng.shuffle(prompts)
            keep = set(prompts[: args.max_prompts])
            pairs = [p for p in pairs if p["prompt"] in keep]

        write_jsonl(args.out_pairs, pairs)

        uniq_prompts = list({p["prompt"] for p in pairs})
        rng.shuffle(uniq_prompts)
        eval_prompts = uniq_prompts[: min(args.eval_prompts, len(uniq_prompts))]
        eval_rows = [{"prompt_id": i, "prompt": p} for i, p in enumerate(eval_prompts)]
        write_jsonl(args.out_eval_prompts, eval_rows)

        print(f"[OK] format=pair | score_fields={score_fields}")
        print(f"[WARN] pair-level input cannot cleanly output fixed per-prompt response sets.")
        print(f"[OK] wrote pairs: {args.out_pairs} (N={len(pairs)})")
        print(f"[OK] wrote eval prompts: {args.out_eval_prompts} (N={len(eval_rows)})")
        return

    # -----------------------------
    # response-level path
    # -----------------------------
    buckets = build_from_response_level(
        rows=rows_iter(),
        prompt_key=args.prompt_key,
        response_key=args.response_key,
        score_fields=score_fields,
        min_k=args.min_k,
    )
    summarize_k_distribution(buckets, "K distribution after initial min_k filter")

    if args.dedup_responses:
        buckets = dedup_prompt_buckets(buckets)
        # after dedup, some prompts may fall below min_k
        buckets = {p: items for p, items in buckets.items() if len(items) >= args.min_k}
        summarize_k_distribution(buckets, "K distribution after dedup")

    buckets = filter_buckets_by_k(
        buckets,
        min_k=args.min_k,
        max_k=args.max_k,
        keep_exact_k=args.keep_exact_k,
    )
    summarize_k_distribution(buckets, "K distribution after exact/min/max filtering")

    buckets = maybe_cap_bucket_size(
        buckets=buckets,
        cap_k=args.cap_k,
        cap_policy=args.cap_policy,
        rng=rng,
    )
    summarize_k_distribution(buckets, "K distribution after optional cap_k")

    prompts = list(buckets.keys())
    rng.shuffle(prompts)

    if args.max_prompts and args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    if not prompts:
        raise ValueError("No prompts left after filtering. Relax keep_exact_k / min_k / max_k / cap settings.")

    # eval prompts sampled from same filtered prompt universe
    eval_prompt_list = prompts[: min(args.eval_prompts, len(prompts))]
    eval_rows = build_eval_rows_from_buckets(eval_prompt_list, buckets)
    write_jsonl(args.out_eval_prompts, eval_rows)

    # build global training pairs
    out_pairs: List[Dict[str, Any]] = []
    pair_id = 0

    for p in prompts:
        items = buckets[p]
        if args.pair_mode == "all":
            idx_pairs = all_pairs_from_bucket(items)
        else:
            idx_pairs = sampled_pairs_from_bucket(items, args.pairs_per_prompt, rng)

        for a, b in idx_pairs:
            ia, ib = items[a], items[b]
            if ia.u >= ib.u:
                chosen, rejected = ia, ib
                delta = float(ia.u - ib.u)
            else:
                chosen, rejected = ib, ia
                delta = float(ib.u - ia.u)

            out_pairs.append({
                "pair_id": pair_id,
                "prompt": p,
                "chosen": chosen.response,
                "rejected": rejected.response,
                "delta": delta,
                "meta": {
                    "u_chosen": chosen.u,
                    "u_rejected": rejected.u,
                    "K_prompt": len(items),
                    "score_fields": score_fields,
                    "source": "response_bucket",
                }
            })
            pair_id += 1

    write_jsonl(args.out_pairs, out_pairs)

    # final summary
    k_vals = [len(buckets[p]) for p in prompts]
    print("\n=== Final summary ===")
    print(f"[OK] format=response | score_fields={score_fields}")
    print(f"[OK] prompts kept: {len(prompts)} | minK={min(k_vals) if k_vals else 'NA'} | maxK={max(k_vals) if k_vals else 'NA'}")
    print(f"[OK] wrote pairs: {args.out_pairs} (N={len(out_pairs)})")
    print(f"[OK] wrote eval prompt-response sets: {args.out_eval_prompts} (N={len(eval_rows)})")
    if args.pair_mode == "all":
        print("[OK] pair_mode=all")
    else:
        print(f"[OK] pair_mode=sample (M={args.pairs_per_prompt} per prompt)")


if __name__ == "__main__":
    main()