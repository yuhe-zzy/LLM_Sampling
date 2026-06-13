#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_cyclic_pairs.py

Build cyclic pairwise training data from response-level data.

For each prompt, choose/order exactly four responses and replace the scalar-score
preference rule with a fixed cyclic rule:

    response_0 > response_1 > response_2 > response_3 > response_0

The output pair JSONL keeps the same prompt/chosen/rejected format consumed by
the current training scripts. The eval JSONL also keeps the existing
prompt/responses format and adds a preference_matrix field for audit/debugging.

Typical usage:
python scripts/build_cyclic_pairs.py \
  --input data/raw/helpsteer.jsonl \
  --out_pairs data/processed/pairs_train_cyclic.jsonl \
  --out_eval_prompts data/processed/eval_prompt_responses_cyclic.jsonl \
  --dedup_responses \
  --keep_exact_k 4 \
  --comparison_mode cycle_edges \
  --order_policy score_desc \
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


CYCLIC_K = 4


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
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def infer_score_fields(example: Dict[str, Any]) -> List[str]:
    candidates = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
    found = [k for k in candidates if k in example and is_number(example[k])]
    if found:
        return found

    blacklist = {"prompt", "response", "text", "chosen", "rejected", "id", "prompt_id", "response_id"}
    return [k for k, v in example.items() if k not in blacklist and is_number(v)]


def detect_input_format(example: Dict[str, Any]) -> str:
    if "chosen" in example and "rejected" in example:
        return "pair"
    pair_keys = {"response_a", "response_b", "y_a", "y_b", "answer_a", "answer_b"}
    if any(k in example for k in pair_keys):
        return "pair"
    if "response" in example or "answer" in example or "output" in example:
        return "response"
    if any(k.endswith("_a") for k in example) and any(k.endswith("_b") for k in example):
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


@dataclass
class RespItem:
    response: str
    u: float
    raw: Dict[str, Any]


def build_from_response_level(
    rows: Iterable[Dict[str, Any]],
    prompt_key: str,
    response_key: str,
    score_fields: List[str],
    min_k: int,
) -> Dict[str, List[RespItem]]:
    buckets: Dict[str, List[RespItem]] = defaultdict(list)
    for row in rows:
        prompt = row.get(prompt_key, None)
        response = row.get(response_key, None)
        if not isinstance(prompt, str) or not isinstance(response, str):
            continue
        u = compute_u_from_flat_scores(row, score_fields)
        if u is None:
            continue
        buckets[prompt].append(RespItem(response=response, u=u, raw=row))
    return {p: items for p, items in buckets.items() if len(items) >= min_k}


def dedup_prompt_buckets(
    buckets: Dict[str, List[RespItem]],
    strip_text: bool = True,
) -> Dict[str, List[RespItem]]:
    new_buckets: Dict[str, List[RespItem]] = {}
    for prompt, items in buckets.items():
        seen = set()
        unique_items = []
        for item in items:
            key = item.response.strip() if strip_text else item.response
            if key in seen:
                continue
            seen.add(key)
            unique_items.append(item)
        new_buckets[prompt] = unique_items
    return new_buckets


def filter_buckets_by_k(
    buckets: Dict[str, List[RespItem]],
    min_k: int = 0,
    max_k: int = 0,
    keep_exact_k: int = 0,
) -> Dict[str, List[RespItem]]:
    out = {}
    for prompt, items in buckets.items():
        k = len(items)
        if keep_exact_k > 0:
            if k == keep_exact_k:
                out[prompt] = items
            continue
        if min_k > 0 and k < min_k:
            continue
        if max_k > 0 and k > max_k:
            continue
        out[prompt] = items
    return out


def maybe_cap_bucket_size(
    buckets: Dict[str, List[RespItem]],
    cap_k: int,
    cap_policy: str,
    rng: random.Random,
) -> Dict[str, List[RespItem]]:
    if cap_k <= 0:
        return buckets

    out = {}
    for prompt, items in buckets.items():
        if len(items) <= cap_k:
            out[prompt] = items
            continue

        if cap_policy == "topu":
            chosen = sorted(items, key=lambda z: z.u, reverse=True)[:cap_k]
        elif cap_policy == "random":
            idx = list(range(len(items)))
            rng.shuffle(idx)
            chosen = [items[i] for i in idx[:cap_k]]
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
                for item in middle[::step]:
                    if len(keep) >= cap_k:
                        break
                    keep.append(item)

            used = {id(item) for item in keep}
            for item in order:
                if len(keep) >= cap_k:
                    break
                if id(item) not in used:
                    keep.append(item)
            chosen = keep[:cap_k]
        else:
            raise ValueError(f"Unknown cap_policy: {cap_policy}")

        out[prompt] = chosen
    return out


def summarize_k_distribution(buckets: Dict[str, List[RespItem]], title: str) -> None:
    ks = [len(items) for items in buckets.values()]
    print(f"\n=== {title} ===")
    if not ks:
        print("[WARN] no prompts")
        return
    cnt = Counter(ks)
    total = len(ks)
    print(f"num_prompts = {total}")
    print(f"minK = {min(ks)}, maxK = {max(ks)}, meanK = {sum(ks) / len(ks):.4f}")
    for k in sorted(cnt):
        print(f"K={k}: {cnt[k]} prompts ({100.0 * cnt[k] / total:.2f}%)")


def ordered_cycle_items(
    items: List[RespItem],
    order_policy: str,
    rng: random.Random,
) -> List[RespItem]:
    indexed = list(enumerate(items))

    if order_policy == "input":
        ordered = indexed
    elif order_policy == "score_desc":
        ordered = sorted(indexed, key=lambda x: (-x[1].u, x[0]))
    elif order_policy == "score_asc":
        ordered = sorted(indexed, key=lambda x: (x[1].u, x[0]))
    elif order_policy == "random":
        ordered = indexed[:]
        rng.shuffle(ordered)
    else:
        raise ValueError(f"Unknown order_policy: {order_policy}")

    return [it for _, it in ordered[:CYCLIC_K]]


def cycle_edges(k: int = CYCLIC_K) -> List[Tuple[int, int]]:
    return [(i, (i + 1) % k) for i in range(k)]


def full_tournament_edges(k: int = CYCLIC_K) -> List[Tuple[int, int]]:
    """
    Complete every unordered pair with a cyclic orientation.

    For k=4 this gives:
        0>1, 1>2, 2>3, 3>0, plus 0>2 and 1>3.

    A perfectly regular cyclic tournament exists only for odd k; this even-k
    rule uses the forward half-turn as a deterministic tie-break.
    """
    edges: List[Tuple[int, int]] = []
    for i in range(k):
        for j in range(i + 1, k):
            forward = (j - i) % k
            if forward <= k // 2:
                edges.append((i, j))
            else:
                edges.append((j, i))
    return edges


def comparison_edges(mode: str, k: int = CYCLIC_K) -> List[Tuple[int, int]]:
    if mode == "cycle_edges":
        return cycle_edges(k)
    if mode == "full_tournament":
        return full_tournament_edges(k)
    raise ValueError(f"Unknown comparison_mode: {mode}")


def preference_matrix_from_edges(edges: Iterable[Tuple[int, int]], k: int = CYCLIC_K) -> List[List[int]]:
    mat = [[0 for _ in range(k)] for _ in range(k)]
    for winner, loser in edges:
        mat[winner][loser] = 1
    return mat


def build_eval_rows(
    prompts: List[str],
    buckets: Dict[str, List[RespItem]],
    pref_mat: List[List[int]],
) -> List[Dict[str, Any]]:
    rows = []
    for prompt_id, prompt in enumerate(prompts):
        items = buckets[prompt]
        rows.append({
            "prompt_id": prompt_id,
            "prompt": prompt,
            "K": len(items),
            "cycle_rule": "response_0 > response_1 > response_2 > response_3 > response_0",
            "preference_matrix": pref_mat,
            "responses": [
                {
                    "response_id": j,
                    "cycle_rank": j,
                    "text": it.response,
                    "u": it.u,
                    "original_u": it.u,
                }
                for j, it in enumerate(items)
            ],
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Path to raw response-level JSONL.")
    ap.add_argument("--out_pairs", type=str, required=True, help="Output JSONL for cyclic pairs.")
    ap.add_argument("--out_eval_prompts", type=str, required=True, help="Output JSONL for eval prompts/responses.")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--input_format", type=str, default="auto", choices=["auto", "response"])
    ap.add_argument("--prompt_key", type=str, default="prompt")
    ap.add_argument("--response_key", type=str, default="response")
    ap.add_argument("--score_fields", type=str, default="", help="Comma-separated score fields. Empty => infer.")

    ap.add_argument("--min_k", type=int, default=CYCLIC_K)
    ap.add_argument("--max_k", type=int, default=0, help="Keep prompts with K <= max_k. 0 means no max filter.")
    ap.add_argument("--keep_exact_k", type=int, default=CYCLIC_K,
                    help="Default keeps only prompts with exactly 4 responses.")
    ap.add_argument("--dedup_responses", action="store_true")
    ap.add_argument("--cap_k", type=int, default=0,
                    help="Optional cap before cyclic selection. Use 4 with --keep_exact_k 0 for K>4 prompts.")
    ap.add_argument("--cap_policy", type=str, default="topu", choices=["topu", "random", "spread"])

    ap.add_argument("--order_policy", type=str, default="score_desc",
                    choices=["score_desc", "score_asc", "input", "random"],
                    help="How to order/select the four responses before applying the cycle.")
    ap.add_argument("--comparison_mode", type=str, default="cycle_edges",
                    choices=["cycle_edges", "full_tournament"],
                    help="cycle_edges emits 4 pairs; full_tournament emits all 6 directed comparisons.")
    ap.add_argument("--fixed_delta", type=float, default=1.0,
                    help="Delta written for every synthetic cyclic comparison.")

    ap.add_argument("--max_prompts", type=int, default=0, help="If >0, keep only first N prompts after shuffle.")
    ap.add_argument("--eval_prompts", type=int, default=500)

    args = ap.parse_args()
    rng = random.Random(args.seed)

    it = read_jsonl(args.input)
    try:
        first = next(it)
    except StopIteration:
        raise ValueError("Empty input file.")

    fmt = args.input_format
    if fmt == "auto":
        fmt = detect_input_format(first)
    if fmt != "response":
        raise ValueError("Cyclic construction requires response-level input with prompt/response rows.")

    if args.score_fields.strip():
        score_fields = [s.strip() for s in args.score_fields.split(",") if s.strip()]
    else:
        score_fields = infer_score_fields(first)
        if not score_fields:
            raise ValueError("Could not infer score fields. Please pass --score_fields a,b,c")

    def rows_iter():
        yield first
        for row in it:
            yield row

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

    cycle_buckets: Dict[str, List[RespItem]] = {}
    skipped_too_small = 0
    truncated = 0
    for prompt, items in buckets.items():
        if len(items) < CYCLIC_K:
            skipped_too_small += 1
            continue
        if len(items) > CYCLIC_K:
            truncated += 1
        cycle_buckets[prompt] = ordered_cycle_items(items, args.order_policy, rng)

    if not cycle_buckets:
        raise ValueError("No prompts left with at least four responses for cyclic construction.")

    prompts = list(cycle_buckets.keys())
    rng.shuffle(prompts)
    if args.max_prompts and args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    edges = comparison_edges(args.comparison_mode, CYCLIC_K)
    pref_mat = preference_matrix_from_edges(edges, CYCLIC_K)

    eval_prompt_list = prompts[: min(args.eval_prompts, len(prompts))]
    eval_rows = build_eval_rows(eval_prompt_list, cycle_buckets, pref_mat)
    write_jsonl(args.out_eval_prompts, eval_rows)

    out_pairs: List[Dict[str, Any]] = []
    pair_id = 0
    for prompt in prompts:
        items = cycle_buckets[prompt]
        for winner_idx, loser_idx in edges:
            winner = items[winner_idx]
            loser = items[loser_idx]
            out_pairs.append({
                "pair_id": pair_id,
                "prompt": prompt,
                "chosen": winner.response,
                "rejected": loser.response,
                "delta": float(args.fixed_delta),
                "meta": {
                    "source": "cyclic_response_bucket",
                    "comparison_mode": args.comparison_mode,
                    "cycle_rule": "response_0 > response_1 > response_2 > response_3 > response_0",
                    "chosen_cycle_rank": winner_idx,
                    "rejected_cycle_rank": loser_idx,
                    "u_chosen_original": winner.u,
                    "u_rejected_original": loser.u,
                    "K_prompt": len(items),
                    "score_fields": score_fields,
                },
            })
            pair_id += 1

    write_jsonl(args.out_pairs, out_pairs)

    print("\n=== Final summary ===")
    print(f"[OK] format=response | score_fields={score_fields}")
    print(f"[OK] prompts kept: {len(prompts)} | K={CYCLIC_K}")
    print(f"[OK] cyclic order policy: {args.order_policy}")
    print(f"[OK] comparison_mode={args.comparison_mode} | pairs_per_prompt={len(edges)}")
    print(f"[OK] skipped prompts with K<4 after filtering: {skipped_too_small}")
    print(f"[OK] truncated prompts with K>4 to first 4 ordered responses: {truncated}")
    print(f"[OK] wrote pairs: {args.out_pairs} (N={len(out_pairs)})")
    print(f"[OK] wrote eval prompt-response sets: {args.out_eval_prompts} (N={len(eval_rows)})")


if __name__ == "__main__":
    main()
