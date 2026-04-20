#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
import os
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model


# -------------------------
# IO
# -------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -------------------------
# Probability helpers
# -------------------------

def entropy_from_probs(p: np.ndarray) -> float:
    p = np.clip(p, 1e-18, 1.0)
    return float(-(p * np.log(p)).sum())


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-18, 1.0)
    q = np.clip(q, 1e-18, 1.0)
    return float((p * (np.log(p) - np.log(q))).sum())


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def normalize_weights(p: torch.Tensor) -> torch.Tensor:
    mean = p.mean().clamp(min=1e-12)
    return p / mean


# -------------------------
# Training dataset
# -------------------------

@dataclass
class PairEx:
    prompt: str
    chosen: str
    rejected: str


class PairDataset(Dataset):
    def __init__(self, rows: List[dict]):
        self.data: List[PairEx] = []
        for r in rows:
            p = r.get("prompt")
            c = r.get("chosen")
            rj = r.get("rejected")
            if isinstance(p, str) and isinstance(c, str) and isinstance(rj, str):
                self.data.append(PairEx(p, c, rj))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return {
            "idx": idx,
            "prompt": ex.prompt,
            "chosen": ex.chosen,
            "rejected": ex.rejected,
        }


def collate(batch: List[dict]) -> dict:
    return {
        "idx": [b["idx"] for b in batch],
        "prompt": [b["prompt"] for b in batch],
        "chosen": [b["chosen"] for b in batch],
        "rejected": [b["rejected"] for b in batch],
    }


# -------------------------
# Eval prompt-response support
# -------------------------

def load_eval_prompt_responses(path: str) -> Tuple[List[str], List[List[str]], List[List[float]], List[int]]:
    """
    Expect each row like:
    {
      "prompt_id": ...,
      "prompt": "...",
      "K": 4,
      "responses": [
        {"response_id": 0, "text": "...", "u": ...},
        ...
      ]
    }
    """
    rows = read_jsonl(path)
    prompts = []
    responses_by_prompt = []
    u_by_prompt = []
    prompt_ids = []

    for r in rows:
        prompt = r["prompt"]
        responses = r["responses"]
        prompt_id = r.get("prompt_id", len(prompt_ids))

        texts = [x["text"] for x in responses]
        us = [float(x.get("u", np.nan)) for x in responses]

        prompts.append(prompt)
        responses_by_prompt.append(texts)
        u_by_prompt.append(us)
        prompt_ids.append(int(prompt_id))

    return prompts, responses_by_prompt, u_by_prompt, prompt_ids


# -------------------------
# Tokenization + logprob
# -------------------------

def build_batch(tokenizer, prompts: List[str], responses: List[str], max_length: int, device) -> Dict[str, torch.Tensor]:
    """
    input_ids = prompt_ids + response_ids (+eos)
    labels = -100 for prompt tokens; ids for response tokens
    """
    assert len(prompts) == len(responses)
    input_ids_list = []
    attn_list = []
    labels_list = []

    eos = tokenizer.eos_token_id

    for p, y in zip(prompts, responses):
        p_ids = tokenizer(p, add_special_tokens=False).input_ids
        y_ids = tokenizer(y, add_special_tokens=False).input_ids
        if eos is not None:
            y_ids = y_ids + [eos]

        ids = p_ids + y_ids
        if max_length and max_length > 0 and len(ids) > max_length:
            ids = ids[-max_length:]

        resp_len = min(len(y_ids), len(ids))
        prompt_len = len(ids) - resp_len

        labels = [-100] * prompt_len + ids[prompt_len:]
        attn = [1] * len(ids)

        input_ids_list.append(torch.tensor(ids, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))
        attn_list.append(torch.tensor(attn, dtype=torch.long))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
    attn = torch.nn.utils.rnn.pad_sequence(attn_list, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attn.to(device),
        "labels": labels.to(device),
    }


def sum_logprob_and_count_from_outputs(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    labels_s = labels[:, 1:].contiguous()
    logits_s = logits[:, :-1, :].contiguous()
    mask = labels_s != -100

    logp = torch.log_softmax(logits_s, dim=-1)
    tgt = labels_s.clamp(min=0)
    gathered = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    gathered = gathered * mask

    s = gathered.sum(dim=1)
    c = mask.sum(dim=1).clamp(min=1)
    return s, c


@torch.no_grad()
def batch_avg_logprob(model, tokenizer, prompts: List[str], responses: List[str], max_length: int, device) -> torch.Tensor:
    batch = build_batch(tokenizer, prompts, responses, max_length, device)
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
    s, c = sum_logprob_and_count_from_outputs(out.logits, batch["labels"])
    return (s / c).float().cpu()


@torch.no_grad()
def batch_sum_logprob(model, tokenizer, prompts: List[str], responses: List[str], max_length: int, device) -> torch.Tensor:
    batch = build_batch(tokenizer, prompts, responses, max_length, device)
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
    s, _ = sum_logprob_and_count_from_outputs(out.logits, batch["labels"])
    return s.float().cpu()


def ipo_loss_from_delta(delta: torch.Tensor, beta: float) -> torch.Tensor:
    beta = max(beta, 1e-6)
    target = 1.0 / (2.0 * beta)
    return (delta - target) ** 2


# -------------------------
# Prompt-level convergence / oscillation
# -------------------------

def update_prompt_convergence(
    prompt_tvs: Optional[np.ndarray],
    stable_counts: np.ndarray,
    converged_mask: np.ndarray,
    current_iter: int,
    min_iters: int,
    patience: int,
    tv_abs_tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if prompt_tvs is None:
        return stable_counts, converged_mask
    if current_iter < min_iters:
        return stable_counts, converged_mask

    active = ~converged_mask
    stable_counts[active] = np.where(
        prompt_tvs[active] <= tv_abs_tol,
        stable_counts[active] + 1,
        0,
    )
    needed = max(1, int(patience))
    newly_converged = active & (stable_counts >= needed)
    converged_mask[newly_converged] = True
    return stable_counts, converged_mask


def detect_oscillation_from_history(
    top1_history: np.ndarray,
    tv_history: np.ndarray,
    converged_mask: np.ndarray,
    min_iters: int,
    current_iter: int,
    osc_window: int,
    osc_min_switches: int,
    osc_tv_floor: float,
) -> np.ndarray:
    """
    Returns boolean mask: prompt is oscillatory.
    Logic:
      - only after enough iters
      - not already converged
      - within the last osc_window iterations, top-1 changes enough times
      - and mean TV over same window stays above a floor
    """
    P = top1_history.shape[1]
    out = np.zeros(P, dtype=bool)

    if current_iter < max(min_iters, osc_window):
        return out

    window = min(osc_window, top1_history.shape[0])
    top_hist = top1_history[-window:, :]   # [W, P]
    tv_hist = tv_history[-window:, :]      # [W, P]

    for p in range(P):
        if converged_mask[p]:
            continue

        top_seq = top_hist[:, p]
        if np.any(top_seq < 0):
            continue

        switches = int(np.sum(top_seq[1:] != top_seq[:-1]))
        mean_tv = float(np.nanmean(tv_hist[:, p]))

        if switches >= osc_min_switches and mean_tv >= osc_tv_floor:
            out[p] = True

    return out


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--pairs_path", type=str, required=True)
    ap.add_argument("--eval_prompts_path", type=str, required=True)

    ap.add_argument("--out_dir", type=str, default="checkpoints_fast")
    ap.add_argument("--log_dir", type=str, default="logs")
    ap.add_argument("--seed", type=int, default=0)

    # iters
    ap.add_argument("--iters", type=int, default=10, help="Used only when --auto_stop 0.")
    ap.add_argument("--auto_stop", type=int, default=1)
    ap.add_argument("--max_iters", type=int, default=50)
    ap.add_argument("--stop_min_iters", type=int, default=15)
    ap.add_argument("--stop_patience", type=int, default=5)
    ap.add_argument("--stop_tv_abs", type=float, default=0.005)

    # oscillation diagnostics
    ap.add_argument("--osc_detect", type=int, default=1,
                    help="1: detect oscillatory prompts and treat them as resolved for stopping.")
    ap.add_argument("--osc_window", type=int, default=8,
                    help="Lookback window for oscillation detection.")
    ap.add_argument("--osc_min_switches", type=int, default=4,
                    help="Min top-1 switches within window to flag oscillation.")
    ap.add_argument("--osc_tv_floor", type=float, default=0.01,
                    help="Mean TV floor within oscillation window.")

    ap.add_argument("--epochs_per_iter", type=int, default=1)

    # theory knobs
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--lambda_on", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1)

    ap.add_argument("--w_clip_min", type=float, default=0.1)
    ap.add_argument("--w_clip_max", type=float, default=10.0)
    ap.add_argument("--max_length", type=int, default=256)

    ap.add_argument("--train_sample_size", type=int, default=50)

    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--score_batch_size", type=int, default=8)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--print_last_k_bench", type=int, default=10)
    ap.add_argument("--dump_each_iter", type=int, default=1)

    args = ap.parse_args()

    args.alpha = float(max(0.0, min(1.0, args.alpha)))
    args.lambda_on = float(max(0.0, min(1.0, args.lambda_on)))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA not available.")

    ensure_dir(args.out_dir)
    ensure_dir(args.log_dir)

    # -------------------------
    # Training data
    # -------------------------
    rows = read_jsonl(args.pairs_path)
    ds = PairDataset(rows)
    N = len(ds)
    print(f"[Train data] N={N}")

    # -------------------------
    # Eval prompt-response data
    # -------------------------
    prompts_uniq, responses_by_prompt, u_by_prompt, prompt_ids_raw = load_eval_prompt_responses(args.eval_prompts_path)
    num_prompts_eval = len(prompts_uniq)
    k_list = [len(x) for x in responses_by_prompt]
    max_k = max(k_list)

    print(f"[Eval prompts] num_prompts={num_prompts_eval} | minK={min(k_list)} | maxK={max(k_list)} | meanK={sum(k_list)/len(k_list):.4f}")

    # Save eval support map once
    support_map_path = os.path.join(
        args.log_dir,
        f"eval_prompt_support_alpha{args.alpha}_lambda{args.lambda_on}_tau{args.tau}_seed{args.seed}.json"
    )
    if not os.path.exists(support_map_path):
        write_json(support_map_path, {
            "num_prompts": num_prompts_eval,
            "prompt_ids": prompt_ids_raw,
            "prompts": prompts_uniq,
            "responses_by_prompt": responses_by_prompt,
            "u_by_prompt": u_by_prompt,
            "k_list": k_list,
        })
        print("[DIAG] wrote eval prompt support map:", support_map_path)

    # flattened eval scoring support
    flat_prompts = []
    flat_resps = []
    flat_prompt_idx = []
    group_offsets = []
    cur = 0
    for pid, (x, ys) in enumerate(zip(prompts_uniq, responses_by_prompt)):
        s = cur
        for y in ys:
            flat_prompts.append(x)
            flat_resps.append(y)
            flat_prompt_idx.append(pid)
            cur += 1
        e = cur
        group_offsets.append((s, e))

    # -------------------------
    # Tokenizer / models
    # -------------------------
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=None,
    ).to(device)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    ref0 = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=None,
    ).to(device)
    ref0.eval()
    for p in ref0.parameters():
        p.requires_grad_(False)

    # -------------------------
    # State trackers
    # -------------------------
    metrics: List[dict] = []

    bench_p_hist: List[np.ndarray] = []
    bench_mu_hist: List[np.ndarray] = []

    prev_q_prompt_matrix: Optional[np.ndarray] = None
    prev_prompt_entropies: Optional[np.ndarray] = None

    prompt_stable_counts = np.zeros(num_prompts_eval, dtype=np.int64)
    prompt_converged_mask = np.zeros(num_prompts_eval, dtype=bool)
    prompt_oscillatory_mask = np.zeros(num_prompts_eval, dtype=bool)
    prompt_first_converged_iter = np.full(num_prompts_eval, -1, dtype=np.int64)
    prompt_first_oscillatory_iter = np.full(num_prompts_eval, -1, dtype=np.int64)

    top1_history = []
    tv_history = []

    dump_dir = os.path.join(
        args.log_dir,
        f"iter_dumps_alpha{args.alpha}_lambda{args.lambda_on}_tau{args.tau}_seed{args.seed}"
    )
    if args.dump_each_iter == 1:
        ensure_dir(dump_dir)

    T = args.max_iters if args.auto_stop == 1 else args.iters
    pct_eps = 1e-12

    # -------------------------
    # Main loop
    # -------------------------
    for t in range(T):
        print(f"\n===== OUTER ITER {t} | alpha={args.alpha} lambda={args.lambda_on} tau={args.tau} beta={args.beta} =====")

        model.eval()
        ref0.eval()

        # -------------------------
        # Pair-level bench diagnostics on a random train subset
        # (kept only as training-distribution diagnostic)
        # -------------------------
        bench_k = min(args.train_sample_size, N)
        rng_bench = random.Random(args.seed + 777 * (t + 1))
        bench_indices = rng_bench.sample(range(N), k=bench_k)
        bench_ds = Subset(ds, bench_indices)
        bench_loader = DataLoader(
            bench_ds,
            batch_size=args.score_batch_size,
            shuffle=False,
            collate_fn=collate,
            drop_last=False,
        )

        s_plus = torch.empty(bench_k, dtype=torch.float32)
        s_minus = torch.empty(bench_k, dtype=torch.float32)
        ptr = 0
        for batch in tqdm(bench_loader, desc=f"scoring@bench_pairs iter {t}", ncols=100):
            B = len(batch["idx"])
            lp_c = batch_avg_logprob(model, tok, batch["prompt"], batch["chosen"], args.max_length, device)
            lp_r = batch_avg_logprob(model, tok, batch["prompt"], batch["rejected"], args.max_length, device)
            s_plus[ptr:ptr+B] = lp_c
            s_minus[ptr:ptr+B] = lp_r
            ptr += B

        margin = (s_plus - s_minus)
        m = args.tau * margin
        m = m - m.max()
        p_un = torch.exp(m).clamp(min=1e-12)
        w_norm = normalize_weights(p_un)
        w_eval = (1.0 - args.lambda_on) + args.lambda_on * w_norm
        w_eval = w_eval.clamp(min=args.w_clip_min, max=args.w_clip_max)

        p_soft_pair = (p_un.double() / p_un.double().sum()).cpu().numpy()
        mu_pair = (w_eval.double() / w_eval.double().sum()).cpu().numpy()
        bench_p_hist.append(p_soft_pair.astype(np.float64))
        bench_mu_hist.append(mu_pair.astype(np.float64))

        # -------------------------
        # Prompt-level fixed-response evaluation
        # -------------------------
                # -------------------------
        # Prompt-level fixed-response evaluation
        # using avg logprob per completion token
        # -------------------------
        flat_scores = np.zeros(len(flat_prompts), dtype=np.float32)
        bs = max(1, int(args.score_batch_size))

        for s in tqdm(range(0, len(flat_prompts), bs), desc=f"scoring@eval_prompts iter {t}", ncols=100):
            e = min(len(flat_prompts), s + bs)
            lp_avg = batch_avg_logprob(
                model,
                tok,
                flat_prompts[s:e],
                flat_resps[s:e],
                args.max_length,
                device,
            )
            flat_scores[s:e] = lp_avg.numpy()

        # build prompt x max_k probability matrix, padded by NaN
        q_prompt_matrix = np.full((num_prompts_eval, max_k), np.nan, dtype=np.float64)
        prompt_entropies = np.zeros(num_prompts_eval, dtype=np.float64)
        prompt_tvs = np.full(num_prompts_eval, np.nan, dtype=np.float64)
        prompt_kls = np.full(num_prompts_eval, np.nan, dtype=np.float64)
        prompt_top1 = np.full(num_prompts_eval, -1, dtype=np.int64)

        for pid, (s, e) in enumerate(group_offsets):
            scores = flat_scores[s:e].astype(np.float64)   # avg logprob scores
            z = scores * float(args.tau)
            z = z - np.max(z)
            p = np.exp(z)
            p = p / np.sum(p)

            k = e - s
            q_prompt_matrix[pid, :k] = p
            prompt_entropies[pid] = entropy_from_probs(p)
            prompt_top1[pid] = int(np.argmax(p))

            if prev_q_prompt_matrix is not None:
                prev_p = prev_q_prompt_matrix[pid, :k].astype(np.float64)
                prev_p = prev_p / np.sum(prev_p)
                prompt_tvs[pid] = total_variation(p, prev_p)
                prompt_kls[pid] = kl_div(p, prev_p)

        prompt_entropy_mean = float(np.mean(prompt_entropies))
        prompt_tv_mean = float(np.nanmean(prompt_tvs)) if np.any(~np.isnan(prompt_tvs)) else float("nan")
        prompt_tv_max = float(np.nanmax(prompt_tvs)) if np.any(~np.isnan(prompt_tvs)) else float("nan")
        prompt_kl_mean = float(np.nanmean(prompt_kls)) if np.any(~np.isnan(prompt_kls)) else float("nan")

        if prev_prompt_entropies is None:
            prompt_entropy_abs_delta = np.full(num_prompts_eval, np.nan, dtype=np.float64)
            prompt_entropy_abs_delta_mean = float("nan")
            prompt_entropy_abs_delta_max = float("nan")
            prompt_entropy_pct_change_mean = float("nan")
        else:
            prompt_entropy_abs_delta = np.abs(prompt_entropies - prev_prompt_entropies)
            prompt_entropy_abs_delta_mean = float(np.mean(prompt_entropy_abs_delta))
            prompt_entropy_abs_delta_max = float(np.max(prompt_entropy_abs_delta))
            pct = 100.0 * (prompt_entropies - prev_prompt_entropies) / np.maximum(np.abs(prev_prompt_entropies), pct_eps)
            prompt_entropy_pct_change_mean = float(np.mean(pct))

        # -------------------------
        # Prompt-level convergence
        # -------------------------
        prev_converged_mask = prompt_converged_mask.copy()
        prompt_stable_counts, prompt_converged_mask = update_prompt_convergence(
            prompt_tvs=None if prev_q_prompt_matrix is None else prompt_tvs,
            stable_counts=prompt_stable_counts,
            converged_mask=prompt_converged_mask,
            current_iter=t,
            min_iters=args.stop_min_iters,
            patience=args.stop_patience,
            tv_abs_tol=args.stop_tv_abs,
        )
        newly_converged_mask = prompt_converged_mask & (~prev_converged_mask)
        prompt_first_converged_iter[newly_converged_mask] = t

        # -------------------------
        # Oscillation diagnostics
        # -------------------------
        top1_history.append(prompt_top1.copy())
        tv_hist_row = np.where(np.isnan(prompt_tvs), -1.0, prompt_tvs)
        tv_history.append(tv_hist_row.copy())

        top1_hist_np = np.stack(top1_history, axis=0)   # [T, P]
        tv_hist_np = np.stack(tv_history, axis=0)       # [T, P]

        if args.osc_detect == 1:
            newly_osc = detect_oscillation_from_history(
                top1_history=top1_hist_np,
                tv_history=tv_hist_np,
                converged_mask=prompt_converged_mask,
                min_iters=args.stop_min_iters,
                current_iter=t,
                osc_window=args.osc_window,
                osc_min_switches=args.osc_min_switches,
                osc_tv_floor=args.osc_tv_floor,
            )
            new_mask = newly_osc & (~prompt_oscillatory_mask)
            prompt_oscillatory_mask[new_mask] = True
            prompt_first_oscillatory_iter[new_mask] = t

        # resolved = converged OR oscillatory
        prompt_resolved_mask = prompt_converged_mask | prompt_oscillatory_mask

        num_converged = int(prompt_converged_mask.sum())
        num_osc = int(prompt_oscillatory_mask.sum())
        num_resolved = int(prompt_resolved_mask.sum())
        frac_resolved = float(num_resolved / max(num_prompts_eval, 1))
        num_unresolved = int(num_prompts_eval - num_resolved)

        unresolved_entropy_mean = float(np.mean(prompt_entropies[~prompt_resolved_mask])) if np.any(~prompt_resolved_mask) else float("nan")
        unresolved_tv_mean = float(np.nanmean(prompt_tvs[~prompt_resolved_mask])) if np.any(~prompt_resolved_mask) and np.any(~np.isnan(prompt_tvs[~prompt_resolved_mask])) else float("nan")
        unresolved_tv_max = float(np.nanmax(prompt_tvs[~prompt_resolved_mask])) if np.any(~prompt_resolved_mask) and np.any(~np.isnan(prompt_tvs[~prompt_resolved_mask])) else float("nan")

        # update prev
        prev_q_prompt_matrix = q_prompt_matrix.copy()
        prev_prompt_entropies = prompt_entropies.copy()

        # -------------------------
        # Print headline
        # -------------------------
        print(
            f"[Metrics@t={t}] "
            f"H_mean={prompt_entropy_mean:.6g} "
            f"| converged={num_converged}/{num_prompts_eval} "
            f"| oscillatory={num_osc}/{num_prompts_eval} "
            f"| resolved={num_resolved}/{num_prompts_eval} ({frac_resolved:.1%}) "
            f"| TV_mean={prompt_tv_mean:.6g} TV_max={prompt_tv_max:.6g} "
            f"| H_delta_mean={prompt_entropy_abs_delta_mean:.6g} "
            f"| KL_mean={prompt_kl_mean:.6g}"
        )

        # -------------------------
        # Save metrics row
        # -------------------------
        metrics.append({
            "iter": t,
            "alpha": args.alpha,
            "lambda": args.lambda_on,
            "tau": args.tau,
            "beta": args.beta,

            "prompt_entropy_mean": prompt_entropy_mean,
            "prompt_tv_mean": prompt_tv_mean,
            "prompt_tv_max": prompt_tv_max,
            "prompt_kl_mean": prompt_kl_mean,

            "prompt_entropy_abs_delta_mean": prompt_entropy_abs_delta_mean,
            "prompt_entropy_abs_delta_max": prompt_entropy_abs_delta_max,
            "prompt_entropy_pct_change_mean": prompt_entropy_pct_change_mean,

            "num_prompts_eval": num_prompts_eval,
            "num_prompts_converged": num_converged,
            "num_prompts_oscillatory": num_osc,
            "num_prompts_resolved": num_resolved,
            "num_prompts_unresolved": num_unresolved,
            "frac_prompts_resolved": frac_resolved,

            "unresolved_entropy_mean": unresolved_entropy_mean,
            "unresolved_prompt_tv_mean": unresolved_tv_mean,
            "unresolved_prompt_tv_max": unresolved_tv_max,

            "train_sample_size": args.train_sample_size,
        })

        # -------------------------
        # Per-iter dumps
        # -------------------------
        if args.dump_each_iter == 1:
            # compact npz: contains the whole pi-star matrix for plotting later
            dump_path = os.path.join(dump_dir, f"iter_{t:04d}.npz")
            np.savez_compressed(
                dump_path,
                iter=np.int32(t),
                q_prompt_matrix=q_prompt_matrix.astype(np.float32),   # this is your prompt-wise pi-star
                prompt_entropies=prompt_entropies.astype(np.float32),
                prompt_tvs=(prompt_tvs.astype(np.float32) if np.any(~np.isnan(prompt_tvs)) else np.full(num_prompts_eval, np.nan, dtype=np.float32)),
                prompt_kls=(prompt_kls.astype(np.float32) if np.any(~np.isnan(prompt_kls)) else np.full(num_prompts_eval, np.nan, dtype=np.float32)),
                prompt_top1=prompt_top1.astype(np.int32),
                prompt_converged_mask=prompt_converged_mask.astype(np.int8),
                prompt_oscillatory_mask=prompt_oscillatory_mask.astype(np.int8),
                prompt_resolved_mask=prompt_resolved_mask.astype(np.int8),
                prompt_stable_counts=prompt_stable_counts.astype(np.int32),
                prompt_first_converged_iter=prompt_first_converged_iter.astype(np.int32),
                prompt_first_oscillatory_iter=prompt_first_oscillatory_iter.astype(np.int32),
            )

            # per-prompt CSV, easier for inspection
            rows_csv = []
            for pid in range(num_prompts_eval):
                row = {
                    "prompt_id": prompt_ids_raw[pid],
                    "prompt_index": pid,
                    "prompt": prompts_uniq[pid],
                    "K": len(responses_by_prompt[pid]),
                    "entropy": prompt_entropies[pid],
                    "tv_delta": prompt_tvs[pid] if not np.isnan(prompt_tvs[pid]) else np.nan,
                    "kl_delta": prompt_kls[pid] if not np.isnan(prompt_kls[pid]) else np.nan,
                    "top1_idx": int(prompt_top1[pid]),
                    "converged": int(prompt_converged_mask[pid]),
                    "oscillatory": int(prompt_oscillatory_mask[pid]),
                    "resolved": int(prompt_resolved_mask[pid]),
                    "stable_count": int(prompt_stable_counts[pid]),
                    "first_converged_iter": int(prompt_first_converged_iter[pid]),
                    "first_oscillatory_iter": int(prompt_first_oscillatory_iter[pid]),
                }
                for j in range(max_k):
                    row[f"prob_{j}"] = q_prompt_matrix[pid, j]
                    row[f"response_{j}"] = responses_by_prompt[pid][j] if j < len(responses_by_prompt[pid]) else ""
                rows_csv.append(row)

            prompt_csv_path = os.path.join(dump_dir, f"iter_{t:04d}_prompt_metrics.csv")
            pd.DataFrame(rows_csv).to_csv(prompt_csv_path, index=False)

        # -------------------------
        # Auto stop
        # -------------------------
        if args.auto_stop == 1 and num_resolved == num_prompts_eval and num_prompts_eval > 0:
            print(
                f"[STOP] All prompts resolved at iter={t} "
                f"(resolved = converged OR oscillatory)."
            )
            break

        # -------------------------
        # Training on random subset
        # -------------------------
        model.train()

        train_k = min(args.train_sample_size, N)
        rng_train = random.Random(args.seed + 999 * (t + 1))
        train_indices = rng_train.sample(range(N), k=train_k)
        train_ds = Subset(ds, train_indices)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            drop_last=False,
        )

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        num_batches = len(train_loader) * args.epochs_per_iter
        total_steps = max(1, math.ceil(num_batches / args.grad_accum))
        warmup_steps = int(args.warmup_ratio * total_steps)
        sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

        opt.zero_grad(set_to_none=True)
        step = 0

        for ep in range(args.epochs_per_iter):
            pbar = tqdm(train_loader, desc=f"train@subset iter {t} ep {ep}", ncols=100)
            for batch in pbar:
                with torch.no_grad():
                    lp_c_pi = batch_avg_logprob(model, tok, batch["prompt"], batch["chosen"], args.max_length, device).to(device)
                    lp_r_pi = batch_avg_logprob(model, tok, batch["prompt"], batch["rejected"], args.max_length, device).to(device)

                    lp_c_ref0 = batch_avg_logprob(ref0, tok, batch["prompt"], batch["chosen"], args.max_length, device).to(device)
                    lp_r_ref0 = batch_avg_logprob(ref0, tok, batch["prompt"], batch["rejected"], args.max_length, device).to(device)

                    lp_c_ref_t = (1.0 - args.alpha) * lp_c_ref0 + args.alpha * lp_c_pi
                    lp_r_ref_t = (1.0 - args.alpha) * lp_r_ref0 + args.alpha * lp_r_pi

                    margin_b = (lp_c_pi - lp_r_pi)
                    mb = args.tau * margin_b
                    mb = mb - mb.max()
                    pb = torch.exp(mb).clamp(min=1e-12)
                    wnb = normalize_weights(pb)
                    wt = ((1.0 - args.lambda_on) + args.lambda_on * wnb).clamp(
                        min=args.w_clip_min, max=args.w_clip_max
                    )
                    wt = wt.detach()

                bc = build_batch(tok, batch["prompt"], batch["chosen"], args.max_length, device)
                out_c = model(input_ids=bc["input_ids"], attention_mask=bc["attention_mask"], labels=bc["labels"])
                s_c, c_c = sum_logprob_and_count_from_outputs(out_c.logits, bc["labels"])
                avg_c = s_c / c_c

                br = build_batch(tok, batch["prompt"], batch["rejected"], args.max_length, device)
                out_r = model(input_ids=br["input_ids"], attention_mask=br["attention_mask"], labels=br["labels"])
                s_r, c_r = sum_logprob_and_count_from_outputs(out_r.logits, br["labels"])
                avg_r = s_r / c_r

                delta = (avg_c - lp_c_ref_t) - (avg_r - lp_r_ref_t)
                loss_vec = ipo_loss_from_delta(delta, args.beta)
                loss = (wt * loss_vec).mean()

                loss.backward()
                step += 1

                if step % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)

                pbar.set_postfix({"loss": float(loss.item())})

    # -------------------------
    # Pair-level diagnostic dump
    # -------------------------
    K = max(0, int(args.print_last_k_bench))
    K = min(K, len(bench_mu_hist))
    if K > 0:
        diag_path = os.path.join(
            args.log_dir,
            f"benchprob_last{K}_alpha{args.alpha}_lambda{args.lambda_on}_tau{args.tau}_seed{args.seed}.npz"
        )
        np.savez_compressed(
            diag_path,
            mu=np.stack(bench_mu_hist[-K:], axis=0).astype(np.float32),
            p_soft=np.stack(bench_p_hist[-K:], axis=0).astype(np.float32),
        )
        print(f"[DIAG] saved pair-level bench distributions to: {diag_path}")

    # -------------------------
    # Summaries
    # -------------------------
    summary_json = os.path.join(
        args.log_dir,
        f"convergence_summary_alpha{args.alpha}_lambda{args.lambda_on}_tau{args.tau}_seed{args.seed}.json"
    )
    write_json(summary_json, {
        "alpha": args.alpha,
        "lambda_on": args.lambda_on,
        "tau": args.tau,
        "beta": args.beta,
        "seed": args.seed,
        "train_sample_size": args.train_sample_size,
        "num_prompts_eval": num_prompts_eval,
        "num_prompts_converged": int(prompt_converged_mask.sum()),
        "num_prompts_oscillatory": int(prompt_oscillatory_mask.sum()),
        "num_prompts_resolved": int((prompt_converged_mask | prompt_oscillatory_mask).sum()),
        "all_prompts_resolved": bool((prompt_converged_mask | prompt_oscillatory_mask).all()) if num_prompts_eval > 0 else False,
        "last_iter_ran": int(metrics[-1]["iter"]) if metrics else -1,
        "prompt_first_converged_iter": prompt_first_converged_iter.tolist(),
        "prompt_first_oscillatory_iter": prompt_first_oscillatory_iter.tolist(),
        "convergence_rule": {
            "type": "prompt_tv_on_fixed_response_set",
            "stop_tv_abs": args.stop_tv_abs,
            "stop_min_iters": args.stop_min_iters,
            "stop_patience": args.stop_patience,
        },
        "oscillation_rule": {
            "enabled": bool(args.osc_detect == 1),
            "osc_window": args.osc_window,
            "osc_min_switches": args.osc_min_switches,
            "osc_tv_floor": args.osc_tv_floor,
        },
    })

    out_csv = os.path.join(
        args.log_dir,
        f"metrics_alpha{args.alpha}_lambda{args.lambda_on}_tau{args.tau}_seed{args.seed}_FAST.csv"
    )
    pd.DataFrame(metrics).to_csv(out_csv, index=False)

    print("[DONE] wrote:", out_csv)
    print("[DONE] wrote:", summary_json)
    if args.dump_each_iter == 1:
        print("[DONE] per-iter dumps in:", dump_dir)


if __name__ == "__main__":
    main()