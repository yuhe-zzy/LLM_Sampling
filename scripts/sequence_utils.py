"""Shared data, response-likelihood, sampling and evaluation helpers."""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_jsonl(path: str):
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


def write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def entropy_from_probs(p: np.ndarray) -> float:
    p = np.clip(p, 1e-18, 1.0)
    return float(-(p * np.log(p)).sum())


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def safe_softmax_np(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if z.size == 0 or not np.isfinite(z).all():
        raise FloatingPointError("Empty or nonfinite scores; refusing a uniform fallback.")
    z = z - np.max(z)
    p = np.exp(z)
    s = np.sum(p)
    if (not np.isfinite(s)) or s <= 0:
        raise FloatingPointError("Invalid softmax normalization.")
    return p / s


@dataclass
class PairEx:
    prompt: str
    chosen: str
    rejected: str


class PairDataset(Dataset):
    def __init__(self, rows):
        self.data = []
        for r in rows:
            p, c, rj = r.get("prompt"), r.get("chosen"), r.get("rejected")
            if isinstance(p, str) and isinstance(c, str) and isinstance(rj, str):
                self.data.append(PairEx(p, c, rj))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return {"idx": idx, "prompt": ex.prompt, "chosen": ex.chosen, "rejected": ex.rejected}


class WeightedPairDataset(Dataset):
    def __init__(self, base_ds, indices, weights, prompt_ids):
        self.base_ds = base_ds
        self.indices = indices
        self.weights = weights
        self.prompt_ids = prompt_ids

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        idx = self.indices[j]
        ex = self.base_ds[idx]
        return {
            "idx": idx,
            "prompt": ex["prompt"],
            "chosen": ex["chosen"],
            "rejected": ex["rejected"],
            "pair_weight": float(self.weights[j]),
            "train_prompt_local_id": int(self.prompt_ids[j]),
        }


def collate(batch):
    out = {
        "idx": [b["idx"] for b in batch],
        "prompt": [b["prompt"] for b in batch],
        "chosen": [b["chosen"] for b in batch],
        "rejected": [b["rejected"] for b in batch],
    }
    if "pair_weight" in batch[0]:
        out["pair_weight"] = torch.tensor([b["pair_weight"] for b in batch], dtype=torch.float32)
    if "train_prompt_local_id" in batch[0]:
        out["train_prompt_local_id"] = torch.tensor(
            [b["train_prompt_local_id"] for b in batch], dtype=torch.int64
        )
    return out


def build_batch(tok, prompts, responses, max_length, device):
    ids_list, attn_list, labels_list = [], [], []
    eos = tok.eos_token_id
    for p, y in zip(prompts, responses):
        p_ids = tok(p, add_special_tokens=False).input_ids
        y_ids = tok(y, add_special_tokens=False).input_ids
        if eos is not None:
            y_ids = y_ids + [eos]
        ids = p_ids + y_ids
        if max_length and max_length > 0 and len(ids) > max_length:
            ids = ids[-max_length:]
        resp_len = min(len(y_ids), len(ids))
        prompt_len = len(ids) - resp_len
        labels = [-100] * prompt_len + ids[prompt_len:]
        attn = [1] * len(ids)
        ids_list.append(torch.tensor(ids, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))
        attn_list.append(torch.tensor(attn, dtype=torch.long))

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(ids_list, batch_first=True, padding_value=pad_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
    attn = torch.nn.utils.rnn.pad_sequence(attn_list, batch_first=True, padding_value=0)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attn.to(device),
        "labels": labels.to(device),
    }


def sum_logprob_and_count_from_outputs(logits, labels):
    labels_s = labels[:, 1:].contiguous()
    logits_s = logits[:, :-1, :].contiguous()
    mask = labels_s != -100
    logp = torch.log_softmax(logits_s, dim=-1)
    tgt = labels_s.clamp(min=0)
    gathered = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mask
    s = gathered.sum(dim=1)
    c = mask.sum(dim=1).clamp(min=1)
    return s, c


@torch.no_grad()
def batch_sequence_logprob(model, tok, prompts, responses, max_length, device):
    batch = build_batch(tok, prompts, responses, max_length, device)
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    s, c = sum_logprob_and_count_from_outputs(out.logits, batch["labels"])
    if not torch.isfinite(s).all():
        raise FloatingPointError("Nonfinite response likelihoods.")
    return s.float().cpu(), c.int().cpu()


def build_prompt_to_pair_indices(ds):
    mp = {}
    for idx, ex in enumerate(ds.data):
        mp.setdefault(ex.prompt, []).append(idx)
    return mp


def score_pair_indices_sequence_margin(model, tok, ds, pair_indices, max_length, device, score_batch_size):
    margins = np.zeros(len(pair_indices), dtype=np.float64)
    bs = max(1, int(score_batch_size))
    for s in range(0, len(pair_indices), bs):
        e = min(len(pair_indices), s + bs)
        chunk = pair_indices[s:e]
        prompts = [ds.data[i].prompt for i in chunk]
        chosens = [ds.data[i].chosen for i in chunk]
        rejects = [ds.data[i].rejected for i in chunk]
        lp_c = batch_sequence_logprob(model, tok, prompts, chosens, max_length, device)[0].numpy()
        lp_r = batch_sequence_logprob(model, tok, prompts, rejects, max_length, device)[0].numpy()
        margins[s:e] = (lp_c - lp_r).astype(np.float64)
    return margins


def build_prompt_aware_training_subset(
    model,
    tok,
    ds,
    prompt_to_pair_indices,
    rng,
    train_prompt_size,
    pairs_per_prompt,
    tau,
    lambda_on,
    mix_eps,
    max_length,
    device,
    score_batch_size,
    weight_floor,
    weight_cap,
):
    prompts_all = list(prompt_to_pair_indices.keys())
    num_prompts = min(train_prompt_size, len(prompts_all))
    sampled_prompts = rng.sample(prompts_all, k=num_prompts)

    chosen_indices, chosen_weights, chosen_prompt_ids = [], [], []
    diag_rows, sampled_pairs_per_prompt = [], {}

    for local_pid, prompt in enumerate(sampled_prompts):
        pair_indices = prompt_to_pair_indices[prompt]
        margins = score_pair_indices_sequence_margin(model, tok, ds, pair_indices, max_length, device, score_batch_size)
        induced = safe_softmax_np(float(tau) * margins)
        uniform = np.ones_like(induced) / len(induced)
        base_mix = (1.0 - float(lambda_on)) * uniform + float(lambda_on) * induced
        mixed = (1.0 - float(mix_eps)) * base_mix + float(mix_eps) * uniform
        mixed = mixed / np.sum(mixed)

        take = min(max(1, pairs_per_prompt), len(pair_indices))
        # Use a uniform proposal and represent q_t exactly once through the
        # loss weight below.  Sampling from q_t here and multiplying the loss
        # by q_t again would instead optimize an unintended q_t^2 objective.
        sampled_local = rng.choices(range(len(pair_indices)), k=take)
        sampled_pairs_per_prompt[prompt] = take

        for j in sampled_local:
            chosen_indices.append(pair_indices[j])
            chosen_weights.append(float(mixed[j]))
            chosen_prompt_ids.append(local_pid)

        for j, gi, mg, ug, bg, mixg in zip(range(len(pair_indices)), pair_indices, margins, induced, base_mix, mixed):
            diag_rows.append({
                "train_prompt_local_id": local_pid,
                "prompt": prompt,
                "pair_global_idx": gi,
                "margin_sequence_logprob": float(mg),
                "induced_pair_prob": float(ug),
                "proposal_pair_prob": float(uniform[j]),
                "base_mix_prob": float(bg),
                "mixed_pair_prob": float(mixg),
                "target_loss_prob": float(mixg),
                "num_pairs_for_prompt": int(len(pair_indices)),
                "pairs_sampled_for_prompt": int(take),
                "tau": float(tau),
                "lambda_on": float(lambda_on),
                "mix_eps": float(mix_eps),
                "effective_lambda_to_induced": float((1.0 - float(mix_eps)) * float(lambda_on)),
            })

    w = np.array(chosen_weights, dtype=np.float64)
    w = w / max(w.mean(), 1e-12)
    w = np.clip(w, weight_floor, weight_cap)

    return (
        WeightedPairDataset(ds, chosen_indices, w.tolist(), chosen_prompt_ids),
        pd.DataFrame(diag_rows),
        sampled_prompts,
        sampled_pairs_per_prompt,
    )


def normalize_text_key(s: str) -> str:
    return " ".join(str(s).strip().split()).lower()


@torch.no_grad()
def generate_candidate_responses(model, tok, prompt, num_return_sequences, max_new_tokens, do_sample, temperature, top_p, device):
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": int(max_new_tokens),
        "num_return_sequences": int(num_return_sequences),
        "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        "eos_token_id": tok.eos_token_id,
    }
    if do_sample:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": float(max(temperature, 1e-5)),
            "top_p": float(top_p),
        })
    else:
        beams = max(1, int(num_return_sequences))
        gen_kwargs.update({
            "do_sample": False,
            "num_beams": beams,
            "early_stopping": True,
        })

    outputs = model.generate(**gen_kwargs)
    prompt_len = input_ids.shape[1]
    texts = []
    for seq in outputs:
        gen_ids = seq[prompt_len:]
        txt = tok.decode(gen_ids, skip_special_tokens=True).strip()
        texts.append(txt)
    return texts


@torch.no_grad()
def build_generated_eval_set(
    model,
    tok,
    prompt_pool,
    prompt_id_pool,
    num_eval_prompts,
    num_candidates_per_prompt,
    keep_top_k,
    max_new_tokens,
    do_sample,
    temperature,
    top_p,
    score_batch_size,
    score_max_length,
    seed,
    device,
):
    rng = random.Random(seed)
    num_eval_prompts = min(int(num_eval_prompts), len(prompt_pool))
    chosen_indices = sorted(rng.sample(range(len(prompt_pool)), k=num_eval_prompts))
    prompts = [prompt_pool[i] for i in chosen_indices]
    prompt_ids = [prompt_id_pool[i] for i in chosen_indices]

    responses_by_prompt = []
    response_sources_by_prompt = []
    generation_rows = []

    model.eval()

    for pid, prompt in enumerate(tqdm(prompts, desc="build_generated_eval_set", ncols=100)):
        raw_texts = generate_candidate_responses(
            model=model,
            tok=tok,
            prompt=prompt,
            num_return_sequences=max(1, int(num_candidates_per_prompt)),
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_p=float(top_p),
            device=device,
        )

        uniq_texts = []
        seen = set()
        for txt in raw_texts:
            key = normalize_text_key(txt)
            if key == "" or key in seen:
                continue
            seen.add(key)
            uniq_texts.append(txt)

        if len(uniq_texts) == 0:
            uniq_texts = [""]

        sum_scores = []
        bs = max(1, int(score_batch_size))
        for s in range(0, len(uniq_texts), bs):
            e = min(len(uniq_texts), s + bs)
            ss, _ = batch_sequence_logprob(
                model, tok, [prompt] * (e - s), uniq_texts[s:e], score_max_length, device
            )
            sum_scores.extend(ss.numpy().tolist())

        cand_rows = []
        for txt, ss in zip(uniq_texts, sum_scores):
            cand_rows.append({
                "text": txt,
                "sum_logprob": float(ss),
            })

        cand_rows = sorted(cand_rows, key=lambda r: r["sum_logprob"], reverse=True)
        kept = cand_rows[: min(int(keep_top_k), len(cand_rows))]

        responses_by_prompt.append([x["text"] for x in kept])
        response_sources_by_prompt.append(["generated_init"] * len(kept))

        for rank, row in enumerate(kept, start=1):
            generation_rows.append({
                "prompt_index": int(pid),
                "prompt_id": int(prompt_ids[pid]),
                "prompt": prompt,
                "response_rank": int(rank),
                "response_text": row["text"],
                "sum_logprob": float(row["sum_logprob"]),
                "num_unique_candidates": int(len(cand_rows)),
                "num_requested_candidates": int(num_candidates_per_prompt),
                "num_kept": int(len(kept)),
            })

    return prompts, prompt_ids, responses_by_prompt, response_sources_by_prompt, generation_rows


def maybe_save_adapter(model, tok, out_dir):
    ensure_dir(out_dir)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
