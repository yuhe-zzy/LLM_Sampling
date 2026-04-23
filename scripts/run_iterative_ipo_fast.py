from __future__ import annotations
import argparse
import json
import math
import os
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_jsonl(path: str):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def entropy_from_probs(p: np.ndarray) -> float:
    p = np.clip(p, 1e-18, 1.0)
    return float(-(p * np.log(p)).sum())


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-18, 1.0)
    q = np.clip(q, 1e-18, 1.0)
    return float((p * (np.log(p) - np.log(q))).sum())


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def safe_softmax_np(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z)
    p = np.exp(z)
    s = np.sum(p)
    if (not np.isfinite(s)) or s <= 0:
        return np.ones_like(z) / len(z)
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
            p, c, rj = r.get('prompt'), r.get('chosen'), r.get('rejected')
            if isinstance(p, str) and isinstance(c, str) and isinstance(rj, str):
                self.data.append(PairEx(p, c, rj))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return {'idx': idx, 'prompt': ex.prompt, 'chosen': ex.chosen, 'rejected': ex.rejected}


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
            'idx': idx,
            'prompt': ex['prompt'],
            'chosen': ex['chosen'],
            'rejected': ex['rejected'],
            'pair_weight': float(self.weights[j]),
            'train_prompt_local_id': int(self.prompt_ids[j]),
        }


def collate(batch):
    out = {
        'idx': [b['idx'] for b in batch],
        'prompt': [b['prompt'] for b in batch],
        'chosen': [b['chosen'] for b in batch],
        'rejected': [b['rejected'] for b in batch],
    }
    if 'pair_weight' in batch[0]:
        out['pair_weight'] = torch.tensor([b['pair_weight'] for b in batch], dtype=torch.float32)
    if 'train_prompt_local_id' in batch[0]:
        out['train_prompt_local_id'] = torch.tensor([b['train_prompt_local_id'] for b in batch], dtype=torch.int64)
    return out


def load_eval_prompt_responses(path):
    rows = read_jsonl(path)
    prompts, responses_by_prompt, u_by_prompt, prompt_ids = [], [], [], []
    for r in rows:
        responses = r['responses']
        prompts.append(r['prompt'])
        responses_by_prompt.append([x['text'] for x in responses])
        u_by_prompt.append([float(x.get('u', np.nan)) for x in responses])
        prompt_ids.append(int(r.get('prompt_id', len(prompt_ids))))
    return prompts, responses_by_prompt, u_by_prompt, prompt_ids


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
        'input_ids': input_ids.to(device),
        'attention_mask': attn.to(device),
        'labels': labels.to(device),
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
def batch_sum_and_avg_logprob(model, tok, prompts, responses, max_length, device):
    batch = build_batch(tok, prompts, responses, max_length, device)
    out = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        labels=batch['labels'],
    )
    s, c = sum_logprob_and_count_from_outputs(out.logits, batch['labels'])
    avg = s / c
    return s.float().cpu(), avg.float().cpu(), c.int().cpu()


@torch.no_grad()
def batch_avg_logprob(model, tok, prompts, responses, max_length, device):
    _, avg, _ = batch_sum_and_avg_logprob(model, tok, prompts, responses, max_length, device)
    return avg


@torch.no_grad()
def token_level_logprobs(model, tok, prompt, response, max_length, device):
    batch = build_batch(tok, [prompt], [response], max_length, device)
    out = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        labels=batch['labels'],
    )

    labels = batch['labels']
    logits = out.logits

    labels_s = labels[:, 1:].contiguous()
    logits_s = logits[:, :-1, :].contiguous()
    mask = labels_s != -100

    logp = torch.log_softmax(logits_s, dim=-1)
    tgt = labels_s.clamp(min=0)
    gathered = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

    tok_lp = gathered[0][mask[0]].detach().cpu().numpy().astype(np.float64)
    full_ids = batch['input_ids'][0].detach().cpu().tolist()
    full_labels = batch['labels'][0].detach().cpu().tolist()
    response_token_ids = [tid for tid, lab in zip(full_ids, full_labels) if lab != -100]
    response_tokens = tok.convert_ids_to_tokens(response_token_ids)

    prefix_sum = np.cumsum(tok_lp).tolist()
    prefix_avg = (np.cumsum(tok_lp) / np.arange(1, len(tok_lp) + 1)).tolist() if len(tok_lp) > 0 else []

    eos_token_id = tok.eos_token_id
    eos_token_index = -1
    eos_logprob = float('nan')
    if eos_token_id is not None and len(response_token_ids) > 0 and response_token_ids[-1] == eos_token_id and len(tok_lp) > 0:
        eos_token_index = len(response_token_ids) - 1
        eos_logprob = float(tok_lp[-1])

    return {
        'token_logprobs': tok_lp.tolist(),
        'token_ids': response_token_ids,
        'tokens': response_tokens,
        'prefix_sum_logprobs': prefix_sum,
        'prefix_avg_logprobs': prefix_avg,
        'sum_logprob': float(np.sum(tok_lp)) if len(tok_lp) > 0 else float('nan'),
        'avg_logprob': float(np.mean(tok_lp)) if len(tok_lp) > 0 else float('nan'),
        'num_tokens': int(len(tok_lp)),
        'eos_token_index': int(eos_token_index),
        'eos_logprob': eos_logprob,
        'truncated_by_max_length': int(max_length > 0 and len(full_ids) >= max_length),
    }


def ipo_loss_from_delta(delta, beta):
    beta = max(beta, 1e-6)
    target = 1.0 / (2.0 * beta)
    return (delta - target) ** 2


def update_prompt_convergence_with_exposure(
    prompt_tvs,
    stable_counts,
    converged_mask,
    cum_exposure_counts,
    recent_exposure_counts,
    current_iter,
    min_iters,
    patience,
    tv_abs_tol,
    min_total_exposure,
    min_recent_exposure,
):
    if prompt_tvs is None or current_iter < min_iters:
        return stable_counts, converged_mask

    active = ~converged_mask
    eligible = (
        active
        & (cum_exposure_counts >= min_total_exposure)
        & (recent_exposure_counts >= min_recent_exposure)
    )

    stable_counts[eligible] = np.where(
        prompt_tvs[eligible] <= tv_abs_tol,
        stable_counts[eligible] + 1,
        0,
    )
    stable_counts[active & (~eligible)] = 0

    newly = active & (stable_counts >= max(1, int(patience)))
    converged_mask[newly] = True
    return stable_counts, converged_mask


def detect_oscillation_from_history(
    top1_history,
    tv_history,
    converged_mask,
    min_iters,
    current_iter,
    osc_window,
    osc_min_switches,
    osc_tv_floor,
):
    P = top1_history.shape[1]
    out = np.zeros(P, dtype=bool)
    if current_iter < max(min_iters, osc_window):
        return out

    window = min(osc_window, top1_history.shape[0])
    top_hist = top1_history[-window:, :]
    tv_hist = tv_history[-window:, :]

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


def build_prompt_to_pair_indices(ds):
    mp = {}
    for idx, ex in enumerate(ds.data):
        mp.setdefault(ex.prompt, []).append(idx)
    return mp


def score_pair_indices_avg_margin(model, tok, ds, pair_indices, max_length, device, score_batch_size):
    margins = np.zeros(len(pair_indices), dtype=np.float64)
    bs = max(1, int(score_batch_size))
    for s in range(0, len(pair_indices), bs):
        e = min(len(pair_indices), s + bs)
        chunk = pair_indices[s:e]
        prompts = [ds.data[i].prompt for i in chunk]
        chosens = [ds.data[i].chosen for i in chunk]
        rejects = [ds.data[i].rejected for i in chunk]
        lp_c = batch_avg_logprob(model, tok, prompts, chosens, max_length, device).numpy()
        lp_r = batch_avg_logprob(model, tok, prompts, rejects, max_length, device).numpy()
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

    chosen_indices = []
    chosen_weights = []
    chosen_prompt_ids = []
    diag_rows = []
    sampled_pairs_per_prompt = {}

    for local_pid, prompt in enumerate(sampled_prompts):
        pair_indices = prompt_to_pair_indices[prompt]
        margins = score_pair_indices_avg_margin(
            model, tok, ds, pair_indices, max_length, device, score_batch_size
        )

        induced = safe_softmax_np(float(tau) * margins)
        uniform = np.ones_like(induced) / len(induced)
        base_mix = (1.0 - float(lambda_on)) * uniform + float(lambda_on) * induced
        mixed = (1.0 - float(mix_eps)) * base_mix + float(mix_eps) * uniform
        mixed = mixed / np.sum(mixed)

        take = min(max(1, pairs_per_prompt), len(pair_indices))
        sampled_local = rng.choices(range(len(pair_indices)), weights=mixed.tolist(), k=take)
        sampled_pairs_per_prompt[prompt] = take

        for j in sampled_local:
            chosen_indices.append(pair_indices[j])
            chosen_weights.append(float(mixed[j]))
            chosen_prompt_ids.append(local_pid)

        for j, gi, mg, ug, bg, mixg in zip(range(len(pair_indices)), pair_indices, margins, induced, base_mix, mixed):
            diag_rows.append({
                'train_prompt_local_id': local_pid,
                'prompt': prompt,
                'pair_global_idx': gi,
                'margin_avglogprob': float(mg),
                'induced_pair_prob': float(ug),
                'base_mix_prob': float(bg),
                'mixed_pair_prob': float(mixg),
                'num_pairs_for_prompt': int(len(pair_indices)),
                'pairs_sampled_for_prompt': int(take),
                'tau': float(tau),
                'lambda_on': float(lambda_on),
                'mix_eps': float(mix_eps),
                'effective_lambda_to_induced': float((1.0 - float(mix_eps)) * float(lambda_on)),
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


def maybe_save_adapter(model, tok, out_dir):
    ensure_dir(out_dir)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


def validate_eval_data(prompts_uniq, responses_by_prompt):
    if len(prompts_uniq) == 0:
        raise ValueError('No eval prompts found in eval_prompts_path.')
    empty_prompt_ids = [i for i, rs in enumerate(responses_by_prompt) if len(rs) == 0]
    if empty_prompt_ids:
        raise ValueError(f'Found eval prompts with zero responses. First few indices: {empty_prompt_ids[:10]}')


def dump_snapshot(
    dump_dir,
    t,
    snapshot_stage,
    prompt_ids_raw,
    prompts_uniq,
    responses_by_prompt,
    u_by_prompt,
    group_offsets,
    max_k,
    flat_sum_scores,
    flat_avg_scores,
    flat_num_tokens,
    q_prompt_matrix_avg,
    q_prompt_matrix_sum,
    prompt_entropies_avg,
    prompt_entropies_sum,
    prompt_tvs_avg,
    prompt_tvs_sum,
    prompt_kls_avg,
    prompt_kls_sum,
    prompt_top1_avg,
    prompt_top1_sum,
    prompt_converged_mask,
    prompt_oscillatory_mask,
    prompt_resolved_mask,
    prompt_stable_counts,
    prompt_first_converged_iter,
    prompt_first_oscillatory_iter,
    prompt_cum_exposure,
    prompt_recent_exposure,
    iter_exposure_vec,
):
    num_prompts_eval = len(prompts_uniq)
    np.savez_compressed(
        os.path.join(dump_dir, f'iter_{t:04d}.npz'),
        iter=np.int32(t),
        snapshot_stage=np.array(snapshot_stage),
        q_prompt_matrix_avg=q_prompt_matrix_avg.astype(np.float32),
        q_prompt_matrix_sum=q_prompt_matrix_sum.astype(np.float32),
        prompt_entropies_avg=prompt_entropies_avg.astype(np.float32),
        prompt_entropies_sum=prompt_entropies_sum.astype(np.float32),
        prompt_tvs_avg=prompt_tvs_avg.astype(np.float32),
        prompt_tvs_sum=prompt_tvs_sum.astype(np.float32),
        prompt_kls_avg=prompt_kls_avg.astype(np.float32),
        prompt_kls_sum=prompt_kls_sum.astype(np.float32),
        prompt_top1_avg=prompt_top1_avg.astype(np.int32),
        prompt_top1_sum=prompt_top1_sum.astype(np.int32),
        flat_sum_scores=flat_sum_scores.astype(np.float32),
        flat_avg_scores=flat_avg_scores.astype(np.float32),
        flat_num_tokens=flat_num_tokens.astype(np.int32),
        prompt_converged_mask=prompt_converged_mask.astype(np.int8),
        prompt_oscillatory_mask=prompt_oscillatory_mask.astype(np.int8),
        prompt_resolved_mask=prompt_resolved_mask.astype(np.int8),
        prompt_stable_counts=prompt_stable_counts.astype(np.int32),
        prompt_first_converged_iter=prompt_first_converged_iter.astype(np.int32),
        prompt_first_oscillatory_iter=prompt_first_oscillatory_iter.astype(np.int32),
        prompt_cum_exposure=prompt_cum_exposure.astype(np.int32),
        prompt_recent_exposure=prompt_recent_exposure.astype(np.int32),
        iter_exposure_vec=iter_exposure_vec.astype(np.int32),
    )

    rows_csv = []
    for pid in range(num_prompts_eval):
        s0, e0 = group_offsets[pid]
        k = e0 - s0
        row = {
            'iter': int(t),
            'snapshot_stage': snapshot_stage,
            'prompt_id': prompt_ids_raw[pid],
            'prompt_index': pid,
            'prompt': prompts_uniq[pid],
            'K': len(responses_by_prompt[pid]),
            'entropy': prompt_entropies_avg[pid],
            'tv_delta': prompt_tvs_avg[pid],
            'kl_delta': prompt_kls_avg[pid],
            'top1_idx': int(prompt_top1_avg[pid]),
            'entropy_avg': prompt_entropies_avg[pid],
            'entropy_sum': prompt_entropies_sum[pid],
            'tv_delta_avg': prompt_tvs_avg[pid],
            'tv_delta_sum': prompt_tvs_sum[pid],
            'kl_delta_avg': prompt_kls_avg[pid],
            'kl_delta_sum': prompt_kls_sum[pid],
            'top1_idx_avg': int(prompt_top1_avg[pid]),
            'top1_idx_sum': int(prompt_top1_sum[pid]),
            'converged': int(prompt_converged_mask[pid]),
            'oscillatory': int(prompt_oscillatory_mask[pid]),
            'resolved': int(prompt_resolved_mask[pid]),
            'stable_count': int(prompt_stable_counts[pid]),
            'first_converged_iter': int(prompt_first_converged_iter[pid]),
            'first_oscillatory_iter': int(prompt_first_oscillatory_iter[pid]),
            'exposure_iter': int(iter_exposure_vec[pid]),
            'cum_exposure': int(prompt_cum_exposure[pid]),
            'recent_exposure': int(prompt_recent_exposure[pid]),
            'exposure_eligible': int(
                (prompt_cum_exposure[pid] >= 0) and (prompt_recent_exposure[pid] >= 0)
            ),
        }
        for j in range(max_k):
            if j < k:
                flat_idx = s0 + j
                row[f'sum_logprob_{j}'] = float(flat_sum_scores[flat_idx])
                row[f'avg_logprob_{j}'] = float(flat_avg_scores[flat_idx])
                row[f'num_tokens_{j}'] = int(flat_num_tokens[flat_idx])
                row[f'prob_sum_{j}'] = float(q_prompt_matrix_sum[pid, j])
                row[f'prob_avg_{j}'] = float(q_prompt_matrix_avg[pid, j])
                row[f'prob_{j}'] = float(q_prompt_matrix_avg[pid, j])
                row[f'response_{j}'] = responses_by_prompt[pid][j]
                row[f'u_{j}'] = u_by_prompt[pid][j] if j < len(u_by_prompt[pid]) else np.nan
            else:
                row[f'sum_logprob_{j}'] = np.nan
                row[f'avg_logprob_{j}'] = np.nan
                row[f'num_tokens_{j}'] = np.nan
                row[f'prob_sum_{j}'] = np.nan
                row[f'prob_avg_{j}'] = np.nan
                row[f'prob_{j}'] = np.nan
                row[f'response_{j}'] = ''
                row[f'u_{j}'] = np.nan
        rows_csv.append(row)

    pd.DataFrame(rows_csv).to_csv(
        os.path.join(dump_dir, f'iter_{t:04d}_prompt_metrics.csv'),
        index=False,
    )


def dump_token_diagnostics(
    dump_dir,
    t,
    snapshot_stage,
    token_diag_prompt_indices,
    token_diag_num_responses,
    prompts_uniq,
    responses_by_prompt,
    prompt_ids_raw,
    model,
    tok,
    device,
    token_diag_max_length,
):
    token_diag_rows = []
    for pid in token_diag_prompt_indices:
        prompt_text = prompts_uniq[pid]
        kdiag = min(token_diag_num_responses, len(responses_by_prompt[pid]))
        for j in range(kdiag):
            response_text = responses_by_prompt[pid][j]
            diag = token_level_logprobs(
                model=model,
                tok=tok,
                prompt=prompt_text,
                response=response_text,
                max_length=token_diag_max_length,
                device=device,
            )
            token_diag_rows.append({
                'iter': t,
                'snapshot_stage': snapshot_stage,
                'prompt_index': pid,
                'prompt_id': prompt_ids_raw[pid],
                'response_index': j,
                'prompt': prompt_text,
                'response': response_text,
                'num_tokens': diag['num_tokens'],
                'sum_logprob': diag['sum_logprob'],
                'avg_logprob': diag['avg_logprob'],
                'eos_token_index': diag['eos_token_index'],
                'eos_logprob': diag['eos_logprob'],
                'truncated_by_max_length': diag['truncated_by_max_length'],
                'token_ids_json': json.dumps(diag['token_ids']),
                'tokens_json': json.dumps(diag['tokens'], ensure_ascii=False),
                'token_logprobs_json': json.dumps(diag['token_logprobs']),
                'prefix_sum_logprobs_json': json.dumps(diag['prefix_sum_logprobs']),
                'prefix_avg_logprobs_json': json.dumps(diag['prefix_avg_logprobs']),
            })
    pd.DataFrame(token_diag_rows).to_csv(
        os.path.join(dump_dir, f'iter_{t:04d}_token_diagnostics.csv'),
        index=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', type=str, required=True)
    ap.add_argument('--pairs_path', type=str, required=True)
    ap.add_argument('--eval_prompts_path', type=str, required=True)
    ap.add_argument('--out_dir', type=str, default='checkpoints_fast')
    ap.add_argument('--log_dir', type=str, default='logs')
    ap.add_argument('--seed', type=int, default=0)

    ap.add_argument('--iters', type=int, default=10)
    ap.add_argument('--auto_stop', type=int, default=1)
    ap.add_argument('--max_iters', type=int, default=50)

    ap.add_argument('--stop_min_iters', type=int, default=15)
    ap.add_argument('--stop_patience', type=int, default=5)
    ap.add_argument('--stop_tv_abs', type=float, default=0.005)

    ap.add_argument('--exposure_window', type=int, default=10)
    ap.add_argument('--min_total_exposure', type=int, default=8)
    ap.add_argument('--min_recent_exposure', type=int, default=4)

    ap.add_argument('--osc_detect', type=int, default=1)
    ap.add_argument('--osc_window', type=int, default=8)
    ap.add_argument('--osc_min_switches', type=int, default=4)
    ap.add_argument('--osc_tv_floor', type=float, default=0.01)

    ap.add_argument('--epochs_per_iter', type=int, default=1)
    ap.add_argument('--alpha', type=float, default=0.0)
    ap.add_argument('--lambda_on', type=float, default=0.0)
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--beta', type=float, default=0.1)

    ap.add_argument('--mix_eps', type=float, default=0.05)
    ap.add_argument('--w_clip_min', type=float, default=0.1)
    ap.add_argument('--w_clip_max', type=float, default=10.0)

    ap.add_argument('--max_length', type=int, default=256)
    ap.add_argument('--train_sample_size', type=int, default=64)
    ap.add_argument('--pairs_per_prompt', type=int, default=2)
    ap.add_argument('--train_prompt_size', type=int, default=0)

    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--grad_accum', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--warmup_ratio', type=float, default=0.03)
    ap.add_argument('--score_batch_size', type=int, default=8)

    ap.add_argument('--lora_r', type=int, default=16)
    ap.add_argument('--lora_alpha', type=int, default=32)
    ap.add_argument('--lora_dropout', type=float, default=0.05)

    ap.add_argument('--dump_each_iter', type=int, default=1)
    ap.add_argument('--save_iter_adapters', type=int, default=1)
    ap.add_argument('--save_initial_adapter', type=int, default=1)
    ap.add_argument('--save_final_adapter', type=int, default=1)

    ap.add_argument('--dump_token_diagnostics', type=int, default=1)
    ap.add_argument('--token_diag_num_prompts', type=int, default=10)
    ap.add_argument('--token_diag_num_responses', type=int, default=1)
    ap.add_argument('--token_diag_seed', type=int, default=123)
    ap.add_argument('--token_diag_max_length', type=int, default=2048)
    args = ap.parse_args()

    args.alpha = float(max(0.0, min(1.0, args.alpha)))
    args.lambda_on = float(max(0.0, min(1.0, args.lambda_on)))
    args.mix_eps = float(max(0.0, min(1.0, args.mix_eps)))

    if args.train_prompt_size < 0:
        raise ValueError('--train_prompt_size must be >= 0')
    if args.pairs_per_prompt < 1:
        raise ValueError('--pairs_per_prompt must be >= 1')
    if args.batch_size < 1 or args.grad_accum < 1 or args.score_batch_size < 1:
        raise ValueError('batch_size, grad_accum, score_batch_size must all be >= 1')
    if args.max_length < 1:
        raise ValueError('--max_length must be >= 1')
    if args.token_diag_max_length < args.max_length:
        print(f'[WARN] token_diag_max_length={args.token_diag_max_length} < max_length={args.max_length}. Token diagnostics may be more truncated than training/eval.')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('CUDA not available.')

    ensure_dir(args.out_dir)
    ensure_dir(args.log_dir)

    ds = PairDataset(read_jsonl(args.pairs_path))
    if len(ds) == 0:
        raise ValueError('No valid training pairs found in pairs_path.')
    prompt_to_pair_indices = build_prompt_to_pair_indices(ds)

    prompts_uniq, responses_by_prompt, u_by_prompt, prompt_ids_raw = load_eval_prompt_responses(
        args.eval_prompts_path
    )
    validate_eval_data(prompts_uniq, responses_by_prompt)

    num_prompts_eval = len(prompts_uniq)
    max_k = max(len(x) for x in responses_by_prompt)

    flat_prompts, flat_resps, group_offsets = [], [], []
    cur = 0
    for x, ys in zip(prompts_uniq, responses_by_prompt):
        s = cur
        for y in ys:
            flat_prompts.append(x)
            flat_resps.append(y)
            cur += 1
        group_offsets.append((s, cur))

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
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
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

    adapters_dir = os.path.join(
        args.out_dir,
        f'adapters_alpha{args.alpha}_lambda{args.lambda_on}_tau{args.tau}_seed{args.seed}',
    )
    ensure_dir(adapters_dir)
    if args.save_initial_adapter == 1:
        maybe_save_adapter(model, tok, os.path.join(adapters_dir, 'iter_init'))

    metrics = []
    prev_q_prompt_matrix_avg = None
    prev_q_prompt_matrix_sum = None
    prev_prompt_entropies_avg = None
    prev_prompt_entropies_sum = None

    prompt_stable_counts = np.zeros(num_prompts_eval, dtype=np.int64)
    prompt_converged_mask = np.zeros(num_prompts_eval, dtype=bool)
    prompt_oscillatory_mask = np.zeros(num_prompts_eval, dtype=bool)
    prompt_first_converged_iter = np.full(num_prompts_eval, -1, dtype=np.int64)
    prompt_first_oscillatory_iter = np.full(num_prompts_eval, -1, dtype=np.int64)

    eval_prompt_to_idx = {p: i for i, p in enumerate(prompts_uniq)}
    prompt_cum_exposure = np.zeros(num_prompts_eval, dtype=np.int64)
    prompt_recent_exposure = np.zeros(num_prompts_eval, dtype=np.int64)
    prompt_exposure_history = deque(maxlen=args.exposure_window)

    top1_history, tv_history = [], []

    dump_dir = os.path.join(
        args.log_dir,
        f'iter_dumps_alpha{args.alpha}_lambda{args.lambda_on}_tau{args.tau}_seed{args.seed}',
    )
    if args.dump_each_iter == 1:
        ensure_dir(dump_dir)

    token_diag_prompt_indices = []
    if args.dump_each_iter == 1 and args.dump_token_diagnostics == 1:
        rng_diag = random.Random(args.token_diag_seed)
        token_diag_prompt_indices = sorted(
            rng_diag.sample(range(num_prompts_eval), k=min(args.token_diag_num_prompts, num_prompts_eval))
        )
        write_json(
            os.path.join(dump_dir, 'token_diag_prompt_indices.json'),
            {
                'prompt_indices': token_diag_prompt_indices,
                'num_prompts': len(token_diag_prompt_indices),
                'num_responses_per_prompt': int(args.token_diag_num_responses),
                'token_diag_max_length': int(args.token_diag_max_length),
                'snapshot_stage': 'pre_update',
            },
        )

    eval_metadata_rows = []
    for pid in range(num_prompts_eval):
        for j, resp in enumerate(responses_by_prompt[pid]):
            eval_metadata_rows.append({
                'prompt_index': pid,
                'prompt_id': prompt_ids_raw[pid],
                'response_index': j,
                'prompt': prompts_uniq[pid],
                'response': resp,
                'u': u_by_prompt[pid][j] if j < len(u_by_prompt[pid]) else np.nan,
            })
    pd.DataFrame(eval_metadata_rows).to_csv(
        os.path.join(dump_dir, 'eval_prompt_response_metadata.csv') if args.dump_each_iter == 1 else os.path.join(args.log_dir, 'eval_prompt_response_metadata.csv'),
        index=False,
    )

    T = args.max_iters if args.auto_stop == 1 else args.iters
    pct_eps = 1e-12
    last_iter_ran = -1

    for t in range(T):
        print(
            f'\n===== OUTER ITER {t} | alpha={args.alpha} lambda={args.lambda_on} '
            f'tau={args.tau} beta={args.beta} mix_eps={args.mix_eps} ====='
        )

        # Save the exact adapter that matches all iter_t dumps in this loop.
        if args.save_iter_adapters == 1:
            maybe_save_adapter(model, tok, os.path.join(adapters_dir, f'iter_{t:04d}_preupdate'))

        model.eval()
        ref0.eval()

        flat_sum_scores = np.zeros(len(flat_prompts), dtype=np.float32)
        flat_avg_scores = np.zeros(len(flat_prompts), dtype=np.float32)
        flat_num_tokens = np.zeros(len(flat_prompts), dtype=np.int32)

        bs = max(1, int(args.score_batch_size))
        for s in tqdm(range(0, len(flat_prompts), bs), desc=f'scoring@eval_prompts iter {t}', ncols=100):
            e = min(len(flat_prompts), s + bs)
            sum_lp, avg_lp, tok_cnt = batch_sum_and_avg_logprob(
                model, tok, flat_prompts[s:e], flat_resps[s:e], args.max_length, device
            )
            flat_sum_scores[s:e] = sum_lp.numpy()
            flat_avg_scores[s:e] = avg_lp.numpy()
            flat_num_tokens[s:e] = tok_cnt.numpy()

        q_prompt_matrix_avg = np.full((num_prompts_eval, max_k), np.nan, dtype=np.float64)
        q_prompt_matrix_sum = np.full((num_prompts_eval, max_k), np.nan, dtype=np.float64)

        prompt_entropies_avg = np.zeros(num_prompts_eval, dtype=np.float64)
        prompt_entropies_sum = np.zeros(num_prompts_eval, dtype=np.float64)

        prompt_tvs_avg = np.full(num_prompts_eval, np.nan, dtype=np.float64)
        prompt_tvs_sum = np.full(num_prompts_eval, np.nan, dtype=np.float64)
        prompt_kls_avg = np.full(num_prompts_eval, np.nan, dtype=np.float64)
        prompt_kls_sum = np.full(num_prompts_eval, np.nan, dtype=np.float64)
        prompt_top1_avg = np.full(num_prompts_eval, -1, dtype=np.int64)
        prompt_top1_sum = np.full(num_prompts_eval, -1, dtype=np.int64)

        for pid, (s, e) in enumerate(group_offsets):
            scores_sum = flat_sum_scores[s:e].astype(np.float64)
            scores_avg = flat_avg_scores[s:e].astype(np.float64)

            p_sum = safe_softmax_np(scores_sum * float(args.tau))
            p_avg = safe_softmax_np(scores_avg * float(args.tau))

            k = e - s
            q_prompt_matrix_sum[pid, :k] = p_sum
            q_prompt_matrix_avg[pid, :k] = p_avg

            prompt_entropies_sum[pid] = entropy_from_probs(p_sum)
            prompt_entropies_avg[pid] = entropy_from_probs(p_avg)
            prompt_top1_sum[pid] = int(np.argmax(p_sum))
            prompt_top1_avg[pid] = int(np.argmax(p_avg))

            if prev_q_prompt_matrix_avg is not None:
                prev_avg = prev_q_prompt_matrix_avg[pid, :k].astype(np.float64)
                prev_avg = prev_avg / np.sum(prev_avg)
                prompt_tvs_avg[pid] = total_variation(p_avg, prev_avg)
                prompt_kls_avg[pid] = kl_div(p_avg, prev_avg)
            if prev_q_prompt_matrix_sum is not None:
                prev_sum = prev_q_prompt_matrix_sum[pid, :k].astype(np.float64)
                prev_sum = prev_sum / np.sum(prev_sum)
                prompt_tvs_sum[pid] = total_variation(p_sum, prev_sum)
                prompt_kls_sum[pid] = kl_div(p_sum, prev_sum)

        # Keep the original experiment logic on avg-normalized conditional probabilities.
        q_prompt_matrix = q_prompt_matrix_avg
        prompt_entropies = prompt_entropies_avg
        prompt_tvs = prompt_tvs_avg
        prompt_kls = prompt_kls_avg
        prompt_top1 = prompt_top1_avg

        prompt_entropy_mean = float(np.mean(prompt_entropies_avg))
        prompt_tv_mean = float(np.nanmean(prompt_tvs_avg)) if np.any(~np.isnan(prompt_tvs_avg)) else float('nan')
        prompt_tv_max = float(np.nanmax(prompt_tvs_avg)) if np.any(~np.isnan(prompt_tvs_avg)) else float('nan')
        prompt_kl_mean = float(np.nanmean(prompt_kls_avg)) if np.any(~np.isnan(prompt_kls_avg)) else float('nan')
        prompt_tv_mean_sum = float(np.nanmean(prompt_tvs_sum)) if np.any(~np.isnan(prompt_tvs_sum)) else float('nan')
        prompt_tv_max_sum = float(np.nanmax(prompt_tvs_sum)) if np.any(~np.isnan(prompt_tvs_sum)) else float('nan')
        prompt_kl_mean_sum = float(np.nanmean(prompt_kls_sum)) if np.any(~np.isnan(prompt_kls_sum)) else float('nan')

        if prev_prompt_entropies_avg is None:
            prompt_entropy_abs_delta_mean = float('nan')
            prompt_entropy_abs_delta_max = float('nan')
            prompt_entropy_pct_change_mean = float('nan')
            prompt_entropy_abs_delta_mean_sum = float('nan')
        else:
            d = np.abs(prompt_entropies_avg - prev_prompt_entropies_avg)
            prompt_entropy_abs_delta_mean = float(np.mean(d))
            prompt_entropy_abs_delta_max = float(np.max(d))
            pct = 100.0 * (prompt_entropies_avg - prev_prompt_entropies_avg) / np.maximum(
                np.abs(prev_prompt_entropies_avg), pct_eps
            )
            prompt_entropy_pct_change_mean = float(np.mean(pct))
            dsum = np.abs(prompt_entropies_sum - prev_prompt_entropies_sum)
            prompt_entropy_abs_delta_mean_sum = float(np.mean(dsum))

        target_prompt_count = (
            args.train_prompt_size
            if args.train_prompt_size > 0
            else int(math.ceil(args.train_sample_size / max(1, args.pairs_per_prompt)))
        )
        rng_train = random.Random(args.seed + 999 * (t + 1))

        train_ds_weighted, train_diag_df, sampled_prompts_this_iter, sampled_pairs_per_prompt = (
            build_prompt_aware_training_subset(
                model,
                tok,
                ds,
                prompt_to_pair_indices,
                rng_train,
                target_prompt_count,
                args.pairs_per_prompt,
                args.tau,
                args.lambda_on,
                args.mix_eps,
                args.max_length,
                device,
                args.score_batch_size,
                args.w_clip_min,
                args.w_clip_max,
            )
        )

        iter_exposure_vec = np.zeros(num_prompts_eval, dtype=np.int64)
        for prompt, cnt in sampled_pairs_per_prompt.items():
            if prompt in eval_prompt_to_idx:
                pid = eval_prompt_to_idx[prompt]
                iter_exposure_vec[pid] += int(cnt)

        prompt_cum_exposure += iter_exposure_vec
        prompt_exposure_history.append(iter_exposure_vec.copy())
        prompt_recent_exposure[:] = 0
        for arr in prompt_exposure_history:
            prompt_recent_exposure += arr

        prev_converged_mask = prompt_converged_mask.copy()
        prompt_stable_counts, prompt_converged_mask = update_prompt_convergence_with_exposure(
            None if prev_q_prompt_matrix_avg is None else prompt_tvs_avg,
            prompt_stable_counts,
            prompt_converged_mask,
            prompt_cum_exposure,
            prompt_recent_exposure,
            t,
            args.stop_min_iters,
            args.stop_patience,
            args.stop_tv_abs,
            args.min_total_exposure,
            args.min_recent_exposure,
        )
        newly = prompt_converged_mask & (~prev_converged_mask)
        prompt_first_converged_iter[newly] = t

        top1_history.append(prompt_top1_avg.copy())
        tv_history.append(np.where(np.isnan(prompt_tvs_avg), -1.0, prompt_tvs_avg).copy())
        top1_hist_np = np.stack(top1_history, axis=0)
        tv_hist_np = np.stack(tv_history, axis=0)

        if args.osc_detect == 1:
            newly_osc = detect_oscillation_from_history(
                top1_hist_np,
                tv_hist_np,
                prompt_converged_mask,
                args.stop_min_iters,
                t,
                args.osc_window,
                args.osc_min_switches,
                args.osc_tv_floor,
            )
            new_mask = newly_osc & (~prompt_oscillatory_mask)
            prompt_oscillatory_mask[new_mask] = True
            prompt_first_oscillatory_iter[new_mask] = t

        prompt_resolved_mask = prompt_converged_mask | prompt_oscillatory_mask
        num_converged = int(prompt_converged_mask.sum())
        num_osc = int(prompt_oscillatory_mask.sum())
        num_resolved = int(prompt_resolved_mask.sum())
        frac_resolved = float(num_resolved / max(num_prompts_eval, 1))
        num_unresolved = int(num_prompts_eval - num_resolved)

        unresolved_entropy_mean = (
            float(np.mean(prompt_entropies_avg[~prompt_resolved_mask]))
            if np.any(~prompt_resolved_mask)
            else float('nan')
        )
        unresolved_tv_mean = (
            float(np.nanmean(prompt_tvs_avg[~prompt_resolved_mask]))
            if np.any(~prompt_resolved_mask) and np.any(~np.isnan(prompt_tvs_avg[~prompt_resolved_mask]))
            else float('nan')
        )
        unresolved_tv_max = (
            float(np.nanmax(prompt_tvs_avg[~prompt_resolved_mask]))
            if np.any(~prompt_resolved_mask) and np.any(~np.isnan(prompt_tvs_avg[~prompt_resolved_mask]))
            else float('nan')
        )

        prev_q_prompt_matrix_avg = q_prompt_matrix_avg.copy()
        prev_q_prompt_matrix_sum = q_prompt_matrix_sum.copy()
        prev_prompt_entropies_avg = prompt_entropies_avg.copy()
        prev_prompt_entropies_sum = prompt_entropies_sum.copy()

        if args.dump_each_iter == 1:
            dump_snapshot(
                dump_dir=dump_dir,
                t=t,
                snapshot_stage='pre_update',
                prompt_ids_raw=prompt_ids_raw,
                prompts_uniq=prompts_uniq,
                responses_by_prompt=responses_by_prompt,
                u_by_prompt=u_by_prompt,
                group_offsets=group_offsets,
                max_k=max_k,
                flat_sum_scores=flat_sum_scores,
                flat_avg_scores=flat_avg_scores,
                flat_num_tokens=flat_num_tokens,
                q_prompt_matrix_avg=q_prompt_matrix_avg,
                q_prompt_matrix_sum=q_prompt_matrix_sum,
                prompt_entropies_avg=prompt_entropies_avg,
                prompt_entropies_sum=prompt_entropies_sum,
                prompt_tvs_avg=prompt_tvs_avg,
                prompt_tvs_sum=prompt_tvs_sum,
                prompt_kls_avg=prompt_kls_avg,
                prompt_kls_sum=prompt_kls_sum,
                prompt_top1_avg=prompt_top1_avg,
                prompt_top1_sum=prompt_top1_sum,
                prompt_converged_mask=prompt_converged_mask,
                prompt_oscillatory_mask=prompt_oscillatory_mask,
                prompt_resolved_mask=prompt_resolved_mask,
                prompt_stable_counts=prompt_stable_counts,
                prompt_first_converged_iter=prompt_first_converged_iter,
                prompt_first_oscillatory_iter=prompt_first_oscillatory_iter,
                prompt_cum_exposure=prompt_cum_exposure,
                prompt_recent_exposure=prompt_recent_exposure,
                iter_exposure_vec=iter_exposure_vec,
            )
            train_diag_df.to_csv(
                os.path.join(dump_dir, f'iter_{t:04d}_train_pair_support.csv'),
                index=False,
            )
            if args.dump_token_diagnostics == 1 and len(token_diag_prompt_indices) > 0:
                dump_token_diagnostics(
                    dump_dir=dump_dir,
                    t=t,
                    snapshot_stage='pre_update',
                    token_diag_prompt_indices=token_diag_prompt_indices,
                    token_diag_num_responses=args.token_diag_num_responses,
                    prompts_uniq=prompts_uniq,
                    responses_by_prompt=responses_by_prompt,
                    prompt_ids_raw=prompt_ids_raw,
                    model=model,
                    tok=tok,
                    device=device,
                    token_diag_max_length=args.token_diag_max_length,
                )

        print(
            f'[Metrics@t={t}] H_avg_mean={prompt_entropy_mean:.6g} | '
            f'H_sum_mean={float(np.mean(prompt_entropies_sum)):.6g} | '
            f'converged={num_converged}/{num_prompts_eval} | '
            f'oscillatory={num_osc}/{num_prompts_eval} | '
            f'resolved={num_resolved}/{num_prompts_eval} ({frac_resolved:.1%}) | '
            f'TV_avg_mean={prompt_tv_mean:.6g} TV_avg_max={prompt_tv_max:.6g} | '
            f'TV_sum_mean={prompt_tv_mean_sum:.6g} TV_sum_max={prompt_tv_max_sum:.6g} | '
            f'cum_exp_mean={float(np.mean(prompt_cum_exposure)):.3f} '
            f'recent_exp_mean={float(np.mean(prompt_recent_exposure)):.3f} | '
            f'train_pairs={len(train_ds_weighted)} train_prompts={target_prompt_count}'
        )

        metrics.append({
            'iter': t,
            'snapshot_stage': 'pre_update',
            'alpha': args.alpha,
            'lambda': args.lambda_on,
            'tau': args.tau,
            'beta': args.beta,
            'mix_eps': args.mix_eps,
            'pairs_per_prompt': args.pairs_per_prompt,
            'target_train_prompt_count': target_prompt_count,
            'actual_train_pairs': len(train_ds_weighted),
            'prompt_entropy_mean': prompt_entropy_mean,
            'prompt_entropy_mean_avg': float(np.mean(prompt_entropies_avg)),
            'prompt_entropy_mean_sum': float(np.mean(prompt_entropies_sum)),
            'prompt_tv_mean': prompt_tv_mean,
            'prompt_tv_max': prompt_tv_max,
            'prompt_kl_mean': prompt_kl_mean,
            'prompt_tv_mean_sum': prompt_tv_mean_sum,
            'prompt_tv_max_sum': prompt_tv_max_sum,
            'prompt_kl_mean_sum': prompt_kl_mean_sum,
            'prompt_entropy_abs_delta_mean': prompt_entropy_abs_delta_mean,
            'prompt_entropy_abs_delta_max': prompt_entropy_abs_delta_max,
            'prompt_entropy_pct_change_mean': prompt_entropy_pct_change_mean,
            'prompt_entropy_abs_delta_mean_sum': prompt_entropy_abs_delta_mean_sum,
            'num_prompts_eval': num_prompts_eval,
            'num_prompts_converged': num_converged,
            'num_prompts_oscillatory': num_osc,
            'num_prompts_resolved': num_resolved,
            'num_prompts_unresolved': num_unresolved,
            'frac_prompts_resolved': frac_resolved,
            'unresolved_entropy_mean': unresolved_entropy_mean,
            'unresolved_prompt_tv_mean': unresolved_tv_mean,
            'unresolved_prompt_tv_max': unresolved_tv_max,
            'train_sample_size': args.train_sample_size,
            'cum_exposure_mean': float(np.mean(prompt_cum_exposure)),
            'cum_exposure_min': int(np.min(prompt_cum_exposure)),
            'cum_exposure_max': int(np.max(prompt_cum_exposure)),
            'recent_exposure_mean': float(np.mean(prompt_recent_exposure)),
            'recent_exposure_min': int(np.min(prompt_recent_exposure)),
            'recent_exposure_max': int(np.max(prompt_recent_exposure)),
            'num_prompts_exposure_eligible': int(np.sum(
                (prompt_cum_exposure >= args.min_total_exposure)
                & (prompt_recent_exposure >= args.min_recent_exposure)
            )),
        })

        if args.auto_stop == 1 and num_resolved == num_prompts_eval and num_prompts_eval > 0:
            print(f'[STOP] All prompts resolved at iter={t}.')
            last_iter_ran = t
            if args.save_iter_adapters == 1:
                maybe_save_adapter(model, tok, os.path.join(adapters_dir, f'iter_{t:04d}_postupdate'))
            break

        model.train()
        train_loader = DataLoader(
            train_ds_weighted,
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
            pbar = tqdm(train_loader, desc=f'train@promptaware iter {t} ep {ep}', ncols=100)
            for batch in pbar:
                with torch.no_grad():
                    lp_c_pi = batch_avg_logprob(model, tok, batch['prompt'], batch['chosen'], args.max_length, device).to(device)
                    lp_r_pi = batch_avg_logprob(model, tok, batch['prompt'], batch['rejected'], args.max_length, device).to(device)
                    lp_c_ref0 = batch_avg_logprob(ref0, tok, batch['prompt'], batch['chosen'], args.max_length, device).to(device)
                    lp_r_ref0 = batch_avg_logprob(ref0, tok, batch['prompt'], batch['rejected'], args.max_length, device).to(device)

                    lp_c_ref_t = (1.0 - args.alpha) * lp_c_ref0 + args.alpha * lp_c_pi
                    lp_r_ref_t = (1.0 - args.alpha) * lp_r_ref0 + args.alpha * lp_r_pi

                bc = build_batch(tok, batch['prompt'], batch['chosen'], args.max_length, device)
                out_c = model(input_ids=bc['input_ids'], attention_mask=bc['attention_mask'], labels=bc['labels'])
                s_c, c_c = sum_logprob_and_count_from_outputs(out_c.logits, bc['labels'])
                avg_c = s_c / c_c

                br = build_batch(tok, batch['prompt'], batch['rejected'], args.max_length, device)
                out_r = model(input_ids=br['input_ids'], attention_mask=br['attention_mask'], labels=br['labels'])
                s_r, c_r = sum_logprob_and_count_from_outputs(out_r.logits, br['labels'])
                avg_r = s_r / c_r

                delta = (avg_c - lp_c_ref_t) - (avg_r - lp_r_ref_t)
                loss_vec = ipo_loss_from_delta(delta, args.beta)
                wt = batch['pair_weight'].to(device)
                loss = (wt * loss_vec).mean()
                loss.backward()
                step += 1

                if step % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)

                pbar.set_postfix({'loss': float(loss.item())})

        if step % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

        if args.save_iter_adapters == 1:
            maybe_save_adapter(model, tok, os.path.join(adapters_dir, f'iter_{t:04d}_postupdate'))
        last_iter_ran = t

    if args.save_final_adapter == 1:
        maybe_save_adapter(model, tok, os.path.join(adapters_dir, 'final'))

    summary_json = os.path.join(
        args.log_dir,
        f'convergence_summary_alpha{args.alpha}_lambda{args.lambda_on}_tau{args.tau}_seed{args.seed}.json',
    )
    write_json(summary_json, {
        'alpha': args.alpha,
        'lambda_on': args.lambda_on,
        'tau': args.tau,
        'beta': args.beta,
        'mix_eps': args.mix_eps,
        'seed': args.seed,
        'train_sample_size': args.train_sample_size,
        'pairs_per_prompt': args.pairs_per_prompt,
        'train_prompt_size': args.train_prompt_size,
        'num_prompts_eval': num_prompts_eval,
        'num_prompts_converged': int(prompt_converged_mask.sum()),
        'num_prompts_oscillatory': int(prompt_oscillatory_mask.sum()),
        'num_prompts_resolved': int((prompt_converged_mask | prompt_oscillatory_mask).sum()),
        'last_iter_ran': int(last_iter_ran),
        'training_rule': {
            'type': 'prompt_aware_pair_sampling_with_lambda_and_mixeps',
            'lambda_on': args.lambda_on,
            'mix_eps': args.mix_eps,
            'pairs_per_prompt': args.pairs_per_prompt,
            'score_mode': 'avg_logprob',
        },
        'sampling_rule': {
            'induced_distribution': 'softmax(tau * avg_margin)',
            'base_mix': '(1-lambda_on) * uniform + lambda_on * induced',
            'final_mix': '(1-mix_eps) * base_mix + mix_eps * uniform',
            'effective_lambda_to_induced': '(1-mix_eps) * lambda_on',
        },
        'exposure_rule': {
            'window': args.exposure_window,
            'min_total_exposure': args.min_total_exposure,
            'min_recent_exposure': args.min_recent_exposure,
        },
        'artifacts': {
            'adapter_dir': adapters_dir,
            'dump_dir': dump_dir if args.dump_each_iter == 1 else None,
            'iter_dump_snapshot_stage': 'pre_update',
            'adapter_names': {
                'matching_iter_dump_adapter': 'iter_XXXX_preupdate',
                'after_training_adapter': 'iter_XXXX_postupdate',
                'initial_adapter': 'iter_init',
                'final_adapter': 'final',
            },
        },
    })

    out_csv = os.path.join(
        args.log_dir,
        f'metrics_alpha{args.alpha}_lambda{args.lambda_on}_tau{args.tau}_seed{args.seed}_FAST.csv',
    )
    pd.DataFrame(metrics).to_csv(out_csv, index=False)

    print('[DONE] wrote:', out_csv)
    print('[DONE] wrote:', summary_json)
    print('[DONE] adapters in:', adapters_dir)
    if args.dump_each_iter == 1:
        print('[DONE] per-iter dumps in:', dump_dir)


if __name__ == '__main__':
    main()
