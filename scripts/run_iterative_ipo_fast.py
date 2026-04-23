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
def batch_avg_logprob(model, tok, prompts, responses, max_length, device):
    batch = build_batch(tok, prompts, responses, max_length, device)
    out = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        labels=batch['labels'],
    )
    s, c = sum_logprob_and_count_from_outputs(out.logits, batch['labels'])
    return (s / c).float().cpu()


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
        mixed = (1.0 - mix_eps) * induced + mix_eps * uniform

        take = min(max(1, pairs_per_prompt), len(pair_indices))
        sampled_local = rng.choices(range(len(pair_indices)), weights=mixed.tolist(), k=take)
        sampled_pairs_per_prompt[prompt] = take

        for j in sampled_local:
            chosen_indices.append(pair_indices[j])
            chosen_weights.append(float(mixed[j]))
            chosen_prompt_ids.append(local_pid)

        for j, gi, mg, ug, mixg in zip(range(len(pair_indices)), pair_indices, margins, induced, mixed):
            diag_rows.append({
                'train_prompt_local_id': local_pid,
                'prompt': prompt,
                'pair_global_idx': gi,
                'margin_avglogprob': float(mg),
                'induced_pair_prob': float(ug),
                'mixed_pair_prob': float(mixg),
                'num_pairs_for_prompt': int(len(pair_indices)),
                'pairs_sampled_for_prompt': int(take),
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
    args = ap.parse_args()

    args.alpha = float(max(0.0, min(1.0, args.alpha)))
    args.lambda_on = float(max(0.0, min(1.0, args.lambda_on)))
    args.mix_eps = float(max(0.0, min(1.0, args.mix_eps)))

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
    prompt_to_pair_indices = build_prompt_to_pair_indices(ds)

    prompts_uniq, responses_by_prompt, u_by_prompt, prompt_ids_raw = load_eval_prompt_responses(
        args.eval_prompts_path
    )
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

    metrics = []
    prev_q_prompt_matrix = None
    prev_prompt_entropies = None

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

    T = args.max_iters if args.auto_stop == 1 else args.iters
    pct_eps = 1e-12

    for t in range(T):
        print(
            f'\n===== OUTER ITER {t} | alpha={args.alpha} lambda={args.lambda_on} '
            f'tau={args.tau} beta={args.beta} mix_eps={args.mix_eps} ====='
        )

        model.eval()
        ref0.eval()

        flat_scores = np.zeros(len(flat_prompts), dtype=np.float32)
        bs = max(1, int(args.score_batch_size))
        for s in tqdm(range(0, len(flat_prompts), bs), desc=f'scoring@eval_prompts iter {t}', ncols=100):
            e = min(len(flat_prompts), s + bs)
            flat_scores[s:e] = batch_avg_logprob(
                model, tok, flat_prompts[s:e], flat_resps[s:e], args.max_length, device
            ).numpy()

        q_prompt_matrix = np.full((num_prompts_eval, max_k), np.nan, dtype=np.float64)
        prompt_entropies = np.zeros(num_prompts_eval, dtype=np.float64)
        prompt_tvs = np.full(num_prompts_eval, np.nan, dtype=np.float64)
        prompt_kls = np.full(num_prompts_eval, np.nan, dtype=np.float64)
        prompt_top1 = np.full(num_prompts_eval, -1, dtype=np.int64)

        for pid, (s, e) in enumerate(group_offsets):
            scores = flat_scores[s:e].astype(np.float64)
            p = safe_softmax_np(scores * float(args.tau))
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
        prompt_tv_mean = float(np.nanmean(prompt_tvs)) if np.any(~np.isnan(prompt_tvs)) else float('nan')
        prompt_tv_max = float(np.nanmax(prompt_tvs)) if np.any(~np.isnan(prompt_tvs)) else float('nan')
        prompt_kl_mean = float(np.nanmean(prompt_kls)) if np.any(~np.isnan(prompt_kls)) else float('nan')

        if prev_prompt_entropies is None:
            prompt_entropy_abs_delta_mean = float('nan')
            prompt_entropy_abs_delta_max = float('nan')
            prompt_entropy_pct_change_mean = float('nan')
        else:
            d = np.abs(prompt_entropies - prev_prompt_entropies)
            prompt_entropy_abs_delta_mean = float(np.mean(d))
            prompt_entropy_abs_delta_max = float(np.max(d))
            pct = 100.0 * (prompt_entropies - prev_prompt_entropies) / np.maximum(
                np.abs(prev_prompt_entropies), pct_eps
            )
            prompt_entropy_pct_change_mean = float(np.mean(pct))

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
            None if prev_q_prompt_matrix is None else prompt_tvs,
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

        top1_history.append(prompt_top1.copy())
        tv_history.append(np.where(np.isnan(prompt_tvs), -1.0, prompt_tvs).copy())
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
            float(np.mean(prompt_entropies[~prompt_resolved_mask]))
            if np.any(~prompt_resolved_mask)
            else float('nan')
        )
        unresolved_tv_mean = (
            float(np.nanmean(prompt_tvs[~prompt_resolved_mask]))
            if np.any(~prompt_resolved_mask) and np.any(~np.isnan(prompt_tvs[~prompt_resolved_mask]))
            else float('nan')
        )
        unresolved_tv_max = (
            float(np.nanmax(prompt_tvs[~prompt_resolved_mask]))
            if np.any(~prompt_resolved_mask) and np.any(~np.isnan(prompt_tvs[~prompt_resolved_mask]))
            else float('nan')
        )

        prev_q_prompt_matrix = q_prompt_matrix.copy()
        prev_prompt_entropies = prompt_entropies.copy()

        print(
            f'[Metrics@t={t}] H_mean={prompt_entropy_mean:.6g} | '
            f'converged={num_converged}/{num_prompts_eval} | '
            f'oscillatory={num_osc}/{num_prompts_eval} | '
            f'resolved={num_resolved}/{num_prompts_eval} ({frac_resolved:.1%}) | '
            f'TV_mean={prompt_tv_mean:.6g} TV_max={prompt_tv_max:.6g} | '
            f'cum_exp_mean={float(np.mean(prompt_cum_exposure)):.3f} '
            f'recent_exp_mean={float(np.mean(prompt_recent_exposure)):.3f} | '
            f'train_pairs={len(train_ds_weighted)} train_prompts={target_prompt_count}'
        )

        metrics.append({
            'iter': t,
            'alpha': args.alpha,
            'lambda': args.lambda_on,
            'tau': args.tau,
            'beta': args.beta,
            'mix_eps': args.mix_eps,
            'pairs_per_prompt': args.pairs_per_prompt,
            'target_train_prompt_count': target_prompt_count,
            'actual_train_pairs': len(train_ds_weighted),
            'prompt_entropy_mean': prompt_entropy_mean,
            'prompt_tv_mean': prompt_tv_mean,
            'prompt_tv_max': prompt_tv_max,
            'prompt_kl_mean': prompt_kl_mean,
            'prompt_entropy_abs_delta_mean': prompt_entropy_abs_delta_mean,
            'prompt_entropy_abs_delta_max': prompt_entropy_abs_delta_max,
            'prompt_entropy_pct_change_mean': prompt_entropy_pct_change_mean,
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

        if args.dump_each_iter == 1:
            np.savez_compressed(
                os.path.join(dump_dir, f'iter_{t:04d}.npz'),
                iter=np.int32(t),
                q_prompt_matrix=q_prompt_matrix.astype(np.float32),
                prompt_entropies=prompt_entropies.astype(np.float32),
                prompt_tvs=prompt_tvs.astype(np.float32),
                prompt_kls=prompt_kls.astype(np.float32),
                prompt_top1=prompt_top1.astype(np.int32),
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
                row = {
                    'prompt_id': prompt_ids_raw[pid],
                    'prompt_index': pid,
                    'prompt': prompts_uniq[pid],
                    'K': len(responses_by_prompt[pid]),
                    'entropy': prompt_entropies[pid],
                    'tv_delta': prompt_tvs[pid],
                    'kl_delta': prompt_kls[pid],
                    'top1_idx': int(prompt_top1[pid]),
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
                        (prompt_cum_exposure[pid] >= args.min_total_exposure)
                        and (prompt_recent_exposure[pid] >= args.min_recent_exposure)
                    ),
                }
                for j in range(max_k):
                    row[f'prob_{j}'] = q_prompt_matrix[pid, j]
                    row[f'response_{j}'] = responses_by_prompt[pid][j] if j < len(responses_by_prompt[pid]) else ''
                rows_csv.append(row)

            pd.DataFrame(rows_csv).to_csv(
                os.path.join(dump_dir, f'iter_{t:04d}_prompt_metrics.csv'),
                index=False,
            )
            train_diag_df.to_csv(
                os.path.join(dump_dir, f'iter_{t:04d}_train_pair_support.csv'),
                index=False,
            )

        if args.auto_stop == 1 and num_resolved == num_prompts_eval and num_prompts_eval > 0:
            print(f'[STOP] All prompts resolved at iter={t}.')
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
        'last_iter_ran': int(metrics[-1]['iter']) if metrics else -1,
        'training_rule': {
            'type': 'prompt_aware_pair_sampling_with_mixing',
            'mix_eps': args.mix_eps,
            'pairs_per_prompt': args.pairs_per_prompt,
            'score_mode': 'avg_logprob',
        },
        'exposure_rule': {
            'window': args.exposure_window,
            'min_total_exposure': args.min_total_exposure,
            'min_recent_exposure': args.min_recent_exposure,
        },
    })

    out_csv = os.path.join(
        args.log_dir,
        f'metrics_alpha{args.alpha}_lambda{args.lambda_on}_tau{args.tau}_seed{args.seed}_FAST.csv',
    )
    pd.DataFrame(metrics).to_csv(out_csv, index=False)

    print('[DONE] wrote:', out_csv)
    print('[DONE] wrote:', summary_json)
    if args.dump_each_iter == 1:
        print('[DONE] per-iter dumps in:', dump_dir)


if __name__ == '__main__':
    main()
