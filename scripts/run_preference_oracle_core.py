from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from run_ipo import (
    PairDataset,
    build_batch as run_build_batch,
    batch_sum_and_avg_logprob,
    build_generated_eval_set,
    build_prompt_aware_training_subset,
    build_prompt_to_pair_indices,
    collate,
    entropy_from_probs,
    maybe_save_adapter,
    read_jsonl,
    safe_softmax_np,
    sum_logprob_and_count_from_outputs,
    total_variation,
    write_json,
    write_jsonl,
)


DEFAULT_ORACLE_MODEL = "nvidia/Llama-3.1-Nemotron-70B-Reward-HF"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_first_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_dtype(name: str) -> torch.dtype:
    name = str(name).lower().strip()
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def dpo_loss_from_delta(delta: torch.Tensor, beta: float) -> torch.Tensor:
    return -F.logsigmoid(float(beta) * delta)


def ipo_loss_from_delta(delta: torch.Tensor, beta: float) -> torch.Tensor:
    beta = max(float(beta), 1e-6)
    target = 1.0 / (2.0 * beta)
    return (delta - target) ** 2


def loss_from_delta(delta: torch.Tensor, beta: float, loss_type: str) -> torch.Tensor:
    if loss_type == "ipo":
        return ipo_loss_from_delta(delta, beta)
    if loss_type == "dpo":
        return dpo_loss_from_delta(delta, beta)
    raise ValueError(f"Unknown loss_type: {loss_type}")


def normalize_text_key(text: str) -> str:
    return " ".join(str(text).strip().split()).lower()


def response_text_from_obj(obj: Any) -> Optional[str]:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in ("text", "response", "output", "answer"):
            val = obj.get(key)
            if isinstance(val, str):
                return val
    return None


def load_prompt_rows(path: str) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
    rows = read_jsonl(path)
    prompts, prompt_ids, kept_rows = [], [], []
    for i, row in enumerate(rows):
        prompt = row.get("prompt")
        if not isinstance(prompt, str):
            continue
        prompts.append(prompt)
        prompt_ids.append(int(row.get("prompt_id", i)))
        kept_rows.append(row)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts, prompt_ids, kept_rows


def load_cyclic_eval_support(
    path: str,
    num_prompts: int,
    seed: int,
    keep_k: int = 4,
) -> Tuple[List[str], List[int], List[List[str]], List[List[str]], List[Dict[str, Any]]]:
    prompts, prompt_ids, rows = load_prompt_rows(path)
    usable: List[Tuple[int, str, int, List[str], Dict[str, Any]]] = []
    for row_idx, (prompt, prompt_id, row) in enumerate(zip(prompts, prompt_ids, rows)):
        raw_responses = row.get("responses")
        if not isinstance(raw_responses, list):
            continue
        responses: List[str] = []
        seen = set()
        for obj in raw_responses:
            text = response_text_from_obj(obj)
            if not isinstance(text, str):
                continue
            key = normalize_text_key(text)
            if key == "" or key in seen:
                continue
            seen.add(key)
            responses.append(text)
        if len(responses) >= keep_k:
            usable.append((row_idx, prompt, prompt_id, responses[:keep_k], row))
    if not usable:
        raise ValueError(
            f"No cyclic eval rows with at least {keep_k} responses found in {path}. "
            "Cyclic evaluation requires JSONL rows with a responses list."
        )
    rng = random.Random(seed)
    take = min(int(num_prompts), len(usable))
    chosen = sorted(rng.sample(range(len(usable)), k=take))
    prompts_out, ids_out, responses_out, source_out, meta_rows = [], [], [], [], []
    for local_pid, idx in enumerate(chosen):
        row_idx, prompt, prompt_id, responses, row = usable[idx]
        prompts_out.append(prompt)
        ids_out.append(prompt_id)
        responses_out.append(responses)
        source_out.append(["dataset_cyclic"] * len(responses))
        meta_rows.append(
            {
                "prompt_index": local_pid,
                "source_row_index": row_idx,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "preference_matrix": row.get("preference_matrix"),
                "cycle_rule": row.get("cycle_rule"),
                "num_available_responses": len(row.get("responses", [])),
                "num_kept": len(responses),
            }
        )
    return prompts_out, ids_out, responses_out, source_out, meta_rows


@torch.no_grad()
def generate_responses_for_prompts_batched(
    model: torch.nn.Module,
    tok,
    prompts: List[str],
    num_return_sequences: int,
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: int,
) -> List[List[str]]:
    if len(prompts) == 0:
        return []
    rng_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    old_padding_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model.eval()
    device = get_first_device(model)
    grouped: List[List[str]] = []
    bs = max(1, int(batch_size))
    nret = max(1, int(num_return_sequences))

    for s in tqdm(range(0, len(prompts), bs), desc="generate_oracle_responses", ncols=100):
        chunk = prompts[s : s + bs]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": int(max_new_tokens),
            "num_return_sequences": nret,
            "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            "eos_token_id": tok.eos_token_id,
        }
        if do_sample:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": float(max(temperature, 1e-5)),
                    "top_p": float(top_p),
                }
            )
        else:
            gen_kwargs.update(
                {
                    "do_sample": False,
                    "num_beams": max(1, nret),
                    "early_stopping": True,
                }
            )
        outputs = model.generate(**gen_kwargs)
        prompt_len = input_ids.shape[1]
        decoded = []
        for seq in outputs:
            decoded.append(tok.decode(seq[prompt_len:], skip_special_tokens=True).strip())
        for i in range(len(chunk)):
            grouped.append(decoded[i * nret : (i + 1) * nret])

    tok.padding_side = old_padding_side
    torch.random.set_rng_state(rng_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    random.setstate(random_state)
    np.random.set_state(np_state)
    return grouped


class HelpfulnessRewardOracle:
    def __init__(
        self,
        model_path: str,
        torch_dtype: str,
        device_map: str,
        max_length: int,
        batch_size: int,
    ) -> None:
        self.model_path = model_path
        self.max_length = int(max_length)
        self.batch_size = max(1, int(batch_size))
        dtype = parse_dtype(torch_dtype)
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        device_map_arg: Any
        if str(device_map).lower() in {"none", ""}:
            device_map_arg = None
        else:
            device_map_arg = device_map
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map_arg,
        )
        if device_map_arg is None:
            self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _format_chat(self, prompt: str, response: str) -> str:
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return f"User: {prompt}\nAssistant: {response}"

    @torch.no_grad()
    def score(self, prompts: List[str], responses: List[str]) -> np.ndarray:
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have equal length")
        if len(prompts) == 0:
            return np.asarray([], dtype=np.float64)
        rewards: List[float] = []
        device = get_first_device(self.model)
        for s in tqdm(range(0, len(prompts), self.batch_size), desc="oracle_score", ncols=100):
            e = min(len(prompts), s + self.batch_size)
            texts = [self._format_chat(p, r) for p, r in zip(prompts[s:e], responses[s:e])]
            enc = self.tok(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = self.model.generate(
                enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
            )
            batch_rewards = out["scores"][0][:, 0].detach().float().cpu().numpy().tolist()
            rewards.extend(float(x) for x in batch_rewards)
        return np.asarray(rewards, dtype=np.float64)


@dataclass
class OracleBaseline:
    prompts: List[str]
    prompt_ids: List[int]
    responses_by_prompt: List[List[str]]
    rewards_by_prompt: List[List[float]]


def load_oracle_baseline_cache(path: str) -> OracleBaseline:
    rows = read_jsonl(path)
    prompts, prompt_ids, responses, rewards = [], [], [], []
    for i, row in enumerate(rows):
        prompt = row.get("prompt")
        if not isinstance(prompt, str):
            continue
        resp_rows = row.get("responses")
        if not isinstance(resp_rows, list):
            continue
        ys, rs = [], []
        for obj in resp_rows:
            if not isinstance(obj, dict):
                continue
            text = obj.get("text")
            reward = obj.get("oracle_reward")
            if isinstance(text, str) and isinstance(reward, (int, float)):
                ys.append(text)
                rs.append(float(reward))
        if ys:
            prompts.append(prompt)
            prompt_ids.append(int(row.get("prompt_id", i)))
            responses.append(ys)
            rewards.append(rs)
    if not prompts:
        raise ValueError(f"No valid oracle baseline rows found in {path}")
    return OracleBaseline(prompts, prompt_ids, responses, rewards)


def write_oracle_baseline_cache(path: str, baseline: OracleBaseline) -> None:
    rows = []
    for pid, (prompt, prompt_id, ys, rs) in enumerate(
        zip(baseline.prompts, baseline.prompt_ids, baseline.responses_by_prompt, baseline.rewards_by_prompt)
    ):
        rows.append(
            {
                "prompt_index": pid,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "responses": [
                    {"response_index": j, "text": y, "oracle_reward": float(r)}
                    for j, (y, r) in enumerate(zip(ys, rs))
                ],
            }
        )
    write_jsonl(path, rows)


def build_or_load_oracle_baseline(
    args,
    ref0,
    tok,
    oracle: HelpfulnessRewardOracle,
    prompt_pool: List[str],
    prompt_id_pool: List[int],
) -> OracleBaseline:
    rng = random.Random(args.oracle_seed)
    n = min(int(args.oracle_num_prompts), len(prompt_pool))
    chosen = sorted(rng.sample(range(len(prompt_pool)), k=n))
    prompts = [prompt_pool[i] for i in chosen]
    prompt_ids = [prompt_id_pool[i] for i in chosen]

    cache_path = args.oracle_baseline_cache_path.strip()
    if cache_path and args.oracle_reuse_baseline_cache == 1 and os.path.exists(cache_path):
        print(f"[Oracle] loading cached pi0 baseline from {cache_path}")
        cached = load_oracle_baseline_cache(cache_path)
        if cached.prompts == prompts and cached.prompt_ids == prompt_ids:
            return cached
        print("[Oracle] cache prompt set does not match this run; rebuilding pi0 baseline.")

    print(f"[Oracle] generating pi0 baseline responses: prompts={len(prompts)} M={args.oracle_num_responses}")
    responses_by_prompt = generate_responses_for_prompts_batched(
        model=ref0,
        tok=tok,
        prompts=prompts,
        num_return_sequences=args.oracle_num_responses,
        batch_size=args.oracle_generation_batch_size,
        max_new_tokens=args.oracle_max_new_tokens,
        do_sample=bool(args.oracle_do_sample),
        temperature=args.oracle_temperature,
        top_p=args.oracle_top_p,
        seed=args.oracle_seed + 17,
    )
    flat_prompts, flat_responses = [], []
    for prompt, ys in zip(prompts, responses_by_prompt):
        for y in ys:
            flat_prompts.append(prompt)
            flat_responses.append(y)
    rewards_flat = oracle.score(flat_prompts, flat_responses)
    rewards_by_prompt: List[List[float]] = []
    cur = 0
    for ys in responses_by_prompt:
        rewards_by_prompt.append([float(x) for x in rewards_flat[cur : cur + len(ys)]])
        cur += len(ys)
    baseline = OracleBaseline(prompts, prompt_ids, responses_by_prompt, rewards_by_prompt)
    if cache_path:
        ensure_dir(os.path.dirname(cache_path) or ".")
        write_oracle_baseline_cache(cache_path, baseline)
        print(f"[Oracle] wrote pi0 baseline cache: {cache_path}")
    return baseline


def evaluate_oracle_checkpoint(
    args,
    model,
    tok,
    oracle: HelpfulnessRewardOracle,
    baseline: OracleBaseline,
    iteration: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    responses_by_prompt = generate_responses_for_prompts_batched(
        model=model,
        tok=tok,
        prompts=baseline.prompts,
        num_return_sequences=args.oracle_num_responses,
        batch_size=args.oracle_generation_batch_size,
        max_new_tokens=args.oracle_max_new_tokens,
        do_sample=bool(args.oracle_do_sample),
        temperature=args.oracle_temperature,
        top_p=args.oracle_top_p,
        seed=args.oracle_seed + 100003 * (iteration + 1),
    )
    flat_prompts, flat_responses = [], []
    for prompt, ys in zip(baseline.prompts, responses_by_prompt):
        for y in ys:
            flat_prompts.append(prompt)
            flat_responses.append(y)
    rewards_flat = oracle.score(flat_prompts, flat_responses)

    win_rates, soft_win_rates, mean_reward_deltas = [], [], []
    response_rows: List[Dict[str, Any]] = []
    cur = 0
    for pid, (prompt, prompt_id, ys, base_rs) in enumerate(
        zip(baseline.prompts, baseline.prompt_ids, responses_by_prompt, baseline.rewards_by_prompt)
    ):
        cur_rewards = np.asarray(rewards_flat[cur : cur + len(ys)], dtype=np.float64)
        cur += len(ys)
        base_rewards = np.asarray(base_rs, dtype=np.float64)
        if len(cur_rewards) == 0 or len(base_rewards) == 0:
            continue
        diff = cur_rewards[:, None] - base_rewards[None, :]
        prompt_wr = float(np.mean(diff > 0.0))
        prompt_tie = float(np.mean(diff == 0.0))
        prompt_swr = float(np.mean(1.0 / (1.0 + np.exp(-diff))))
        win_rates.append(prompt_wr)
        soft_win_rates.append(prompt_swr)
        mean_reward_deltas.append(float(np.mean(cur_rewards) - np.mean(base_rewards)))
        for j, (text, reward) in enumerate(zip(ys, cur_rewards.tolist())):
            response_rows.append(
                {
                    "iter": int(iteration),
                    "prompt_index": int(pid),
                    "prompt_id": int(prompt_id),
                    "model_role": "checkpoint",
                    "response_index": int(j),
                    "oracle_reward": float(reward),
                    "response_length_chars": int(len(text)),
                    "prompt": prompt,
                    "response": text,
                    "prompt_win_rate_vs_pi0": prompt_wr,
                    "prompt_soft_win_rate_vs_pi0": prompt_swr,
                    "prompt_tie_rate_vs_pi0": prompt_tie,
                }
            )
    if not win_rates:
        stats = {
            "oracle_win_rate": float("nan"),
            "oracle_soft_win_rate": float("nan"),
            "oracle_reward_delta_mean": float("nan"),
            "oracle_num_prompts": 0,
            "oracle_num_checkpoint_responses": 0,
            "oracle_num_baseline_responses": 0,
            "oracle_avg_response_len_chars": float("nan"),
        }
    else:
        lengths = [row["response_length_chars"] for row in response_rows]
        stats = {
            "oracle_win_rate": float(np.mean(win_rates)),
            "oracle_soft_win_rate": float(np.mean(soft_win_rates)),
            "oracle_reward_delta_mean": float(np.mean(mean_reward_deltas)),
            "oracle_win_rate_prompt_std": float(np.std(win_rates)),
            "oracle_soft_win_rate_prompt_std": float(np.std(soft_win_rates)),
            "oracle_num_prompts": int(len(win_rates)),
            "oracle_num_checkpoint_responses": int(sum(len(x) for x in responses_by_prompt)),
            "oracle_num_baseline_responses": int(sum(len(x) for x in baseline.responses_by_prompt)),
            "oracle_avg_response_len_chars": float(np.mean(lengths)) if lengths else float("nan"),
        }
    return stats, response_rows


def flatten_eval_support(
    prompts: List[str], responses_by_prompt: List[List[str]]
) -> Tuple[List[str], List[str], List[Tuple[int, int]]]:
    flat_prompts, flat_resps, group_offsets = [], [], []
    cur = 0
    for prompt, ys in zip(prompts, responses_by_prompt):
        start = cur
        for y in ys:
            flat_prompts.append(prompt)
            flat_resps.append(y)
            cur += 1
        group_offsets.append((start, cur))
    return flat_prompts, flat_resps, group_offsets


def dump_prompt_metrics(
    dump_dir: str,
    t: int,
    preference_case: str,
    prompt_ids: List[int],
    prompts: List[str],
    responses_by_prompt: List[List[str]],
    sources_by_prompt: List[List[str]],
    group_offsets: List[Tuple[int, int]],
    max_k: int,
    flat_avg_scores: np.ndarray,
    q_matrix: np.ndarray,
    entropies: np.ndarray,
    tvs: np.ndarray,
    top1: np.ndarray,
    top1_initial: Optional[np.ndarray],
) -> None:
    rows = []
    for pid, (start, end) in enumerate(group_offsets):
        k = end - start
        row: Dict[str, Any] = {
            "iter": int(t),
            "preference_case": preference_case,
            "prompt_index": int(pid),
            "prompt_id": int(prompt_ids[pid]),
            "prompt": prompts[pid],
            "K": int(k),
            "entropy": float(entropies[pid]),
            "tv_delta": float(tvs[pid]) if np.isfinite(tvs[pid]) else np.nan,
            "top1_idx": int(top1[pid]),
            "top1_initial_idx": int(top1_initial[pid]) if top1_initial is not None else -1,
            "top1_flipped_vs_initial": int(top1_initial is not None and top1[pid] != top1_initial[pid]),
        }
        for j in range(max_k):
            if j < k:
                flat_idx = start + j
                row[f"avg_logprob_{j}"] = float(flat_avg_scores[flat_idx])
                row[f"prob_{j}"] = float(q_matrix[pid, j])
                row[f"response_{j}"] = responses_by_prompt[pid][j]
                row[f"response_source_{j}"] = sources_by_prompt[pid][j]
            else:
                row[f"avg_logprob_{j}"] = np.nan
                row[f"prob_{j}"] = np.nan
                row[f"response_{j}"] = ""
                row[f"response_source_{j}"] = ""
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(dump_dir, f"iter_{t:04d}_prompt_metrics.csv"), index=False)


def add_common_args(ap: argparse.ArgumentParser, loss_type: str) -> None:
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--pairs_path", type=str, required=True)
    ap.add_argument("--eval_prompts_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=f"checkpoints_{loss_type}_oracle")
    ap.add_argument("--log_dir", type=str, default=f"logs_{loss_type}_oracle")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preference_case", type=str, required=True, choices=["transitive", "cyclic"])

    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--max_iters", type=int, default=100)
    ap.add_argument("--auto_stop", type=int, default=0)

    ap.add_argument("--epochs_per_iter", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--lambda_on", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--mix_eps", type=float, default=0.05)
    ap.add_argument("--w_clip_min", type=float, default=0.1)
    ap.add_argument("--w_clip_max", type=float, default=10.0)

    ap.add_argument("--max_length", type=int, default=1537)
    ap.add_argument("--train_sample_size", type=int, default=1000)
    ap.add_argument("--pairs_per_prompt", type=int, default=2)
    ap.add_argument("--train_prompt_size", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--score_batch_size", type=int, default=8)
    ap.add_argument("--model_torch_dtype", type=str, default="float16")

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--save_iter_adapters", type=int, default=0)
    ap.add_argument("--save_initial_adapter", type=int, default=0)
    ap.add_argument("--save_final_adapter", type=int, default=0)
    ap.add_argument("--dump_each_iter", type=int, default=1)

    ap.add_argument("--generated_eval_num_prompts", type=int, default=500)
    ap.add_argument("--generated_eval_num_candidates", type=int, default=10)
    ap.add_argument("--generated_eval_keep_top_k", type=int, default=5)
    ap.add_argument("--generated_eval_max_new_tokens", type=int, default=256)
    ap.add_argument("--generated_eval_do_sample", type=int, default=1)
    ap.add_argument("--generated_eval_temperature", type=float, default=0.8)
    ap.add_argument("--generated_eval_top_p", type=float, default=0.95)
    ap.add_argument("--generated_eval_seed", type=int, default=123)
    ap.add_argument("--cyclic_eval_keep_k", type=int, default=4)
    ap.add_argument("--cycle_burn_in", type=int, default=10)

    ap.add_argument("--enable_oracle", type=int, default=1)
    ap.add_argument("--oracle_model_path", type=str, default=DEFAULT_ORACLE_MODEL)
    ap.add_argument("--oracle_torch_dtype", type=str, default="bfloat16")
    ap.add_argument("--oracle_device_map", type=str, default="auto")
    ap.add_argument("--oracle_max_length", type=int, default=4096)
    ap.add_argument("--oracle_batch_size", type=int, default=4)
    ap.add_argument("--oracle_eval_every", type=int, default=1)
    ap.add_argument("--oracle_num_prompts", type=int, default=500)
    ap.add_argument("--oracle_num_responses", type=int, default=4)
    ap.add_argument("--oracle_generation_batch_size", type=int, default=8)
    ap.add_argument("--oracle_max_new_tokens", type=int, default=256)
    ap.add_argument("--oracle_do_sample", type=int, default=1)
    ap.add_argument("--oracle_temperature", type=float, default=0.8)
    ap.add_argument("--oracle_top_p", type=float, default=0.95)
    ap.add_argument("--oracle_seed", type=int, default=777)
    ap.add_argument("--oracle_baseline_cache_path", type=str, default="")
    ap.add_argument("--oracle_reuse_baseline_cache", type=int, default=1)


def validate_args(args) -> None:
    args.alpha = float(max(0.0, min(1.0, args.alpha)))
    args.lambda_on = float(max(0.0, min(1.0, args.lambda_on)))
    args.mix_eps = float(max(0.0, min(1.0, args.mix_eps)))
    if args.batch_size < 1 or args.grad_accum < 1 or args.score_batch_size < 1:
        raise ValueError("batch_size, grad_accum, and score_batch_size must be >= 1")
    if args.train_prompt_size < 0:
        raise ValueError("--train_prompt_size must be >= 0")
    if args.pairs_per_prompt < 1:
        raise ValueError("--pairs_per_prompt must be >= 1")
    if args.max_length < 1:
        raise ValueError("--max_length must be >= 1")
    if args.generated_eval_num_prompts < 1:
        raise ValueError("--generated_eval_num_prompts must be >= 1")
    if args.preference_case == "cyclic" and args.cyclic_eval_keep_k < 2:
        raise ValueError("--cyclic_eval_keep_k must be >= 2")
    if args.enable_oracle == 1:
        if args.oracle_num_prompts < 1 or args.oracle_num_responses < 1:
            raise ValueError("oracle_num_prompts and oracle_num_responses must be >= 1")
        if args.oracle_eval_every < 1:
            raise ValueError("--oracle_eval_every must be >= 1")


def run_experiment(loss_type: str) -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap, loss_type)
    args = ap.parse_args()
    validate_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these experiments.")

    ensure_dir(args.out_dir)
    ensure_dir(args.log_dir)

    ds = PairDataset(read_jsonl(args.pairs_path))
    if len(ds) == 0:
        raise ValueError("No valid training pairs found in pairs_path.")
    prompt_to_pair_indices = build_prompt_to_pair_indices(ds)
    prompt_pool, prompt_id_pool, _prompt_rows = load_prompt_rows(args.eval_prompts_path)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_device = torch.device("cuda:0")
    model_dtype = parse_dtype(args.model_torch_dtype)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=model_dtype,
        device_map=None,
    ).to(train_device)
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
        torch_dtype=model_dtype,
        device_map=None,
    ).to(train_device)
    ref0.eval()
    for p in ref0.parameters():
        p.requires_grad_(False)

    if args.preference_case == "transitive":
        prompts_eval, prompt_ids_eval, responses_by_prompt, sources_by_prompt, generation_rows = build_generated_eval_set(
            model=model,
            tok=tok,
            prompt_pool=prompt_pool,
            prompt_id_pool=prompt_id_pool,
            num_eval_prompts=args.generated_eval_num_prompts,
            num_candidates_per_prompt=args.generated_eval_num_candidates,
            keep_top_k=args.generated_eval_keep_top_k,
            max_new_tokens=args.generated_eval_max_new_tokens,
            do_sample=bool(args.generated_eval_do_sample),
            temperature=args.generated_eval_temperature,
            top_p=args.generated_eval_top_p,
            score_batch_size=args.score_batch_size,
            score_max_length=args.max_length,
            seed=args.generated_eval_seed,
            device=train_device,
        )
        eval_support_meta = [{"mode": "pi0_generated", "num_generation_rows": len(generation_rows)}]
    else:
        prompts_eval, prompt_ids_eval, responses_by_prompt, sources_by_prompt, eval_support_meta = load_cyclic_eval_support(
            path=args.eval_prompts_path,
            num_prompts=args.generated_eval_num_prompts,
            seed=args.generated_eval_seed,
            keep_k=args.cyclic_eval_keep_k,
        )
        generation_rows = []

    num_prompts_eval = len(prompts_eval)
    max_k = max(len(x) for x in responses_by_prompt)
    flat_prompts, flat_resps, group_offsets = flatten_eval_support(prompts_eval, responses_by_prompt)

    run_tag = (
        f"{loss_type}_{args.preference_case}_alpha{args.alpha}_lambda{args.lambda_on}"
        f"_tau{args.tau}_beta{args.beta}_seed{args.seed}"
    )
    adapters_dir = os.path.join(args.out_dir, f"adapters_{run_tag}")
    dump_dir = os.path.join(args.log_dir, f"iter_dumps_{run_tag}")
    ensure_dir(adapters_dir)
    if args.dump_each_iter == 1:
        ensure_dir(dump_dir)
    if args.save_initial_adapter == 1:
        maybe_save_adapter(model, tok, os.path.join(adapters_dir, "iter_init"))

    eval_meta_rows = []
    for pid, (prompt_id, prompt, ys, srcs) in enumerate(
        zip(prompt_ids_eval, prompts_eval, responses_by_prompt, sources_by_prompt)
    ):
        for j, (y, src) in enumerate(zip(ys, srcs)):
            eval_meta_rows.append(
                {
                    "preference_case": args.preference_case,
                    "prompt_index": pid,
                    "prompt_id": prompt_id,
                    "response_index": j,
                    "prompt": prompt,
                    "response": y,
                    "response_source": src,
                }
            )
    pd.DataFrame(eval_meta_rows).to_csv(os.path.join(args.log_dir, f"eval_support_{run_tag}.csv"), index=False)
    write_jsonl(
        os.path.join(args.log_dir, f"eval_support_{run_tag}.jsonl"),
        [
            {
                "prompt_index": i,
                "prompt_id": int(prompt_ids_eval[i]),
                "prompt": prompts_eval[i],
                "responses": [
                    {"response_index": j, "text": y, "source": sources_by_prompt[i][j]}
                    for j, y in enumerate(responses_by_prompt[i])
                ],
            }
            for i in range(num_prompts_eval)
        ],
    )
    if generation_rows:
        pd.DataFrame(generation_rows).to_csv(
            os.path.join(args.log_dir, f"generated_eval_generation_scores_{run_tag}.csv"),
            index=False,
        )

    oracle = None
    oracle_baseline = None
    oracle_response_rows: List[Dict[str, Any]] = []
    if args.enable_oracle == 1:
        torch.cuda.empty_cache()
        print(
            f"[Oracle] loading {args.oracle_model_path} with device_map={args.oracle_device_map}. "
            "This is the multi-GPU evaluator."
        )
        oracle = HelpfulnessRewardOracle(
            model_path=args.oracle_model_path,
            torch_dtype=args.oracle_torch_dtype,
            device_map=args.oracle_device_map,
            max_length=args.oracle_max_length,
            batch_size=args.oracle_batch_size,
        )
        oracle_baseline = build_or_load_oracle_baseline(args, ref0, tok, oracle, prompts_eval, prompt_ids_eval)

    metrics: List[Dict[str, Any]] = []
    prev_q: Optional[np.ndarray] = None
    prev_entropy: Optional[np.ndarray] = None
    initial_top1: Optional[np.ndarray] = None
    q_history: List[np.ndarray] = []
    global_train_batch_step = 0
    T = int(args.max_iters if args.auto_stop == 1 else args.iters)
    pct_eps = 1e-12

    metrics_csv = os.path.join(args.log_dir, f"metrics_{run_tag}.csv")
    oracle_resp_csv = os.path.join(args.log_dir, f"oracle_response_scores_{run_tag}.csv")

    for t in range(T):
        print(
            f"\n===== {loss_type.upper()} {args.preference_case.upper()} OUTER ITER {t} | "
            f"alpha={args.alpha} lambda={args.lambda_on} tau={args.tau} beta={args.beta} ====="
        )
        if args.save_iter_adapters == 1:
            maybe_save_adapter(model, tok, os.path.join(adapters_dir, f"iter_{t:04d}_preupdate"))

        model.eval()
        ref0.eval()
        flat_avg_scores = np.zeros(len(flat_prompts), dtype=np.float32)
        bs = max(1, int(args.score_batch_size))
        for s in tqdm(range(0, len(flat_prompts), bs), desc=f"score_eval_support@{t}", ncols=100):
            e = min(len(flat_prompts), s + bs)
            _, avg_lp, _ = batch_sum_and_avg_logprob(
                model, tok, flat_prompts[s:e], flat_resps[s:e], args.max_length, train_device
            )
            flat_avg_scores[s:e] = avg_lp.numpy()

        q_matrix = np.full((num_prompts_eval, max_k), np.nan, dtype=np.float64)
        entropies = np.zeros(num_prompts_eval, dtype=np.float64)
        tvs = np.full(num_prompts_eval, np.nan, dtype=np.float64)
        top1 = np.full(num_prompts_eval, -1, dtype=np.int64)
        for pid, (start, end) in enumerate(group_offsets):
            scores = flat_avg_scores[start:end].astype(np.float64)
            probs = safe_softmax_np(scores * float(args.tau))
            k = end - start
            q_matrix[pid, :k] = probs
            entropies[pid] = entropy_from_probs(probs)
            top1[pid] = int(np.argmax(probs))
            if prev_q is not None:
                prev_probs = prev_q[pid, :k].astype(np.float64)
                prev_probs = prev_probs / np.sum(prev_probs)
                tvs[pid] = total_variation(probs, prev_probs)

        if initial_top1 is None:
            initial_top1 = top1.copy()
        top1_flip_rate = float(np.mean(top1 != initial_top1)) if initial_top1 is not None else float("nan")
        entropy_mean = float(np.mean(entropies))
        entropy_min = float(np.min(entropies))
        entropy_max = float(np.max(entropies))
        tv_mean = float(np.nanmean(tvs)) if np.any(~np.isnan(tvs)) else float("nan")
        tv_max = float(np.nanmax(tvs)) if np.any(~np.isnan(tvs)) else float("nan")
        if prev_entropy is None:
            entropy_abs_delta_mean = float("nan")
            entropy_abs_delta_max = float("nan")
            entropy_pct_change_mean = float("nan")
        else:
            delta_h = entropies - prev_entropy
            entropy_abs_delta_mean = float(np.mean(np.abs(delta_h)))
            entropy_abs_delta_max = float(np.max(np.abs(delta_h)))
            entropy_pct_change_mean = float(np.mean(100.0 * delta_h / np.maximum(np.abs(prev_entropy), pct_eps)))

        q_history.append(q_matrix.copy())
        cycle_strength_mean = float("nan")
        cycle_strength_max = float("nan")
        if args.preference_case == "cyclic" and t >= int(args.cycle_burn_in):
            hist = np.stack(q_history[int(args.cycle_burn_in) :], axis=0)
            var_by_prompt_resp = np.nanvar(hist, axis=0)
            cs_by_prompt = np.nanmean(var_by_prompt_resp, axis=1)
            cycle_strength_mean = float(np.nanmean(cs_by_prompt))
            cycle_strength_max = float(np.nanmax(cs_by_prompt))

        oracle_stats: Dict[str, Any] = {
            "oracle_win_rate": float("nan"),
            "oracle_soft_win_rate": float("nan"),
            "oracle_reward_delta_mean": float("nan"),
            "oracle_win_rate_prompt_std": float("nan"),
            "oracle_soft_win_rate_prompt_std": float("nan"),
            "oracle_num_prompts": 0,
            "oracle_num_checkpoint_responses": 0,
            "oracle_num_baseline_responses": 0,
            "oracle_avg_response_len_chars": float("nan"),
        }
        if oracle is not None and oracle_baseline is not None and t % int(args.oracle_eval_every) == 0:
            oracle_stats, new_oracle_rows = evaluate_oracle_checkpoint(args, model, tok, oracle, oracle_baseline, t)
            oracle_response_rows.extend(new_oracle_rows)
            pd.DataFrame(oracle_response_rows).to_csv(oracle_resp_csv, index=False)
            print(
                f"[Oracle@t={t}] WR={oracle_stats['oracle_win_rate']:.4f} "
                f"SWR={oracle_stats['oracle_soft_win_rate']:.4f} "
                f"dR={oracle_stats['oracle_reward_delta_mean']:.4f}"
            )

        target_prompt_count = (
            args.train_prompt_size
            if args.train_prompt_size > 0
            else int(math.ceil(args.train_sample_size / max(1, args.pairs_per_prompt)))
        )
        rng_train = random.Random(args.seed + 999 * (t + 1))
        train_ds_weighted, train_diag_df, sampled_prompts, sampled_pairs_per_prompt = build_prompt_aware_training_subset(
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
            train_device,
            args.score_batch_size,
            args.w_clip_min,
            args.w_clip_max,
        )

        if args.dump_each_iter == 1:
            dump_prompt_metrics(
                dump_dir,
                t,
                args.preference_case,
                prompt_ids_eval,
                prompts_eval,
                responses_by_prompt,
                sources_by_prompt,
                group_offsets,
                max_k,
                flat_avg_scores,
                q_matrix,
                entropies,
                tvs,
                top1,
                initial_top1,
            )
            train_diag_df.to_csv(os.path.join(dump_dir, f"iter_{t:04d}_train_pair_support.csv"), index=False)

        metric_row: Dict[str, Any] = {
            "iter": int(t),
            "snapshot_stage": "pre_update",
            "loss_type": loss_type,
            "preference_case": args.preference_case,
            "alpha": args.alpha,
            "lambda": args.lambda_on,
            "tau": args.tau,
            "beta": args.beta,
            "mix_eps": args.mix_eps,
            "target_train_prompt_count": int(target_prompt_count),
            "actual_train_pairs": int(len(train_ds_weighted)),
            "sampled_train_prompts": int(len(sampled_prompts)),
            "train_sample_size": int(args.train_sample_size),
            "pairs_per_prompt": int(args.pairs_per_prompt),
            "prompt_entropy_mean": entropy_mean,
            "prompt_entropy_min": entropy_min,
            "prompt_entropy_max": entropy_max,
            "prompt_tv_mean": tv_mean,
            "prompt_tv_max": tv_max,
            "top1_flip_rate_vs_initial": top1_flip_rate,
            "prompt_entropy_abs_delta_mean": entropy_abs_delta_mean,
            "prompt_entropy_abs_delta_max": entropy_abs_delta_max,
            "prompt_entropy_pct_change_mean": entropy_pct_change_mean,
            "cycle_strength_mean": cycle_strength_mean,
            "cycle_strength_max": cycle_strength_max,
            "num_prompts_eval": int(num_prompts_eval),
            "max_eval_support_k": int(max_k),
            "train_loss_mean": float("nan"),
            "train_loss_std": float("nan"),
            "train_loss_min": float("nan"),
            "train_loss_max": float("nan"),
            "train_loss_last": float("nan"),
            "train_num_batches": 0,
            "optimizer_steps_in_iter": 0,
            **oracle_stats,
        }

        print(
            f"[Metrics@t={t}] H={entropy_mean:.6g} TV={tv_mean:.6g} "
            f"top1_flip={top1_flip_rate:.3f} CS={cycle_strength_mean:.6g} "
            f"train_pairs={len(train_ds_weighted)}"
        )

        prev_q = q_matrix.copy()
        prev_entropy = entropies.copy()

        iter_loss_values: List[float] = []
        model.train()
        train_loader = DataLoader(
            train_ds_weighted,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            drop_last=False,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        num_batches = len(train_loader) * int(args.epochs_per_iter)
        total_steps = max(1, math.ceil(num_batches / int(args.grad_accum)))
        warmup_steps = int(args.warmup_ratio * total_steps)
        sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
        opt.zero_grad(set_to_none=True)
        step = 0
        optimizer_step_in_iter = 0

        for ep in range(int(args.epochs_per_iter)):
            pbar = tqdm(train_loader, desc=f"train_{loss_type}@iter{t}_ep{ep}", ncols=100)
            for batch in pbar:
                with torch.no_grad():
                    lp_c_pi = batch_sum_and_avg_logprob(
                        model, tok, batch["prompt"], batch["chosen"], args.max_length, train_device
                    )[1].to(train_device)
                    lp_r_pi = batch_sum_and_avg_logprob(
                        model, tok, batch["prompt"], batch["rejected"], args.max_length, train_device
                    )[1].to(train_device)
                    lp_c_ref0 = batch_sum_and_avg_logprob(
                        ref0, tok, batch["prompt"], batch["chosen"], args.max_length, train_device
                    )[1].to(train_device)
                    lp_r_ref0 = batch_sum_and_avg_logprob(
                        ref0, tok, batch["prompt"], batch["rejected"], args.max_length, train_device
                    )[1].to(train_device)
                    lp_c_ref_t = (1.0 - args.alpha) * lp_c_ref0 + args.alpha * lp_c_pi
                    lp_r_ref_t = (1.0 - args.alpha) * lp_r_ref0 + args.alpha * lp_r_pi

                bc = run_build_batch(tok, batch["prompt"], batch["chosen"], args.max_length, train_device)
                out_c = model(input_ids=bc["input_ids"], attention_mask=bc["attention_mask"], labels=bc["labels"])
                s_c, c_c = sum_logprob_and_count_from_outputs(out_c.logits, bc["labels"])
                avg_c = s_c / c_c

                br = run_build_batch(tok, batch["prompt"], batch["rejected"], args.max_length, train_device)
                out_r = model(input_ids=br["input_ids"], attention_mask=br["attention_mask"], labels=br["labels"])
                s_r, c_r = sum_logprob_and_count_from_outputs(out_r.logits, br["labels"])
                avg_r = s_r / c_r

                delta = (avg_c - lp_c_ref_t) - (avg_r - lp_r_ref_t)
                loss_vec = loss_from_delta(delta, args.beta, loss_type)
                wt = batch["pair_weight"].to(train_device)
                loss = (wt * loss_vec).mean()
                loss_value = float(loss.detach().item())
                iter_loss_values.append(loss_value)
                global_train_batch_step += 1
                loss.backward()
                step += 1

                if step % int(args.grad_accum) == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    optimizer_step_in_iter += 1
                pbar.set_postfix({"loss": loss_value})

        if step % int(args.grad_accum) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            optimizer_step_in_iter += 1

        if iter_loss_values:
            metric_row.update(
                {
                    "train_loss_mean": float(np.mean(iter_loss_values)),
                    "train_loss_std": float(np.std(iter_loss_values)),
                    "train_loss_min": float(np.min(iter_loss_values)),
                    "train_loss_max": float(np.max(iter_loss_values)),
                    "train_loss_last": float(iter_loss_values[-1]),
                    "train_num_batches": int(len(iter_loss_values)),
                    "optimizer_steps_in_iter": int(optimizer_step_in_iter),
                }
            )

        metrics.append(metric_row)
        pd.DataFrame(metrics).to_csv(metrics_csv, index=False)
        if args.save_iter_adapters == 1:
            maybe_save_adapter(model, tok, os.path.join(adapters_dir, f"iter_{t:04d}_postupdate"))

    if args.save_final_adapter == 1:
        maybe_save_adapter(model, tok, os.path.join(adapters_dir, "final"))

    summary_path = os.path.join(args.log_dir, f"summary_{run_tag}.json")
    write_json(
        summary_path,
        {
            "loss_type": loss_type,
            "preference_case": args.preference_case,
            "alpha": args.alpha,
            "lambda_on": args.lambda_on,
            "tau": args.tau,
            "beta": args.beta,
            "seed": args.seed,
            "pairs_path": args.pairs_path,
            "eval_prompts_path": args.eval_prompts_path,
            "num_prompts_eval": num_prompts_eval,
            "eval_support_mode": "pi0_generated" if args.preference_case == "transitive" else "dataset_cyclic_responses",
            "eval_support_meta": eval_support_meta,
            "oracle_enabled": bool(args.enable_oracle),
            "oracle_model_path": args.oracle_model_path if args.enable_oracle == 1 else None,
            "oracle_num_prompts": args.oracle_num_prompts if args.enable_oracle == 1 else 0,
            "oracle_num_responses": args.oracle_num_responses if args.enable_oracle == 1 else 0,
            "oracle_eval_every": args.oracle_eval_every if args.enable_oracle == 1 else 0,
            "artifacts": {
                "metrics_csv": metrics_csv,
                "oracle_response_scores_csv": oracle_resp_csv if args.enable_oracle == 1 else None,
                "adapters_dir": adapters_dir,
                "dump_dir": dump_dir if args.dump_each_iter == 1 else None,
            },
        },
    )
    print("[DONE] wrote:", metrics_csv)
    if args.enable_oracle == 1:
        print("[DONE] wrote:", oracle_resp_csv)
    print("[DONE] wrote:", summary_path)
