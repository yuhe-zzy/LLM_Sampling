"""Isolated fixed-panel cyclic IPO/DPO diagnostics. Preview-only by default.

Adapted from the existing run_ipo.py/run_dpo.py data, tokenization, LoRA and
pair-loss pipeline. Original training scripts and results are never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from history_math import (build_outer_state, centered, describe_distribution, finite,
                          load_panels, sample_pairs, support_hash)


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def resolve_config(plan_path, run_id):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    matches = [row for row in plan["runs"] if row["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown/duplicate run_id {run_id}")
    cfg = dict(plan["common"], **matches[0])
    cfg["protocol"] = plan["protocol"]
    cfg["plan_status"] = plan["status"]
    return cfg


def validate_config(c):
    if c["method"] not in ("ipo", "dpo"):
        raise ValueError("Unknown loss")
    if c["scheme"] not in ("ordinary", "lagged_reference", "lagged_sampling"):
        raise ValueError("Unknown scheme")
    if not 0 <= c["alpha"] < 1 or not 0 <= c["lambda_current"] < 1:
        raise ValueError("alpha/lambda_current must be in [0,1)")
    if not 0 <= c["nu"] <= c["alpha"] or not np.isfinite(c["kappa"]) or c["kappa"] < 0:
        raise ValueError("Invalid history coefficients")
    if c["scheme"] == "ordinary" and (c["nu"] or c["kappa"]):
        raise ValueError("Ordinary baseline cannot use history coefficients")
    if c["scheme"] == "lagged_reference" and (c["nu"] <= 0 or c["kappa"]):
        raise ValueError("Reference ablation needs positive nu and zero kappa")
    if c["scheme"] == "lagged_sampling" and (c["kappa"] <= 0 or c["nu"]):
        raise ValueError("Sampling ablation needs positive kappa and zero nu")
    for key in ("beta_train", "lr"):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"Invalid {key}")
    for key in ("num_prompts", "keep_k", "iters", "pairs_per_prompt", "batch_size",
                "grad_accum", "epochs_per_iter", "max_length", "score_batch_size",
                "lora_r", "lora_alpha", "checkpoint_every"):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seed", "support_seed"):
        if type(c[key]) is not int or c[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if c["keep_k"] != 4 or c["lora_dropout"] != 0:
        raise ValueError("This protocol fixes K=4 and dropout=0 across all arms")
    if not 0 <= c["warmup_ratio"] < 1:
        raise ValueError("Invalid warmup_ratio")
    expected = dict(coverage="uniform_fixed_panel", pair_law="mu_i_mu_j_conditioned_distinct",
                    support_probability="softmax_sequence_sum", optimizer_reset="each_outer_iteration")
    if any(c.get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported protocol: changing strings alone cannot change behavior")
    if c["dtype"] not in ("bfloat16", "float32"):
        raise ValueError("Only bfloat16/float32 supported; no unscaled fp16 training")
    if Path(c["run_id"]).name != c["run_id"] or "/" in c["run_id"] or "\\" in c["run_id"]:
        raise ValueError("run_id must be a filename component")


def encode_response(tok, prompt, response, max_length):
    prefix = tok(prompt, add_special_tokens=False).input_ids
    answer = tok(response, add_special_tokens=False).input_ids
    if tok.eos_token_id is not None:
        answer = answer + [tok.eos_token_id]
    if not prefix or not answer:
        raise ValueError("Empty tokenized prompt/response")
    if len(answer) >= max_length:
        raise ValueError("Response would be truncated; increase max_length explicitly")
    kept_prefix = prefix[-(max_length - len(answer)):]
    return dict(input_ids=kept_prefix + answer,
                labels=[-100] * len(kept_prefix) + answer,
                response_tokens=len(answer), truncated_prompt_tokens=len(prefix) - len(kept_prefix))


def encode_panel(tok, panel, max_length):
    encoded = [encode_response(tok, panel["prompt"], answer, max_length)
               for answer in panel["responses"]]
    # Candidate-dependent truncation would compare different conditioning prompts.
    prefix_lengths = [len(row["input_ids"]) - row["response_tokens"] for row in encoded]
    common_length = min(prefix_lengths)
    for row, length in zip(encoded, prefix_lengths):
        removed = length - common_length
        row["input_ids"] = row["input_ids"][removed:]
        row["labels"] = row["labels"][removed:]
        row["truncated_prompt_tokens"] += removed
    return encoded


def write_metrics(root, rows):
    columns = sorted(set().union(*(row.keys() for row in rows)))
    temp = root / "metrics.csv.tmp"
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, root / "metrics.csv")


def build_batch(encoded, pad_id, device):
    import torch
    length = max(len(row["input_ids"]) for row in encoded)
    ids = torch.full((len(encoded), length), pad_id, dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for index, row in enumerate(encoded):
        n = len(row["input_ids"])
        ids[index, :n] = torch.tensor(row["input_ids"], device=device)
        labels[index, :n] = torch.tensor(row["labels"], device=device)
        attention[index, :n] = 1
    return dict(input_ids=ids, attention_mask=attention), labels


def sequence_scores(logits, labels):
    import torch
    targets = labels[:, 1:]
    shifted = logits[:, :-1]
    sums, counts = [], []
    # Chunk FP32 reductions rather than making a full FP32 vocabulary tensor.
    for row in range(len(labels)):
        valid = targets[row] != -100
        count = valid.sum()
        if count.item() == 0:
            raise ValueError("No response tokens after causal shifting")
        positions = torch.nonzero(valid, as_tuple=False).flatten()
        parts = []
        for chunk in positions.split(128):
            values = shifted[row, chunk].float()
            actual = targets[row, chunk]
            selected = values.gather(1, actual[:, None]).squeeze(1)
            parts.append((selected - torch.logsumexp(values, dim=-1)).sum())
        sums.append(torch.stack(parts).sum())
        counts.append(count)
    result = torch.stack(sums)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Non-finite sequence log probabilities")
    return result, torch.stack(counts)


def pair_loss(delta, preference, beta_train, method):
    import torch
    if method == "ipo":
        target = 1.0 / (2 * beta_train)
        return preference * (delta - target).square() + (1 - preference) * (delta + target).square()
    if method == "dpo":
        return torch.nn.functional.binary_cross_entropy_with_logits(
            beta_train * delta, preference, reduction="none")
    raise ValueError("Unknown loss")


def score_panel(model, encoded, pad_id, device, batch_size, shape):
    import torch
    model.eval()
    scores, counts = [], []
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            batch, labels = build_batch(encoded[start:start + batch_size], pad_id, device)
            result = model(**batch, use_cache=False)
            sums, lengths = sequence_scores(result.logits, labels)
            scores.extend(sums.cpu().double().tolist())
            counts.extend(lengths.cpu().tolist())
    return finite(scores, "panel scores").reshape(shape), np.asarray(counts).reshape(shape)


def train_round(model, encoded, matrices, outer_state, cfg, pad_id, device, round_index):
    import torch
    from transformers import get_linear_schedule_with_warmup
    pairs = sample_pairs(outer_state["mu"], cfg["pairs_per_prompt"], cfg["seed"], round_index)
    batches = [pairs[s:s + cfg["batch_size"]] for s in range(0, len(pairs), cfg["batch_size"])]
    batches = batches * cfg["epochs_per_iter"]
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["lr"])
    steps = math.ceil(len(batches) / cfg["grad_accum"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg["warmup_ratio"] * steps), steps)
    loss_sum, count, gradient_norms = 0.0, 0, []
    model.train()
    for group_start in range(0, len(batches), cfg["grad_accum"]):
        group = batches[group_start:group_start + cfg["grad_accum"]]
        group_examples = sum(len(batch) for batch in group)
        optimizer.zero_grad(set_to_none=True)
        for batch_pairs in group:
            examples, ref_margins, probabilities, weights = [], [], [], []
            for prompt, left, right, weight in batch_pairs:
                examples.extend([encoded[prompt * cfg["keep_k"] + left], encoded[prompt * cfg["keep_k"] + right]])
                reference = outer_state["effective_reference"][prompt]
                ref_margins.append(reference[left] - reference[right])
                probabilities.append(matrices[prompt, left, right])
                weights.append(weight)
            batch, labels = build_batch(examples, pad_id, device)
            output = model(**batch, use_cache=False)
            scores, _ = sequence_scores(output.logits, labels)
            scores = scores.reshape(-1, 2)
            ref = torch.tensor(ref_margins, dtype=torch.float32, device=device)
            probs = torch.tensor(probabilities, dtype=torch.float32, device=device)
            importance = torch.tensor(weights, dtype=torch.float32, device=device)
            delta = scores[:, 0] - scores[:, 1] - ref
            losses = importance * pair_loss(delta, probs, cfg["beta_train"], cfg["method"])
            if not torch.isfinite(losses).all():
                raise FloatingPointError("Non-finite loss; stopping instead of masking it")
            (losses.sum() / group_examples).backward()
            loss_sum += float(losses.detach().sum())
            count += len(batch_pairs)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        gradient_norms.append(float(norm))
        optimizer.step()
        scheduler.step()
        if any(not torch.isfinite(p).all() for p in parameters):
            raise FloatingPointError("Non-finite adapter parameters")
    return dict(train_loss=loss_sum / count, train_pairs=count, optimizer_steps=steps,
                grad_norm_mean=float(np.mean(gradient_norms)), grad_norm_max=max(gradient_norms))


def save_snapshot(directory, step, scores, initial, previous, lengths, outer_state=None):
    metrics = describe_distribution(scores, initial, previous)
    data = dict(sequence_sum_logprob=scores, response_token_count=lengths, **metrics)
    if outer_state is not None:
        data.update({"training_" + k: v for k, v in outer_state.items()})
    path = directory / f"step_{step:04d}.npz"
    with open(str(path) + ".tmp", "wb") as handle:
        np.savez_compressed(handle, **data)
    os.replace(str(path) + ".tmp", path)
    row = dict(step=step, panel_entropy_mean=float(metrics["panel_entropy"].mean()),
               relative_sequence_entropy_mean=float(metrics["relative_sequence_entropy"].mean()),
               tv_mean=float(metrics["tv"].mean()),
               panel_log_mass_mean=float(metrics["panel_log_mass"].mean()))
    if outer_state is not None:
        target_error = centered(scores) - outer_state["target_logits"]
        row.update(operator_residual_rms=float(np.sqrt(np.mean(target_error ** 2))),
                   operator_residual_max=float(np.max(np.abs(target_error))),
                   bt_solver_residual_max=float(np.max(outer_state["solver_residual"])))
    return row


def train(cfg, panels):
    # No imports/model loading reach this branch without explicit --execute.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Training requires an approved Slurm allocation; no login-node training")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This diagnostic requires exactly one visible allocated GPU")
    if cfg["dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires BF16 support")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    root = Path(cfg["output_root"]) / cfg["run_id"]
    root.mkdir(parents=True, exist_ok=False)
    snapshots = root / "snapshots"
    snapshots.mkdir()
    versions = {package: version(package) for package in ("torch", "transformers", "peft", "numpy")}
    manifest = dict(config=cfg, support_sha256=support_hash(panels), versions=versions,
                    created_at_utc=datetime.now(timezone.utc).isoformat(),
                    slurm_job_id=os.environ["SLURM_JOB_ID"], state="INITIALIZING",
                    feedback_operator=("actual_BT_optimizer_extrapolation_not_logit_PsiPO"
                                       if cfg["method"] == "dpo" else "identity_payoff"))
    manifest["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path(__file__).parent.glob("*.py")}
    write_json(root / "manifest.json", manifest)
    write_json(root / "support.json", panels)
    step_completed = -1
    try:
        tok = AutoTokenizer.from_pretrained(cfg["model_path"], local_files_only=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        if tok.pad_token_id is None:
            raise ValueError("Tokenizer has neither pad nor eos token")
        encoded = [row for panel in panels for row in encode_panel(tok, panel, cfg["max_length"])]
        write_json(root / "tokenization_audit.json", dict(
            truncated_prompt_tokens=[x["truncated_prompt_tokens"] for x in encoded],
            response_token_counts=[x["response_tokens"] for x in encoded],
            response_truncation_allowed=False, response_eos_included=tok.eos_token_id is not None,
            prompt_context_shared_within_panel=True))
        dtype = torch.bfloat16 if cfg["dtype"] == "bfloat16" else torch.float32
        base = AutoModelForCausalLM.from_pretrained(cfg["model_path"], torch_dtype=dtype,
                                                   local_files_only=True).to("cuda")
        base.config.use_cache = False
        model = get_peft_model(base, LoraConfig(
            r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        shape = (len(panels), cfg["keep_k"])
        initial, lengths = score_panel(model, encoded, tok.pad_token_id, "cuda", cfg["score_batch_size"], shape)
        current, previous = initial.copy(), initial.copy()
        matrices = np.asarray([p["preference_matrix"] for p in panels])
        rows = [save_snapshot(snapshots, 0, initial, initial, None, lengths)]
        model.save_pretrained(root / "adapter_initial")
        tok.save_pretrained(root / "adapter_initial")
        write_metrics(root, rows)
        step_completed = 0
        manifest["last_complete_step"] = step_completed
        manifest["state"] = "RUNNING"
        write_json(root / "manifest.json", manifest)
        for outer_iter in range(cfg["iters"]):
            started = time.monotonic()
            state = build_outer_state(initial, current, previous, matrices, cfg["method"],
                                      cfg["alpha"], cfg["lambda_current"], cfg["beta_train"],
                                      cfg["nu"], cfg["kappa"])
            diagnostics = train_round(model, encoded, matrices, state, cfg, tok.pad_token_id, "cuda", outer_iter)
            updated, new_lengths = score_panel(model, encoded, tok.pad_token_id, "cuda", cfg["score_batch_size"], shape)
            if not np.array_equal(lengths, new_lengths):
                raise RuntimeError("Token support changed between iterations")
            step = outer_iter + 1
            row = save_snapshot(snapshots, step, updated, initial, current, lengths, state)
            row.update(diagnostics, outer_wall_seconds=time.monotonic() - started)
            rows.append(row)
            previous, current = current.copy(), updated.copy()
            write_metrics(root, rows)
            if step % cfg["checkpoint_every"] == 0 or step == cfg["iters"]:
                checkpoint = root / "adapters" / f"step_{step:04d}"
                model.save_pretrained(checkpoint)
                tok.save_pretrained(checkpoint)
            step_completed = step
            manifest["last_complete_step"] = step_completed
            write_json(root / "manifest.json", manifest)
            print(json.dumps(row, allow_nan=False), flush=True)
        manifest["state"] = "COMPLETED"
        write_json(root / "manifest.json", manifest)
    except BaseException as exc:
        manifest.update(state="FAILED", last_complete_step=step_completed,
                        error=f"{type(exc).__name__}: {exc}")
        write_json(root / "manifest.json", manifest)
        raise


def main(required_scheme=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("experiment_plan.json"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--eval-path", help="Override data location for read-only validation")
    parser.add_argument("--model-path", help="Override the local initial-model directory")
    parser.add_argument("--output-root", help="Override the directory for run artifacts")
    parser.add_argument("--check-data", action="store_true", help="Validate panels without importing Torch")
    parser.add_argument("--execute", action="store_true", help="Explicitly train inside an approved 1-GPU Slurm allocation")
    args = parser.parse_args()
    if args.check_data and args.execute:
        parser.error("--check-data and --execute are mutually exclusive")
    cfg = resolve_config(args.plan, args.run_id)
    if args.eval_path:
        cfg["eval_path"] = args.eval_path
    if args.model_path:
        cfg["model_path"] = args.model_path
    if args.output_root:
        cfg["output_root"] = args.output_root
    validate_config(cfg)
    if required_scheme is not None and cfg["scheme"] != required_scheme:
        parser.error(f"This entry point requires scheme={required_scheme}")
    print(json.dumps(cfg, indent=2), flush=True)
    if not args.check_data and not args.execute:
        print("PREVIEW ONLY: no model loaded, no output directory created, no experiment started.")
        return
    panels = load_panels(cfg["eval_path"], cfg["num_prompts"], cfg["support_seed"], cfg["keep_k"])
    print(json.dumps(dict(panel_count=len(panels), support_sha256=support_hash(panels))), flush=True)
    if args.check_data:
        print("DATA CHECK ONLY: no model loaded and no experiment started.")
        return
    train(cfg, panels)


if __name__ == "__main__":
    main()
