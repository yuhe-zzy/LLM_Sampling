#!/usr/bin/env python3
"""Reconstruct sequence-ratio entropy; old CSV score conversion is read-only."""
import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


ITER_RE = re.compile(r"iter_(\d+)_prompt_metrics\.csv$")


def entropy(probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-300, 1.0)
    return float(-(probs * np.log(probs)).sum())


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(logits).all():
        raise FloatingPointError("Nonfinite sequence logits")
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    return weights / weights.sum()


def response_token_count(tokenizer, prompt: str, response: str, max_length: int) -> int:
    prompt_ids = tokenizer(str(prompt), add_special_tokens=False).input_ids
    response_ids = tokenizer(str(response), add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None:
        response_ids = response_ids + [tokenizer.eos_token_id]
    total_length = len(prompt_ids) + len(response_ids)
    kept_length = min(total_length, max_length) if max_length > 0 else total_length
    response_length = min(len(response_ids), kept_length)
    prompt_length = kept_length - response_length
    # The causal shift removes position zero. It only affects the response when
    # truncation has removed the entire prompt.
    return max(1, response_length - int(prompt_length == 0))


def find_metrics_file(run_dir: Path) -> Path:
    files = sorted(run_dir.glob("metrics_*.csv"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one metrics CSV in {run_dir}, found {len(files)}")
    return files[0]


def find_dump_dir(run_dir: Path) -> Path:
    dirs = sorted(path for path in run_dir.glob("iter_dumps_*") if path.is_dir())
    if len(dirs) != 1:
        raise RuntimeError(f"Expected one iteration dump directory in {run_dir}, found {len(dirs)}")
    return dirs[0]


def load_dumps(dump_dir: Path):
    dumps = []
    for path in dump_dir.glob("iter_*_prompt_metrics.csv"):
        match = ITER_RE.search(path.name)
        if match:
            dumps.append((int(match.group(1)), path))
    return sorted(dumps)


def sequence_scores(row, k, counts):
    if "sequence_logprob_0" in row.index:
        scores = np.asarray([row[f"sequence_logprob_{j}"] for j in range(k)], dtype=float)
    else:
        # Import-only adapter for archived CSVs; never used by a trainer.
        scores = np.asarray([row[f"avg_logprob_{j}"] for j in range(k)], dtype=float) * counts[:k]
    if not np.isfinite(scores).all():
        raise FloatingPointError("Nonfinite sequence likelihoods in saved data")
    return scores


def build_reference_state(tokenizer, frame: pd.DataFrame, max_length: int):
    frame = frame.sort_values("prompt_index").reset_index(drop=True)
    max_k = int(frame["K"].max())
    counts = np.full((len(frame), max_k), np.nan, dtype=np.float64)
    reference_sum = np.full((len(frame), max_k), np.nan, dtype=np.float64)
    responses = []

    for row_index, row in frame.iterrows():
        k = int(row["K"])
        row_responses = []
        for candidate_index in range(k):
            response = str(row[f"response_{candidate_index}"])
            counts[row_index, candidate_index] = response_token_count(
                tokenizer, row["prompt"], response, max_length
            )
            count_key = f"response_token_count_{candidate_index}"
            if count_key in row.index and int(row[count_key]) != counts[row_index, candidate_index]:
                raise RuntimeError("Saved token counts disagree with tokenizer/truncation settings")
            row_responses.append(response)
        responses.append(row_responses)
        if "reference_sequence_logprob_0" in row.index:
            reference_sum[row_index, :k] = [row[f"reference_sequence_logprob_{j}"] for j in range(k)]
        else:
            reference_sum[row_index, :k] = sequence_scores(row, k, counts[row_index])
    return frame, counts, reference_sum, responses


def validate_support(frame: pd.DataFrame, reference: pd.DataFrame, responses) -> pd.DataFrame:
    frame = frame.sort_values("prompt_index").reset_index(drop=True)
    if len(frame) != len(reference):
        raise RuntimeError("Evaluation prompt count changed across iteration dumps")
    if not np.array_equal(frame["prompt_id"].to_numpy(), reference["prompt_id"].to_numpy()):
        raise RuntimeError("Evaluation prompt ordering changed across iteration dumps")
    for row_index, row in frame.iterrows():
        if row["prompt"] != reference.iloc[row_index]["prompt"]:
            raise RuntimeError("Evaluation prompt text changed across iteration dumps")
        if int(row["K"]) != len(responses[row_index]):
            raise RuntimeError("Evaluation support size changed across iteration dumps")
        for candidate_index, expected in enumerate(responses[row_index]):
            if str(row[f"response_{candidate_index}"]) != expected:
                raise RuntimeError("Evaluation response support changed across iteration dumps")
    return frame


def prompt_probabilities(frame, counts, reference_sum, tau: float):
    probabilities = []
    direct = "relative_sequence_prob_0" in frame.columns
    for row_index, row in frame.iterrows():
        k = int(row["K"])
        if direct:
            probs = np.asarray(
                [float(row[f"relative_sequence_prob_{j}"]) for j in range(k)],
                dtype=np.float64,
            )
            if not np.isfinite(probs).all() or np.any(probs < 0) or probs.sum() <= 0:
                raise FloatingPointError("Invalid saved relative-sequence probabilities")
            probs = probs / probs.sum()
        else:
            current_sum = sequence_scores(row, k, counts[row_index])
            relative_logits = (current_sum - reference_sum[row_index, :k]) * float(tau)
            probs = softmax(relative_logits)
        probabilities.append(probs)
    if direct:
        mode = "saved_relative_sequence_probabilities"
    elif "sequence_logprob_0" in frame.columns:
        mode = "reconstructed_from_sequence_scores"
    else:
        mode = "imported_archived_scores_times_token_count"
    return probabilities, mode


def reconstruct_run(run_dir: Path, tokenizer, max_length: int) -> Path:
    metrics_path = find_metrics_file(run_dir)
    metrics = pd.read_csv(metrics_path)
    dump_dir = find_dump_dir(run_dir)
    dumps = load_dumps(dump_dir)
    if not dumps or dumps[0][0] != 0:
        raise RuntimeError(f"Iteration zero dump is required in {dump_dir}")

    reference, counts, reference_sum, responses = build_reference_state(
        tokenizer, pd.read_csv(dumps[0][1]), max_length
    )
    initial_top1 = None
    previous_probs = None
    rows = []

    for iteration, path in dumps:
        frame = validate_support(pd.read_csv(path), reference, responses)
        probs_by_prompt, reconstruction_mode = prompt_probabilities(
            frame, counts, reference_sum, tau=float(metrics["tau"].iloc[0])
        )
        entropies = np.asarray([entropy(probs) for probs in probs_by_prompt])
        top1 = np.asarray([int(np.argmax(probs)) for probs in probs_by_prompt])
        if initial_top1 is None:
            initial_top1 = top1.copy()

        if previous_probs is None:
            tvs = np.full(len(probs_by_prompt), np.nan)
        else:
            tvs = np.asarray(
                [0.5 * np.abs(cur - prev).sum() for cur, prev in zip(probs_by_prompt, previous_probs)]
            )

        metric_row = metrics.loc[metrics["iter"] == iteration]
        oracle_wr = float(metric_row["oracle_win_rate"].iloc[0]) if len(metric_row) else math.nan
        rows.append(
            {
                "iter": iteration,
                "loss_type": metrics["loss_type"].iloc[0],
                "alpha": float(metrics["alpha"].iloc[0]),
                "lambda": float(metrics["lambda"].iloc[0]),
                "beta": float(metrics["beta"].iloc[0]),
                **({"seed": int(metrics["seed"].iloc[0])} if "seed" in metrics else {}),
                "primary_entropy_metric": "prompt_relative_sequence_entropy_mean",
                "relative_sequence_score_definition": "log_pi_t_minus_log_pi_0",
                "reconstruction_mode": reconstruction_mode,
                "prompt_relative_sequence_entropy_mean": float(np.mean(entropies)),
                "prompt_relative_sequence_entropy_min": float(np.min(entropies)),
                "prompt_relative_sequence_entropy_max": float(np.max(entropies)),
                "prompt_relative_sequence_tv_mean": (
                    float(np.nanmean(tvs)) if np.any(np.isfinite(tvs)) else math.nan
                ),
                "prompt_relative_sequence_tv_max": (
                    float(np.nanmax(tvs)) if np.any(np.isfinite(tvs)) else math.nan
                ),
                "relative_sequence_top1_flip_rate_vs_initial": float(np.mean(top1 != initial_top1)),
                "oracle_win_rate": oracle_wr,
                "eval_response_token_count_mean": float(np.nanmean(counts)),
                "eval_response_token_count_median": float(np.nanmedian(counts)),
                "num_prompts_eval": len(frame),
                "max_eval_support_k": int(frame["K"].max()),
            }
        )
        previous_probs = probabilities_copy(probs_by_prompt)

    output_path = run_dir / "relative_sequence_metrics.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def probabilities_copy(probabilities):
    return [probs.copy() for probs in probabilities]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--run_glob", default="*sequencesum_v2")
    parser.add_argument("--max_length", type=int, default=1537)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    run_dirs = sorted(path for path in Path(args.logs_root).glob(args.run_glob) if path.is_dir())
    if not run_dirs:
        raise RuntimeError(f"No runs matched {args.run_glob!r} under {args.logs_root}")

    failures = []
    for run_dir in run_dirs:
        try:
            output_path = reconstruct_run(run_dir, tokenizer, args.max_length)
            frame = pd.read_csv(output_path)
            print(
                f"{run_dir.name}: through iter {int(frame['iter'].max())}, "
                f"H_rel_seq={frame['prompt_relative_sequence_entropy_mean'].iloc[-1]:.6f}"
            )
        except Exception as exc:
            failures.append((run_dir, exc))
            print(f"ERROR {run_dir}: {exc}")

    if failures:
        raise RuntimeError(f"Failed to reconstruct {len(failures)} run(s)")


if __name__ == "__main__":
    main()
