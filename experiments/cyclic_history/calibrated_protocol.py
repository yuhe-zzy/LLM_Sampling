"""Frozen support and fail-closed CPU prediction gates for the new micropilot."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

from history_math import centered, support_hash
from population_calibration import analyze, balanced_cycle, basis, local_matrices

PROTOCOL = "cyclic_history_calibrated_sequence_v2"
EMPIRICAL_PROTOCOL = "cyclic_history_empirical_full_refresh_v1"
TREND_PROTOCOL = "cyclic_history_empirical_partial_refresh_trend_v1"
UNSTABLE_MIN = 1.03
STABLE_MAX = .98


def transform_panels(panels, cfg):
    calibration = cfg["calibration"]
    if support_hash(panels) != calibration["source_panel_sha256"]:
        raise ValueError("Calibrated source responses/prompts/preferences changed")
    result = copy.deepcopy(panels)
    roles = calibration["response_roles"]
    orientations = cfg["orientations"]
    if len(roles) != len(result) or len(orientations) != len(result):
        raise ValueError("Role/orientation count mismatch")
    for row, role, orientation in zip(result, roles, orientations):
        row["preference_matrix"] = balanced_cycle(cfg["cycle_probability"], orientation, role).tolist()
    if support_hash(result) != cfg["transformed_support_sha256"]:
        raise ValueError("Transformed support does not match frozen calibration")
    return result


def contract_hash(cfg):
    # Output paths and the explicit approval flag do not affect the mathematics.
    keys = ("method", "scheme", "alpha", "lambda_current", "beta_train", "nu", "kappa",
            "cycle_probability", "orientations", "panel_ids", "transformed_support_sha256",
            "max_length", "support_probability", "pair_law", "pair_mode", "coverage",
            "seed", "support_seed", "iters", "epochs_per_iter", "batch_size", "grad_accum",
            "lr", "warmup_ratio", "lora_r", "lora_alpha", "lora_dropout", "dtype",
            "optimizer_reset", "num_prompts", "keep_k", "pairs_per_prompt", "prediction_role")
    raw = json.dumps({k: cfg[k] for k in keys}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def verify_support_calibration(panels, cfg, fresh_scores=None, fresh_lengths=None):
    """Check the frozen data and initialization, independently of stability theory."""
    if cfg["calibration_contract_sha256"] != contract_hash(cfg):
        raise ValueError("Parameters changed after calibration; regenerate the plan")
    c = cfg["calibration"]
    scores = np.asarray(c["initial_sequence_scores"], dtype=float)
    if scores.shape != (len(panels), 4) or not np.isfinite(scores).all():
        raise ValueError("Invalid calibrated sequence scores")
    if fresh_scores is not None:
        if not np.allclose(centered(fresh_scores), centered(scores), rtol=0,
                           atol=c["initial_centered_score_atol"]):
            raise ValueError("Fresh Qwen initial scores differ from calibration; do not train")
        if not np.array_equal(fresh_lengths, np.asarray(c["response_token_counts"])):
            raise ValueError("Tokenization/EOS lengths changed after calibration")
        scores = fresh_scores
    if support_hash(panels) != cfg["transformed_support_sha256"]:
        raise ValueError("Wrong calibrated training support")
    return scores


def verify_calibration(panels, cfg, fresh_scores=None, fresh_lengths=None):
    scores = verify_support_calibration(panels, cfg, fresh_scores, fresh_lengths)
    predictions = []
    for panel, reference in zip(panels, scores):
        result = analyze(cfg["method"], panel["preference_matrix"], reference, cfg["alpha"],
                         cfg["lambda_current"], cfg["beta_train"], cfg["nu"], cfg["kappa"])
        radii = result["radii"]
        role = cfg["prediction_role"]
        if role == "ordinary_stable":
            passed = radii["ordinary"] <= STABLE_MAX
        else:
            passed = radii["ordinary"] >= UNSTABLE_MIN
            if cfg["scheme"] != "ordinary":
                passed = passed and radii[cfg["scheme"]] <= STABLE_MAX
        if not passed:
            raise ValueError(f"Prediction gate failed for prompt {panel['prompt_id']}: {radii}")
        predictions.append(result)
    return predictions


def calibrated_metrics(scores, previous_scores, fixed_logits):
    error = centered(scores) - np.asarray(fixed_logits)
    step = centered(scores) - centered(previous_scores)
    return dict(fixed_point_residual_rms=float(np.sqrt(np.mean(error ** 2))),
                centered_step_rms=float(np.sqrt(np.mean(step ** 2))))


def mode_coordinates(panels, cfg, predictions):
    modes = []
    q = basis(4)
    for panel, reference, prediction in zip(panels, cfg["calibration"]["initial_sequence_scores"], predictions):
        matrix = local_matrices(cfg["method"], panel["preference_matrix"], reference,
                                prediction["fixed_logits"], cfg["alpha"], cfg["lambda_current"],
                                cfg["beta_train"])["ordinary"]
        values, vectors = np.linalg.eig(matrix)
        selected = int(np.argmax(values.imag))
        if values[selected].imag <= 1e-8:
            raise ValueError("No resolved cyclic mode for this calibrated panel")
        left = np.linalg.inv(vectors)[selected] @ q.T
        modes.append(left / np.linalg.norm(left))
    return dict(fixed_logits=np.asarray([p["fixed_logits"] for p in predictions]),
                left_modes=np.asarray(modes))
