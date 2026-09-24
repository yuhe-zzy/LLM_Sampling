"""Finite-panel history updates. No model imports, GPU allocation, or Slurm calls."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np


def finite(values, name):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError(f"Non-finite {name}; refusing uniform fallback")
    return values


def centered(values):
    values = finite(values, "log scores")
    return values - values.mean(axis=-1, keepdims=True)


def softmax(values):
    values = finite(values, "softmax input")
    weights = np.exp(values - values.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def logsumexp(values):
    values = finite(values, "logsumexp input")
    maximum = values.max(axis=-1)
    return maximum + np.log(np.exp(values - maximum[..., None]).sum(axis=-1))


def validate_preferences(matrix):
    p = finite(matrix, "preference matrix").copy()
    if p.ndim != 2 or p.shape[0] != p.shape[1] or len(p) < 3:
        raise ValueError("Expected a square preference matrix with K >= 3")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Preference probabilities must be in [0,1]")
    # The legacy builder stores zeros on the unused diagonal, not 0.5.
    if not np.all(np.isclose(np.diag(p), 0) | np.isclose(np.diag(p), .5)):
        raise ValueError("Invalid diagonal")
    np.fill_diagonal(p, .5)
    if not np.allclose(p + p.T, 1, atol=1e-10, rtol=0):
        raise ValueError("Incomplete/nonreciprocal P: missing edges cannot be filled as ties")
    return p


def strongly_connected(p):
    reach = np.asarray(p > 0, dtype=bool).copy()
    for k in range(len(p)):
        reach |= reach[:, k, None] & reach[None, k, :]
    return bool(reach.all())


def load_panels(path, count=500, seed=123, keep_k=4):
    candidates = []
    with open(path, encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            responses = [r if isinstance(r, str) else r["text"] for r in row["responses"]]
            if len(responses) != keep_k or len(set(responses)) != keep_k:
                raise ValueError(f"Row {row_index}: need exactly {keep_k} distinct responses")
            if not isinstance(row["prompt"], str) or not row["prompt"].strip():
                raise ValueError(f"Row {row_index}: invalid prompt")
            if any(not isinstance(r, str) or not r.strip() for r in responses):
                raise ValueError(f"Row {row_index}: invalid response")
            p = validate_preferences(row["preference_matrix"])
            if p.shape != (keep_k, keep_k):
                raise ValueError("Response/matrix shape mismatch")
            if not strongly_connected(p):
                raise ValueError("Cyclic runner requires a strongly connected win graph")
            candidates.append(dict(prompt_id=int(row.get("prompt_id", row_index)),
                                   prompt=row["prompt"], responses=responses,
                                   preference_matrix=p.tolist()))
    if count < 1 or count > len(candidates):
        raise ValueError(f"Requested {count} panels, available {len(candidates)}")
    if len({p["prompt_id"] for p in candidates}) != len(candidates):
        raise ValueError("Duplicate prompt IDs")
    if len({p["prompt"] for p in candidates}) != len(candidates):
        raise ValueError("Duplicate prompt texts")
    indices = sorted(random.Random(seed).sample(range(len(candidates)), count))
    return [candidates[i] for i in indices]


def support_hash(panels):
    raw = json.dumps(panels, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sampler(log_scores, lambda_current):
    if not 0 <= lambda_current < 1:
        raise ValueError("Require 0 <= lambda_current < 1 for full-support uniform coverage")
    q = softmax(log_scores)
    return (1 - lambda_current) / q.shape[-1] + lambda_current * q


def pair_distribution(mu):
    mu = finite(mu, "sampler")
    if mu.ndim != 1 or np.any(mu <= 0) or not np.isclose(mu.sum(), 1):
        raise ValueError("Sampler must have strictly positive normalized mass")
    i, j = np.triu_indices(len(mu), 1)
    weights = mu[i] * mu[j]
    weights /= weights.sum()
    return i, j, weights


def sigmoid(x):
    exp_abs = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + exp_abs), exp_abs / (1 + exp_abs))


def bt_objective_gradient_hessian(v, p, mu):
    i, j, w = pair_distribution(mu)
    margins = v[i] - v[j]
    probability = sigmoid(margins)
    objective = float(np.sum(w * (np.logaddexp(0, margins) - p[i, j] * margins)))
    grad = np.zeros_like(v)
    np.add.at(grad, i, w * (probability - p[i, j]))
    np.add.at(grad, j, -w * (probability - p[i, j]))
    hessian = np.zeros((len(v), len(v)))
    curvature = w * probability * (1 - probability)
    np.add.at(hessian, (i, i), curvature)
    np.add.at(hessian, (j, j), curvature)
    np.add.at(hessian, (i, j), -curvature)
    np.add.at(hessian, (j, i), -curvature)
    return objective, grad, hessian


def bt_scores(p, mu, tolerance=1e-10, max_steps=100):
    """Actual population DPO/BT projection; never elementwise logit(P)."""
    p = validate_preferences(p)
    if not strongly_connected(p):
        raise ValueError("BT optimum may be infinite: win graph not strongly connected")
    v = np.zeros(len(p), dtype=np.float64)
    for iteration in range(max_steps):
        loss, grad, hessian = bt_objective_gradient_hessian(v, p, mu)
        residual = float(np.max(np.abs(grad)))
        if residual < tolerance:
            return centered(v), dict(residual=residual, iterations=iteration, objective=loss)
        direction = np.zeros_like(v)
        try:
            direction[:-1] = np.linalg.solve(hessian[:-1, :-1], -grad[:-1])
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Singular BT solve; no silent regularization") from exc
        slope = float(grad @ direction)
        if not np.isfinite(direction).all() or slope >= 0:
            raise RuntimeError("Invalid Newton direction in BT solve")
        rate = 1.0
        for _ in range(40):
            candidate = v + rate * direction
            new_loss, new_grad, _ = bt_objective_gradient_hessian(candidate, p, mu)
            roundoff = 16 * np.finfo(float).eps * max(1., abs(loss))
            # At machine precision the loss can round upward even as the
            # stationarity residual vanishes. Keep the original solve tolerance.
            residual_improves_at_roundoff = (
                new_loss <= loss + roundoff and np.max(np.abs(new_grad)) < .5 * residual)
            if new_loss <= loss + 1e-4 * rate * slope or residual_improves_at_roundoff:
                v = candidate
                break
            rate *= .5
        else:
            raise RuntimeError("BT line search failed")
    raise RuntimeError(f"BT solve did not converge within {max_steps} iterations")


def population_delta(method, p, mu, beta_train):
    if not np.isfinite(beta_train) or beta_train <= 0:
        raise ValueError("beta_train must be finite and positive")
    p = validate_preferences(p)
    pair_distribution(mu)
    if method == "ipo":
        # E[(d_i-d_j - (2Y-1)/(2 beta_train))^2], pair law mu x mu.
        return centered((p - .5) @ mu / beta_train), dict(residual=0.0, iterations=0)
    if method == "dpo":
        scores, info = bt_scores(p, mu)
        return scores / beta_train, info
    raise ValueError(f"Unknown objective {method}")


def build_outer_state(initial, current, previous, matrices, method, alpha,
                      lambda_current, beta_train, nu=0.0, kappa=0.0):
    if not 0 <= alpha < 1 or not 0 <= nu <= alpha or not np.isfinite(kappa) or kappa < 0:
        raise ValueError("Invalid history coefficients")
    if nu and kappa:
        raise ValueError("Do not combine the two interventions in this ablation")
    initial, current, previous = [finite(x, "cached scores") for x in (initial, current, previous)]
    if initial.shape != current.shape or previous.shape != current.shape:
        raise ValueError("History shapes differ")
    if np.asarray(matrices).shape != current.shape + (current.shape[-1],):
        raise ValueError("Preference/history shapes differ")
    mu = sampler(current, lambda_current)
    old_mu = sampler(previous, lambda_current)
    reference = (1 - alpha) * initial + (alpha - nu) * current + nu * previous
    delta, old_delta, residuals = [], [], []
    for p, m, old_m in zip(matrices, mu, old_mu):
        d, info = population_delta(method, p, m, beta_train)
        old_d, old_info = population_delta(method, p, old_m, beta_train) if kappa else (d, info)
        delta.append(d)
        old_delta.append(old_d)
        residuals.append(max(info["residual"], old_info["residual"]))
    delta, old_delta = np.asarray(delta), np.asarray(old_delta)
    # Translation equivariance of positive IPO/DPO losses implements score
    # extrapolation without negative loss weights or signed sampling probabilities.
    offset = kappa * (delta - old_delta)
    effective_reference = reference + offset
    target = centered(effective_reference + delta)
    return dict(mu=mu, reference=reference, effective_reference=effective_reference,
                feedback_delta=delta, previous_feedback_delta=old_delta, offset=offset,
                target_logits=target, solver_residual=np.asarray(residuals))


def sample_pairs(mu, pairs_per_prompt, seed, outer_iter):
    """Uniform pair proposal, exact positive importance weights, common RNG across arms."""
    if pairs_per_prompt < 1:
        raise ValueError("pairs_per_prompt must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([seed, outer_iter, 271828]))
    rows = []
    for prompt_index, masses in enumerate(mu):
        i, j, target = pair_distribution(masses)
        for pair in rng.integers(len(i), size=pairs_per_prompt):
            rows.append((prompt_index, int(i[pair]), int(j[pair]), float(len(i) * target[pair])))
    rng.shuffle(rows)
    return rows


def describe_distribution(scores, initial_scores, previous_scores=None):
    scores = finite(scores, "snapshot")
    q = softmax(scores)
    entropy = -np.sum(q * np.log(np.maximum(q, np.finfo(float).tiny)), axis=-1)
    relative = softmax(scores - initial_scores)
    relative_entropy = -np.sum(relative * np.log(np.maximum(relative, np.finfo(float).tiny)), axis=-1)
    tv = np.zeros(len(q)) if previous_scores is None else .5 * np.abs(q - softmax(previous_scores)).sum(-1)
    return dict(panel_probability=q, panel_entropy=entropy,
                relative_sequence_entropy=relative_entropy, tv=tv,
                panel_log_mass=logsumexp(scores), centered_logits=centered(scores))
