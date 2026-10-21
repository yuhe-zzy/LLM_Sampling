"""CPU-only local analysis of the actual IPO and pairwise DPO population maps.

General DPO roots are checked, not assumed unique. No logit(P) substitution,
model loading, scheduler calls, or changes to the sampling convention occur.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import root

from history_math import (bt_scores, centered, finite, population_delta, sampler,
                          sigmoid, softmax, validate_preferences)


def basis(k):
    return np.linalg.qr(np.eye(k)[:, :-1] - np.eye(k)[:, -1, None])[0]


def balanced_cycle(probability=.8, orientation=1, roles=None):
    if not .5 < probability < 1 or orientation not in (-1, 1):
        raise ValueError("Require a soft cycle with probability in (.5,1)")
    roles = np.arange(4) if roles is None else np.asarray(roles)
    if sorted(roles.tolist()) != list(range(4)):
        raise ValueError("Roles must be a permutation of four response indices")
    p = np.full((4, 4), .5)
    for a, b in zip(roles, np.roll(roles, -1)):
        p[a, b] = .5 + orientation * (probability - .5)
        p[b, a] = 1 - p[a, b]
    return p


def feedback_jacobian(method, p, mu, beta_train):
    """Derivative of optimal centered log-ratio feedback with respect to mu."""
    p = validate_preferences(p)
    q = basis(len(p))
    if method == "ipo":
        return q @ q.T @ (p - .5) / beta_train
    if method != "dpo":
        raise ValueError("Unknown objective")
    v, _ = bt_scores(p, mu, tolerance=1e-12)
    probability = sigmoid(v[:, None] - v[None, :])
    error = probability - p
    weights = mu[:, None] * mu[None, :] * probability * (1 - probability)
    hessian = np.diag(weights.sum(1)) - weights
    return -q @ np.linalg.solve(q.T @ hessian @ q,
                                q.T @ (mu[:, None] * error)) / beta_train


def local_matrices(method, p, reference, x, alpha, lam, beta_train, nu=0., kappa=0.):
    q = basis(len(p))
    probability = softmax(x)
    jac = np.diag(probability) - np.outer(probability, probability)
    b = q.T @ feedback_jacobian(method, p, sampler(x, lam), beta_train) @ (lam * jac) @ q
    identity, zero = np.eye(len(p) - 1), np.zeros((len(p) - 1, len(p) - 1))
    return dict(ordinary=alpha * identity + b,
                lagged_reference=np.block([[(alpha - nu) * identity + b, nu * identity],
                                           [identity, zero]]),
                oracle_feedback_extrapolation=np.block(
                    [[alpha * identity + (1 + kappa) * b, -kappa * b], [identity, zero]]))


def spectral_radius(matrix):
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def fixed_point(method, p, reference, alpha, lam, beta_train):
    """Find and independently verify a centered fixed point.

    DPO is solved via its BT stationarity equations at
    v=beta_train*(1-alpha)*(x-reference), avoiding nested Newton solves while
    searching. Verification still calls the same BT optimizer as training.
    A failed solve raises, rather than declaring the panel stable.
    """
    p = validate_preferences(p)
    reference = centered(reference)
    if not 0 <= alpha < 1 or not 0 <= lam < 1 or beta_train <= 0:
        raise ValueError("Invalid population coefficients")
    q = basis(len(p))

    def equations(y, gain):
        x = q @ y
        probability = softmax(x)
        mu = sampler(x, lam)
        jmu = lam * (np.diag(probability) - np.outer(probability, probability))
        if method == "ipo":
            delta = centered((p - .5) @ mu / beta_train)
            value = (1 - alpha) * (x - reference) - gain * delta
            jac = (1 - alpha) * np.eye(len(p)) - gain * (p - .5) @ jmu / beta_train
        elif method == "dpo":
            # Continuation starts at a zero cyclic/quality signal, not infinite beta.
            target = .5 + gain * (p - .5)
            v = beta_train * (1 - alpha) * (x - reference)
            bt = sigmoid(v[:, None] - v[None, :])
            error = bt - target
            first = error @ mu
            value = mu * first
            w = mu[:, None] * mu[None, :] * bt * (1 - bt)
            h = np.diag(w.sum(1)) - w
            jac = (np.diag(first) + mu[:, None] * error) @ jmu + beta_train * (1 - alpha) * h
        else:
            raise ValueError("Unknown objective")
        return q.T @ value, q.T @ jac @ q

    def attempt(y, gain):
        answer = root(lambda z: equations(z, gain)[0], y,
                      jac=lambda z: equations(z, gain)[1], tol=1e-10)
        return answer.x, float(np.max(np.abs(equations(answer.x, gain)[0])))

    y, residual = attempt(q.T @ reference, 1.)
    if residual > 1e-10:
        y, gain, increment = q.T @ reference, 0., .025
        while gain < 1.:
            next_gain = min(1., gain + increment)
            candidate, residual = attempt(y, next_gain)
            if residual <= 1e-10:
                y, gain = candidate, next_gain
                increment = min(.05, 1.5 * increment)
            else:
                increment *= .5
                if increment < 1e-6:
                    raise RuntimeError(f"Unresolved {method} fixed point at gain {gain:g}")
    x = centered(q @ y)
    delta, _ = population_delta(method, p, sampler(x, lam), beta_train)
    actual_residual = float(np.max(np.abs((1 - alpha) * (x - reference) - delta)))
    if actual_residual > 1e-7:
        raise RuntimeError(f"Actual outer-map residual {actual_residual:g} exceeds tolerance")
    return x, actual_residual


def analyze(method, p, reference, alpha, lam, beta_train, nu=.45, kappa=.5):
    x, residual = fixed_point(method, p, reference, alpha, lam, beta_train)
    matrices = local_matrices(method, p, reference, x, alpha, lam, beta_train, nu, kappa)
    return dict(fixed_logits=x.tolist(), fixed_probability=softmax(x).tolist(),
                fixed_point_residual=residual,
                radii={name: spectral_radius(m) for name, m in matrices.items()},
                ordinary_eigenvalues=[[float(v.real), float(v.imag)]
                                     for v in np.linalg.eigvals(matrices["ordinary"])])


def exact_trajectory(method, p, reference, initial, alpha, lam, beta_train,
                     steps=200, nu=0., kappa=0.):
    reference, current = centered(reference), centered(initial)
    previous = current.copy()
    output = [current.copy()]
    for _ in range(steps):
        delta, _ = population_delta(method, p, sampler(current, lam), beta_train)
        old_delta, _ = population_delta(method, p, sampler(previous, lam), beta_train)
        updated = centered((1 - alpha) * reference + (alpha - nu) * current + nu * previous
                           + (1 + kappa) * delta - kappa * old_delta)
        previous, current = current, finite(updated, "exact population trajectory")
        output.append(current.copy())
    return np.array(output)
