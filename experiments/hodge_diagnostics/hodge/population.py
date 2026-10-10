"""Exact population MRS-PsiPO on a finite panel (JMLR-B-v3, Sections 3-6).

Centered-logit recursion (thm:master-recursion):

    x_{t+1} = alpha x_t + b + g C softmax(x_t),
    b = (1-alpha) r + beta u + beta (1-lambda) C pi0,   g = beta lambda,

with the paper's beta multiplying the payoff (the KL coefficient is 1/beta).
Mapping to training code: DPO's KL coefficient beta_DPO = 1/beta; the IPO loss
(h - 1/(2 tau))^2 gives beta = 1/tau (LLM_Sampling's beta_train = tau).
With uniform reference and coverage, C pi0 = 0 and r = 0, so b = beta u.
"""
from __future__ import annotations

import numpy as np

from .core import Hodge, cyclic_frequencies, softmax_jacobian


def softmax(x):
    z = np.exp(x - np.max(x))
    return z / z.sum()


def centered(x):
    x = np.asarray(x, dtype=float)
    return x - x.mean()


def forcing(h: Hodge, alpha, beta, lam, ref=None, pi0=None):
    """State-independent forcing b and cyclic gain g of the master recursion."""
    K = h.K
    ref = np.full(K, 1 / K) if ref is None else np.asarray(ref, dtype=float)
    pi0 = np.full(K, 1 / K) if pi0 is None else np.asarray(pi0, dtype=float)
    r = centered(np.log(ref))
    b = (1 - alpha) * r + beta * h.u + beta * (1 - lam) * h.C @ pi0
    return centered(b), beta * lam


def _newton(C, b, alpha, gain, x, tol, max_iter):
    K = len(b)

    def residual(y):
        return (1 - alpha) * y - b - gain * C @ softmax(y)

    g = residual(x)
    for _ in range(max_iter):
        if np.max(np.abs(g)) < tol:
            break
        jac = (1 - alpha) * np.eye(K) - gain * C @ softmax_jacobian(softmax(x))
        step = np.linalg.solve(jac, -g)
        t = 1.0
        while t > 1e-10:
            trial = centered(x + t * step)
            g_trial = residual(trial)
            if np.linalg.norm(g_trial) < (1 - 1e-4 * t) * np.linalg.norm(g):
                break
            t *= 0.5
        x, g = trial, g_trial
    return x, float(np.max(np.abs(g)))


def fixed_point(C, b, alpha, gain, tol=1e-12, max_iter=100, steps=200):
    """Unique interior fixed point for alpha < 1 (thm:kl-contraction, part 1).

    Solves (1-alpha) x = b + g C softmax(x) on the centered subspace by Newton;
    the Newton matrix (1-alpha) I - g C J(x) is invertible because C J has an
    imaginary spectrum. Near alpha = 1 the equation is a quantal-response
    condition at temperature 1 - alpha and Newton can stall far from the
    solution, so the fallback continues the unique solution branch from
    alpha = 0, tracking y = (1 - alpha) x on a geometric grid of temperatures.
    """
    if not 0 <= alpha < 1:
        raise ValueError("A unique fixed point requires 0 <= alpha < 1")
    b = centered(b)
    x, res = _newton(C, b, alpha, gain, b / (1 - alpha), tol, max_iter)
    if res < tol:
        return x
    y = b + gain * C @ softmax(b)
    for eps in np.geomspace(1.0, 1 - alpha, steps)[1:]:
        x, res = _newton(C, b, 1 - eps, gain, y / eps, tol, max_iter)
        y = eps * x
    if res < 1e-8:
        return x
    raise RuntimeError(f"Fixed point not found; residual {res:.2e}")


def ordinary_stability(h: Hodge, alpha, beta, lam, ref=None, pi0=None):
    """Fixed point, realized gain gamma = g omega_max(pi*), and index alpha^2 + gamma^2."""
    b, gain = forcing(h, alpha, beta, lam, ref, pi0)
    x = fixed_point(h.C, b, alpha, gain)
    pi = softmax(x)
    omega = float(cyclic_frequencies(h.C, pi)[0])
    gamma = gain * omega
    return dict(x_star=x, pi_star=pi, omega=omega, gamma=gamma,
                index=alpha ** 2 + gamma ** 2,
                entropy=float(-np.sum(np.where(pi > 0, pi * np.log(np.where(pi > 0, pi, 1.0)), 0.0))),
                min_prob=float(pi.min()))


def jacobian(C, x, alpha, gain):
    """Fixed-point Jacobian alpha I + g C J (eq. baseline-jacobian)."""
    return alpha * np.eye(len(x)) + gain * C @ softmax_jacobian(softmax(x))


# Exact local thresholds of the history-aware schemes -------------------------

def gamma2_ordinary(alpha):
    return 1 - alpha ** 2


def Q(alpha, kappa):
    return 1 + 2 * (1 - 2 * alpha) * kappa + (1 - alpha) * (3 - alpha) * kappa ** 2


def gamma2_lagged_sampling(alpha, kappa):
    """Gamma_kappa(alpha) of lem:optimistic-local: stable iff gamma^2 < Gamma."""
    if kappa == 0:
        return 1 - alpha ** 2
    A = kappa ** 2 * (2 * kappa + 1)
    q = Q(alpha, kappa)
    return max((np.sqrt(q ** 2 + 4 * A * (1 - alpha ** 2)) - q) / (2 * A), 0.0)


def kappa_exp(alpha):
    """Largest kappa that enlarges the ordinary region (eq. kappa-expansion); alpha > 1/2."""
    return (alpha * np.sqrt(2 / (1 - alpha)) - 1) / (1 + alpha)


KAPPA_STAR_FULL_REFRESH = (1 + np.sqrt(5)) / 4


def gamma2_lagged_reference(alpha, nu):
    """R_nu(alpha) of thm:lagged-reference-stability: stable iff gamma^2 < R."""
    return (1 - alpha) * (1 + alpha - 2 * nu) * ((1 + nu) / (1 - nu)) ** 2


# Trajectories -----------------------------------------------------------------

def simulate(C, b, gain, x1, T, weights, kappa=0.0, r=None):
    """History-weighted recursion (eq. history-centered).

    x_{t+1} = sum_k w_k x_{t+1-k} + b + g C [(1+kappa) s(x_t) - kappa s(x_{t-1})],
    with sum_k w_k = alpha, an ordinary first step, and pre-initial policies at
    the reference r (centered log reference). Returns the (T+1) x K array x_1..x_{T+1}.
    """
    w = np.asarray(weights, dtype=float)
    K = len(b)
    r = np.zeros(K) if r is None else np.asarray(r, dtype=float)
    history = [r] * (len(w) - 1) + [np.asarray(x1, dtype=float)]
    xs = [history[-1]]
    prev_s = None
    for t in range(T):
        s = softmax(history[-1])
        feedback = s if (prev_s is None or kappa == 0) else (1 + kappa) * s - kappa * prev_s
        memory = sum(w[k] * history[-1 - k] for k in range(len(w)))
        x_next = centered(memory + b + gain * C @ feedback)
        history.append(x_next)
        xs.append(x_next)
        prev_s = s
    return np.array(xs)
