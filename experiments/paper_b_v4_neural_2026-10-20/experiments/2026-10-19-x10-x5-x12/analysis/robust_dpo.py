"""Numerically robust population DPO round (2026-10-19, post hoc check for X12).
dpo_map.bt_fit solves the mu-weighted Bradley-Terry equations with a Hessian whose entries are mu_i mu_j; near a vertex these are ~1e-20
and the least-squares solve loses the low-mass coordinates. Here the same equations, divided by mu_i,
g_i(phi) = sum_j mu_j (sigma(phi_i - phi_j) - P_ij) = 0, are solved by Newton on this row-scaled system, which stays well conditioned
(each row is dominated by the high-mass answers). The solution is the same phi (T1), so the round is unchanged mathematically."""
import numpy as np
sig = lambda z: 1 / (1 + np.exp(-z))
sm = lambda x: np.exp(x - x.max()) / np.exp(x - x.max()).sum()
Pi = lambda v: v - v.mean()


def bt_fit_robust(P, mu, iters=200):
    K = len(mu); phi = np.zeros(K); mu = np.maximum(mu, 1e-300)
    for _ in range(iters):
        D = phi[:, None] - phi[None, :]; S = sig(D); np.fill_diagonal(S, .5)
        g = (mu[None, :] * (S - P)).sum(1) - mu * (0.5 - np.diag(P))           # diagonal terms cancel (P_ii = S_ii = 1/2)
        Jm = -(mu[None, :] * S * (1 - S)); np.fill_diagonal(Jm, 0); np.fill_diagonal(Jm, -Jm.sum(1))
        # sum_i mu_i g_i = 0 identically, so the equation of the highest-mass answer is implied by the others (and is the
        # ill-conditioned one near a vertex): drop it and fix the gauge sum(phi) = 0 instead
        keep = np.arange(K) != int(np.argmax(mu))
        A = np.vstack([Jm[keep], np.ones(K)]); b = -np.concatenate([g[keep], [phi.sum()]])
        step = np.linalg.solve(A, b)
        lam = 1.0
        while lam > 1e-4 and np.abs(step * lam).max() > 20: lam /= 2
        phi = phi + lam * step
        if np.abs(g[keep]).max() < 1e-13 and np.abs(step).max() < 1e-12: break
    return Pi(phi)


def dpo_round_robust(x, r, P, alpha, beta, lam, pi0):
    mu = (1 - lam) * pi0 + lam * sm(x)
    return Pi(alpha * x + (1 - alpha) * r + beta * bt_fit_robust(P, mu))
