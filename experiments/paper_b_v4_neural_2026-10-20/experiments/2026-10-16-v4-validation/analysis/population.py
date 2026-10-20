"""Exact population predictions for the panel experiments (JMLR-B-v4 validation).

Given P*[N,K,K], the initial panel scores s_ref[N,K] (reference = initial policy on the panel), coverage pi0
(uniform), and (alpha, beta, lam, link), this module runs the exact MRS-PsiPO recursion on every panel and
computes the theory quantities used in the predictions: the target (fixed point) at sharpness B = beta/(1-alpha),
the realized cyclic frequency omega_max at the target, gamma_B = B*lam*omega_max, alpha_loc, Condorcet winners and
maximal lotteries, and the dominance threshold of cor:dominance.
"""
import numpy as np
from scipy.optimize import linprog

sm = lambda x: np.exp(x - x.max(-1, keepdims=True)) / np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True)


def link(P, kind):
    if kind == 'identity':
        return P - 0.5
    Pc = np.clip(P, 1e-6, 1 - 1e-6)
    return np.log(Pc / (1 - Pc))


def hodge(A):
    K = A.shape[-1]
    u = A.sum(-1) / K
    C = A - (u[..., :, None] - u[..., None, :])
    return u, C


def trajectory(A, s_ref, alpha, beta, lam, T, pi0=None, nu=0.0):
    """Exact recursion on raw (uncentered) panel scores: s_{t+1} = (alpha-nu) s_t + nu s_{t-1} + (1-alpha) s_ref + beta*A mu_t."""
    N, K = s_ref.shape
    pi0 = np.full((N, K), 1.0 / K) if pi0 is None else pi0
    s, s_prev = s_ref.copy(), s_ref.copy()
    out = [s.copy()]
    for _ in range(T):
        mu = (1 - lam) * pi0 + lam * sm(s)
        g = np.einsum('nij,nj->ni', A, mu)
        s_new = (alpha - nu) * s + nu * s_prev + (1 - alpha) * s_ref + beta * g
        s_prev, s = s, s_new
        out.append(s.copy())
    return np.stack(out)                      # [T+1, N, K]


def target(A, s_ref, B, lam, pi0=None, iters=4000):
    """Fixed point x* = r + B(d + lam C pi*) per panel, by Newton on 1-perp (falls back to damped iteration)."""
    N, K = s_ref.shape
    pi0 = np.full((N, K), 1.0 / K) if pi0 is None else pi0
    Q = np.linalg.svd(np.eye(K) - np.ones((K, K)) / K)[0][:, :K - 1]
    X = np.zeros((N, K)); res = np.zeros(N)
    for n in range(N):
        r = s_ref[n] - s_ref[n].mean()
        u, C = hodge(A[n])
        d = u + (1 - lam) * C @ pi0[n]
        F = lambda z: Q.T @ (Q @ z - r - B * (d + lam * C @ sm(Q @ z)))
        z = Q.T @ r
        for _ in range(200):
            x = Q @ z; p = sm(x); J = np.diag(p) - np.outer(p, p); f = F(z)
            if np.linalg.norm(f) < 1e-10:
                break
            dz = np.linalg.solve(np.eye(K - 1) - B * lam * Q.T @ C @ J @ Q, -f); step = 1.0
            while step > 1e-8 and np.linalg.norm(F(z + step * dz)) >= np.linalg.norm(f):
                step /= 2
            z = z + step * dz
        X[n] = Q @ z; res[n] = np.linalg.norm(F(z))
    return X, res


def omega_max(C, p):
    J = np.diag(p) - np.outer(p, p)
    return np.abs(np.linalg.eigvals(C @ J).imag).max()


def condorcet_winner(P):
    K = P.shape[0]
    for w in range(K):
        if all(P[w, j] > 0.5 for j in range(K) if j != w):
            return w
    return -1


def maximal_lottery(A):
    K = A.shape[0]
    res = linprog(np.zeros(K), A_ub=A, b_ub=np.zeros(K), A_eq=np.ones((1, K)), b_eq=[1], bounds=[(0, None)] * K)
    return res.x


def margins(A, lam, pi0=None):
    """Worst- and best-case margins Gamma[w,j], Gamma_bar[w,j] over reachable samplers (prop:reachable-margin)."""
    K = A.shape[0]
    pi0 = np.full(K, 1.0 / K) if pi0 is None else pi0
    G = np.zeros((K, K)); Gb = np.zeros((K, K))
    for w in range(K):
        for j in range(K):
            row = A[w] - A[j]
            G[w, j] = (1 - lam) * row @ pi0 + lam * row.min()
            Gb[w, j] = (1 - lam) * row @ pi0 + lam * row.max()
    return G, Gb


def entropy(s):
    p = sm(s)
    return -(p * np.log(p + 1e-300)).sum(-1)
