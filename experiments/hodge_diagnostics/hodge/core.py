"""Hodge diagnostics for finite-panel preference matrices.

Notation follows JMLR-B-v3 (Sections 3-4):

    P*   reciprocal preference probabilities, P*_ij + P*_ji = 1, P*_ii = 1/2;
         NaN marks a pair that was never compared
    Psi  symmetric link, identity (IPO) or logit (DPO); c = Psi(1/2)
    A    Psi(P*) - c 11^T, the skew-symmetric flow; unobserved pairs are filled
         at indifference (Assumption 1), so A_ij = 0 there
    u    A 1 / K                       (eq. hodge-components)
    G    u 1^T - 1 u^T                 potential flow
    C    A - G                         cyclic residual, C^T = -C, C 1 = 0

The decomposition is taken on the complete, uniformly weighted graph, which is
the C that enters the master recursion (thm:master-recursion). On an incomplete panel this
C depends on the indifference fill; only curls of fully observed triangles are
identified from the data (Appendix, sparse comparisons).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

LINKS = ("identity", "logit")


def validate_preference(P, atol=1e-8):
    """Return a float copy of P with a 1/2 diagonal after checking reciprocity."""
    P = np.array(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1] or P.shape[0] < 2:
        raise ValueError("P must be a square matrix with K >= 2")
    np.fill_diagonal(P, 0.5)
    observed = ~np.isnan(P)
    if not np.array_equal(observed, observed.T):
        raise ValueError("Observed pairs must be symmetric")
    values = P[observed]
    if np.any((values < -atol) | (values > 1 + atol)):
        raise ValueError("Preference probabilities must lie in [0, 1]")
    if not np.allclose(np.where(observed, P + P.T, 1.0), 1.0, atol=atol):
        raise ValueError("P must be reciprocal: P_ij + P_ji = 1")
    P[observed] = np.clip(P[observed], 0.0, 1.0)
    return P


def observed_mask(P):
    mask = ~np.isnan(np.asarray(P, dtype=float))
    np.fill_diagonal(mask, False)
    return mask


def flow(P, link, clip=None):
    """Skew flow A = Psi(P) - Psi(1/2), zero on unobserved pairs.

    The logit link needs probabilities strictly inside (0, 1). Smooth counts in
    the construction (pseudo-counts) or pass ``clip`` as an explicit, reported
    choice; nothing is clipped silently.
    """
    P = validate_preference(P)
    observed = observed_mask(P)
    A = np.zeros_like(P)
    if link == "identity":
        A[observed] = P[observed] - 0.5
    elif link == "logit":
        q = P[observed]
        if clip is not None:
            q = np.clip(q, clip, 1 - clip)
        if np.any((q <= 0) | (q >= 1)):
            raise ValueError("logit link needs 0 < P < 1; smooth the counts or pass clip")
        A[observed] = np.log(q) - np.log1p(-q)
    else:
        raise ValueError(f"Unknown link {link!r}; expected one of {LINKS}")
    return A


@dataclass(frozen=True)
class Hodge:
    A: np.ndarray
    u: np.ndarray
    G: np.ndarray
    C: np.ndarray

    @property
    def K(self):
        return len(self.u)


def hodge(A, atol=1e-10):
    """Complete-graph Hodge decomposition A = G + C (eq. hodge-components)."""
    A = np.asarray(A, dtype=float)
    if not np.allclose(A, -A.T, atol=atol):
        raise ValueError("A must be skew-symmetric")
    u = A.sum(axis=1) / len(A)
    G = u[:, None] - u[None, :]
    return Hodge(A=A, u=u, G=G, C=A - G)


def triangle_curls(A, mask=None):
    """Cyclic sums A_ij + A_jk + A_ki over triangles i<j<k.

    With ``mask`` (observed pairs), only fully observed triangles are returned:
    these are the cyclic quantities the data identify without a fill convention.
    """
    K = len(A)
    if K < 3:
        return np.zeros(0)
    i, j, k = np.array(list(combinations(range(K), 3))).T
    curls = A[i, j] + A[j, k] + A[k, i]
    if mask is not None:
        curls = curls[mask[i, j] & mask[j, k] & mask[i, k]]
    return curls


def cyclic_constants(C):
    """L_C (half the largest row range, thm:kl-contraction) and the spectral norm ||C||_2."""
    L_C = 0.5 * float(np.max(C.max(axis=1) - C.min(axis=1)))
    return L_C, float(np.linalg.norm(C, 2))


def softmax_jacobian(pi):
    pi = np.asarray(pi, dtype=float)
    return np.diag(pi) - np.outer(pi, pi)


def cyclic_frequencies(C, pi):
    """Nonnegative frequencies omega_l, where i*omega_l are the eigenvalues of C J(pi).

    Computed through the similar skew matrix J^{1/2} C J^{1/2}, whose singular
    values are the |omega_l| in pairs (thm:local-spectrum).
    """
    w, V = np.linalg.eigh(softmax_jacobian(pi))
    root = (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T
    S = root @ C @ root
    return np.linalg.svd(0.5 * (S - S.T), compute_uv=False)


def omega_max(C, pi):
    return float(cyclic_frequencies(C, pi)[0])


def graph_stats(P):
    mask = observed_mask(P)
    K = len(mask)
    degree = mask.sum(axis=1)
    edges = int(mask.sum()) // 2
    return dict(K=K, edges=edges, d=int(degree.max()), mean_degree=float(degree.mean()),
                completeness=edges / (K * (K - 1) / 2))


def ordinal_stats(P, tol=1e-12):
    """Condorcet 3-cycles and stochastic-transitivity violations on observed triples."""
    P = validate_preference(P)
    mask = observed_mask(P)
    K = len(P)
    triples = cycles = sst_viol = wst_viol = sst_checks = 0
    for i, j, k in combinations(range(K), 3):
        if not (mask[i, j] and mask[j, k] and mask[i, k]):
            continue
        triples += 1
        beats = lambda a, b: P[a, b] > 0.5 + tol
        if (beats(i, j) and beats(j, k) and beats(k, i)) or (beats(j, i) and beats(k, j) and beats(i, k)):
            cycles += 1
        for a, b, c in ((i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)):
            if P[a, b] >= 0.5 and P[b, c] >= 0.5 and (P[a, b] > 0.5 + tol or P[b, c] > 0.5 + tol):
                sst_checks += 1
                if P[a, c] < max(P[a, b], P[b, c]) - tol:
                    sst_viol += 1
                if P[a, c] < 0.5 - tol:
                    wst_viol += 1
    return dict(observed_triples=triples, condorcet_triples=cycles,
                has_condorcet_cycle=cycles > 0,
                sst_violation_rate=sst_viol / sst_checks if sst_checks else 0.0,
                wst_violation_rate=wst_viol / sst_checks if sst_checks else 0.0)


def panel_summary(P, link, clip=None):
    """All static Hodge diagnostics of one panel under one link."""
    A = flow(P, link, clip=clip)
    h = hodge(A)
    graph = graph_stats(P)
    K, d = graph["K"], graph["d"]
    energy_A = float(np.sum(A * A))
    energy_C = float(np.sum(h.C * h.C))
    curls = triangle_curls(A)
    observed_curls = triangle_curls(A, observed_mask(P))
    L_C, C2 = cyclic_constants(h.C)
    a = float(np.max(np.abs(A)))
    return dict(
        link=link, **graph,
        a=a,
        norm_u=float(np.linalg.norm(h.u)),
        energy_A=energy_A,
        cyclic_share=energy_C / energy_A if energy_A > 0 else 0.0,
        max_abs_curl=float(np.max(np.abs(curls))) if curls.size else 0.0,
        max_abs_observed_curl=float(np.max(np.abs(observed_curls))) if observed_curls.size else np.nan,
        L_C=L_C, C_norm2=C2,
        omega_uniform=C2 / K,
        fact_L_C_bound=a * (1 + 2 * d / K),
        fact_C2_bound=2 * a * d,
        **ordinal_stats(P),
    )


def fit_bt(W, ridge=1e-3, tol=1e-10, max_iter=200):
    """Bradley--Terry MLE of centered scores from win counts W (ties counted as halves).

    Maximizes sum_ij W_ij log sigma(s_i - s_j) - ridge/2 ||s||^2. The small ridge
    keeps the estimate finite when the win graph is not strongly connected.
    """
    W = np.asarray(W, dtype=float)
    N = W + W.T
    K = len(W)
    s = np.zeros(K)
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-(s[:, None] - s[None, :])))
        grad = (W - N * p).sum(axis=1) - ridge * s
        curvature = N * p * (1 - p)
        H = curvature - np.diag(curvature.sum(axis=1) + ridge)
        step = np.linalg.lstsq(H, -grad, rcond=None)[0]  # minimum norm: s stays centered
        s = s + step
        if np.max(np.abs(step)) < tol:
            return s - s.mean()
    raise RuntimeError("BT fit did not converge")


def bt_null_cyclic_share(W, link, pseudo, draws=200, ridge=1e-3, seed=0):
    """Parametric bootstrap of the cyclic share under a fitted Bradley--Terry truth.

    Simulates the observed number of votes on every observed pair from the BT
    fit and recomputes the share with the same link and smoothing. Under the
    logit link a BT truth has C = 0 exactly, so the null measures sampling noise;
    under the identity link it also contains the link-mismatch residual.
    """
    from .constructions import vote_matrix

    W = np.asarray(W, dtype=float)
    N = np.rint(W + W.T).astype(int)
    s = fit_bt(W, ridge=ridge)
    p = 1 / (1 + np.exp(-(s[:, None] - s[None, :])))
    rng = np.random.default_rng(seed)
    iu, ju = np.triu_indices(len(W), 1)
    keep = N[iu, ju] > 0
    iu, ju = iu[keep], ju[keep]
    shares = np.empty(draws)
    for b in range(draws):
        wins = rng.binomial(N[iu, ju], p[iu, ju])
        Wb = np.zeros_like(W)
        Wb[iu, ju] = wins
        Wb[ju, iu] = N[iu, ju] - wins
        shares[b] = panel_summary(vote_matrix(Wb, pseudo), link)["cyclic_share"]
    return shares
