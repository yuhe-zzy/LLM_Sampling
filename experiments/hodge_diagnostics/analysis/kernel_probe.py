"""Exploratory checks for the shared-parameter linearizations (candidate results R1, R2).

Setting: stacked centered logits over N prompts, J = blockdiag(J_x), C = blockdiag(C_x),
prompt weights W, shared features Phi (lazy regime), NTK Theta = Phi Phi^T.

(1) R1 matrix: alpha I + g P_Phi C J on range(Phi), with P_Phi the J-weighted projection.
    Its eigenvalues are alpha + i g omega~ with omega~ <= max_x omega_x (Cauchy interlacing
    for the compression of the skew J^{1/2} C J^{1/2}). CORRECTION (2026-09-25): this is the
    full local Jacobian of exact restricted optimization only when the restricted class
    represents the tabular fixed point (realizable case). Otherwise a curvature term
    proportional to the projection residual enters, and rates below or above alpha occur
    (see hodge/coherence.py, where the exact restricted map is iterated instead).
(2) R2: a finite kernel step z' = z + Theta W J (T(z) - z), normalized so D Theta D <= I
    (no over-relaxation, D = (W J)^{1/2}). With V = z^T H^{-1} z one gets
    Delta V <= -[2(1-alpha) - h((1-alpha)^2 + gamma^2)] ||z||^2 for H <= h I, so
    alpha^2 + gamma^2 < 1 stays sufficient for h <= 1. The probe counts violations.

Numerical checks only; neither statement has been independently verified as a theorem.
"""
import numpy as np

from _common import block_diag, psd_sqrt

rng = np.random.default_rng(0)


def skew(K):
    A = rng.normal(size=(K, K))
    Pi = np.eye(K) - np.ones((K, K)) / K
    return Pi @ (A - A.T) @ Pi


def jac(pi):
    return np.diag(pi) - np.outer(pi, pi)


N, K, p = 6, 4, 10
res = dict(trials=0, r1_max_real_part_deviation=0.0, r1_max_frequency_ratio=0.0,
           r2_unstable_while_tabular_stable=0, r2_stable_while_tabular_unstable=0)
for _ in range(4000):
    C = block_diag(*[skew(K) for _ in range(N)])
    J = block_diag(*[jac(rng.dirichlet(np.ones(K))) for _ in range(N)])
    W = np.kron(np.diag(rng.dirichlet(np.ones(N)) * N), np.eye(K))
    Phi = rng.normal(size=(N * K, p))
    Jh = psd_sqrt(J)
    omega = np.linalg.svd(Jh @ C @ Jh, compute_uv=False)[0]
    alpha, g = rng.uniform(0.3, 0.99), rng.uniform(0.1, 3.0) / max(omega, 1e-9)
    tabular = alpha ** 2 + (g * omega) ** 2
    B = np.linalg.solve(Phi.T @ W @ J @ Phi, Phi.T @ W @ J @ C @ J @ Phi)
    ev = np.linalg.eigvals(alpha * np.eye(p) + g * B)
    res["r1_max_real_part_deviation"] = max(res["r1_max_real_part_deviation"], float(np.max(np.abs(ev.real - alpha))))
    res["r1_max_frequency_ratio"] = max(res["r1_max_frequency_ratio"], float(np.max(np.abs(np.linalg.eigvals(B).imag)) / omega))
    Theta = Phi @ Phi.T
    D = psd_sqrt(W @ J)
    Theta = Theta / np.linalg.eigvalsh(D @ Theta @ D).max()
    M = np.eye(N * K) + Theta @ W @ J @ ((alpha - 1) * np.eye(N * K) + g * C @ J)
    U, s, _ = np.linalg.svd(Theta @ W @ J)
    U = U[:, s > 1e-10]
    r = np.max(np.abs(np.linalg.eigvals(U.T @ M @ U)))
    res["trials"] += 1
    res["r2_unstable_while_tabular_stable"] += int(tabular < 1 and r > 1 + 1e-9)
    res["r2_stable_while_tabular_unstable"] += int(tabular > 1 and r < 1 - 1e-9)
print(res)
