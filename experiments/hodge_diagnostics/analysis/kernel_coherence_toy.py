"""Toy version of R3: shared features act as a coherence filter.

N prompts carry the pilot's cyclic tournament residual C0. With features that encode only
the response position (shared by all prompts), identical residuals keep the tabular
frequency, while residuals whose response order is permuted per prompt average out.
Adding a few generic features lets part of the shuffled residual through.
"""
import numpy as np

from _common import block_diag
from hodge.coherence import compressed_frequency

rng = np.random.default_rng(1)
K = 4
A0 = np.zeros((K, K))
for w, l in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]:
    A0[w, l], A0[l, w] = 0.5, -0.5
u0 = A0.sum(1) / K
C0 = A0 - (u0[:, None] - u0[None, :])
omega = np.linalg.norm(C0, 2) / K


def omega_generic(Cblocks, Phi):
    Nn = len(Cblocks)
    C = block_diag(*Cblocks)
    J = np.kron(np.eye(Nn), (np.eye(K) - np.ones((K, K)) / K) / K)
    G = Phi.T @ J @ Phi
    S = Phi.T @ J @ C @ J @ Phi
    return float(np.max(np.abs(np.linalg.eigvals(np.linalg.pinv(G) @ S).imag)))


for N in (10, 100, 500):
    perms = [rng.permutation(K) for _ in range(N)]
    same = np.repeat(C0[None], N, axis=0)
    shuffled = np.array([C0[np.ix_(p, p)] for p in perms])
    L = np.tile(np.arange(K), (N, 1))
    Q = np.linalg.svd(np.eye(K) - 1.0 / K)[0][:, : K - 1]
    extra = np.hstack([np.kron(np.ones((N, 1)), Q), rng.normal(size=(N * K, 8)) * 0.3])
    print(f"N={N:4d}: tabular omega {omega:.3f} | shared position only: identical {compressed_frequency(L, same, K):.3f}, "
          f"shuffled {compressed_frequency(L, shuffled, K):.3f} | plus 8 generic features: shuffled "
          f"{omega_generic(list(shuffled), extra):.3f}")
