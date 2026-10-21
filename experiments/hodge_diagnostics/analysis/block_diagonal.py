"""Stacking prompts into one block-diagonal matrix.

The correct decomposition of a block-diagonal panel is per block (the comparison graph is a
disjoint union of complete graphs), so the stacked cyclic share is the energy-weighted mean
of the per-prompt shares. Feeding the 4N x 4N matrix to the complete-graph formula with
cross-prompt pairs at indifference instead labels (1 - 1/N) of every block's potential as
cyclic, and the share tends to one. Stability of the stacked tabular system is governed by
the worst block, max_b omega_b.
"""
import numpy as np

from _common import DATA, block_diag
from hodge.constructions import attribute_vote
from hodge.core import flow, hodge, omega_max
from hodge.datasets import LOADERS

panels = LOADERS["ultrafeedback"](DATA)
rng = np.random.default_rng(0)
for N in (10, 100, 1000):
    idx = rng.choice(len(panels), N, replace=False)
    As = [flow(attribute_vote(panels[i].ratings, pseudo=0.5), "logit") for i in idx]
    Cs = [hodge(A).C for A in As]
    per_block = sum(np.sum(C ** 2) for C in Cs) / sum(np.sum(A ** 2) for A in As)
    big = block_diag(*As)
    hb = hodge(big)
    naive = np.sum(hb.C ** 2) / np.sum(big ** 2)
    worst = max(omega_max(C, np.full(4, 0.25)) for C in Cs)
    print(f"N={N:5d}: per-block share {per_block:.3f} | complete-graph share on 4N {naive:.3f} | max_b omega_b {worst:.3f}")
