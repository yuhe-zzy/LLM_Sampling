"""Where does the cyclic-history pilot sit relative to the ordinary frontier?

The pilot (experiments/cyclic_history/experiment_plan.json) uses the full four-response
tournament of scripts/build_cyclic_pairs.py on every prompt, IPO (identity link),
alpha = 0.9, lambda_current = 0.8, beta_train = 1 (paper beta = 1/beta_train), and uniform
coverage. This prints the exact local index alpha^2 + gamma^2 at the unique fixed point for
a flat panel reference and for a spread reference that mimics sequence-sum log-likelihood
gaps of 5 nats between consecutive responses (the pilot samples from softmax of raw
sequence sums, so a spread reference concentrates the sampler).
"""
import numpy as np

import _common  # noqa: F401
from hodge.core import flow, hodge
from hodge.population import ordinary_stability

P = np.full((4, 4), 0.5)
for w, l in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]:
    P[w, l], P[l, w] = 1.0, 0.0
h = hodge(flow(P, "identity"))
LAM = 0.8
for name, logref in (("flat panel reference", np.zeros(4)), ("reference with 5-nat steps", np.array([0., -5., -10., -15.]))):
    ref = np.exp(logref - logref.max()); ref /= ref.sum()
    print(f"\n{name}; lambda = {LAM}; rows beta_train (paper beta = 1/beta_train), columns alpha")
    print("beta_train " + " ".join(f"{a:>7}" for a in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)))
    for bt in (1.0, 0.5, 0.25):
        row = [ordinary_stability(h, a, 1 / bt, LAM, ref=ref)["index"] for a in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)]
        print(f"{bt:>10} " + " ".join(f"{v:7.3f}{'*' if v > 1 else ' '}" for v in row))
print("\n* beyond the frontier. The planned pilot point is alpha = 0.9, beta_train = 1.")
