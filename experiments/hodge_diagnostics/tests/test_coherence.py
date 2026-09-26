"""Coherence statistic and exact restricted dynamics."""
import numpy as np
import pytest

from hodge.coherence import SharedPolicy, compressed_frequency, permutation_null
from hodge.core import flow, hodge, omega_max
from hodge.population import fixed_point, forcing, softmax
from test_core import R_CYC, random_reciprocal


def test_identical_residuals_are_fully_coherent():
    rng = np.random.default_rng(20)
    C = hodge(flow(random_reciprocal(4, rng), "identity")).C
    N = 50
    L = np.tile(np.arange(4), (N, 1))
    Cs = np.repeat(C[None], N, axis=0)
    assert compressed_frequency(L, Cs, 4) == pytest.approx(np.linalg.norm(C, 2) / 4)
    assert np.median(permutation_null(L, Cs, 4, 50, 0)) < 0.5 * np.linalg.norm(C, 2) / 4


def test_realizable_shared_policy_reproduces_tabular_dynamics():
    """One prompt with identity labels: the restricted map is the tabular map, so the
    simulated limit is the tabular fixed point and the contraction rate is sqrt(alpha^2 + gamma^2)."""
    rng = np.random.default_rng(21)
    h = hodge(flow(random_reciprocal(4, rng), "logit"))
    alpha, beta, lam = 0.8, 1.0, 1.0
    policy = SharedPolicy(np.arange(4)[None], h.u[None], h.C[None], 4)
    sim = policy.simulate(alpha, beta, lam, T=800)
    b, g = forcing(h, alpha, beta, lam)
    x = fixed_point(h.C, b, alpha, g)
    assert sim["label"] == "converged"
    assert np.abs(sim["probs"][0] - softmax(x)).sum() < 1e-8
    gamma = g * omega_max(h.C, softmax(x))
    assert sim["rate"] == pytest.approx(np.sqrt(alpha ** 2 + gamma ** 2), rel=0.03)


def test_realizable_three_cycle_keeps_cycling():
    policy = SharedPolicy(np.arange(3)[None], np.zeros((1, 3)), R_CYC[None], 3)
    assert policy.simulate(0.9, 0.93, 1.0, T=600)["label"] == "cycling"
    assert policy.simulate(0.9, 0.65, 1.0, T=600)["label"] == "converged"
