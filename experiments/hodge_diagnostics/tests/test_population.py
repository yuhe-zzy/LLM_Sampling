"""Population recursion and stability thresholds of JMLR-B-v3, Sections 4-6."""
import numpy as np
import pytest

from hodge.core import flow, hodge, omega_max
from hodge.population import (KAPPA_STAR_FULL_REFRESH, centered, fixed_point, forcing,
                              gamma2_lagged_reference, gamma2_lagged_sampling, jacobian, kappa_exp,
                              ordinary_stability, simulate, softmax)
from test_core import R_CYC, random_reciprocal


def roots_inside(coeffs):
    return np.max(np.abs(np.roots(coeffs))) < 1


def test_fixed_point_and_exact_jacobian():
    rng = np.random.default_rng(10)
    for K in (3, 5):
        h = hodge(flow(random_reciprocal(K, rng), "logit"))
        ref = rng.dirichlet(np.ones(K))
        pi0 = rng.dirichlet(np.ones(K))
        alpha, beta, lam = 0.8, 2.0, 0.7
        b, gain = forcing(h, alpha, beta, lam, ref, pi0)
        x = fixed_point(h.C, b, alpha, gain)
        assert np.allclose(alpha * x + b + gain * h.C @ softmax(x), x, atol=1e-10)
        eps = 1e-6
        F = lambda y: centered(alpha * y + b + gain * h.C @ softmax(y))
        basis = np.linalg.qr(np.vstack([np.ones(K), np.eye(K)[:-1]]).T)[0][:, 1:]
        numeric = np.column_stack([(F(x + eps * v) - F(x - eps * v)) / (2 * eps) for v in basis.T])
        assert np.allclose(numeric, jacobian(h.C, x, alpha, gain) @ basis, atol=1e-7)
        mult = np.linalg.eigvals(basis.T @ jacobian(h.C, x, alpha, gain) @ basis)
        assert np.allclose(mult.real, alpha, atol=1e-9)
        assert np.max(np.abs(mult.imag)) == pytest.approx(gain * omega_max(h.C, softmax(x)))


def test_three_cycle_frontier_by_simulation():
    """alpha = 0.9: g = 0.65 decays, g = 0.93 keeps cycling (discussion after cor:generic-nonconvergence)."""
    b = np.zeros(3)
    x1 = centered(np.array([0.02, -0.01, 0.0]))
    for g, converges in ((0.65, True), (0.93, False)):
        xs = simulate(R_CYC, b, g, x1, T=3000, weights=(0.9,))
        tail = np.linalg.norm(xs[-300:], axis=1)
        assert (tail.max() < 1e-6) == converges
        if not converges:
            assert tail.min() > 0.05
        index = 0.81 + (g * omega_max(R_CYC, np.full(3, 1 / 3))) ** 2
        assert (index < 1) == converges


def test_lagged_sampling_threshold_matches_characteristic_roots():
    rng = np.random.default_rng(11)
    for _ in range(2000):
        alpha, kappa, gamma = rng.uniform(0, 1), rng.uniform(0, 2), rng.uniform(0, 1.2)
        G = gamma2_lagged_sampling(alpha, kappa)
        if abs(gamma ** 2 - G) < 1e-6:
            continue
        stable = roots_inside([1, -(alpha + 1j * (1 + kappa) * gamma), 1j * kappa * gamma])
        assert stable == (gamma ** 2 < G)


def test_lagged_sampling_constants():
    assert gamma2_lagged_sampling(1.0, 1.0) == pytest.approx(1 / 3)
    assert np.sqrt(gamma2_lagged_sampling(1.0, KAPPA_STAR_FULL_REFRESH)) == pytest.approx(0.601, abs=5e-4)
    assert gamma2_lagged_sampling(1.0, 0.5) == 0
    for alpha in np.linspace(0.05, 0.95, 19):
        base = 1 - alpha ** 2
        for kappa in np.linspace(0.01, 3, 60):
            gains = gamma2_lagged_sampling(alpha, kappa) > base + 1e-12
            assert gains == (alpha > 0.5 and kappa < kappa_exp(alpha))
    # kappa = 1 destabilizes exactly below 2(sqrt 2 - 1)
    assert kappa_exp(2 * (np.sqrt(2) - 1)) == pytest.approx(1.0)


def test_lagged_reference_threshold_matches_characteristic_roots():
    rng = np.random.default_rng(12)
    for _ in range(2000):
        alpha = rng.uniform(0.01, 0.99)
        nu, gamma = rng.uniform(0, alpha), rng.uniform(0, 2)
        R = gamma2_lagged_reference(alpha, nu)
        if abs(gamma ** 2 - R) < 1e-6:
            continue
        assert roots_inside([1, -(alpha - nu + 1j * gamma), -nu]) == (gamma ** 2 < R)
    for alpha in (0.3, 0.9):
        assert gamma2_lagged_reference(alpha, 0) == pytest.approx(1 - alpha ** 2)
        assert gamma2_lagged_reference(alpha, alpha) == pytest.approx((1 + alpha) ** 2)
        values = [gamma2_lagged_reference(alpha, nu) for nu in np.linspace(0, alpha, 30)]
        assert np.all(np.diff(values) > 0)


def test_surgeries_share_the_ordinary_limit_point():
    """prop:history-anchor, part 1, on a random stable panel."""
    rng = np.random.default_rng(13)
    h = hodge(flow(random_reciprocal(5, rng), "identity"))
    alpha, beta, lam = 0.9, 1.5, 1.0
    b, gain = forcing(h, alpha, beta, lam)
    stats = ordinary_stability(h, alpha, beta, lam)
    assert stats["index"] < 1
    x1 = np.zeros(5)
    for weights, kappa in (((alpha,), 0.0), ((alpha,), 0.25), ((alpha / 2, alpha / 2), 0.0)):
        xs = simulate(h.C, b, gain, x1, T=4000, weights=weights, kappa=kappa)
        assert np.allclose(xs[-1], stats["x_star"], atol=1e-8)


def test_additive_horizons():
    """prop:adaptation-horizon and prop:history-anchor, part 2 (C = 0, x_1 = r = 0)."""
    K, beta, alpha = 4, 1.0, 0.8
    u = centered(np.array([0.3, 0.1, -0.1, -0.3]))
    C = np.zeros((K, K))
    M = lambda T: sum(alpha ** k for k in range(T))
    ordinary = simulate(C, beta * u, 0.0, np.zeros(K), T=12, weights=(alpha,), kappa=0.7)
    maxlag = simulate(C, beta * u, 0.0, np.zeros(K), T=12, weights=(0.0, alpha))
    for T in range(1, 13):
        assert np.allclose(ordinary[T], beta * M(T) * u)
        assert np.allclose(maxlag[T], beta * M(-(-T // 2)) * u)
