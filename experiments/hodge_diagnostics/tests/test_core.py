"""Static Hodge diagnostics against the worked numbers of JMLR-B-v3, Section 3."""
import numpy as np
import pytest

from hodge.constructions import attribute_bt, single_score_bt, single_score_hard, vote_matrix
from hodge.core import (cyclic_constants, cyclic_frequencies, fit_bt, flow, graph_stats, hodge,
                        omega_max, ordinal_stats, panel_summary, softmax_jacobian, triangle_curls)

R_CYC = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], dtype=float)


def random_reciprocal(K, rng, missing=0.0):
    P = np.full((K, K), 0.5)
    for i in range(K):
        for j in range(i + 1, K):
            if rng.random() < missing:
                P[i, j] = P[j, i] = np.nan
            else:
                P[i, j] = rng.uniform(0.02, 0.98)
                P[j, i] = 1 - P[i, j]
    return P


def test_decomposition_identities():
    rng = np.random.default_rng(0)
    for K in (3, 4, 7):
        h = hodge(flow(random_reciprocal(K, rng), "logit"))
        assert np.allclose(h.C, -h.C.T)
        assert np.allclose(h.C @ np.ones(K), 0)
        assert abs(h.u.sum()) < 1e-12
        assert abs(np.sum(h.G * h.C)) < 1e-10  # Frobenius-orthogonal


def test_bt_identity_link_example():
    """s = (1, 0, -1): identity link gives u = (0.204, 0, -0.204) and C = 0.027 R_cyc."""
    P = single_score_bt([1.0, 0.0, -1.0])
    A = flow(P, "identity")
    assert np.allclose([A[0, 1], A[0, 2], A[1, 2]], [0.231, 0.381, 0.231], atol=5e-4)
    h = hodge(A)
    assert np.allclose(h.u, [0.204, 0.0, -0.204], atol=5e-4)
    assert np.allclose(h.C, 0.027 * R_CYC, atol=5e-4)
    assert triangle_curls(A)[0] == pytest.approx(0.081, abs=5e-4)
    assert np.allclose(hodge(flow(P, "logit")).C, 0, atol=1e-12)  # cor:additive


def test_identity_link_tanh_formula():
    rng = np.random.default_rng(1)
    for _ in range(20):
        a, b = rng.uniform(0.01, 4, size=2)
        A = flow(single_score_bt([a + b, b, 0.0]), "identity")
        expected = 0.5 * np.tanh(a / 2) * np.tanh(b / 2) * np.tanh((a + b) / 2)
        assert triangle_curls(A)[0] == pytest.approx(expected, rel=1e-10)


def test_two_annotator_logit_mixture_example():
    """s1 = (2, 0, 0), s2 = (0, 0, 1): the logit-transformed mixture has cyclic sum 0.030."""
    P = attribute_bt(np.array([[2.0, 0.0], [0.0, 0.0], [0.0, 1.0]]))
    assert triangle_curls(flow(P, "logit"))[0] == pytest.approx(0.030, abs=5e-4)


def test_balanced_three_cycle_constants():
    for link, a in (("identity", 0.3), ("logit", 0.2)):
        q = 0.5 + a
        P = np.array([[0.5, q, 1 - q], [1 - q, 0.5, q], [q, 1 - q, 0.5]])
        A = flow(P, link)
        a_psi = A[0, 1]
        h = hodge(A)
        assert np.allclose(h.u, 0)
        L_C, C2 = cyclic_constants(h.C)
        assert L_C == pytest.approx(abs(a_psi))
        assert C2 == pytest.approx(np.sqrt(3) * abs(a_psi))
        assert omega_max(h.C, np.full(3, 1 / 3)) == pytest.approx(abs(a_psi) / np.sqrt(3))


def test_three_response_frequency_formula():
    """omega_max = 3|q| sqrt(pi1 pi2 pi3) for C = q R_cyc (eq. three-frequency)."""
    rng = np.random.default_rng(2)
    for _ in range(10):
        pi = rng.dirichlet(np.ones(3))
        q = rng.normal()
        assert omega_max(q * R_CYC, pi) == pytest.approx(3 * abs(q) * np.sqrt(np.prod(pi)))


def test_frequencies_match_eigenvalues_of_CJ():
    rng = np.random.default_rng(3)
    for K in (4, 6):
        C = hodge(flow(random_reciprocal(K, rng), "identity")).C
        pi = rng.dirichlet(np.ones(K))
        eig = np.linalg.eigvals(C @ softmax_jacobian(pi))
        assert np.max(np.abs(eig.real)) < 1e-10
        assert omega_max(C, pi) == pytest.approx(np.max(np.abs(eig.imag)))
        assert np.allclose(np.sort(cyclic_frequencies(C, pi)), np.sort(np.abs(eig.imag)), atol=1e-10)


def test_transitive_hard_labels_have_cyclic_residual_under_identity():
    """SST and no Condorcet cycle, yet C != 0: ordinal transitivity does not imply C = 0."""
    P = single_score_hard([3.0, 2.0, 1.0, 0.0])
    stats = ordinal_stats(P)
    assert not stats["has_condorcet_cycle"] and stats["sst_violation_rate"] == 0
    assert np.max(np.abs(hodge(flow(P, "identity")).C)) > 0.1


def test_repo_cyclic_tournament():
    """LLM_Sampling/scripts/build_cyclic_pairs.py full tournament, K = 4."""
    P = np.full((4, 4), 0.5)
    for w, l in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]:
        P[w, l], P[l, w] = 1.0, 0.0
    h = hodge(flow(P, "identity"))
    assert np.allclose(h.u, np.array([1, 1, -1, -1]) / 8)
    assert np.allclose(h.C[0], [0, 0.5, 0.25, -0.75])
    assert cyclic_constants(h.C)[1] == pytest.approx(np.sqrt(1.25))


def test_fact_bounds_on_sparse_panels():
    rng = np.random.default_rng(4)
    for _ in range(50):
        K = int(rng.integers(3, 9))
        P = random_reciprocal(K, rng, missing=0.5)
        for link in ("identity", "logit"):
            s = panel_summary(P, link)
            assert s["L_C"] <= s["fact_L_C_bound"] + 1e-12
            assert s["C_norm2"] <= s["fact_C2_bound"] + 1e-12


def test_graph_stats_and_votes():
    W = np.array([[0, 3, 0], [1, 0, 2], [0, 2, 0]], dtype=float)
    P = vote_matrix(W)
    assert np.isnan(P[0, 2]) and P[0, 1] == pytest.approx(0.75)
    g = graph_stats(P)
    assert g["edges"] == 2 and g["d"] == 2 and g["completeness"] == pytest.approx(2 / 3)


def test_bt_fit_recovers_scores():
    s = np.array([1.0, 0.2, -0.4, -0.8])
    p = 1 / (1 + np.exp(-(s[:, None] - s[None, :])))
    W = 1e5 * p
    np.fill_diagonal(W, 0)
    assert np.allclose(fit_bt(W, ridge=0.0), s - s.mean(), atol=1e-6)
