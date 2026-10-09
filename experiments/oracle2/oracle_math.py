"""Oracle2 probabilities and prompt-level statistics; no model or scheduler imports."""
import itertools
import numpy as np


def finite(x):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite oracle input')
    return x


def sigmoid(x):
    x = finite(x)
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + z), z / (1 + z))


def probability(n_difference, s_difference, weight=.6, temperatures=(1., 1.)):
    if not 0 < weight < 1 or len(temperatures) != 2 or any(
            not np.isfinite(t) or t <= 0 for t in temperatures):
        raise ValueError('Invalid frozen mixture specification')
    return weight * sigmoid(finite(n_difference) / temperatures[0]) + (
        1 - weight) * sigmoid(finite(s_difference) / temperatures[1])


def preference_matrix(n_scores, s_scores, weight=.6, temperatures=(1., 1.)):
    n, s = finite(n_scores), finite(s_scores)
    if n.shape != s.shape or n.ndim != 1 or len(n) < 3:
        raise ValueError('Need aligned one-dimensional reward vectors, K>=3')
    p = probability(n[:, None] - n, s[:, None] - s, weight, temperatures)
    np.fill_diagonal(p, .5)
    if not np.allclose(p + p.T, 1, atol=1e-14, rtol=0):
        raise ValueError('Nonreciprocal comparison')
    return p


def classify_panel(p, margin=.02):
    p = finite(p)
    if p.ndim != 2 or p.shape[0] != p.shape[1] or not np.allclose(p+p.T, 1):
        raise ValueError('Invalid comparison matrix')
    if not 0 <= margin < .5 or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid probability or margin')
    triangles, majority_cycles = [], 0
    for a, b, c in itertools.combinations(range(len(p)), 3):
        for cycle in ((a, b, c), (a, c, b)):
            i, j, k = cycle
            m = float(min(p[i, j], p[j, k], p[k, i]) - .5)
            majority_cycles += int(m > 0)
            if m > margin:
                triangles.append(dict(vertices=list(cycle), min_edge_margin=m))
    edges = p[np.triu_indices(len(p), 1)]
    all_decisive = bool(np.all(np.abs(edges - .5) >= margin))
    group = 'cyclic' if triangles else ('transitive' if all_decisive and not majority_cycles else 'ambiguous')
    return dict(group=group, robust_triangles=triangles, majority_triangle_count=majority_cycles,
                minimum_pair_margin=float(np.min(np.abs(edges - .5))))


def generated_wr(current_n, current_s, baseline_n, baseline_s, weight=.6, temperatures=(1., 1.)):
    arrays = [finite(x) for x in (current_n, current_s, baseline_n, baseline_s)]
    if any(x.ndim != 1 or not len(x) for x in arrays):
        raise ValueError('Expected nonempty response score vectors')
    n, s, bn, bs = arrays
    if n.shape != s.shape or bn.shape != bs.shape:
        raise ValueError('Component response IDs must align')
    p = probability(n[:, None] - bn, s[:, None] - bs, weight, temperatures)
    return dict(oracle2_expected_win_rate=float(p.mean()),
                oracle2_majority_win_rate=float(((p > .5) + .5 * (p == .5)).mean()),
                pair_count=int(p.size))


def wr_steps(iterations=100, every=10):
    if iterations < 1 or every < 1 or iterations % every:
        raise ValueError('Use a complete regular checkpoint grid')
    return list(range(0, iterations + 1, every))
