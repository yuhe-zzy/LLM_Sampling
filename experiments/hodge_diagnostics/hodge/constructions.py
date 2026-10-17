"""Constructions of a reciprocal panel matrix P* from ratings or votes.

Every construction is a modeling choice that changes the Hodge components, so
each is named and reported separately:

    single_score_hard   one aggregate score per response, P*_ij in {0, 1/2, 1};
                        the orientation LLM_Sampling/scripts/build_pairs.py uses
    single_score_bt     one aggregate score, P*_ij = sigma(scale (s_i - s_j)); a
                        single Bradley--Terry annotator, so C = 0 under logit
    attribute_vote      each rated attribute is one annotator casting a vote
                        (ties count 1/2); P*_ij is the vote share
    attribute_bt        each attribute is one Bradley--Terry annotator;
                        P*_ij = mean_k sigma(scale (R_ik - R_jk))
    votes               empirical vote shares from pairwise judgments

Pseudo-counts shrink shares toward 1/2 so the logit link stays finite; the
identity-link results are reported without smoothing unless stated.
"""
from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 0.5 * (1 + np.tanh(0.5 * np.asarray(x, dtype=float)))


def _pairwise(scores, fn):
    s = np.asarray(scores, dtype=float)
    P = fn(s[:, None] - s[None, :])
    P[np.isnan(s[:, None] - s[None, :])] = np.nan
    np.fill_diagonal(P, 0.5)
    return P


def single_score_hard(scores):
    return _pairwise(scores, lambda d: np.where(d > 0, 1.0, np.where(d < 0, 0.0, 0.5)))


def single_score_bt(scores, scale=1.0):
    return _pairwise(scores, lambda d: sigmoid(scale * d))


def attribute_vote(R, weights=None, pseudo=0.0):
    """Vote share over attributes; R is K x m with NaN for a missing rating."""
    R = np.asarray(R, dtype=float)
    K, m = R.shape
    w = np.ones(m) if weights is None else np.asarray(weights, dtype=float)
    diff = R[:, None, :] - R[None, :, :]
    present = ~np.isnan(diff)
    vote = np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, 0.5))
    mass = (present * w).sum(axis=2)
    wins = (np.where(present, vote, 0.0) * w).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = (wins + pseudo) / (mass + 2 * pseudo)
    P[mass == 0] = np.nan
    np.fill_diagonal(P, 0.5)
    return P


def attribute_bt(R, scale=1.0, weights=None):
    """Mixture of Bradley--Terry annotators, one per attribute (prop:cyclic-sources)."""
    R = np.asarray(R, dtype=float)
    K, m = R.shape
    w = np.ones(m) if weights is None else np.asarray(weights, dtype=float)
    diff = R[:, None, :] - R[None, :, :]
    present = ~np.isnan(diff)
    mass = (present * w).sum(axis=2)
    total = (np.where(present, sigmoid(scale * np.nan_to_num(diff)), 0.0) * w).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = total / mass
    P[mass == 0] = np.nan
    np.fill_diagonal(P, 0.5)
    return P


def vote_matrix(W, pseudo=0.0):
    """Vote shares from a win-count matrix W (ties already split as halves)."""
    W = np.asarray(W, dtype=float)
    N = W + W.T
    with np.errstate(invalid="ignore", divide="ignore"):
        P = (W + pseudo) / (N + 2 * pseudo)
    P[N == 0] = np.nan
    np.fill_diagonal(P, 0.5)
    return P


def rating_constructions(R, weights=None, bt_scale=1.0, pseudo=0.5):
    """The four rating-based constructions used in the reports, keyed by name."""
    R = np.asarray(R, dtype=float)
    w = np.ones(R.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    score = np.nansum(R * w, axis=1) / np.sum(~np.isnan(R) * w, axis=1)
    return {
        "single_score_hard": (single_score_hard(score), {"identity"}),
        "single_score_bt": (single_score_bt(score, bt_scale), {"identity", "logit"}),
        "attribute_vote": (attribute_vote(R, w, pseudo=0.0), {"identity"}),
        "attribute_vote_smoothed": (attribute_vote(R, w, pseudo=pseudo), {"logit"}),
        "attribute_bt": (attribute_bt(R, bt_scale, w), {"identity", "logit"}),
    }
