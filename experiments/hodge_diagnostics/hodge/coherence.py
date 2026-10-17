"""Cross-prompt coherence of cyclic residuals under parameter sharing.

A policy that shares parameters across prompts sees the per-prompt residuals C_x only
through shared response features. With identity features E_x (K x M one-hot: the model
that wrote the response, or its within-prompt length rank), the locally compressed cyclic
operator at uniform policies is

    G = sum_x E_x^T J_x E_x,   S = sum_x E_x^T J_x C_x J_x E_x,
    omega~ = largest |Im eig(G^+ S)| = ||G^{+1/2} S G^{+1/2}||_2.

Identical residuals on every prompt give omega~ = omega_x; residuals whose orientation is
unrelated across prompts average out. The permutation null shuffles the label assignment
inside every prompt: each |C_x| and each label set (hence G) is unchanged, and only the
alignment across prompts is destroyed.

The restricted dynamics use the exact round map of the maximally shared policy
pi(.|x) = softmax(E_x theta), theta in R^M, with uniform reference and coverage:

    theta_{t+1} = argmin_theta sum_x KL(softmax(E_x theta) || softmax(alpha E_x theta_t + beta g_x(theta_t))),
    g_x = u_x + lambda C_x pi_x(theta_t).

The shared class cannot represent every prompt's potential, so the fixed point carries a
projection residual and the local Jacobian is computed numerically from the exact map
rather than from the realizable-case formula alpha I + beta lambda G^{-1} S.

    python -m hodge.coherence
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .constructions import attribute_vote, vote_matrix
from .core import flow, hodge
from .datasets import HELPSTEER_ALL5, LOADERS

ROOT = Path(__file__).resolve().parents[1]


# Labeled panels ---------------------------------------------------------------

def _stack(labels, flows, names, meta=None):
    L = np.array(labels, dtype=int)
    A = np.array(flows)
    parts = [hodge(a) for a in A]
    return dict(L=L, A=A, U=np.array([h.u for h in parts]), C=np.array([h.C for h in parts]),
                names=list(names), meta=meta or {})


def ultrafeedback_by_model(root):
    panels = [p for p in LOADERS["ultrafeedback"](root) if p.K == 4 and len(set(p.labels)) == 4]
    names = sorted({m for p in panels for m in p.labels})
    index = {m: k for k, m in enumerate(names)}
    out = _stack([[index[m] for m in p.labels] for p in panels],
                 [flow(attribute_vote(p.ratings, pseudo=0.5), "logit") for p in panels], names)
    out["source"] = np.array([p.meta["source"] for p in panels])
    return out


def helpsteer_by_length_rank(root, construction="votes"):
    """K = 4 HelpSteer panels labeled by within-prompt length rank.

    construction: "votes" (attributes as annotators, observed), "bt" (soft per-attribute BT
    mixture), or "surrogate" (the attribute mean given as five identical votes: transitive).
    """
    from .constructions import attribute_bt

    groups = defaultdict(lambda: defaultdict(list))
    for split in ("train", "validation"):
        with gzip.open(Path(root) / "helpsteer" / f"{split}.jsonl.gz", "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                groups[(split, row["prompt"])][row["response"]].append([float(row[a]) for a in HELPSTEER_ALL5])
    labels, flows = [], []
    for responses in groups.values():
        if len(responses) != 4:
            continue
        texts = list(responses)
        R = np.array([np.mean(responses[t], axis=0) for t in texts])
        labels.append(np.argsort(np.argsort([len(t) for t in texts], kind="stable"), kind="stable"))
        if construction == "bt":
            P = attribute_bt(R)
        elif construction == "surrogate":
            P = attribute_vote(np.repeat(R.mean(axis=1, keepdims=True), R.shape[1], axis=1), pseudo=0.5)
        else:
            P = attribute_vote(R, pseudo=0.5)
        flows.append(flow(P, "logit"))
    return _stack(labels, flows, ["shortest", "second", "third", "longest"])


def mt_bench_by_model(root, split="human"):
    panels = LOADERS[f"mt_bench_{'human' if split == 'human' else 'gpt4'}"](root)
    return _stack([list(range(p.K)) for p in panels],
                  [flow(vote_matrix(p.wins, 0.5), "logit") for p in panels], panels[0].labels)


# Static compressed frequency ----------------------------------------------------

def compressed_frequency(L, C, M):
    """omega~ at uniform policies; J_x = Pi/K so J C J = C/K^2 and E^T J E = E^T Pi E / K."""
    N, K = L.shape
    Pi = np.eye(K) - 1.0 / K
    rows, cols = L[:, :, None], L[:, None, :]
    G = np.zeros((M, M))
    np.add.at(G, (np.broadcast_to(rows, (N, K, K)), np.broadcast_to(cols, (N, K, K))),
              np.broadcast_to(Pi / K, (N, K, K)))
    S = np.zeros((M, M))
    np.add.at(S, (np.broadcast_to(rows, (N, K, K)), np.broadcast_to(cols, (N, K, K))), C / K ** 2)
    w, V = np.linalg.eigh(G)
    keep = w > 1e-9 * w.max()
    root = V[:, keep] / np.sqrt(w[keep])
    T = root.T @ S @ root
    return float(np.linalg.svd(0.5 * (T - T.T), compute_uv=False)[0])


def permutation_null(L, C, M, draws, seed):
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for b in range(draws):
        perm = np.argsort(rng.random(L.shape), axis=1)
        out[b] = compressed_frequency(np.take_along_axis(L, perm, axis=1), C, M)
    return out


def static_coherence(data, draws=500, seed=0):
    L, C = data["L"], data["C"]
    M, K = len(data["names"]), L.shape[1]
    omega_x = np.linalg.norm(C, ord=2, axis=(1, 2)) / K
    observed = compressed_frequency(L, C, M)
    null = permutation_null(L, C, M, draws, seed)
    return dict(panels=int(len(L)), labels=M, K=K, omega_tilde=observed,
                omega_prompt_median=float(np.median(omega_x)), omega_prompt_p90=float(np.quantile(omega_x, 0.9)),
                omega_prompt_energy_rms=float(np.sqrt(np.mean(omega_x ** 2))),
                coherence_ratio=observed / float(np.sqrt(np.mean(omega_x ** 2))),
                null_median=float(np.median(null)), null_p95=float(np.quantile(null, 0.95)),
                p_value=float((1 + np.sum(null >= observed)) / (1 + draws)))


# Label-level HodgeRank -----------------------------------------------------------

def _aggregate(L, A, M, weights=None):
    N, K = L.shape
    w = np.ones(N) if weights is None else weights
    ii, jj = np.where(~np.eye(K, dtype=bool))
    SA = np.zeros((M, M))
    n = np.zeros((M, M))
    for i, j in zip(ii, jj):
        np.add.at(SA, (L[:, i], L[:, j]), w * A[:, i, j])
        np.add.at(n, (L[:, i], L[:, j]), w)
    with np.errstate(invalid="ignore", divide="ignore"):
        Y = np.where(n > 0, SA / n, 0.0)
    return Y, n


def label_hodgerank(data, min_count=100, boot=200, seed=0, top=8):
    L, A, names = data["L"], data["A"], data["names"]
    M = len(names)
    Y, n = _aggregate(L, A, M)
    Lap = np.diag(n.sum(axis=1)) - n
    s = np.linalg.lstsq(Lap, (n * Y).sum(axis=1), rcond=None)[0]
    s -= s.mean()
    R = np.where(n > 0, Y - (s[:, None] - s[None, :]), 0.0)
    residual_share = float(np.sum(n * R ** 2) / np.sum(n * Y ** 2))
    strong = n >= min_count
    triangles = []
    for a in range(M):
        for b in range(a + 1, M):
            for c in range(b + 1, M):
                if strong[a, b] and strong[b, c] and strong[a, c]:
                    triangles.append((a, b, c, Y[a, b] + Y[b, c] + Y[c, a]))
    triangles.sort(key=lambda t: -abs(t[3]))
    chosen = triangles[:top]
    rng = np.random.default_rng(seed)
    boots = np.empty((boot, len(chosen)))
    for k in range(boot):
        Yb, _ = _aggregate(L, A, M, weights=rng.multinomial(len(L), np.full(len(L), 1 / len(L))).astype(float))
        boots[k] = [Yb[a, b] + Yb[b, c] + Yb[c, a] for a, b, c, _ in chosen]
    table = pd.DataFrame([dict(cycle=f"{names[a]} > {names[b]} > {names[c]} > {names[a]}" if curl > 0 else
                               f"{names[a]} > {names[c]} > {names[b]} > {names[a]}",
                               curl=abs(curl), ci_low=float(np.quantile(np.sign(curl) * boots[:, k], 0.025)),
                               ci_high=float(np.quantile(np.sign(curl) * boots[:, k], 0.975)),
                               co_occurrences=int(min(n[a, b], n[b, c], n[a, c])))
                          for k, (a, b, c, curl) in enumerate(chosen)])
    ranking = [names[k] for k in np.argsort(-s)]
    return dict(residual_share=residual_share, triangles_tested=len(triangles), ranking=ranking,
                scores=dict(zip(names, np.round(s, 3).tolist()))), table


# Exact restricted dynamics --------------------------------------------------------

class SharedPolicy:
    """pi(.|x) = softmax(theta[L_x]) with uniform reference and coverage."""

    def __init__(self, L, U, C, M):
        self.L, self.U, self.C, self.M = L, U, C, M
        N, K = L.shape
        self._rows = np.broadcast_to(L[:, :, None], (N, K, K))
        self._cols = np.broadcast_to(L[:, None, :], (N, K, K))
        self._eye = np.eye(K)[None]

    def probs(self, theta):
        Z = theta[self.L]
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def payoff(self, theta, beta, lam):
        return beta * (self.U + lam * np.einsum("nij,nj->ni", self.C, self.probs(theta)))

    def kl(self, theta, target):
        Z = theta[self.L]
        P = self.probs(theta)

        def lse(X):
            m = X.max(axis=1)
            return m + np.log(np.exp(X - m[:, None]).sum(axis=1))

        return float(np.sum(np.sum(P * (Z - target), axis=1) - lse(Z) + lse(target)))

    def gauss_newton_step(self, theta, target):
        """Fisher-preconditioned step G^+ sum_x E_x^T J_x (target_x - E_x theta); in the
        realizable case it is the exact tabular displacement."""
        P = self.probs(theta)
        V = target - theta[self.L]
        W = P * V - P * np.sum(P * V, axis=1, keepdims=True)
        r = np.zeros(self.M)
        np.add.at(r, self.L, W)
        J = P[:, :, None] * self._eye - P[:, :, None] * P[:, None, :]
        G = np.zeros((self.M, self.M))
        np.add.at(G, (self._rows, self._cols), J)
        step = np.linalg.lstsq(G, r - r.mean(), rcond=None)[0]
        return step - step.mean()

    def inner_solve(self, theta, target, tol=1e-11, max_iter=200):
        """Exact round update: minimize sum_x KL(softmax(E_x theta) || softmax(target_x))."""
        f = self.kl(theta, target)
        for _ in range(max_iter):
            step = self.gauss_newton_step(theta, target)
            if np.max(np.abs(step)) < tol:
                break
            t = 1.0
            while t > 1e-12:
                trial = theta + t * step
                f_trial = self.kl(trial, target)
                if f_trial <= f + 1e-12 * (1.0 + abs(f)):  # KL is a difference of large terms; allow roundoff
                    break
                t *= 0.5
            else:
                break
            theta, f = trial, f_trial
        return theta - theta.mean()

    def round(self, theta, alpha, beta, lam):
        return self.inner_solve(theta, alpha * theta[self.L] + self.payoff(theta, beta, lam))

    def simulate(self, alpha, beta, lam, T=400, window=50, seed=0, jitter=1e-3):
        """Iterate the exact restricted map from the uniform policy and classify the tail.

        tv_t is the prompt-averaged total-variation step. The label is converged when the
        last-window steps are below 1e-8 and cycling when they all stay above 1e-4; the rate
        is the geometric-mean ratio of successive steps in the linear regime 1e-12 < tv < 1e-6
        (NaN when the run never enters it, e.g. while cycling).
        """
        theta = jitter * np.random.default_rng(seed).normal(size=self.M)  # generic start near uniform
        theta -= theta.mean()
        P_prev = self.probs(theta)
        tv = []
        for _ in range(T):
            theta = self.round(theta, alpha, beta, lam)
            P = self.probs(theta)
            tv.append(float(0.5 * np.abs(P - P_prev).sum(axis=1).mean()))
            P_prev = P
        tv = np.array(tv)
        tail = tv[-window:]
        live = tv[(tv > 1e-12) & (tv < 1e-6)]  # asymptotic linear regime
        rate = float(np.exp(np.mean(np.log(live[1:] / live[:-1])))) if len(live) > 10 else float("nan")
        label = "converged" if tail.max() < 1e-8 else ("cycling" if tail.min() > 1e-4 else "slow or intermittent")
        entropy = -np.sum(P * np.log(np.maximum(P, 1e-300)), axis=1)
        return dict(label=label, tail_tv_max=float(tail.max()), tail_tv_min=float(tail.min()), rate=rate,
                    prompt_entropy_median=float(np.median(entropy)), theta=theta, probs=P, tv=tv)


def restricted_dynamics(data, grid, lam=1.0, sample=None, seed=0, L=None):
    Lx, U, C = (data["L"] if L is None else L), data["U"], data["C"]
    if sample and len(Lx) > sample:
        idx = np.random.default_rng(seed).choice(len(Lx), sample, replace=False)
        Lx, U, C = Lx[idx], U[idx], C[idx]
    policy = SharedPolicy(Lx, U, C, len(data["names"]))
    rows = []
    for alpha, beta in grid:
        sim = policy.simulate(alpha, beta, lam)
        rows.append(dict(alpha=alpha, beta_lambda=beta * lam, panels=len(Lx), label=sim["label"],
                         tail_tv_max=sim["tail_tv_max"], tail_tv_min=sim["tail_tv_min"], rate=sim["rate"],
                         prompt_entropy_median=sim["prompt_entropy_median"],
                         top_label=data["names"][int(np.argmax(sim["theta"]))]))
    return pd.DataFrame(rows)


GRID = [(0.5, 1.0), (0.5, 10.0), (0.9, 1.0), (0.9, 10.0), (0.99, 1.0), (0.99, 10.0)]


def main():
    root = ROOT / "data"
    out = ROOT / "outputs" / "coherence"
    out.mkdir(parents=True, exist_ok=True)
    datasets = {
        "ultrafeedback_by_model": ultrafeedback_by_model(root),
        "helpsteer_by_length_rank": helpsteer_by_length_rank(root),
        "mt_bench_human_by_model": mt_bench_by_model(root, "human"),
        "mt_bench_gpt4_by_model": mt_bench_by_model(root, "gpt4"),
    }
    summary = {}
    for name, data in datasets.items():
        static = static_coherence(data)
        rank, triangles = label_hodgerank(data, min_count=100 if name.startswith("ultra") else 20)
        triangles.to_csv(out / f"triangles_{name}.csv", index=False)
        dyn = restricted_dynamics(data, GRID, sample=8000)
        perm = np.argsort(np.random.default_rng(7).random(data["L"].shape), axis=1)
        shuffled = restricted_dynamics(data, GRID, sample=8000, L=np.take_along_axis(data["L"], perm, axis=1))
        dyn["label_shuffled"] = shuffled["label"]
        dyn["rate_shuffled"] = shuffled["rate"]
        dyn.to_csv(out / f"restricted_{name}.csv", index=False)
        summary[name] = dict(static=static, hodgerank=rank)
        print(f"\n### {name}\n", json.dumps(static, indent=1), "\n", json.dumps(rank, indent=1)[:900])
        print(triangles.round(3).to_string(index=False))
        print(dyn.round(5).to_string(index=False))
    uf = datasets["ultrafeedback_by_model"]
    by_source = {}
    for source in sorted(set(uf["source"])):
        mask = uf["source"] == source
        sub = dict(uf, L=uf["L"][mask], A=uf["A"][mask], U=uf["U"][mask], C=uf["C"][mask])
        by_source[source] = static_coherence(sub, draws=200)
    summary["ultrafeedback_by_source"] = by_source
    print("\n### ultrafeedback by source\n", pd.DataFrame(by_source).T[["panels", "omega_tilde", "omega_prompt_energy_rms",
                                                                         "coherence_ratio", "null_p95", "p_value"]])
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


# Transitive surrogates ------------------------------------------------------------

def _with_flows(data, flows):
    parts = [hodge(a) for a in flows]
    return dict(data, A=np.array(flows), U=np.array([h.u for h in parts]), C=np.array([h.C for h in parts]))


def surrogate_comparison(root, draws=200, seed=0):
    """Is the coherent residual reproduced by a perfectly transitive surrogate?

    For rating panels the surrogate keeps each panel's attribute-mean order but gives every
    attribute the same vote (saturated, transitive labels); the soft alternative is the
    per-attribute Bradley--Terry mixture. For vote panels the surrogate keeps the observed
    number of votes on every observed pair and awards them all to the higher BT score.
    """
    from .constructions import attribute_bt
    from .core import fit_bt

    rows = []

    def record(name, construction, data):
        s = static_coherence(data, draws=draws, seed=seed)
        rows.append(dict(dataset=name, construction=construction, omega_tilde=s["omega_tilde"],
                         omega_prompt_rms=s["omega_prompt_energy_rms"], coherence_ratio=s["coherence_ratio"],
                         null_p95=s["null_p95"], p_value=s["p_value"]))

    panels = [p for p in LOADERS["ultrafeedback"](root) if p.K == 4 and len(set(p.labels)) == 4]
    uf = ultrafeedback_by_model(root)
    record("ultrafeedback_by_model", "aspect votes, logit (observed)", uf)
    record("ultrafeedback_by_model", "aspect BT mixture, logit (soft)",
           _with_flows(uf, [flow(attribute_bt(p.ratings), "logit") for p in panels]))
    means = [np.nanmean(p.ratings, axis=1, keepdims=True) for p in panels]
    record("ultrafeedback_by_model", "transitive surrogate: mean score as 4 identical votes, logit",
           _with_flows(uf, [flow(attribute_vote(np.repeat(m, 4, axis=1), pseudo=0.5), "logit") for m in means]))
    for construction, label in (("votes", "attribute votes, logit (observed)"), ("bt", "attribute BT mixture, logit (soft)"),
                                ("surrogate", "transitive surrogate: mean score as 5 identical votes, logit")):
        record("helpsteer_by_length_rank", label, helpsteer_by_length_rank(root, construction))
    for split in ("human", "gpt4"):
        data = mt_bench_by_model(root, split)
        name = f"mt_bench_{split}_by_model"
        record(name, "votes, logit (observed)", data)
        loader = LOADERS["mt_bench_human" if split == "human" else "mt_bench_gpt4"](root)
        flows = []
        for p in loader:
            s = fit_bt(p.wins)
            N = p.wins + p.wins.T
            W = np.where(s[:, None] > s[None, :] + 1e-9, N, np.where(np.abs(s[:, None] - s[None, :]) <= 1e-9, N / 2, 0.0))
            np.fill_diagonal(W, 0.0)
            flows.append(flow(vote_matrix(W, 0.5), "logit"))
        record(name, "transitive surrogate: same votes to the BT winner, logit", _with_flows(data, flows))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    if "--surrogates" in sys.argv:
        table = surrogate_comparison(ROOT / "data")
        table.to_csv(ROOT / "outputs" / "coherence" / "surrogates.csv", index=False)
        print(table.round(4).to_string(index=False))
    else:
        main()
