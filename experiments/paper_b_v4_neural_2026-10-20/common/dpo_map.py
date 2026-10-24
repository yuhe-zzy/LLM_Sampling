"""Population DPO round on general (possibly intransitive) expected labels, and its comparison with the logit-link MRS map.
T1: the round minimizer solves the mu-weighted Bradley-Terry moment equations sum_j mu_j sigma(phi_i - phi_j) = (P* mu)_i;
    DPO map x' = alpha x + (1-alpha) r + beta * Pi phi(mu(x)).
T1 Jacobian: D phi = L_W^+ Diag(mu) (P* - S), L_W the Laplacian with weights mu_i mu_j sigma'(phi_i - phi_j)."""
import numpy as np
sig = lambda z: 1 / (1 + np.exp(-z))
sm = lambda x: np.exp(x - x.max()) / np.exp(x - x.max()).sum()
Pi = lambda v: v - v.mean()


def bt_fit_lstsq(P, mu, iters=100):
    """Original solver (kept for reference). Its Hessian has entries mu_i mu_j; near a vertex these are tiny and the least-squares
    solve can leave the low-mass coordinates unsolved (moment residual up to 0.9 found on 2026-10-19). Use bt_fit."""
    K = len(mu); phi = np.zeros(K); W = np.outer(mu, mu); np.fill_diagonal(W, 0)
    for _ in range(iters):
        D = phi[:, None] - phi[None, :]; S = sig(D)
        g = (W * (S - P)).sum(1)                       # gradient (up to factor 2)
        H = -(W * S * (1 - S)); np.fill_diagonal(H, -H.sum(1))
        step = np.linalg.lstsq(H + 1e-15 * np.eye(K), -g, rcond=None)[0]
        phi = Pi(phi + step)
        if np.abs(g).max() < 1e-14: break
    return phi




def bt_fit(P, mu, iters=200):
    """Solve sum_j mu_j sigma(phi_i-phi_j) = (P mu)_i (the weighted BT MLE), centered phi. Fixed 2026-10-19: Newton on the equations
    divided by mu_i, g_i = sum_j mu_j (sigma(phi_i - phi_j) - P_ij); the equation of the highest-mass answer is implied by the others
    (sum_i mu_i g_i = 0) and is replaced by the gauge sum(phi) = 0. Well conditioned near vertices; same solution as before elsewhere."""
    K = len(mu); phi = np.zeros(K); mu = np.maximum(np.asarray(mu, float), 1e-300)
    for _ in range(iters):
        S = sig(phi[:, None] - phi[None, :]); np.fill_diagonal(S, .5)
        g = (mu[None, :] * (S - P)).sum(1) - mu * (0.5 - np.diag(P))
        Jm = -(mu[None, :] * S * (1 - S)); np.fill_diagonal(Jm, 0); np.fill_diagonal(Jm, -Jm.sum(1))
        keep = np.arange(K) != int(np.argmax(mu))
        step = np.linalg.solve(np.vstack([Jm[keep], np.ones(K)]), -np.concatenate([g[keep], [phi.sum()]]))
        lam = 1.0
        while lam > 1e-4 and np.abs(step * lam).max() > 20: lam /= 2
        phi = phi + lam * step
        if np.abs(g[keep]).max() < 1e-13 and np.abs(step).max() < 1e-12: break
    return Pi(phi)


def dpo_round(x, r, P, alpha, beta, lam, pi0):
    mu = (1 - lam) * pi0 + lam * sm(x)
    return Pi(alpha * x + (1 - alpha) * r + beta * bt_fit(P, mu))


def logit_round(x, r, P, alpha, beta, lam, pi0):
    A = np.log(np.clip(P, 1e-12, 1) / np.clip(1 - P, 1e-12, 1)); mu = (1 - lam) * pi0 + lam * sm(x)
    return Pi(alpha * x + (1 - alpha) * r + beta * (A @ mu))


def dphi_formula(P, mu):
    phi = bt_fit(P, mu); S = sig(phi[:, None] - phi[None, :]); W = np.outer(mu, mu) * S * (1 - S); np.fill_diagonal(W, 0)
    L = np.diag(W.sum(1)) - W
    return np.linalg.pinv(L) @ np.diag(mu) @ (P - S)


if __name__ == '__main__':
    rng = np.random.default_rng(1)
    K = 4
    # random reciprocal P* with cycles
    worst = 0
    for trial in range(20):
        U = rng.uniform(0.05, 0.95, (K, K)); P = np.triu(U, 1); P = P + np.tril(1 - P.T, -1); np.fill_diagonal(P, 0.5)
        mu = rng.dirichlet(np.ones(K))
        D = dphi_formula(P, mu)
        # finite difference along simplex directions (dmu in 1-perp)
        E = np.eye(K) - 1 / K; fd = np.zeros((K, K))
        for k in range(K):
            e = E[:, k] * 1e-6; fd[:, k] = (bt_fit(P, mu + e) - bt_fit(P, mu - e)) / 2e-6
        worst = max(worst, np.abs(Pi(D @ E.T) - np.apply_along_axis(Pi, 0, fd)).max())
        # moment equations check
        phi = bt_fit(P, mu); res = (mu[None, :] * sig(phi[:, None] - phi[None, :])).sum(1) - P @ mu
        worst = max(worst, np.abs(res).max())
    print('T1 Jacobian formula vs finite differences (and moment residual), max abs error over 20 random P*:', worst)
    # balanced 3-cycle at uniform: C_eff = 4 (p - 1/2) Ccyc
    for p in (0.6, 0.8, 0.9):
        P = np.array([[0.5, p, 1 - p], [1 - p, 0.5, p], [p, 1 - p, 0.5]]); mu = np.ones(3) / 3
        D = dphi_formula(P, mu); print(f'p={p}: D phi =\n', np.round(D, 6), '\n 4(p-1/2)=', 4 * (p - 0.5), ' logit p=', np.log(p / (1 - p)))
