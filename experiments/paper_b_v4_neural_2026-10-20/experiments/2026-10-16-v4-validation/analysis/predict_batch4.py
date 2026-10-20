"""Frozen predictions for batch 4 (Qwen2.5-1.5B-Instruct replication) on a 64-panel subset of panels_q15
(first 32 balanced and first 32 decisive panel ids of oracle_planted_q15.npz), before any batch-4 run.
R1: IPO soft, same target B=8 (alpha 0 / beta 8 and alpha 0.8 / beta 1.6), 30 rounds.
R2: DPO hard vs soft (expected labels), alpha 0.8, beta 1, lambda 0.8, 30 rounds.
R3: planted cycle, IPO, B=12, alpha 0.2 (beta 9.6) vs 0.9 (beta 1.2), lambda 1, 60 rounds.
Usage: python predict_batch4.py <panels_q15.json> <oracle_planted_q15.npz> <out.json>"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from population import entropy, hodge, link, omega_max, sm, target, trajectory  # noqa: E402
D = json.load(open(sys.argv[1])); Z = np.load(sys.argv[2])
kind_all = Z['kind']; ids_all = Z['ids']
sel = np.sort(np.concatenate([np.where(kind_all == 'balanced')[0][:32], np.where(kind_all == 'decisive')[0][:32]]))
ids = [int(ids_all[i]) for i in sel]; pid = {p['id']: p for p in D['panels']}
s0 = np.array([pid[i]['s0'] for i in ids]); kind = kind_all[sel]
RM = 'Skywork-Reward-Llama-3.1-8B-v0.2-d4117fbfd81b'
R = np.array([pid[i]['rm'][RM] for i in ids]); K = R.shape[1]; iu = np.triu_indices(K, 1)
gaps = R[:, :, None] - R[:, None, :]; T = 0.5 * float(np.std(gaps[:, iu[0], iu[1]]))
Psoft = 1 / (1 + np.exp(-gaps / T)); Ppl = Z['P'][sel]
out = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), ids=ids, T_soft=T, criteria={
    'R1': 'final KL to the B=8 target <= 0.02 for both runs; pairwise KL between final policies <= 0.02',
    'R2': 'soft-label DPO final entropy within 0.05 of target; hard-label DPO entropy < 0.05 by t=5 and top-two log-odds gap increasing from t=5 to t=30',
    'R3': 'alpha=0.2: median probability amplitude of balanced panels (last 30 rounds, detrended) within +-25% of the population value and >= 2x the decisive median; alpha=0.9: balanced median <= 0.10'})
X8, _ = target(link(Psoft, 'identity'), s0, 8.0, 0.8); out['R1_target_entropy'] = float(entropy(X8).mean())
X5, _ = target(link(Psoft, 'logit'), s0, 5.0, 0.8); out['R2_soft_target_entropy'] = float(entropy(X5).mean())
def pamp(traj, T_=30):
    t = np.arange(T_) - (T_ - 1) / 2; x = sm(traj[-T_:]); sl = (t[:, None, None] * (x - x.mean(0))).sum(0) / (t ** 2).sum()
    return (x - x.mean(0) - sl[None] * t[:, None, None]).std(0).max(-1)
A = Ppl - 0.5
for name, (a, b) in {'R3_a02': (0.2, 9.6), 'R3_a09': (0.9, 1.2)}.items():
    X, _ = target(A, s0, 12.0, 1.0)
    gB = np.array([12.0 * omega_max(hodge(A[n])[1], sm(X[n])) for n in range(len(ids))])
    rad = np.sqrt(a ** 2 + ((1 - a) * gB) ** 2)
    amp = pamp(trajectory(A, s0, a, b, 1.0, 60))
    out[name] = {k: dict(gammaB_median=float(np.median(gB[kind == k])), share_unstable=float((rad[kind == k] > 1).mean()), pop_amp_median=float(np.median(amp[kind == k]))) for k in ('balanced', 'decisive')}
json.dump(out, open(sys.argv[3], 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'ids'}, indent=1))
