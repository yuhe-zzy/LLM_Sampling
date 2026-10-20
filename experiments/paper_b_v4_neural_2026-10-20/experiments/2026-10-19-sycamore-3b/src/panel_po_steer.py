"""Copy for the Sycamore 3B dynamics runs (2026-10-19): identical to experiments/2026-10-19-x14-steer/vm/panel_po_steer.py except that
the micro-batch size CH can be set with the environment variable PANEL_CH, and the peak GPU memory is printed at the end.

X14 (2026-10-19): panel_po_ctrl.py plus two P-free steering rules. The controller reads only the policy's own panel scores and
sampled judgments; it never reads P.
  --d5_round R --d5_n n --d5_action debias|comp   at the start of round R+1, judge every pair of every panel n times at s_0 and
        at s_R (Bernoulli draws from the run's label source), estimate the judge bias eps_hat by the conditional likelihood
        (Fisher's noncentral hypergeometric law given each pair's total wins), and act only if |eps_hat| > 1.96 SE:
        debias: labels' log-odds get - eps_hat (s_i - s_j) of the current policy from then on;
        comp:   alpha <- alpha - beta*eps_hat (if >= 0), else alpha <- 0 and beta <- alpha_planned / eps_hat.
  --switch_entropy H   full refresh until the mean panel entropy of the current policy first falls to H or below; then
        every panel switches to the anchor alpha = 1 - beta / B_t, with B_t = beta * (rounds completed).
Decisions are written to steer.json in the run directory.

Original description (2026-10-17): theory-matched iterative preference optimization on fixed panels, with feedback controls (extends the v4 validation script).

Controls (all off by default, so the defaults reproduce panel_po.py):
  --judge_bias eps     labels' log-odds get + eps*(s_i - s_j) of the CURRENT policy (on-policy judge bias; eps < 0 = popularity penalty)
  --judge_frozen       the bias uses the initial scores s_0 instead (frozen judge)
  --beta_ctrl drift    per-panel step beta_n = kappa * max(D*_n - KL(pi_t || pi_0), 0), D*_n from --drift_targets npz (ids, D)
  --beta_ctrl perround global beta adapted so that the mean KL(pi_{t+1} || pi_t) stays near --kl_round
  --replay rho         sampler uses m_t = (1-rho) m_{t-1} + rho pi_t (replay of past policies)
  --adaptive_W W       adaptive refresh at fixed target --B: every W rounds, if the mean policy step did not fall by 20%,
                       alpha <- alpha + 0.3 (1 - alpha) (cap 0.95), beta = (1 - alpha) B
  --wavg_from r        keep the running average of the weights after rounds r..T and score the panels with it at the end

Original description (2026-10-16):

Each outer round t:
  s_t(i)   = exact sequence log-probability of panel response i under the current policy (no grad)
  pi_t     = softmax over the panel of s_t;  mu_t = (1-lam) * coverage + lam * pi_t   (comparison sampler)
  r_t(i)   = (alpha - nu) * s_t(i) + nu * s_{t-1}(i) + (1-alpha) * s_ref(i)            (geometric reference)
  inner    : minimize over theta, for S Adam steps from a fresh optimizer,
             sum_x sum_{i != j} mu_i mu_j loss(h_i - h_j; P*_ij) / sum mu_i mu_j,   h_i = s_theta(i) - r_t(i)
             IPO: (h_i - h_j - beta * (P*_ij - 1/2))^2           (identity link; exact population IPO round)
             DPO: -P* log sig((h_i-h_j)/beta) - (1-P*) log sig((h_j-h_i)/beta)   (beta = 1/beta_DPO)
Labels enter as expectations (soft targets), so a round is the population update up to optimization error.
Everything needed to compare with the exact population recursion is saved: s_t for every round, P*, s_ref.

Usage: python panel_po.py --panels panels.json --policy <dir> --out run_dir --objective ipo --oracle soft
       --alpha 0.9 --beta 1 --lam 1 --rounds 80 --inner 40 --lr 1e-5
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument('--panels', required=True)
ap.add_argument('--policy', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--objective', choices=['ipo', 'dpo'], default='ipo')
ap.add_argument('--oracle', default='soft', help='soft | hard | jury | npz:<path with P[N,K,K]>')
ap.add_argument('--rm', default='Skywork-Reward-Llama-3.1-8B-v0.2-d4117fbfd81b', help='reward model for soft/hard oracles')
ap.add_argument('--temp', type=float, default=0.0, help='soft-label temperature; 0 = per-run std of within-panel reward gaps')
ap.add_argument('--alpha', type=float, required=True)
ap.add_argument('--beta', type=float, required=True)
ap.add_argument('--lam', type=float, default=1.0)
ap.add_argument('--nu', type=float, default=0.0, help='lagged reference weight on s_{t-1}')
ap.add_argument('--coverage', choices=['uniform', 'ref'], default='uniform')
ap.add_argument('--form', choices=['pairwise', 'payoff'], default='pairwise',
                help='pairwise: IPO/DPO loss with pair weights mu_i mu_j (the sampled-pair objective); payoff (IPO only): uniform pair weights and target beta*((A mu)_i-(A mu)_j), the same population optimum without vanishing weights')
ap.add_argument('--rounds', type=int, default=80)
ap.add_argument('--inner', type=int, default=40)
ap.add_argument('--lr', type=float, default=1e-5)
ap.add_argument('--n_prompts', type=int, default=0)
ap.add_argument('--sampled', type=int, default=0, help='if > 0: each round draws this many pairs per prompt from mu x mu (i != j) with labels ~ Bernoulli(P*), as in practice')
ap.add_argument('--ids', default='', help='comma-separated panel ids to use')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--dtype', default='fp32')
ap.add_argument('--judge_bias', type=float, default=0.0)
ap.add_argument('--judge_frozen', action='store_true')
ap.add_argument('--beta_ctrl', choices=['none', 'drift', 'perround', 'switch'], default='none')
ap.add_argument('--alpha_anchor', type=float, default=0.875, help='switch: refresh weight a panel switches to once its drift reaches D*_n')
ap.add_argument('--kappa', type=float, default=2.0)
ap.add_argument('--drift_targets', default='')
ap.add_argument('--kl_round', type=float, default=0.02)
ap.add_argument('--replay', type=float, default=1.0)
ap.add_argument('--adaptive_W', type=int, default=0)
ap.add_argument('--B', type=float, default=0.0)
ap.add_argument('--wavg_from', type=int, default=0)
ap.add_argument('--d5_round', type=int, default=0)
ap.add_argument('--d5_n', type=int, default=16)
ap.add_argument('--d5_action', choices=['debias', 'comp', 'none'], default='none')
ap.add_argument('--d5_seed', type=int, default=5)
ap.add_argument('--switch_entropy', type=float, default=0.0)
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
os.makedirs(args.out, exist_ok=True)
dev = 'cuda'

D = json.load(open(args.panels))
panels = D['panels']
if args.ids:
    keep = set(int(x) for x in args.ids.split(','))
    panels = [p for p in panels if p['id'] in keep]
if args.n_prompts:
    panels = panels[:args.n_prompts]
N, K = len(panels), len(panels[0]['responses'])

# ---- oracle P*[N,K,K]
if args.oracle.startswith('npz:'):
    Z = np.load(args.oracle[4:])
    idmap = {int(i): n for n, i in enumerate(Z['ids'])}
    P = np.stack([Z['P'][idmap[p['id']]] for p in panels])
else:
    R = np.array([p['rm'][args.rm] for p in panels])                       # [N,K]
    gaps = (R[:, :, None] - R[:, None, :])
    if args.oracle == 'soft':
        T = args.temp or float(np.std(gaps[:, np.triu_indices(K, 1)[0], np.triu_indices(K, 1)[1]]))
        P = 1 / (1 + np.exp(-gaps / T))
        args.temp = T
    elif args.oracle == 'hard':
        P = (gaps > 0).astype(float) + 0.5 * (gaps == 0)
    elif args.oracle == 'jury':
        names = sorted(panels[0]['rm'])
        votes = [(np.array([p['rm'][n] for p in panels])[:, :, None] - np.array([p['rm'][n] for p in panels])[:, None, :]) for n in names]
        P = np.mean([(v > 0) + 0.5 * (v == 0) for v in votes], axis=0)
    else:
        raise SystemExit('unknown oracle')
P = P.astype(np.float64)
for n in range(N):
    np.fill_diagonal(P[n], 0.5)

tok = AutoTokenizer.from_pretrained(args.policy)
dtype = torch.float32 if args.dtype == 'fp32' else torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(args.policy, dtype=dtype).to(dev)
eos = '<|im_end|>'

# ---- tokenize panels once: input ids and the mask of response tokens (incl. EOS)
seqs, masks = [], []
for p in panels:
    pid = tok(tok.apply_chat_template([{'role': 'user', 'content': p['prompt_full']}], add_generation_prompt=True, tokenize=False), add_special_tokens=False)['input_ids']
    for r in p['responses']:
        rid = tok(r + eos, add_special_tokens=False)['input_ids']
        seqs.append(pid + rid); masks.append([0] * len(pid) + [1] * len(rid))
L = max(len(s) for s in seqs)
inp = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
att = torch.zeros((len(seqs), L), dtype=torch.long)
rmask = torch.zeros((len(seqs), L), dtype=torch.float32)
for i, (s, m) in enumerate(zip(seqs, masks)):
    inp[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1; rmask[i, :len(m)] = torch.tensor(m, dtype=torch.float32)
inp, att, rmask = inp.to(dev), att.to(dev), rmask.to(dev)
CH = int(os.environ.get('PANEL_CH', '128'))              # rows per micro-batch (Sycamore 3B runs may lower it for memory)
# positions whose next token is a response token: hidden state at l-1 predicts token l
tgt_mask = rmask[:, 1:] > 0
seq_of_pos = torch.arange(inp.shape[0], device=dev)[:, None].expand(-1, L - 1)


def seqlogp_rows(rows, grad=False):
    """Exact response log-probabilities for the given sequence rows; LM head only at response positions."""
    with torch.set_grad_enabled(grad), torch.autocast('cuda', dtype=torch.bfloat16):
        hid = model.model(input_ids=inp[rows], attention_mask=att[rows]).last_hidden_state[:, :-1]
        m = tgt_mask[rows]
        logits = model.lm_head(hid[m])
    lp = torch.log_softmax(logits.float(), -1).gather(-1, inp[rows, 1:][m][:, None])[:, 0]
    pos = torch.arange(len(rows), device=dev)[:, None].expand(-1, L - 1)[m]
    return torch.zeros(len(rows), device=dev, dtype=lp.dtype).index_add(0, pos, lp)


def seqlogp(grad=False, prompts=None):
    prompts = torch.arange(N, device=dev) if prompts is None else prompts
    rows = (prompts[:, None] * K + torch.arange(K, device=dev)[None]).reshape(-1)
    outs = [seqlogp_rows(rows[a:a + CH], grad) for a in range(0, len(rows), CH)]
    return torch.cat(outs).view(len(prompts), K)


Pt = torch.tensor(P, device=dev, dtype=torch.float32)
At = Pt - 0.5
s_ref = seqlogp().detach()
cov = torch.full((N, K), 1.0 / K, device=dev) if args.coverage == 'uniform' else torch.softmax(s_ref, -1)
s_prev = s_ref.clone()
hist = dict(s=[s_ref.cpu().numpy()], loss_first=[], loss_last=[], secs=[], betas=[], alphas=[])
logitP = torch.log(Pt.clamp(1e-6, 1 - 1e-6)) - torch.log1p(-Pt.clamp(1e-6, 1 - 1e-6))
alpha_t = args.alpha; steps_hist = []
bvec = torch.full((N,), args.beta, device=dev)
if args.adaptive_W:
    assert args.B > 0; bvec[:] = (1 - alpha_t) * args.B
avec = torch.full((N,), args.alpha, device=dev)
if args.beta_ctrl in ('drift', 'switch'):
    Zd = np.load(args.drift_targets); dmap = {int(i): float(d) for i, d in zip(Zd['ids'], Zd['D'])}
    Dstar = torch.tensor([dmap[p['id']] for p in panels], device=dev)
m_rep = torch.softmax(s_ref, -1)
pi_ref = torch.softmax(s_ref, -1)
wsum, wcount = None, 0
steer = dict(d5=None, switch=None); bias_corr = 0.0; switched = False


def d5_estimate(s0, s1, n, seed):
    """Judge every pair n times at s0 and s1 (the run's own label source); conditional-likelihood eps_hat and its SE."""
    rng = np.random.default_rng(seed); iu = np.triu_indices(K, 1); lp = logitP.cpu().numpy().astype(np.float64)
    def labels(s):
        sb = s0 if args.judge_frozen else s
        pe = 1 / (1 + np.exp(-(lp + args.judge_bias * (sb[:, :, None] - sb[:, None, :]))))
        return rng.binomial(n, pe[:, iu[0], iu[1]]).ravel()
    k0, k1 = labels(s0), labels(s1)
    dx = ((s1[:, iu[0]] - s1[:, iu[1]]) - (s0[:, iu[0]] - s0[:, iu[1]])).ravel(); S = k0 + k1
    keep = (S > 0) & (S < 2 * n) & (np.abs(dx) > 1e-9); dx, k1, S = dx[keep], k1[keep], S[keep]
    lc = np.array([math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) for k in range(n + 1)])
    ks = np.arange(n + 1)[None, :]; ok = (ks >= np.maximum(0, S - n)[:, None]) & (ks <= np.minimum(n, S)[:, None])
    lb = np.where(ok, lc[ks.clip(0, n)] + lc[(S[:, None] - ks).clip(0, n)], -np.inf)
    e = 0.0; h = 1.0
    for _ in range(100):
        z = lb + e * dx[:, None] * ks; z = z - z.max(1, keepdims=True); w = np.exp(z); w /= w.sum(1, keepdims=True)
        m1 = (w * ks).sum(1); m2 = (w * ks * ks).sum(1); g = (dx * (k1 - m1)).sum(); h = (dx * dx * (m2 - m1 * m1)).sum()
        st = float(np.clip(g / max(h, 1e-12), -1, 1)); e += st
        if abs(st) < 1e-10:
            break
    return float(e), float(1 / math.sqrt(max(h, 1e-12))), int(2 * n * s0.shape[0] * len(iu[0]))
offdiag = (1 - torch.eye(K, device=dev))[None]
t0 = time.time()
for t in range(1, args.rounds + 1):
    s_t = seqlogp().detach()
    pi_t = torch.softmax(s_t, -1)
    if args.d5_round and t == args.d5_round + 1:            # D5 on the scores before round 1 and after round d5_round
        eh, se, calls = d5_estimate(hist['s'][0].astype(np.float64), hist['s'][args.d5_round].astype(np.float64), args.d5_n, args.d5_seed)
        act = abs(eh) > 1.96 * se; steer['d5'] = dict(round=t, eps_hat=eh, se=se, calls=calls, acted=bool(act), action=args.d5_action)
        if act and args.d5_action == 'debias':
            bias_corr = eh
        elif act and args.d5_action == 'comp':
            a_new = args.alpha - args.beta * eh
            if a_new >= 0:
                alpha_t = a_new
            else:
                alpha_t = 0.0; bvec[:] = args.alpha / eh
        steer['d5'].update(alpha=alpha_t, beta=float(bvec[0]), bias_correction=bias_corr)
        print('D5', json.dumps(steer['d5']), flush=True)
    if args.switch_entropy and not switched:
        ent_t = float((-(pi_t * torch.log(pi_t + 1e-30)).sum(-1)).mean())
        if ent_t <= args.switch_entropy and t > 1:
            switched = True; B_sw = args.beta * (t - 1); alpha_t = 1 - args.beta / B_sw
            steer['switch'] = dict(round=t, mean_entropy=ent_t, B=B_sw, alpha=alpha_t); print('SWITCH', json.dumps(steer['switch']), flush=True)
    m_rep = (1 - args.replay) * m_rep + args.replay * pi_t
    mu = (1 - args.lam) * cov + args.lam * m_rep
    if args.beta_ctrl == 'drift':
        drift = (pi_t * (torch.log(pi_t + 1e-30) - torch.log(pi_ref + 1e-30))).sum(-1)
        bvec = args.kappa * (Dstar - drift).clamp(min=0.0)
    if args.beta_ctrl == 'switch':                            # refresh until the drift budget is used, then anchor (latched)
        drift = (pi_t * (torch.log(pi_t + 1e-30) - torch.log(pi_ref + 1e-30))).sum(-1)
        avec = torch.where(drift >= Dstar, torch.full_like(avec, args.alpha_anchor), avec)
        r_t = avec[:, None] * s_t + (1 - avec[:, None]) * s_ref
    else:
        r_t = (alpha_t - args.nu) * s_t + args.nu * s_prev + (1 - alpha_t) * s_ref
    sb = s_ref if args.judge_frozen else s_t
    if args.judge_bias != 0.0:      # policy-dependent label bias in log-odds: P_eff = sigma(logit P* + eps (s_i - s_j)), minus any D5 correction
        Peff = torch.sigmoid(logitP + args.judge_bias * (sb[:, :, None] - sb[:, None, :]) - bias_corr * (s_t[:, :, None] - s_t[:, None, :]))
        Peff = Peff * offdiag + 0.5 * torch.eye(K, device=dev)[None]
    else:
        Peff = Pt
    Aeff = Peff - 0.5
    active = bvec > 1e-6                                     # panels with zero step are left out of the loss
    bsafe = torch.where(active, bvec, torch.ones_like(bvec))
    W = (mu[:, :, None] * mu[:, None, :] if args.form == 'pairwise' else torch.ones_like(mu)[:, :, None] * torch.ones_like(mu)[:, None, :]) * offdiag
    if args.sampled:
        # practice-like data: n pairs per prompt drawn from mu x mu restricted to i != j, labels ~ Bernoulli(P*);
        # counts[n, i, j] = number of sampled pairs (i, j) whose label says i is preferred
        probs = (W / W.sum((1, 2), keepdim=True)).reshape(N, K * K)
        draws = torch.multinomial(probs, args.sampled, replacement=True)
        ii, jj = draws // K, draws % K
        y = (torch.rand(draws.shape, device=dev) < Pt[torch.arange(N, device=dev)[:, None], ii, jj]).float()
        win_i = torch.where(y > 0, ii, jj); win_j = torch.where(y > 0, jj, ii)
        counts = torch.zeros(N, K, K, device=dev)
        counts.index_put_((torch.arange(N, device=dev)[:, None].expand_as(win_i), win_i, win_j), torch.ones_like(y), accumulate=True)
        Wemp = counts / args.sampled
    g = (Aeff * mu[:, None, :]).sum(-1)                     # payoff vector (A mu)_i
    Tpay = g[:, :, None] - g[:, None, :]
    W = W / W.sum((1, 2), keepdim=True)
    W = W * active[:, None, None].float()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    PC = max(1, CH // K)                                   # prompts per micro-batch
    for k in range(args.inner):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for a in range(0, N, PC):
            idx = torch.arange(a, min(a + PC, N), device=dev)
            h = seqlogp(grad=True, prompts=idx) - r_t[idx]
            d = h[:, :, None] - h[:, None, :]
            if args.sampled:   # empirical loss on sampled pairs with hard labels (winner i over loser j)
                if args.objective == 'ipo':
                    loss_mat = (d - args.beta * 0.5) ** 2
                else:
                    loss_mat = -torch.nn.functional.logsigmoid(d / args.beta)
                lchunk = (Wemp[idx] * loss_mat).sum((1, 2)).sum() / N
            else:
                bb = bsafe[idx][:, None, None]
                if args.objective == 'ipo':
                    loss_mat = (d - bb * (Aeff[idx] if args.form == 'pairwise' else Tpay[idx])) ** 2
                else:
                    z = d / bb
                    loss_mat = -(Peff[idx] * torch.nn.functional.logsigmoid(z) + (1 - Peff[idx]) * torch.nn.functional.logsigmoid(-z))
                lchunk = (W[idx] * loss_mat).sum((1, 2)).sum() / N
            lchunk.backward()
            tot += lchunk.item()
        opt.step()
        loss = torch.tensor(tot)
        if k == 0:
            hist['loss_first'].append(tot)
    hist['loss_last'].append(loss.item())
    s_prev = s_t
    s_new = seqlogp().detach()
    hist['s'].append(s_new.cpu().numpy()); hist['secs'].append(time.time() - t0)
    hist['betas'].append(bvec.cpu().numpy().copy()); hist['alphas'].append(avec.cpu().numpy().copy() if args.beta_ctrl == 'switch' else alpha_t)
    pin_new = torch.softmax(s_new, -1)
    if args.beta_ctrl == 'perround':                          # PPO-style controller on the per-round KL
        klr = float((pin_new * (torch.log(pin_new + 1e-30) - torch.log(pi_t + 1e-30))).sum(-1).mean())
        fac = 1.5 if klr < args.kl_round / 1.5 else (1 / 1.5 if klr > args.kl_round * 1.5 else 1.0)
        bvec = (bvec * fac).clamp(1e-3, 8.0)
    if args.adaptive_W:
        steps_hist.append(float((pin_new - pi_t).abs().max(-1).values.mean()))
        if t % args.adaptive_W == 0 and t >= 2 * args.adaptive_W:
            W_ = args.adaptive_W
            if np.mean(steps_hist[-W_:]) > 0.8 * np.mean(steps_hist[-2 * W_:-W_]):
                alpha_t = min(0.95, alpha_t + 0.3 * (1 - alpha_t))
                bvec[:] = (1 - alpha_t) * args.B
    if args.wavg_from and t >= args.wavg_from:
        with torch.no_grad():
            if wsum is None:
                wsum = {k_: v.detach().float().clone() for k_, v in model.state_dict().items()}
            else:
                for k_, v in model.state_dict().items():
                    wsum[k_] += v.detach().float()
        wcount += 1
    if t % 5 == 0 or t == 1:
        pin = torch.softmax(s_new, -1)
        ent = float((-(pin * torch.log(pin + 1e-30)).sum(-1)).mean())
        print(f't={t} loss {hist["loss_first"][-1]:.4g}->{loss.item():.4g} entropy {ent:.4f} top {pin.max(-1).values.mean():.3f} '
              f'{time.time() - t0:.0f}s', flush=True)
    del opt; torch.cuda.empty_cache()

extra = {}
if wsum is not None:                                     # score the panels with the average of the weights over rounds wavg_from..T
    sd = model.state_dict()
    with torch.no_grad():
        for k_ in sd:
            sd[k_].copy_((wsum[k_] / wcount).to(sd[k_].dtype))
    extra['s_wavg'] = seqlogp().detach().cpu().numpy(); extra['wavg_count'] = np.array(wcount)
np.savez_compressed(os.path.join(args.out, 'run.npz'), s=np.stack(hist['s']), P=P, ids=np.array([p['id'] for p in panels]),
                    loss_first=np.array(hist['loss_first']), loss_last=np.array(hist['loss_last']), secs=np.array(hist['secs']),
                    betas=np.array(hist['betas']), alphas=np.array(hist['alphas']), **extra)
json.dump(vars(args), open(os.path.join(args.out, 'config.json'), 'w'), indent=1)
json.dump(steer, open(os.path.join(args.out, 'steer.json'), 'w'), indent=1)
print('done', args.out, f'{time.time() - t0:.0f}s', f'peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB')
