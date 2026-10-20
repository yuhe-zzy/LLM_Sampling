"""Theory-matched iterative preference optimization on fixed panels (JMLR-B-v4 validation, 2026-10-16).

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
CH = 128
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
hist = dict(s=[s_ref.cpu().numpy()], loss_first=[], loss_last=[], secs=[])
offdiag = (1 - torch.eye(K, device=dev))[None]
t0 = time.time()
for t in range(1, args.rounds + 1):
    s_t = seqlogp().detach()
    pi_t = torch.softmax(s_t, -1)
    mu = (1 - args.lam) * cov + args.lam * pi_t
    r_t = (args.alpha - args.nu) * s_t + args.nu * s_prev + (1 - args.alpha) * s_ref
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
    g = (At * mu[:, None, :]).sum(-1)                       # payoff vector (A mu)_i
    Tpay = g[:, :, None] - g[:, None, :]
    W = W / W.sum((1, 2), keepdim=True)
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
                if args.objective == 'ipo':
                    loss_mat = (d - args.beta * (At[idx] if args.form == 'pairwise' else Tpay[idx])) ** 2
                else:
                    z = d / args.beta
                    loss_mat = -(Pt[idx] * torch.nn.functional.logsigmoid(z) + (1 - Pt[idx]) * torch.nn.functional.logsigmoid(-z))
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
    if t % 5 == 0 or t == 1:
        pin = torch.softmax(s_new, -1)
        ent = float((-(pin * torch.log(pin + 1e-30)).sum(-1)).mean())
        print(f't={t} loss {hist["loss_first"][-1]:.4g}->{loss.item():.4g} entropy {ent:.4f} top {pin.max(-1).values.mean():.3f} '
              f'{time.time() - t0:.0f}s', flush=True)
    del opt; torch.cuda.empty_cache()

np.savez_compressed(os.path.join(args.out, 'run.npz'), s=np.stack(hist['s']), P=P, ids=np.array([p['id'] for p in panels]),
                    loss_first=np.array(hist['loss_first']), loss_last=np.array(hist['loss_last']), secs=np.array(hist['secs']))
json.dump(vars(args), open(os.path.join(args.out, 'config.json'), 'w'), indent=1)
print('done', args.out, f'{time.time() - t0:.0f}s')
