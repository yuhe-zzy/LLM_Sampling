"""X15 (2026-10-19): specialist panels. For each prompt the base policy samples M short answers (as in v4's build_panels.py);
eligible candidates are distinct complete answers with normalized probability >= --min_prob among the prompt's distinct answers.
Three transitive criteria score every candidate: Q = Skywork-8B reward (scored here), C = -characters, N = -s0.
A K=3 panel is a triple (a_Q, a_C, a_N) that the three criteria rank in rotation (Latin square); a K=4 panel adds an eligible
answer that all three criteria rank below the triple. Outputs panels3.json and panels4.json with the v4 panel fields.
Usage: python build_specialist_panels.py --policy <dir> --rms <skywork> [<others>...] --out_dir <dir> [--M 48 --min_prob 0.05]"""
import argparse, glob, itertools, json, math, os, sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

SUFFIX = " Answer with a single short phrase of at most six words, and nothing else."
ap = argparse.ArgumentParser()
ap.add_argument('--policy', required=True)
ap.add_argument('--data', default=os.path.expanduser('~/v4val/data/novelty-bench'))
ap.add_argument('--out_dir', required=True)
ap.add_argument('--M', type=int, default=48)
ap.add_argument('--max_new', type=int, default=20)
ap.add_argument('--min_prob', type=float, default=0.05)
ap.add_argument('--max_prompts', type=int, default=600)
ap.add_argument('--rms', nargs='+', required=True, help='first = Skywork-8B (criterion Q); others scored for the record')
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()
torch.manual_seed(args.seed); dev = 'cuda'; os.makedirs(args.out_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(args.data, 'data', '*.parquet'))); prompts = []
for f in files:
    df = pd.read_parquet(f); col = 'prompt' if 'prompt' in df.columns else df.columns[0]; src = os.path.basename(f).split('-')[0]
    for p in df[col].tolist():
        if isinstance(p, str) and 10 <= len(p) <= 400: prompts.append((src, p.strip()))
seen = set(); uniq = []
for s, p in prompts:
    if p not in seen: seen.add(p); uniq.append((s, p))
sys.path.insert(0, os.path.expanduser('~/v4val'))
from prompts_short import short_prompts  # noqa: E402
uniq = [('short', q) for q in short_prompts()] + [x for x in uniq if x[0] == 'curated'] + [x for x in uniq if x[0] != 'curated']
prompts = uniq[:args.max_prompts]
print('prompts', len(prompts), flush=True)

tok = AutoTokenizer.from_pretrained(args.policy); tok.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(args.policy, dtype=torch.bfloat16).to(dev).eval()
eos_id = tok.convert_tokens_to_ids('<|im_end|>')


def chat_ids(p):
    text = tok.apply_chat_template([{'role': 'user', 'content': p + SUFFIX}], add_generation_prompt=True, tokenize=False)
    return tok(text, add_special_tokens=False)['input_ids']


@torch.no_grad()
def seq_logprob(prompt_ids, resp_ids_list):
    out = []
    for b in range(0, len(resp_ids_list), 16):
        chunk = resp_ids_list[b:b + 16]; seqs = [prompt_ids + r for r in chunk]; L = max(len(s) for s in seqs)
        inp = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long); att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs): inp[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
        inp, att = inp.to(dev), att.to(dev)
        lp = torch.log_softmax(model(input_ids=inp, attention_mask=att).logits.float()[:, :-1], -1).gather(-1, inp[:, 1:, None])[..., 0]
        n0 = len(prompt_ids)
        out += [lp[i, n0 - 1:n0 - 1 + len(r)].sum().item() for i, r in enumerate(chunk)]
    return out


# ---- stage A: sample and score candidates by s0
cand_all = []   # per prompt: dict(src, prompt, texts, s0, eligible mask)
for pi, (src, p) in enumerate(prompts):
    pid = chat_ids(p); inp = torch.tensor([pid], device=dev)
    with torch.no_grad():
        gen = model.generate(inp, attention_mask=torch.ones_like(inp), do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                             max_new_tokens=args.max_new, num_return_sequences=args.M, pad_token_id=tok.pad_token_id)
    cands = {}
    for g in gen[:, len(pid):].tolist():
        if eos_id not in g: continue
        r = g[:g.index(eos_id) + 1]; text = tok.decode(r[:-1]).strip()
        if not text or len(text.split()) > 8: continue
        key = text.lower().strip(' .!"\'')
        if key not in cands: cands[key] = (text, r)
    if len(cands) < 3: continue
    texts = [v[0] for v in cands.values()]; rids = [v[1] for v in cands.values()]
    lps = seq_logprob(pid, rids); m = max(lps); z = sum(math.exp(v - m) for v in lps); probs = [math.exp(v - m) / z for v in lps]
    elig = [i for i in range(len(texts)) if probs[i] >= args.min_prob]
    if len(elig) < 3: continue
    cand_all.append(dict(source=src, prompt=p, texts=[texts[i] for i in elig], s0=[lps[i] for i in elig], p_all=[probs[i] for i in elig]))
    if pi % 50 == 0: print(pi, len(cand_all), flush=True)
print('prompts with >=3 eligible candidates', len(cand_all), flush=True)
del model; torch.cuda.empty_cache()

# ---- stage B: reward scores on eligible candidates
rm_names = []
for rm_dir in args.rms:
    name = os.path.basename(rm_dir.rstrip('/')); rm_names.append(name)
    rtok = AutoTokenizer.from_pretrained(rm_dir)
    rm = AutoModelForSequenceClassification.from_pretrained(rm_dir, dtype=torch.bfloat16, num_labels=1).to(dev).eval()
    if rtok.pad_token is None: rtok.pad_token = rtok.eos_token
    rm.config.pad_token_id = rtok.pad_token_id
    for P in cand_all:
        sc = []
        for r in P['texts']:
            c = rtok.apply_chat_template([{'role': 'user', 'content': P['prompt'] + SUFFIX}, {'role': 'assistant', 'content': r}], tokenize=False)
            enc = rtok(c, return_tensors='pt', add_special_tokens=False).to(dev)
            with torch.no_grad(): sc.append(rm(**enc).logits[0, 0].float().item())
        P.setdefault('rm', {})[name] = sc
    print('scored with', name, flush=True); del rm; torch.cuda.empty_cache()
json.dump(dict(policy=args.policy, suffix=SUFFIX, args=vars(args), rms=args.rms, prompts=cand_all),
          open(os.path.join(args.out_dir, 'candidates.json'), 'w'), indent=1)

# ---- stage C: specialist selection
Qn = rm_names[0]
def crit(P):
    return dict(Q=np.array(P['rm'][Qn]), C=-np.array([len(t) for t in P['texts']], float), N=-np.array(P['s0']))
# provisional gap scales from all eligible candidates
gaps = {k: [] for k in 'QCN'}
for P in cand_all:
    c = crit(P)
    for k in 'QCN':
        v = c[k]; gaps[k] += [abs(v[i] - v[j]) for i in range(len(v)) for j in range(i + 1, len(v))]
T = {k: float(np.std(gaps[k])) if np.std(gaps[k]) > 0 else 1.0 for k in 'QCN'}
print('provisional T', T, flush=True)
ROT = [('Q', 'C', 'N'), ('C', 'N', 'Q'), ('N', 'Q', 'C')]   # criterion k ranks its own answer first, then the next in the rotation


def is_rotation(c, a):   # a = (iQ, iC, iN); forward rotation Q: aQ>aC>aN, C: aC>aN>aQ, N: aN>aQ>aC
    idx = dict(Q=a[0], C=a[1], N=a[2])
    for k, (f, s, t) in zip('QCN', ROT):
        v = c[k]
        if not (v[idx[f]] > v[idx[s]] > v[idx[t]]): return False
    return True


def min_std_gap(c, a):
    idx = dict(Q=a[0], C=a[1], N=a[2]); g = []
    for k, (f, s, t) in zip('QCN', ROT):
        v = c[k]; g += [(v[idx[f]] - v[idx[s]]) / T[k], (v[idx[s]] - v[idx[t]]) / T[k]]
    return min(g)


panels3, panels4 = [], []
for P in cand_all:
    c = crit(P); n = len(P['texts']); best = None
    for a in itertools.permutations(range(n), 3):
        for mirror in (False, True):
            aa = (a[0], a[2], a[1]) if mirror else a      # mirror rotation = relabel roles
            if is_rotation(c, aa):
                g = min_std_gap(c, aa)
                if best is None or g > best[0]: best = (g, aa, mirror)
    if best is None: continue
    g, aa, mirror = best; sel = list(aa)
    s0 = [P['s0'][i] for i in sel]; m = max(s0); z = sum(math.exp(v - m) for v in s0); p0 = [math.exp(v - m) / z for v in s0]
    if min(p0) < args.min_prob: continue
    rec = dict(id=len(panels3), source=P['source'], prompt=P['prompt'], prompt_full=P['prompt'] + SUFFIX,
               responses=[P['texts'][i] for i in sel], s0=s0, p0=p0, rm={nm: [P['rm'][nm][i] for i in sel] for nm in rm_names},
               roles=['Q', 'C', 'N'], mirror=mirror, min_std_gap=g, chars=[len(P['texts'][i]) for i in sel])
    panels3.append(rec)
    # K=4: an eligible answer every criterion ranks below the triple
    losers = [i for i in range(n) if i not in sel and all(c[k][i] < min(c[k][j] for j in sel) for k in 'QCN')]
    if losers:
        l = max(losers, key=lambda i: P['s0'][i]); sel4 = sel + [l]
        s04 = [P['s0'][i] for i in sel4]; m4 = max(s04); z4 = sum(math.exp(v - m4) for v in s04); p04 = [math.exp(v - m4) / z4 for v in s04]
        if min(p04) >= args.min_prob:
            panels4.append(dict(id=len(panels4), source=P['source'], prompt=P['prompt'], prompt_full=P['prompt'] + SUFFIX,
                                responses=[P['texts'][i] for i in sel4], s0=s04, p0=p04, rm={nm: [P['rm'][nm][i] for i in sel4] for nm in rm_names},
                                roles=['Q', 'C', 'N', 'L'], mirror=mirror, min_std_gap=g, chars=[len(P['texts'][i]) for i in sel4], from_k3=rec['id']))
print('K=3 panels', len(panels3), 'K=4 panels', len(panels4), flush=True)
for K, pans in ((3, panels3), (4, panels4)):
    json.dump(dict(policy=args.policy, suffix=SUFFIX, args=vars(args), rms=args.rms, T_provisional=T, panels=pans),
              open(os.path.join(args.out_dir, f'panels{K}.json'), 'w'), indent=1)
