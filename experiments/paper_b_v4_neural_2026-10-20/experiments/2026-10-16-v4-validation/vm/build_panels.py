"""Build fixed short-answer panels for theory-matched iterative PO (JMLR-B-v4 validation, 2026-10-16).

For each prompt (NoveltyBench curated + a WildChat subset), the policy model samples M short answers; distinct
answers that end with EOS are scored by their exact sequence log-probability; the K most probable form the panel,
kept only if every normalized initial panel probability is at least --min_prob (interior support).
Every panel response is then scored by the reward models listed in --rms.

Usage: python build_panels.py --policy <dir> --out panels.json [--K 4 --M 48 --max_new 20 --min_prob 0.05]
"""
import argparse
import glob
import json
import math
import os

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

SUFFIX = " Answer with a single short phrase of at most six words, and nothing else."

ap = argparse.ArgumentParser()
ap.add_argument('--policy', required=True)
ap.add_argument('--data', default=os.path.expanduser('~/v4val/data/novelty-bench'))
ap.add_argument('--out', required=True)
ap.add_argument('--K', type=int, default=4)
ap.add_argument('--M', type=int, default=48)
ap.add_argument('--max_new', type=int, default=20)
ap.add_argument('--min_prob', type=float, default=0.05)
ap.add_argument('--max_prompts', type=int, default=600)
ap.add_argument('--rms', nargs='*', default=[])
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()
torch.manual_seed(args.seed)
dev = 'cuda'

# ---- prompts
files = sorted(glob.glob(os.path.join(args.data, 'data', '*.parquet')))
prompts = []
for f in files:
    df = pd.read_parquet(f)
    col = 'prompt' if 'prompt' in df.columns else df.columns[0]
    src = os.path.basename(f).split('-')[0]
    for p in df[col].tolist():
        if isinstance(p, str) and 10 <= len(p) <= 400:
            prompts.append((src, p.strip()))
seen = set(); uniq = []
for s, p in prompts:
    if p not in seen:
        seen.add(p); uniq.append((s, p))
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts_short import short_prompts  # noqa: E402
uniq = [('short', q) for q in short_prompts()] + [x for x in uniq if x[0] == 'curated'] + [x for x in uniq if x[0] != 'curated']
prompts = uniq[:args.max_prompts]
print('prompts', len(prompts), {s: sum(1 for x, _ in prompts if x == s) for s in set(x for x, _ in prompts)}, flush=True)

tok = AutoTokenizer.from_pretrained(args.policy)
tok.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(args.policy, dtype=torch.bfloat16).to(dev).eval()
eos_id = tok.convert_tokens_to_ids('<|im_end|>')


def chat_ids(p):
    text = tok.apply_chat_template([{'role': 'user', 'content': p + SUFFIX}], add_generation_prompt=True, tokenize=False)
    return tok(text, add_special_tokens=False)['input_ids']


@torch.no_grad()
def seq_logprob(prompt_ids, resp_ids_list):
    """Exact sum of log-probabilities of each response (including the EOS token) given the prompt."""
    seqs = [prompt_ids + r for r in resp_ids_list]
    L = max(len(s) for s in seqs)
    inp = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        inp[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
    inp, att = inp.to(dev), att.to(dev)
    logits = model(input_ids=inp, attention_mask=att).logits.float()
    lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, inp[:, 1:, None])[..., 0]
    out = []
    n0 = len(prompt_ids)
    for i, r in enumerate(resp_ids_list):
        out.append(lp[i, n0 - 1:n0 - 1 + len(r)].sum().item())
    return out


panels = []
for pi, (src, p) in enumerate(prompts):
    pid = chat_ids(p)
    inp = torch.tensor([pid], device=dev)
    with torch.no_grad():
        gen = model.generate(inp, attention_mask=torch.ones_like(inp), do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                             max_new_tokens=args.max_new, num_return_sequences=args.M, pad_token_id=tok.pad_token_id)
    cands = {}
    for g in gen[:, len(pid):].tolist():
        if eos_id not in g:
            continue
        r = g[:g.index(eos_id) + 1]
        text = tok.decode(r[:-1]).strip()
        if not text or len(text.split()) > 8:
            continue
        key = text.lower().strip(' .!"\'')
        if key not in cands:
            cands[key] = (text, r)
    if len(cands) < args.K:
        continue
    texts = [v[0] for v in cands.values()]; rids = [v[1] for v in cands.values()]
    lps = seq_logprob(pid, rids)
    order = sorted(range(len(texts)), key=lambda i: -lps[i])[:args.K]
    s0 = [lps[i] for i in order]
    m = max(s0); z = sum(math.exp(v - m) for v in s0); probs = [math.exp(v - m) / z for v in s0]
    if min(probs) < args.min_prob:
        continue
    panels.append(dict(id=len(panels), source=src, prompt=p, prompt_full=p + SUFFIX,
                       responses=[texts[i] for i in order], s0=s0, p0=probs))
    if pi % 50 == 0:
        print(pi, len(panels), flush=True)
print('panels kept', len(panels), 'of', len(prompts), flush=True)
del model; torch.cuda.empty_cache()

# ---- reward models
for rm_dir in args.rms:
    name = os.path.basename(rm_dir.rstrip('/'))
    rtok = AutoTokenizer.from_pretrained(rm_dir)
    rm = AutoModelForSequenceClassification.from_pretrained(rm_dir, dtype=torch.bfloat16, num_labels=1).to(dev).eval()
    if rtok.pad_token is None:
        rtok.pad_token = rtok.eos_token
    rm.config.pad_token_id = rtok.pad_token_id
    for P in panels:
        convs = [rtok.apply_chat_template([{'role': 'user', 'content': P['prompt_full']}, {'role': 'assistant', 'content': r}], tokenize=False)
                 for r in P['responses']]
        sc = []
        for c in convs:  # one at a time: no padding, so the last token is unambiguous
            enc = rtok(c, return_tensors='pt', add_special_tokens=False).to(dev)
            with torch.no_grad():
                sc.append(rm(**enc).logits[0, 0].float().item())
        P.setdefault('rm', {})[name] = sc
    print('scored with', name, flush=True)
    del rm; torch.cuda.empty_cache()

json.dump(dict(policy=args.policy, suffix=SUFFIX, args=vars(args), panels=panels), open(args.out, 'w'), indent=1)
print('wrote', args.out, len(panels))
