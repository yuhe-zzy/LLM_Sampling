"""X11 stage 1b (2026-10-19): style panels. For each prompt the policy writes one short answer in each of three styles
(rigorous, creative, concise; the style instruction is appended to the prompt) plus its most probable plain answer. Panels keep
the plain prompt (the style is hidden from the policy at training time), are scored by exact log-probability under the plain
prompt, and kept when the four answers are distinct and every normalized initial probability is at least --min_prob.
Each answer is then scored by every reward model under the plain prompt and under the three rubric contexts of rubric_scores.py.
Usage: python style_panels.py --policy <dir> --out style_panels.json --rms <dir> ... [--M 16 --min_prob 0.01]"""
import argparse, glob, json, math, os, sys
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

SUFFIX = " Answer with a single short phrase of at most six words, and nothing else."
STYLES = {'rigor': ' Make it precise and accurately descriptive.',
          'creativity': ' Make it creative and original, even whimsical.',
          'concision': ' Make it as short and punchy as possible.'}
RUBRICS = {'rigor': ' I care most that the answer is clear, precise and accurately descriptive.',
           'creativity': ' I care most that the answer is creative and original, even unusual.',
           'concision': ' I care most that the answer is as short and punchy as possible.'}
ap = argparse.ArgumentParser()
ap.add_argument('--policy', required=True)
ap.add_argument('--data', default=os.path.expanduser('~/v4val/data/novelty-bench'))
ap.add_argument('--out', required=True)
ap.add_argument('--M', type=int, default=16)
ap.add_argument('--max_new', type=int, default=20)
ap.add_argument('--min_prob', type=float, default=0.01)
ap.add_argument('--max_prompts', type=int, default=600)
ap.add_argument('--rms', nargs='*', default=[])
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()
torch.manual_seed(args.seed); dev = 'cuda'

files = sorted(glob.glob(os.path.join(args.data, 'data', '*.parquet'))); prompts = []
for f in files:
    df = pd.read_parquet(f); col = 'prompt' if 'prompt' in df.columns else df.columns[0]; src = os.path.basename(f).split('-')[0]
    for p in df[col].tolist():
        if isinstance(p, str) and 10 <= len(p) <= 400: prompts.append((src, p.strip()))
seen = set(); uniq = []
for s, p in prompts:
    if p not in seen: seen.add(p); uniq.append((s, p))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts_short import short_prompts  # noqa: E402
uniq = [('short', q) for q in short_prompts()] + [x for x in uniq if x[0] == 'curated'] + [x for x in uniq if x[0] != 'curated']
prompts = uniq[:args.max_prompts]

tok = AutoTokenizer.from_pretrained(args.policy); tok.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(args.policy, dtype=torch.bfloat16).to(dev).eval()
eos_id = tok.convert_tokens_to_ids('<|im_end|>')
chat_ids = lambda text: tok(tok.apply_chat_template([{'role': 'user', 'content': text}], add_generation_prompt=True, tokenize=False), add_special_tokens=False)['input_ids']


@torch.no_grad()
def seq_logprob(prompt_ids, resp_ids_list):
    seqs = [prompt_ids + r for r in resp_ids_list]; L = max(len(s) for s in seqs)
    inp = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long); att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs): inp[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
    inp, att = inp.to(dev), att.to(dev)
    lp = torch.log_softmax(model(input_ids=inp, attention_mask=att).logits.float()[:, :-1], -1).gather(-1, inp[:, 1:, None])[..., 0]
    n0 = len(prompt_ids); return [lp[i, n0 - 1:n0 - 1 + len(r)].sum().item() for i, r in enumerate(resp_ids_list)]


@torch.no_grad()
def best_answer(text):
    """Most probable valid short answer (EOS-terminated, at most 8 words) among M samples from the given prompt."""
    pid = chat_ids(text); inp = torch.tensor([pid], device=dev)
    gen = model.generate(inp, attention_mask=torch.ones_like(inp), do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                         max_new_tokens=args.max_new, num_return_sequences=args.M, pad_token_id=tok.pad_token_id)
    cands = {}
    for g in gen[:, len(pid):].tolist():
        if eos_id not in g: continue
        r = g[:g.index(eos_id) + 1]; t = tok.decode(r[:-1]).strip()
        if not t or len(t.split()) > 8: continue
        cands.setdefault(t.lower().strip(' .!"\''), (t, r))
    if not cands: return None
    vals = list(cands.items()); lps = seq_logprob(pid, [v[1][1] for v in vals]); i = max(range(len(vals)), key=lambda k: lps[k])
    return vals[i][0], vals[i][1][0], vals[i][1][1]


panels = []; stats = dict(prompts=len(prompts), no_answer=0, not_distinct=0, low_prob=0, kept_at_0p05=0)
for pi, (src, p) in enumerate(prompts):
    got = {'plain': best_answer(p + SUFFIX)}
    for s, st in STYLES.items(): got[s] = best_answer(p + SUFFIX + st)
    if any(v is None for v in got.values()): stats['no_answer'] += 1; continue
    keys = [v[0] for v in got.values()]
    if len(set(keys)) < 4: stats['not_distinct'] += 1; continue
    order = ['plain', 'rigor', 'creativity', 'concision']
    s0 = seq_logprob(chat_ids(p + SUFFIX), [got[k][2] for k in order])
    m = max(s0); z = sum(math.exp(v - m) for v in s0); probs = [math.exp(v - m) / z for v in s0]
    if min(probs) < args.min_prob: stats['low_prob'] += 1; continue
    stats['kept_at_0p05'] += int(min(probs) >= 0.05)
    panels.append(dict(id=len(panels), source=src, prompt=p, prompt_full=p + SUFFIX, styles=order,
                       responses=[got[k][1] for k in order], s0=s0, p0=probs))
    if pi % 50 == 0: print(pi, len(panels), flush=True)
stats['kept'] = len(panels); print('stats', stats, flush=True)
del model; torch.cuda.empty_cache()

for rm_dir in args.rms:
    name = os.path.basename(rm_dir.rstrip('/')); rtok = AutoTokenizer.from_pretrained(rm_dir)
    rm = AutoModelForSequenceClassification.from_pretrained(rm_dir, dtype=torch.bfloat16, num_labels=1).to(dev).eval()
    if rtok.pad_token is None: rtok.pad_token = rtok.eos_token
    rm.config.pad_token_id = rtok.pad_token_id
    for P in panels:
        for key, text in [('plain', '')] + list(RUBRICS.items()):
            sc = []
            for r in P['responses']:
                c = rtok.apply_chat_template([{'role': 'user', 'content': P['prompt_full'] + text}, {'role': 'assistant', 'content': r}], tokenize=False)
                enc = rtok(c, return_tensors='pt', add_special_tokens=False).to(dev)
                with torch.no_grad(): sc.append(rm(**enc).logits[0, 0].float().item())
            P.setdefault('rubric', {}).setdefault(name, {})[key] = sc
            if key == 'plain': P.setdefault('rm', {})[name] = sc
    print('scored with', name, flush=True); del rm; torch.cuda.empty_cache()
json.dump(dict(policy=args.policy, suffix=SUFFIX, styles=STYLES, rubrics=RUBRICS, args=vars(args), stats=stats, panels=panels), open(args.out, 'w'), indent=1)
print('wrote', args.out, len(panels))
