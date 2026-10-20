"""X12 (2026-10-19): a pairwise LLM judge on the v4 panels and the style panels, both presentation orders.
For each panel and each ordered pair (i in slot A, j in slot B), q[i, j] = P(judge answers "A") from the next-token logits restricted
to the two letters. Order-averaged soft label: p_ij = (q[i, j] + 1 - q[j, i]) / 2.
Usage: python llm_judge.py --judge <dir> --panels a.json [b.json ...] --out llm_judge.json"""
import argparse, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = 'You are an impartial judge of response quality.'
TEMPLATE = ('Here is a user request and two candidate responses.\n\nRequest: {req}\n\nResponse A: {a}\n\nResponse B: {b}\n\n'
            'Which response is better overall? Answer with a single letter: A or B.')
ap = argparse.ArgumentParser()
ap.add_argument('--judge', required=True)
ap.add_argument('--panels', nargs='+', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(args.judge)
model = AutoModelForCausalLM.from_pretrained(args.judge, dtype=torch.bfloat16).to('cuda').eval()
ida, idb = tok.encode('A', add_special_tokens=False), tok.encode('B', add_special_tokens=False)
assert len(ida) == 1 and len(idb) == 1, (ida, idb)
ida, idb = ida[0], idb[0]
out = dict(judge=args.judge, system=SYSTEM, template=TEMPLATE, files={})
t0 = time.time()
for f in args.panels:
    panels = json.load(open(f))['panels']; res = []
    for P in panels:
        K = len(P['responses']); q = [[0.5] * K for _ in range(K)]; top = [[None] * K for _ in range(K)]
        for i in range(K):
            for j in range(K):
                if i == j: continue
                msg = [{'role': 'system', 'content': SYSTEM},
                       {'role': 'user', 'content': TEMPLATE.format(req=P['prompt_full'], a=P['responses'][i], b=P['responses'][j])}]
                text = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
                ids = tok(text, return_tensors='pt', add_special_tokens=False)['input_ids'].to('cuda')
                with torch.no_grad():
                    lg = model(input_ids=ids).logits[0, -1].float()
                pa = torch.softmax(torch.stack([lg[ida], lg[idb]]), 0)[0].item()
                q[i][j] = pa
                top[i][j] = tok.decode(int(lg.argmax()))                 # sanity: is the top token one of the letters?
        res.append(dict(id=P['id'], q=q, top=top))
    out['files'][f] = res
    print(f, len(res), 'panels', round(time.time() - t0, 1), 's', flush=True)
json.dump(out, open(args.out, 'w'))
print('wrote', args.out)
