"""X11 stage 1 (2026-10-19): score every panel response with each reward model under three rubric contexts.
The rubric is the context a labeller sees; the policy is later trained on the plain prompt, which hides it.
Usage: python rubric_scores.py --panels panels_q05.json --rms <dir> [<dir> ...] --out rubric_scores.json"""
import argparse, json, os, time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RUBRICS = {
    'rigor': ' I care most that the answer is clear, precise and accurately descriptive.',
    'creativity': ' I care most that the answer is creative and original, even unusual.',
    'concision': ' I care most that the answer is as short and punchy as possible.',
}
ap = argparse.ArgumentParser()
ap.add_argument('--panels', required=True)
ap.add_argument('--rms', nargs='+', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()
panels = json.load(open(args.panels))['panels']
out = dict(rubrics=RUBRICS, ids=[p['id'] for p in panels], scores={})
dev = 'cuda'
for rm_dir in args.rms:
    name = os.path.basename(rm_dir.rstrip('/')); t0 = time.time()
    rtok = AutoTokenizer.from_pretrained(rm_dir)
    rm = AutoModelForSequenceClassification.from_pretrained(rm_dir, dtype=torch.bfloat16, num_labels=1).to(dev).eval()
    if rtok.pad_token is None:
        rtok.pad_token = rtok.eos_token
    rm.config.pad_token_id = rtok.pad_token_id
    out['scores'][name] = {}
    for key, text in [('plain', '')] + list(RUBRICS.items()):
        S = []
        for P in panels:
            sc = []
            for r in P['responses']:   # one at a time: no padding, as in build_panels.py
                c = rtok.apply_chat_template([{'role': 'user', 'content': P['prompt_full'] + text}, {'role': 'assistant', 'content': r}], tokenize=False)
                enc = rtok(c, return_tensors='pt', add_special_tokens=False).to(dev)
                with torch.no_grad():
                    sc.append(rm(**enc).logits[0, 0].float().item())
            S.append(sc)
        out['scores'][name][key] = S
        print(name, key, 'done', round(time.time() - t0, 1), flush=True)
    del rm; torch.cuda.empty_cache()
json.dump(out, open(args.out, 'w'))
print('wrote', args.out)
