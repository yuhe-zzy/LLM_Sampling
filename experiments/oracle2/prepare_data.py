"""Deterministic prompt-disjoint real panels, selected before reward scoring."""
import argparse
from collections import Counter, defaultdict
import hashlib
import random
import unicodedata
from pathlib import Path

from common import jsonl, load_json, save_jsonl, sha256, verify_model_lock, write_json


def prompt_key(text):
    return hashlib.sha256(' '.join(unicodedata.normalize('NFKC', text).split()).encode()).hexdigest()


def split_panels(panels, sizes, seed):
    if len(panels) < sum(sizes.values()):
        raise ValueError(f'Only {len(panels)} eligible real panels for {sum(sizes.values())} required')
    if len({p['prompt_key'] for p in panels}) != len(panels):
        raise ValueError('Duplicate normalized prompts')
    order = list(panels)
    random.Random(seed).shuffle(order)
    output, start = {}, 0
    for split, count in sizes.items():
        output[split] = [dict(p, split=split) for p in order[start:start+count]]
        start += count
    return output


def main():
    from transformers import AutoTokenizer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    a = p.parse_args()
    plan = load_json(a.plan)
    spec = plan['dataset']
    lock = load_json(plan['model_lock'])
    verify_model_lock(lock)
    root = Path(plan['data_root'])
    if root.exists():
        raise FileExistsError('Do not replace a frozen or partially prepared dataset')
    tokenizers = {k: AutoTokenizer.from_pretrained(v['path'], local_files_only=True,
                                                  trust_remote_code=False) for k, v in lock.items()}
    raw = defaultdict(set)
    for row in jsonl(spec['raw_provenance']):
        raw[row['prompt']].add(row['response'])
    rejected = Counter()
    eligible, seen = [], set()
    for row in jsonl(spec['source']):
        prompt = row['prompt']
        responses = [r['text'] if isinstance(r, dict) else r for r in row['responses']]
        key = prompt_key(prompt)
        if key in seen:
            rejected['normalized_duplicate_prompt'] += 1
            continue
        seen.add(key)
        if len(responses) != spec['keep_k'] or len({r.strip() for r in responses}) != spec['keep_k']:
            rejected['not_four_distinct_responses'] += 1
            continue
        if not prompt.strip() or any(not r.strip() for r in responses):
            raise ValueError('Empty source text')
        if any(r not in raw[prompt] for r in responses):
            raise ValueError(f'Candidate not present in raw data for prompt ID {row["prompt_id"]}')
        tok = tokenizers['policy']
        if tok.eos_token_id is None:
            raise ValueError('Policy tokenizer must have EOS')
        prompt_tokens = len(tok.encode(prompt, add_special_tokens=False))
        response_tokens = [len(tok.encode(r, add_special_tokens=False)) + 1 for r in responses]
        if prompt_tokens < 1 or max(prompt_tokens + max(response_tokens),
                                     prompt_tokens + plan['evaluation']['max_new_tokens']) > spec['max_policy_length']:
            rejected['policy_length'] += 1
            continue
        judge_lengths = {}
        for name in ('nemotron', 'skywork'):
            judge_lengths[name] = [len(tokenizers[name].apply_chat_template(
                [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': r}],
                tokenize=True, add_generation_prompt=False)) for r in responses]
        if max(max(v) for v in judge_lengths.values()) > spec['max_judge_length']:
            rejected['judge_length'] += 1
            continue
        eligible.append(dict(prompt_id=int(row['prompt_id']), prompt_key=key, prompt=prompt,
                             responses=responses, policy_prompt_tokens=prompt_tokens,
                             response_tokens=response_tokens, judge_lengths=judge_lengths))
    sizes = {k: spec[k] for k in ('calibration', 'train', 'evaluation')}
    report = dict(eligible_panels=len(eligible), rejected=dict(rejected), requested=sizes,
                  selection='seeded shuffle after content/length checks; before reward scoring',
                  plan_sha256=sha256(a.plan), model_lock_sha256=sha256(plan['model_lock']),
                  source_sha256=sha256(spec['source']), raw_source_sha256=sha256(spec['raw_provenance']))
    print(report, flush=True)
    splits = split_panels(eligible, sizes, spec['split_seed'])
    root.mkdir(parents=True, exist_ok=False)
    records = []
    for split, panels in splits.items():
        save_jsonl(root / f'{split}.jsonl', panels)
        for panel in panels:
            for index, response in enumerate(panel['responses']):
                records.append(dict(id=f'{split}:{panel["prompt_key"]}:{index}', split=split,
                                    prompt_key=panel['prompt_key'], response_index=index,
                                    prompt=panel['prompt'], response=response))
    save_jsonl(root / 'candidate_records.jsonl', records)
    report['files_sha256'] = {path.name: sha256(path) for path in root.glob('*.jsonl')}
    report['state'] = 'PREPARED_REAL_UNSCORED'
    write_json(root / 'data_manifest.json', report)


if __name__ == '__main__':
    main()
