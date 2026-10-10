"""Merge immutable generation banks and summarize true generated-response oracle2 WR."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import numpy as np

from common import jsonl, load_json, save_jsonl, sha256, write_json
from oracle_math import generated_wr


def collect(plan_path, destination):
    plan = load_json(plan_path)
    generations = Path(plan['output_root']) / 'generations'
    rows, sources = [], {}
    expected = plan['dataset']['evaluation'] * plan['evaluation']['responses_per_prompt']
    for name in ['baseline'] + [r['run_id'] for r in plan['runs']]:
        root = generations / name
        manifest = load_json(root / 'manifest.json')
        if manifest['state'] != 'COMPLETE' or manifest['plan_sha256'] != sha256(plan_path):
            raise ValueError(f'Incomplete or mismatched generation bank {name}')
        expected_files = {f'step_{s:04d}.jsonl' for s in ([0] if name == 'baseline' else plan['evaluation']['steps'])}
        if set(manifest['files']) != expected_files:
            raise ValueError('Missing or extra checkpoint bank')
        for filename, info in sorted(manifest['files'].items()):
            path = root / filename
            if sha256(path) != info['sha256']:
                raise ValueError('Generation bank modified')
            batch = jsonl(path)
            if len(batch) != expected:
                raise ValueError('Incomplete prompt/response bank')
            rows.extend(batch)
            sources[str(path)] = sha256(path)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate generated response IDs')
    save_jsonl(destination, rows)
    write_json(Path(str(destination)+'.manifest.json'), dict(plan_sha256=sha256(plan_path),
               count=len(rows), records_sha256=sha256(destination), sources=sources))


def summarize(plan_path, records_path, scores_root, destination):
    plan = load_json(plan_path)
    manifest = load_json(scores_root / 'manifest.json')
    if manifest['state'] != 'COMPLETE' or manifest['records_sha256'] != sha256(records_path):
        raise ValueError('Incomplete or wrong generated-response score cache')
    if manifest['model_lock_sha256'] != sha256(plan['model_lock']):
        raise ValueError('Judge changed')
    records = jsonl(records_path)
    scores = {}
    for name in ('nemotron', 'skywork'):
        path = scores_root / f'{name}.jsonl'
        if sha256(path) != manifest['components'][name]['sha256']:
            raise ValueError('Score cache modified')
        scored = jsonl(path)
        scores[name] = {r['id']: r['reward'] for r in scored}
        if len(scores[name]) != len(scored) or set(scores[name]) != {r['id'] for r in records}:
            raise ValueError('Reward ID set mismatch')
    grouped = defaultdict(list)
    for row in records:
        grouped[(row['run_id'], row['step'], row['prompt_key'])].append(row)
    per_prompt = []
    for (run, step, key), current in sorted(grouped.items()):
        if run == 'baseline':
            continue
        base = grouped[('baseline', 0, key)]
        if len(current) != 4 or len(base) != 4 or len({r['draw'] for r in current}) != 4:
            raise ValueError('Need four aligned independent responses per prompt/bank')
        values = [[scores[name][r['id']] for r in bank] for bank in (current, base)
                  for name in ('nemotron', 'skywork')]
        metrics = generated_wr(*values, plan['oracle']['nemotron_weight'], plan['oracle']['temperatures'])
        per_prompt.append(dict(run_id=run, step=step, prompt_key=key, group=current[0]['group'], **metrics))
    summary = []
    for run in [r['run_id'] for r in plan['runs']]:
        for step in plan['evaluation']['steps']:
            batch = [r for r in per_prompt if r['run_id'] == run and r['step'] == step]
            if len(batch) != plan['dataset']['evaluation']:
                raise ValueError('Missing evaluation prompts or checkpoint')
            for group in ('all', 'cyclic', 'transitive', 'ambiguous'):
                subset = [r for r in batch if group == 'all' or r['group'] == group]
                result = dict(run_id=run, step=step, group=group, prompt_count=len(subset))
                for metric in ('oracle2_expected_win_rate', 'oracle2_majority_win_rate'):
                    values = [r[metric] for r in subset]
                    result[metric] = float(np.mean(values)) if values else None
                    result[metric+'_prompt_se'] = float(np.std(values, ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
                summary.append(result)
    destination.mkdir(parents=True, exist_ok=False)
    for name, rows in (('wr_per_prompt.csv', per_prompt), ('wr_summary.csv', summary)):
        with (destination / name).open('x', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(destination / 'manifest.json', dict(plan_sha256=sha256(plan_path),
        records_sha256=sha256(records_path), scores_manifest_sha256=sha256(scores_root / 'manifest.json'),
        warning='WR against fixed initial-policy generations is not proof of convergence or a global quality order'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['collect', 'summarize'])
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--scores', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.action == 'collect':
        collect(a.plan, a.records)
    else:
        if a.scores is None or a.output is None:
            p.error('summarize requires --scores and --output')
        summarize(a.plan, a.records, a.scores, a.output)
