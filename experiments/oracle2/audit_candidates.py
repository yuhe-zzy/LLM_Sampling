"""Freeze real mixed preference matrices; publish aggregate audit, never force cycles."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np

from common import jsonl, load_json, save_jsonl, sha256, write_json
from history_math import bt_scores
from oracle_math import classify_panel, preference_matrix, sigmoid


def audit(plan_path, scores_root):
    plan = load_json(plan_path)
    data = Path(plan['data_root'])
    output = data / 'scored'
    output.mkdir(exist_ok=False)
    manifest = load_json(scores_root / 'manifest.json')
    if manifest['state'] != 'COMPLETE' or manifest['records_sha256'] != sha256(data / 'candidate_records.jsonl'):
        raise ValueError('Candidate scores missing/incomplete or belong to another input')
    if manifest['model_lock_sha256'] != sha256(plan['model_lock']):
        raise ValueError('Judge identity differs from frozen plan')
    scores = {}
    for name in ('nemotron', 'skywork'):
        if sha256(scores_root / f'{name}.jsonl') != manifest['components'][name]['sha256']:
            raise ValueError('Reward file modified')
        rows = jsonl(scores_root / f'{name}.jsonl')
        scores[name] = {r['id']: r['reward'] for r in rows}
        if len(scores[name]) != len(rows):
            raise ValueError('Duplicate reward IDs')
    ids = {r['id'] for r in jsonl(data / 'candidate_records.jsonl')}
    if any(set(v) != ids for v in scores.values()):
        raise ValueError('Score/input ID sets differ')
    report = dict(plan_sha256=sha256(plan_path), model_lock_sha256=sha256(plan['model_lock']),
                  data_manifest_sha256=sha256(data / 'data_manifest.json'),
                  score_manifest_sha256=sha256(scores_root / 'manifest.json'), splits={},
                  status='AWAITING_REVIEW_NOT_TRAINING_AUTHORIZATION',
                  interpretation='Groups concern only the four pretraining real candidates')
    for split in ('calibration', 'train', 'evaluation'):
        panels = jsonl(data / f'{split}.jsonl')
        groups = Counter()
        alternative = {str(w): Counter() for w in plan['oracle']['diagnostic_only_weights']}
        pair_n, pair_s, margins, solver_failures = [], [], [], []
        maximum_solver_residual = 0.
        for panel in panels:
            keys = [f'{split}:{panel["prompt_key"]}:{i}' for i in range(4)]
            n = np.array([scores['nemotron'][k] for k in keys])
            s = np.array([scores['skywork'][k] for k in keys])
            p = preference_matrix(n, s, plan['oracle']['nemotron_weight'], plan['oracle']['temperatures'])
            info = classify_panel(p, plan['dataset']['cycle_edge_margin'])
            groups[info['group']] += 1
            panel.update(preference_matrix=p.tolist(), oracle2_group=info['group'],
                         cycle_audit=info, nemotron_scores=n.tolist(), skywork_scores=s.tolist())
            for w in plan['oracle']['diagnostic_only_weights']:
                alternative[str(w)][classify_panel(preference_matrix(n, s, w, plan['oracle']['temperatures']),
                    plan['dataset']['cycle_edge_margin'])['group']] += 1
            i, j = np.triu_indices(4, 1)
            pair_n.extend(sigmoid((n[i]-n[j])/plan['oracle']['temperatures'][0]))
            pair_s.extend(sigmoid((s[i]-s[j])/plan['oracle']['temperatures'][1]))
            margins.extend(np.abs(p[i, j]-.5))
            if split == 'train':
                mus = [np.full(4, .25)] + [(1-plan['training']['lambda_current'])/4 +
                    plan['training']['lambda_current']*np.eye(4)[k] for k in range(4)]
                try:
                    for mu in mus:
                        _, solver = bt_scores(p, mu)
                        maximum_solver_residual = max(maximum_solver_residual, solver['residual'])
                except (RuntimeError, ValueError) as exc:
                    solver_failures.append(dict(prompt_id=panel['prompt_id'], error=str(exc)))
        pn, ps = np.array(pair_n), np.array(pair_s)
        report['splits'][split] = dict(count=len(panels), groups=dict(groups),
            diagnostic_only_weight_groups={w: dict(c) for w, c in alternative.items()},
            component_strict_disagreement_fraction=float(np.mean((pn-.5)*(ps-.5) < 0)),
            nemotron_saturated_pair_fraction=float(np.mean((pn < .01) | (pn > .99))),
            skywork_saturated_pair_fraction=float(np.mean((ps < .01) | (ps > .99))),
            median_mixed_pair_margin=float(np.median(margins)),
            bt_solver_failures=solver_failures, bt_solver_max_residual=maximum_solver_residual)
        save_jsonl(output / f'{split}.jsonl', panels)
    report['files_sha256'] = {p.name: sha256(p) for p in output.glob('*.jsonl')}
    write_json(output / 'audit.json', report)
    print(__import__('json').dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--scores', type=Path, required=True)
    a = p.parse_args()
    audit(a.plan, a.scores)
