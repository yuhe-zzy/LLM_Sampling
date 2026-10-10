"""Verify downloaded fixed-target probes without changing raw data or server jobs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent.parent
sys.path.insert(0, str(CODE))
from history_math import centered, pair_distribution, build_outer_state

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results-dir', type=Path, default=HERE)
parser.add_argument('--stage-a-results-dir', type=Path)
parser.add_argument('--output-dir', type=Path)
args = parser.parse_args()
HERE = args.results_dir.resolve()
OUT = (args.output_dir or args.results_dir).resolve()
PARENT = args.stage_a_results_dir or HERE.parent/'cyclic_history_stage_a_analysis_20260928'
OUT.mkdir(parents=True, exist_ok=True)


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    assert all(np.isfinite(x).all() for x in data.values())
    return data


def objective(scores, pref, state, cfg):
    out = []
    for x, p, mu, ref in zip(scores, pref, state['mu'], state['effective_reference']):
        i, j, w = pair_distribution(mu)
        d = x[i]-x[j]-ref[i]+ref[j]
        beta = cfg['beta_train']
        loss = (p[i,j]*(d-1/(2*beta))**2+(1-p[i,j])*(d+1/(2*beta))**2
                if cfg['method'] == 'ipo' else np.logaddexp(0,beta*d)-p[i,j]*beta*d)
        out.append(float(w @ loss))
    return np.asarray(out)


results = []
hashes = {}
for root in sorted((HERE/'raw').iterdir()):
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['state'] == 'COMPLETED' and manifest['completed_blocks'] == 6
    cfg = manifest['config']
    for name, expected in manifest['source_sha256'].items():
        assert hashlib.sha256((CODE/name).read_bytes()).hexdigest() == expected
    panels = json.loads((root/'support.json').read_text(encoding='utf-8'))
    matrices = np.array([p['preference_matrix'] for p in panels])
    state = read_npz(root/'frozen_outer_state.npz')
    parent = PARENT/'raw'/root.name/'snapshots'
    initial, previous, current = [read_npz(parent/f'step_{t:04d}.npz')['sequence_sum_logprob']
                                  for t in (0,19,20)]
    rebuilt = build_outer_state(initial, current, previous, matrices, cfg['method'],
                                cfg['alpha'],cfg['lambda_current'],cfg['beta_train'])
    for key in state:
        np.testing.assert_allclose(state[key],rebuilt[key],rtol=0,atol=1e-7)
    with (root/'metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r['block']) for r in rows] == list(range(7))
    target = state['target_logits']
    optimum = objective(target,matrices,state,cfg)
    denominator = np.sqrt(np.mean((centered(current)-target)**2))
    detail = []
    per_prompt_errors = []
    for block,row in enumerate(rows):
        z = read_npz(root/'snapshots'/f'block_{block:02d}.npz')
        np.testing.assert_array_equal(z['target_logits'],target)
        scores = z['sequence_sum_logprob']
        error = centered(scores)-target
        np.testing.assert_allclose(error,z['target_error'],rtol=0,atol=1e-12)
        rms = float(np.sqrt(np.mean(error**2)))
        loss = objective(scores,matrices,state,cfg)
        np.testing.assert_allclose([rms,rms/denominator,loss.mean(),(loss-optimum).mean()],
            [float(row[k]) for k in ('target_error_rms','error_to_intended_ratio',
                                     'pair_objective_mean','excess_pair_objective_mean')],atol=1e-10)
        per_prompt_errors.append(np.sqrt(np.mean(error**2,axis=1)))
        detail.append(dict(epochs=block*10,error_rms=rms,normalized_error=rms/denominator,
                           objective=float(loss.mean()),excess_objective=float((loss-optimum).mean())))
    results.append(dict(run=root.name,beta=cfg['beta_train'],measurements=detail,
        prompts_with_larger_error_60_vs_10=int(np.sum(per_prompt_errors[6]>per_prompt_errors[1])),
        per_prompt=[dict(prompt_id=p['prompt_id'],error10=per_prompt_errors[1][j],
                        error30=per_prompt_errors[3][j],error60=per_prompt_errors[6][j])
                    for j,p in enumerate(panels)]))
    for path in root.rglob('*'):
        if path.is_file():
            hashes[path.relative_to(HERE).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
(OUT/'verified_summary.json').write_text(json.dumps(results,indent=2)+'\n')
(OUT/'raw_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(json.dumps([dict(run=r['run'],beta=r['beta'],checkpoints=[m for m in r['measurements']
                    if m['epochs'] in (0,10,30,60)],worsened_prompts=r['prompts_with_larger_error_60_vs_10'])
                  for r in results],indent=2))
