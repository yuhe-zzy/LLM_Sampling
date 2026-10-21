"""Derive exactly two user-requested empirical full-refresh reference runs."""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CODE = ROOT.parent.parent
sys.path.insert(0, str(CODE))
from calibrated_protocol import EMPIRICAL_PROTOCOL, contract_hash
from run_cyclic_history import validate_config


def build():
    source = ROOT.parent/'cyclic_history_stage_b100_launch_20260929/experiment_plan_b100.json'
    original = json.loads(source.read_text())
    plan = copy.deepcopy(original)
    plan['protocol'] = EMPIRICAL_PROTOCOL
    plan['status'] = 'USER_APPROVED_TWO_FULL_REFRESH_REFERENCE90_ARMS'
    plan['common'].update(alpha=1., output_root=
        '/work/users/y/u/yuhe32/ipo_runs/cyclic_history_reference90_20260929')
    plan['runs'] = []
    for method in ('ipo', 'dpo'):
        row = copy.deepcopy(next(r for r in original['runs'] if r['run_id'] == f'{method}_reference_b100_s0'))
        row.update(run_id=f'{method}_reference90_a1_b100_s0', nu=.9,
                   prediction_role='empirical_unclassified')
        # Alpha=.9 theory predictions do not describe the requested alpha=1 map.
        row.pop('predictions', None)
        cfg = dict(plan['common'], **row, protocol=plan['protocol'])
        row['calibration_contract_sha256'] = contract_hash(cfg)
        cfg['calibration_contract_sha256'] = row['calibration_contract_sha256']
        validate_config(cfg)
        plan['runs'].append(row)
    plan['launch_provenance'] = dict(source_plan_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        reference_weights=dict(initial=0.,current=.1,previous=.9),
        initialization='fresh initial model; previous=current=initial at outer step 0',
        scope='Empirical relative trajectories; no fixed-point or spectral-stability gate',
        comparison_caveat='Both alpha and nu differ from Stage B; not a nu-only causal contrast',
        preserved='Same six aligned panels, beta, lambda_current, seed, tokenizer, inner solver and horizon',
        excluded='No ordinary alpha=1 baseline, no mixed orientation, no additional seeds or sweeps')
    target = ROOT/'experiment_plan.json'
    with target.open('x', encoding='utf-8') as handle:
        json.dump(plan, handle, indent=2)
        handle.write('\n')
    print(target)


if __name__ == '__main__':
    build()
