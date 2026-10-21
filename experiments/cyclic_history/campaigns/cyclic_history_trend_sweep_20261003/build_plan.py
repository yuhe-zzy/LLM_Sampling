"""Five one-factor base settings, two interventions each, shared ordinary controls."""
import copy
import csv
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CODE = HERE.parent.parent
sys.path.insert(0, str(CODE))
from calibrated_protocol import TREND_PROTOCOL, contract_hash
from run_cyclic_history import validate_config

REMOTE_ROOT = '/work/users/y/u/yuhe32/ipo_runs/cyclic_history_trend_sweep_20261003'
BASES = [
    dict(base_id='center', alpha=.9, lambda_current=.8, beta_scale=1.),
    dict(base_id='alpha08', alpha=.8, lambda_current=.8, beta_scale=1.),
    dict(base_id='alpha099', alpha=.99, lambda_current=.8, beta_scale=1.),
    dict(base_id='coverage05', alpha=.9, lambda_current=.5, beta_scale=1.),
    dict(base_id='beta15', alpha=.9, lambda_current=.8, beta_scale=1.5),
]
ARMS = ('ordinary', 'reference_half', 'reference_max', 'feedback_half', 'feedback_one')
REUSE_TASKS = {('ipo','ordinary'):0, ('ipo','reference_half'):1, ('ipo','feedback_half'):2,
               ('dpo','ordinary'):3, ('dpo','reference_half'):4, ('dpo','feedback_half'):5}


def make_plan():
    prior = json.loads((CODE/'campaigns/cyclic_history_stage_b100_launch_20260929/experiment_plan_b100.json').read_text())
    plan = dict(protocol=TREND_PROTOCOL, status='USER_APPROVED_TEN_PER_INTERVENTION_PER_OBJECTIVE',
                common=copy.deepcopy(prior['common']), bases=BASES, comparisons=[], reused_runs=[], runs=[])
    plan['common']['output_root'] = REMOTE_ROOT
    for base in BASES:
        for method, beta in [('ipo',.2), ('dpo',.8)]:
            prior_id = f'{method}_ordinary_b100_s0'
            template = copy.deepcopy(next(row for row in prior['runs'] if row['run_id']==prior_id))
            for arm in ARMS:
                row = copy.deepcopy(template)
                row.pop('predictions', None)
                row.update(alpha=base['alpha'],lambda_current=base['lambda_current'],
                    beta_train=round(beta*base['beta_scale'],8),nu=0.,kappa=0.,
                    prediction_role='empirical_unclassified',
                    run_id=f'{method}_{base["base_id"]}_{arm}_b100_s0')
                row['scheme'] = ('lagged_reference' if arm.startswith('reference') else
                                 'oracle_feedback_extrapolation' if arm.startswith('feedback') else 'ordinary')
                if arm.startswith('reference'):
                    row['nu'] = base['alpha'] * (.5 if arm=='reference_half' else 1.)
                if arm.startswith('feedback'):
                    row['kappa'] = .5 if arm=='feedback_half' else 1.
                cfg = dict(plan['common'], **row)
                cfg['protocol'] = TREND_PROTOCOL
                row['calibration_contract_sha256'] = contract_hash(cfg)
                cfg['calibration_contract_sha256'] = row['calibration_contract_sha256']
                validate_config(cfg)
                record = dict(base_id=base['base_id'],method=method,arm=arm,run_id=row['run_id'],
                              alpha=row['alpha'],lambda_current=row['lambda_current'],
                              beta_train=row['beta_train'],nu=row['nu'],kappa=row['kappa'],
                              baseline_run_id=f'{method}_{base["base_id"]}_ordinary_b100_s0')
                if base['base_id']=='center' and (method,arm) in REUSE_TASKS:
                    old_arm = dict(ordinary='ordinary',reference_half='reference',feedback_half='feedback')[arm]
                    old_id = f'{method}_{old_arm}_b100_s0'
                    record.update(status='REUSE_COMPLETED',task_index=None,existing_run_id=old_id,
                        existing_job_id=f'4605560_{REUSE_TASKS[method,arm]}',
                        existing_root='/work/users/y/u/yuhe32/ipo_runs/cyclic_history_stage_b100_20260929',
                        existing_source_commit='f0fe034cc0d8e3bef35b0f5e02806ee81340da76')
                    plan['reused_runs'].append(dict(comparison=record, expected_config=cfg))
                else:
                    record.update(status='NEW',task_index=len(plan['runs']))
                    plan['runs'].append(row)
                plan['comparisons'].append(record)
    return plan


def main():
    plan = make_plan()
    (HERE/'experiment_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    fields = list(dict.fromkeys(k for r in plan['comparisons'] for k in r))
    with (HERE/'comparison_grid.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(plan['comparisons'])
    print(json.dumps(dict(new_runs=len(plan['runs']),reused=len(plan['reused_runs']),
                          comparisons=len(plan['comparisons']),array='0-43%6')))


if __name__=='__main__':
    main()
