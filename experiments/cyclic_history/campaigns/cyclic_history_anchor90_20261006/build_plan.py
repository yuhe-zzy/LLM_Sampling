"""Two matched ordinary arms with 90% initial-reference anchoring."""
import copy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CODE = HERE.parent.parent
sys.path.insert(0, str(CODE))
from calibrated_protocol import TREND_PROTOCOL, contract_hash
from run_cyclic_history import validate_config

OUTPUT = '/work/users/y/u/yuhe32/ipo_runs/cyclic_history_anchor90_20261006'
RUNS = [f'{method}_anchor90_a01_b100_s0' for method in ('ipo', 'dpo')]


def make_plan():
    prior = json.loads((CODE/'campaigns/cyclic_history_stage_b100_launch_20260929/experiment_plan_b100.json').read_text())
    plan = dict(protocol=TREND_PROTOCOL, status='USER_APPROVED_TWO_ANCHOR90_ARMS_20261006',
                common=copy.deepcopy(prior['common']), runs=[])
    plan['common'].update(alpha=.1, output_root=OUTPUT)
    for method, run_id in zip(('ipo', 'dpo'), RUNS):
        row = copy.deepcopy(next(r for r in prior['runs'] if r['run_id']==f'{method}_ordinary_b100_s0'))
        row.pop('predictions', None)
        row.update(run_id=run_id, alpha=.1, nu=0., kappa=0., scheme='ordinary',
                   prediction_role='empirical_unclassified')
        cfg = dict(plan['common'], **row, protocol=TREND_PROTOCOL)
        row['calibration_contract_sha256'] = contract_hash(cfg)
        cfg['calibration_contract_sha256'] = row['calibration_contract_sha256']
        validate_config(cfg)
        plan['runs'].append(row)
    return plan


if __name__ == '__main__':
    (HERE/'experiment_plan.json').write_text(json.dumps(make_plan(), indent=2)+'\n')
    print('Prepared exactly two new arms: IPO and DPO, alpha=.1, nu=kappa=0')
