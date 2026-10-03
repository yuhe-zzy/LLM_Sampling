"""Derive the explicitly approved 100-round, six-arm Stage B observation plan."""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CODE = ROOT.parent.parent
sys.path.insert(0, str(CODE))
from calibrated_protocol import contract_hash
from run_cyclic_history import validate_config

SOURCE = CODE/'calibration/2026-09-28/experiment_plan.json'
RUNS = [(method, arm) for method in ('ipo', 'dpo')
        for arm in ('ordinary', 'reference', 'feedback')]


def main():
    original = json.loads(SOURCE.read_text(encoding='utf-8'))
    plan = copy.deepcopy(original)
    plan['status'] = 'USER_APPROVED_STAGE_B100_WITH_MATCHED_ORDINARY_BASELINES'
    plan['common'].update(iters=100,
        eval_path='/work/users/y/u/yuhe32/ipo/data/processed/eval_prompt_responses_cyclic_1000.jsonl',
        model_path='/work/users/y/u/yuhe32/ipo/model/Qwen2.5-1.5B',
        output_root='/work/users/y/u/yuhe32/ipo_runs/cyclic_history_stage_b100_20260929')
    plan['runs'] = []
    for method, arm in RUNS:
        old = next(row for row in original['runs'] if row['run_id'] == f'{method}_{arm}_calibrated_s0')
        row = copy.deepcopy(old)
        row['run_id'] = f'{method}_{arm}_b100_s0'
        cfg = dict(plan['common'], **row, protocol=plan['protocol'])
        row['calibration_contract_sha256'] = contract_hash(cfg)
        cfg['calibration_contract_sha256'] = row['calibration_contract_sha256']
        validate_config(cfg)
        before = dict(original['common'], **old, protocol=original['protocol'])
        differences = {key for key in cfg if cfg[key] != before.get(key)}
        assert differences == {'iters', 'run_id', 'eval_path', 'model_path', 'output_root',
                               'calibration_contract_sha256'}, differences
        plan['runs'].append(row)
    plan['launch_provenance'] = dict(source_plan_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        source_git_commit='f0fe034cc0d8e3bef35b0f5e02806ee81340da76',
        approved_outer_states='0 through 100 inclusive; 100 outer updates from initial model',
        approved_arms='IPO/DPO ordinary, lagged reference, oracle feedback extrapolation',
        interpretation='Relative LLM phenomena; exact target fit is not a run/promotion gate',
        preserved='Same six prompts, preference matrices, alpha/lambda/beta, seed, inner solver',
        excluded='No beta-stable controls, mixed orientations, or further fixed-target probes')
    output = ROOT/'experiment_plan_b100.json'
    if output.exists():
        raise FileExistsError('Do not overwrite a prepared/possibly submitted plan')
    output.write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(plan=str(output), runs=[r['run_id'] for r in plan['runs']],
                          outer_updates=100, inner_epochs=plan['common']['epochs_per_iter']), indent=2))


if __name__ == '__main__':
    main()
