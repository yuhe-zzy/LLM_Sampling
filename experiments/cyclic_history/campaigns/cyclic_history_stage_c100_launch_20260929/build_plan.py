"""Derive the remaining Stage C arms, matched to the completed B100 controls."""
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
RUNS = [(method, 'mixed') for method in ('ipo', 'dpo')]


def main():
    original = json.loads(SOURCE.read_text(encoding='utf-8'))
    plan = copy.deepcopy(original)
    plan['status'] = 'USER_APPROVED_REMAINING_STAGE_C100_ORIENTATION_CONTROLS'
    plan['common'].update(iters=100,
        eval_path='/work/users/y/u/yuhe32/ipo/data/processed/eval_prompt_responses_cyclic_1000.jsonl',
        model_path='/work/users/y/u/yuhe32/ipo/model/Qwen2.5-1.5B',
        output_root='/work/users/y/u/yuhe32/ipo_runs/cyclic_history_stage_c100_20260929')
    baseline = json.loads((ROOT.parent/'cyclic_history_stage_b100_launch_20260929/experiment_plan_b100.json').read_text())
    plan['runs'] = []
    for method, arm in RUNS:
        old = next(row for row in original['runs'] if row['run_id'] == f'{method}_{arm}_calibrated_s0')
        row = copy.deepcopy(old)
        row['run_id'] = f'{method}_{arm}_c100_s0'
        cfg = dict(plan['common'], **row, protocol=plan['protocol'])
        row['calibration_contract_sha256'] = contract_hash(cfg)
        cfg['calibration_contract_sha256'] = row['calibration_contract_sha256']
        validate_config(cfg)
        before = dict(original['common'], **old, protocol=original['protocol'])
        differences = {key for key in cfg if cfg[key] != before.get(key)}
        assert differences == {'iters', 'run_id', 'eval_path', 'model_path', 'output_root',
                               'calibration_contract_sha256'}, differences
        control = next(r for r in baseline['runs'] if r['run_id'] == f'{method}_ordinary_b100_s0')
        control_cfg = dict(baseline['common'], **control, protocol=baseline['protocol'])
        contrast = {key for key in cfg if cfg[key] != control_cfg.get(key)}
        assert contrast == {'run_id', 'output_root', 'orientations', 'predictions',
                            'transformed_support_sha256', 'calibration_contract_sha256'}, contrast
        assert cfg['orientations'] == [-1, 1, -1, 1, -1, 1]
        plan['runs'].append(row)
    plan['launch_provenance'] = dict(source_plan_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        source_git_commit='f0fe034cc0d8e3bef35b0f5e02806ee81340da76',
        approved_outer_states='0 through 100 inclusive; 100 outer updates from initial model',
        approved_arms='Remaining Stage C: IPO/DPO ordinary with mixed cycle orientations',
        matched_controls='Completed 4605560_0 (IPO) and 4605560_3 (DPO); do not rerun',
        baseline_plan_sha256=hashlib.sha256((ROOT.parent/'cyclic_history_stage_b100_launch_20260929/experiment_plan_b100.json').read_bytes()).hexdigest(),
        interpretation='Relative LLM phenomena; exact target fit is not a run/promotion gate',
        preserved='Same six prompt texts/responses, alpha/lambda/beta, seed, inner solver; only reverse P for prompts 54/612/867',
        excluded='No repeats of A/B, no extra seed, no parameter sweep or fixed-target probes')
    output = ROOT/'experiment_plan_c100.json'
    if output.exists():
        raise FileExistsError('Do not overwrite a prepared/possibly submitted plan')
    output.write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(plan=str(output), runs=[r['run_id'] for r in plan['runs']],
                          outer_updates=100, inner_epochs=plan['common']['epochs_per_iter']), indent=2))


if __name__ == '__main__':
    main()
