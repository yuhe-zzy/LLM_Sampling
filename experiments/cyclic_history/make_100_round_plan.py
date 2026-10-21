"""Export portable B/C reproduction configs; never train or submit jobs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from calibrated_protocol import contract_hash
from run_cyclic_history import validate_config

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'calibration/2026-09-28/experiment_plan.json'
ARMS = {'B': ('ordinary', 'reference', 'feedback'), 'C': ('mixed',)}


def plan_fingerprint(plan):
    # Git normalizes line endings; provenance should identify the JSON values.
    normalized = json.dumps(plan, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(normalized).hexdigest()


def make_plan(stage, model_path='model/Qwen2.5-1.5B',
              eval_path='data/processed/eval_prompt_responses_cyclic_1000.jsonl',
              output_root=None):
    if stage not in ARMS:
        raise ValueError('Only the executed B100/C100 configurations are supported')
    original = json.loads(SOURCE.read_text(encoding='utf-8'))
    plan = copy.deepcopy(original)
    plan['status'] = 'REPRODUCTION_CONFIG_ONLY_REQUIRES_NEW_EXECUTION_APPROVAL'
    plan['common'].update(iters=100, model_path=model_path, eval_path=eval_path,
                          output_root=output_root or f'outputs/cyclic_history_stage_{stage.lower()}100')
    plan['runs'] = []
    for method in ('ipo', 'dpo'):
        for arm in ARMS[stage]:
            original_run = next(r for r in original['runs']
                                if r['run_id'] == f'{method}_{arm}_calibrated_s0')
            run = copy.deepcopy(original_run)
            run['run_id'] = f'{method}_{arm}_{stage.lower()}100_s0'
            cfg = dict(plan['common'], **run, protocol=plan['protocol'])
            run['calibration_contract_sha256'] = contract_hash(cfg)
            validate_config(dict(cfg, calibration_contract_sha256=run['calibration_contract_sha256']))
            plan['runs'].append(run)
    plan['provenance'] = dict(original_plan_canonical_json_sha256=plan_fingerprint(original),
                              training_source_commit='f0fe034cc0d8e3bef35b0f5e02806ee81340da76',
                              recorded_array={'B': '4605560', 'C': '4606367'}[stage],
                              scope='Portable reproduction; does not submit or authorize duplicate runs')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=tuple(ARMS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-path', default='model/Qwen2.5-1.5B')
    parser.add_argument('--eval-path', default='data/processed/eval_prompt_responses_cyclic_1000.jsonl')
    parser.add_argument('--output-root')
    args = parser.parse_args()
    plan = make_plan(args.stage, args.model_path, args.eval_path, args.output_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(plan, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(config=str(args.output), runs=[r['run_id'] for r in plan['runs']],
                          outer_updates=100, submitted=False), indent=2))


if __name__ == '__main__':
    main()
