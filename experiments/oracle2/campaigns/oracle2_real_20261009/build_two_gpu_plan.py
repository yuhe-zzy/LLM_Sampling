"""Freeze the authorized two-GPU attempt without changing the scientific setup."""
import json
from pathlib import Path


def build():
    here = Path(__file__).resolve().parent
    plan = json.loads((here / 'plan_attempt2.json').read_text())
    for field in ('output_root', 'data_root', 'model_lock'):
        plan[field] = plan[field].replace('oracle2_real_20261009_v2', 'oracle2_real_20261009_v3_2gpu')
    plan['oracle']['scoring_gpus'] = 2
    return plan


if __name__ == '__main__':
    destination = Path(__file__).with_name('plan_two_gpu.json')
    with destination.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(build(), handle, indent=2)
        handle.write('\n')
