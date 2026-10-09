"""Read-only campaign inspection. Does not submit, cancel, retry or schedule anything."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import subprocess


def inspect(root):
    plan = json.loads((root/'plan.json').read_text())
    queue_path = root/'source/experiments/oracle2/campaigns/oracle2_real_20261009/queue.py'
    spec = importlib.util.spec_from_file_location('oracle2_queue', queue_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    queue = module.owned_snapshot()
    gpu_count = 0
    for line in queue['owned_scontrol']:
        value = re.search(r'\bAllocTRES=(\S+)', line)
        if value:
            match = re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)', value[1])
            gpu_count += int(match[1]) if match else 0
    queue['allocated_gpus'] = gpu_count
    result = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), queue=queue, jobs={},
                  phases={}, errors=[], runs={})
    for receipt in root.glob('*_submission_receipt.json'):
        data = json.loads(receipt.read_text())
        job = data['job_id'].split(';')[0]
        result['jobs'][data['phase']] = subprocess.check_output(
            ['sacct', '-j', job, '-X', '-n', '-P', '-o', 'JobID,State,ExitCode,AllocTRES,Elapsed'], text=True).strip()
    output = Path(plan['output_root'])
    for name in ('candidate_scores', 'evaluation_scores'):
        directory = output/name
        if (directory/'manifest.json').exists():
            status = json.loads((directory/'manifest.json').read_text())
            status['written_rows'] = {}
            for component in ('nemotron', 'skywork'):
                path = directory/f'{component}.jsonl'
                if path.exists():
                    with path.open() as handle:
                        status['written_rows'][component] = sum(1 for _ in handle)
            result['phases'][name] = status
    audit = Path(plan['data_root'])/'scored/audit.json'
    if audit.exists():
        result['candidate_audit'] = json.loads(audit.read_text())
    for arm in plan['runs']:
        path = output/arm['run_id']/'manifest.json'
        if path.exists():
            data = json.loads(path.read_text())
            result['runs'][arm['run_id']] = {k: data[k] for k in ('state', 'last_complete_step', 'error') if k in data}
    pattern = re.compile(r'Traceback|CUDA out of memory|OutOfMemoryError|oom-kill|OUT_OF_MEMORY|CUDNN_STATUS_ALLOC_FAILED|Nonfinite|Non-finite')
    for path in (root/'logs').glob('*'):
        if path.is_file():
            for line in path.read_text(errors='replace').splitlines():
                if pattern.search(line):
                    result['errors'].append(dict(file=path.name, line=line[:400]))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = inspect(a.root)
    text = json.dumps(result, indent=2)
    print(text, flush=True)
    if a.output:
        with a.output.open('x', encoding='utf-8') as handle:
            handle.write(text)
