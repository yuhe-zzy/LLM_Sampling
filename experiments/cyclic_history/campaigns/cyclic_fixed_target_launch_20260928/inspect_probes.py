"""Read-only account allocation, result, and error check for the fixed-target probes."""
import csv
import datetime
import json
from pathlib import Path
import re
import subprocess

import numpy as np

ROOT = Path('/work/users/y/u/yuhe32/ipo_runs/cyclic_fixed_target_20260928')
LOGS = Path('/work/users/y/u/yuhe32/ipo_runs/slurm_logs')
RUNS = ['ipo_ordinary_calibrated_s0', 'ipo_stable_calibrated_s0',
        'dpo_ordinary_calibrated_s0', 'dpo_stable_calibrated_s0']
JOB = '4605171'


def command(args):
    return subprocess.check_output(args, text=True).strip()


full = command(['squeue', '-a', '-r', '-h', '-o', '%i|%u|%U|%T|%b|%R|%j'])
owned, total = [], 0
for row in full.splitlines():
    f = row.split('|')
    if f[1].strip() != 'yuhe32' and f[2].strip() != '448057':
        continue
    detail = command(['scontrol', '-a', 'show', 'job', '-o', f[0].strip()])
    tres = re.search(r'\bAllocTRES=(\S+)', detail)
    gpu = re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)', tres[1]) if tres else None
    count = int(gpu[1]) if gpu else 0
    total += count
    owned.append(dict(queue=row, allocated_gpus=count, detail=detail))

results = []
for index, run in enumerate(RUNS):
    folder = ROOT/(run+'_fixed_t20')
    item = dict(task=index, run=run)
    if (folder/'manifest.json').exists():
        manifest = json.loads((folder/'manifest.json').read_text())
        item.update(state=manifest['state'], blocks=manifest['completed_blocks'],
                    checkpoint_replay_max_abs_error=manifest.get('checkpoint_replay_max_abs_error'),
                    error=manifest.get('error'))
    if (folder/'metrics.csv').exists():
        with (folder/'metrics.csv').open() as f:
            rows = list(csv.DictReader(f))
        item['latest_metrics'] = rows[-1]
        item['measurement_blocks'] = [r['block'] for r in rows]
    finite = True
    for path in (folder/'snapshots').glob('*.npz'):
        with np.load(path, allow_pickle=False) as z:
            finite = finite and all(np.isfinite(z[key]).all() for key in z.files)
    item['written_snapshots_finite'] = bool(finite)
    item['error_matches'] = []
    for suffix in ('out', 'err'):
        path = LOGS/f'hist_fixed_target_0928-{JOB}_{index}.{suffix}'
        if path.exists():
            item['error_matches'].extend(line for line in path.read_text(errors='replace').splitlines()
                if re.search(r'Traceback|CUDA out of memory|OutOfMemoryError|oom-kill|OUT_OF_MEMORY|'
                             r'CUDNN_STATUS_ALLOC_FAILED|FloatingPointError|AssertionError', line))
    results.append(item)
print(json.dumps(dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    allocated_gpus=total, owned_jobs=owned, results=results,
    accounting=command(['sacct', '-X', '-j', JOB, '-n', '-P',
                        '--format=JobID,State,ExitCode,Elapsed,AllocTRES'])), indent=2))
