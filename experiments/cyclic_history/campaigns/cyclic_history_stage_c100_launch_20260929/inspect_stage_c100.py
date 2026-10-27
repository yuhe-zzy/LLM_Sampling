"""Read-only startup/progress inspection for the two remaining Stage C arms."""
import csv
import json
from pathlib import Path
import re

import numpy as np

from queue_stage_c100 import ROOT, OUTPUT, RUNS, account_snapshot, command

receipt = json.loads((ROOT/'submission_receipt.json').read_text())
jid = receipt['job_id']
account = account_snapshot()
rows = []
for index, run in enumerate(RUNS):
    folder = OUTPUT/run
    item = dict(task=index, run=run)
    if (folder/'manifest.json').exists():
        m = json.loads((folder/'manifest.json').read_text())
        item.update(state=m['state'], last_complete_step=m.get('last_complete_step'),
                    iters=m['config']['iters'], calibration_gate=m.get('calibration_gate'),
                    error=m.get('error'))
    if (folder/'metrics.csv').exists():
        with (folder/'metrics.csv').open() as f:
            metrics = list(csv.DictReader(f))
        steps = [int(r['step']) for r in metrics]
        item['steps_contiguous_from_zero'] = steps == list(range(steps[-1]+1))
        item['latest_metrics'] = metrics[-1]
    item['error_matches'] = []
    for suffix in ('out','err'):
        path = Path('/work/users/y/u/yuhe32/ipo_runs/slurm_logs')/f'hist_C100_0929-{jid}_{index}.{suffix}'
        if path.exists():
            item['error_matches'].extend(line for line in path.read_text(errors='replace').splitlines()
                if re.search(r'Traceback|CUDA out of memory|OutOfMemoryError|oom-kill|OUT_OF_MEMORY|'
                             r'CUDNN_STATUS_ALLOC_FAILED|FloatingPointError|AssertionError|non[- ]?finite',line,re.I))
    snapshots = sorted((folder/'snapshots').glob('step_*.npz'))
    item['snapshot_count'] = len(snapshots)
    if snapshots:
        with np.load(snapshots[-1],allow_pickle=False) as data:
            item['latest_snapshot_finite'] = bool(all(np.isfinite(data[k]).all() for k in data.files))
    rows.append(item)
print(json.dumps(dict(job_id=jid, account=account, runs=rows,
    accounting=command(['sacct','-X','-j',jid,'-n','-P','--format=JobID,State,ExitCode,Elapsed,AllocTRES'])),indent=2))
