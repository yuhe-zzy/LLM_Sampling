"""Read-only inspection, no scheduler mutation or training."""
import csv
import json
from pathlib import Path
import re
import numpy as np

from queue_reference90 import ROOT, OUTPUT, RUNS, account_snapshot, command

receipt = json.loads((ROOT/'submission_receipt.json').read_text())
job = receipt['job_id']
report = dict(account=account_snapshot(), receipt=receipt,
              sacct=command(['sacct','-j',job,'--format=JobID,State,ExitCode,Elapsed,AllocTRES%80','-P']),runs=[])
for index, run in enumerate(RUNS):
    directory = OUTPUT/run
    item = dict(run_id=run)
    manifest = directory/'manifest.json'
    if manifest.exists():
        m = json.loads(manifest.read_text())
        item.update({k:m.get(k) for k in ('state','last_complete_step','calibration_gate',
            'population_prediction','reference_coefficients','error')})
    metrics = directory/'metrics.csv'
    if metrics.exists():
        with metrics.open() as f:
            rows = list(csv.DictReader(f))
        item['metrics_rows'] = len(rows)
        item['last_metrics'] = rows[-1] if rows else None
        item['contiguous_steps'] = [int(r['step']) for r in rows] == list(range(len(rows)))
    snapshots = sorted((directory/'snapshots').glob('step_*.npz'))
    invalid = []
    for path in snapshots:
        with np.load(path,allow_pickle=False) as snapshot:
            for key in snapshot.files:
                x = snapshot[key]
                if np.issubdtype(x.dtype,np.number) and not np.isfinite(x).all():
                    invalid.append([path.name,key])
    item.update(snapshot_count=len(snapshots),nonfinite=invalid)
    logs = {}
    for extension in ('out','err'):
        path = Path('/work/users/y/u/yuhe32/ipo_runs/slurm_logs')/f'hist_ref90_0929-{job}_{index}.{extension}'
        if path.exists():
            text = path.read_text(errors='replace')
            pattern = r'Traceback|CUDA out of memory|OutOfMemoryError|oom-kill|OUT_OF_MEMORY|CUDNN_STATUS_ALLOC_FAILED|FloatingPointError|Non-finite|nonfinite'
            logs[extension] = [line[:500] for line in text.splitlines() if re.search(pattern,line,re.I)]
    item['error_matches'] = logs
    report['runs'].append(item)
print(json.dumps(report,indent=2))
