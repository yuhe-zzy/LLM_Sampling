"""Read-only account, Slurm, metric, snapshot and error inspection."""
import csv
import json
from pathlib import Path
import re
import numpy as np
from queue_sweep import ROOT, OUTPUT, account_snapshot, command

receipt=json.loads((ROOT/'submission_receipt.json').read_text())
job=receipt['job_id']
report=dict(account=account_snapshot(),receipt=receipt,
    sacct=command(['sacct','-j',job,'--format=JobID,State,ExitCode,Elapsed,AllocTRES%80','-P']),runs=[])
for index,run in enumerate(receipt['run_ids']):
    path=OUTPUT/run
    item=dict(task=index,run_id=run)
    if (path/'manifest.json').exists():
        m=json.loads((path/'manifest.json').read_text())
        item.update({key:m.get(key) for key in ('state','last_complete_step','calibration_gate','error')})
    if (path/'metrics.csv').exists():
        with (path/'metrics.csv').open() as f:
            rows=list(csv.DictReader(f))
        item['metric_count']=len(rows)
        item['contiguous_steps']=[int(r['step']) for r in rows]==list(range(len(rows)))
        item['last_metrics']=rows[-1] if rows else None
    snapshots=sorted((path/'snapshots').glob('step_*.npz'))
    bad=[]
    for name in snapshots:
        with np.load(name,allow_pickle=False) as data:
            for key in data.files:
                if np.issubdtype(data[key].dtype,np.number) and not np.isfinite(data[key]).all():
                    bad.append([name.name,key])
    item.update(snapshot_count=len(snapshots),nonfinite=bad)
    item['error_matches']={}
    for extension in ('out','err'):
        log=Path('/work/users/y/u/yuhe32/ipo_runs/slurm_logs')/f'hist_trend_1003-{job}_{index}.{extension}'
        if log.exists():
            pattern=r'Traceback|CUDA out of memory|OutOfMemoryError|oom-kill|OUT_OF_MEMORY|CUDNN_STATUS_ALLOC_FAILED|FloatingPointError|Non-finite|nonfinite'
            item['error_matches'][extension]=[line[:500] for line in log.read_text(errors='replace').splitlines()
                                              if re.search(pattern,line,re.I)]
    report['runs'].append(item)
print(json.dumps(report,indent=2))
