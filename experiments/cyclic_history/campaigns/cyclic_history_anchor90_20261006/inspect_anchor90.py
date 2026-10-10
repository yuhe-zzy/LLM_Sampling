"""Read-only inspection; complete metrics bound snapshot validation during writes."""
import csv
import json
from pathlib import Path
import re
import numpy as np
from queue_anchor90 import ROOT, OUTPUT, account_snapshot, command


def inspect():
    receipt = json.loads((ROOT/'submission_receipt.json').read_text())
    job = receipt['job_id']
    report = dict(account=account_snapshot(), receipt=receipt,
        sacct=command(['sacct', '-j', job, '--format=JobID,State,ExitCode,Elapsed,AllocTRES%80', '-P']), runs=[])
    for index, run in enumerate(receipt['run_ids']):
        path = OUTPUT/run
        item = dict(task=index, run_id=run)
        if (path/'manifest.json').exists():
            manifest = json.loads((path/'manifest.json').read_text())
            item.update({k:manifest.get(k) for k in ('state', 'last_complete_step', 'calibration_gate', 'error', 'reference_coefficients')})
        if (path/'metrics.csv').exists():
            with (path/'metrics.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            steps = [int(row['step']) for row in rows]
            item.update(metric_count=len(rows), contiguous_steps=steps==list(range(len(rows))),
                        last_metrics=rows[-1] if rows else None, nonfinite=[], missing_snapshots=[])
            for step in steps:
                name = path/'snapshots'/f'step_{step:04d}.npz'
                if not name.exists():
                    item['missing_snapshots'].append(step)
                    continue
                with np.load(name, allow_pickle=False) as data:
                    for key in data.files:
                        if np.issubdtype(data[key].dtype, np.number) and not np.isfinite(data[key]).all():
                            item['nonfinite'].append([name.name, key])
        item['error_matches'] = {}
        for extension in ('out', 'err'):
            log = Path('/work/users/y/u/yuhe32/ipo_runs/slurm_logs')/f'hist_anchor_1006-{job}_{index}.{extension}'
            if log.exists():
                pattern = r'Traceback|CUDA out of memory|OutOfMemoryError|oom-kill|OUT_OF_MEMORY|CUDNN_STATUS_ALLOC_FAILED|FloatingPointError|Non-finite|nonfinite'
                item['error_matches'][extension] = [line[:500] for line in log.read_text(errors='replace').splitlines()
                                                   if re.search(pattern, line, re.I)]
        report['runs'].append(item)
    return report


if __name__ == '__main__':
    print(json.dumps(inspect(), indent=2))
