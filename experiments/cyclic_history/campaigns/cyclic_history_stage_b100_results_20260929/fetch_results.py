"""Read-only SSH inspection and retrieval of Stage B numeric artifacts."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import tarfile

import paramiko

HERE = Path(__file__).resolve().parent
ROOT = '/work/users/y/u/yuhe32/ipo_runs/cyclic_history_stage_b100_20260929'
LAUNCH = '/work/users/y/u/yuhe32/ipo/diagnostics/history_stage_b100_20260929'
RUNS = [f'{method}_{arm}_b100_s0' for method in ('ipo', 'dpo')
        for arm in ('ordinary', 'reference', 'feedback')]
ARTIFACTS = ['manifest.json', 'metrics.csv', 'support.json', 'tokenization_audit.json',
             'verified_population_predictions.json', 'snapshots']
DEST = HERE / 'raw'
if DEST.exists():
    raise FileExistsError('Raw artifacts already exist; do not overwrite')

c = paramiko.SSHClient()
c.load_system_host_keys()
c.load_host_keys(str(Path.home() / '.ssh/known_hosts'))
c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect('sycamore.unc.edu', username='yuhe32', password=os.environ['SYCAMORE_PASSWORD'],
          look_for_keys=False, allow_agent=False, timeout=20, banner_timeout=20, auth_timeout=20)
try:
    cmd = '/work/users/y/u/yuhe32/h100env312/bin/python ' + LAUNCH + '/inspect_stage_b100.py'
    _, out, err = c.exec_command(cmd, timeout=120)
    status = out.read()
    errors = err.read().decode(errors='replace')
    if out.channel.recv_exit_status():
        raise RuntimeError(errors)
    status = json.loads(status)
    (HERE / 'server_status.json').write_text(json.dumps(status, indent=2) + '\n')

    names = [run + '/' + name for run in RUNS for name in ARTIFACTS]
    cmd = 'tar -czf - -C ' + shlex.quote(ROOT) + ' ' + ' '.join(map(shlex.quote, names))
    _, out, err = c.exec_command(cmd, timeout=120)
    data = out.read()
    errors = err.read().decode(errors='replace')
    if out.channel.recv_exit_status():
        raise RuntimeError(errors)
    DEST.mkdir()
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
        for member in tar.getmembers():
            target = (DEST / member.name).resolve()
            if not target.is_relative_to(DEST.resolve()) or not (member.isfile() or member.isdir()):
                raise ValueError('Unexpected archive path/type')
        tar.extractall(DEST)
    logs = HERE / 'slurm_logs'
    logs.mkdir()
    with c.open_sftp() as s:
        for task in range(6):
            for suffix in ('out', 'err'):
                name = f'hist_B100_0929-4605560_{task}.{suffix}'
                s.get('/work/users/y/u/yuhe32/ipo_runs/slurm_logs/' + name, str(logs / name))
    hashes = {p.relative_to(HERE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in (DEST, logs) for p in folder.rglob('*') if p.is_file()}
    receipt = dict(downloaded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   remote_root=ROOT, job_id='4605560', run_ids=RUNS, files_sha256=hashes,
                   archive_sha256=hashlib.sha256(data).hexdigest())
    (HERE / 'download_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'files': len(hashes), 'compressed_bytes': len(data),
                      'allocated_gpus': status['account']['allocated_gpus'],
                      'steps': {r['run']: r.get('last_complete_step') for r in status['runs']}}))
finally:
    c.close()
