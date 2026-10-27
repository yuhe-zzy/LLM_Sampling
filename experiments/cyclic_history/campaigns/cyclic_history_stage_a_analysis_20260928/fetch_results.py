"""Download completed Stage A artifacts without changing server files or jobs."""
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
ROOT = '/work/users/y/u/yuhe32/ipo_runs/cyclic_history_calibrated_20260928/cyclic_history_calibrated'
RUNS = ['ipo_ordinary_calibrated_s0', 'ipo_stable_calibrated_s0',
        'dpo_ordinary_calibrated_s0', 'dpo_stable_calibrated_s0']
ARTIFACTS = ['manifest.json', 'metrics.csv', 'support.json', 'tokenization_audit.json',
             'verified_population_predictions.json', 'snapshots']
DEST = HERE / 'raw'
if DEST.exists():
    raise FileExistsError('Raw artifacts already exist; reuse rather than overwrite')

c = paramiko.SSHClient()
c.load_system_host_keys()
c.load_host_keys(str(Path.home() / '.ssh/known_hosts'))
c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect('sycamore.unc.edu', username='yuhe32', password=os.environ['SYCAMORE_PASSWORD'],
          look_for_keys=False, allow_agent=False, timeout=20)
try:
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
    hashes = {p.relative_to(DEST).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in DEST.rglob('*') if p.is_file()}
    receipt = dict(downloaded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   remote_root=ROOT, job_id='4603433', run_ids=RUNS, files_sha256=hashes,
                   archive_sha256=hashlib.sha256(data).hexdigest())
    (HERE / 'download_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'files': len(hashes), 'compressed_bytes': len(data), 'runs': RUNS}))
finally:
    c.close()
