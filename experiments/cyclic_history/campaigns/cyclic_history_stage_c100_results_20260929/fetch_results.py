"""Read-only Stage C retrieval; raw private artifacts stay outside Git."""
import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import tarfile

import paramiko

ROOT = '/work/users/y/u/yuhe32/ipo_runs/cyclic_history_stage_c100_20260929'
LAUNCH = '/work/users/y/u/yuhe32/ipo/diagnostics/history_stage_c100_20260929'
RUNS = ['ipo_mixed_c100_s0', 'dpo_mixed_c100_s0']
ARTIFACTS = ['manifest.json', 'metrics.csv', 'support.json', 'tokenization_audit.json',
             'verified_population_predictions.json', 'snapshots']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    dest = args.output_dir.resolve()
    if dest.exists() and any(p.is_file() for p in dest.rglob('*')):
        raise FileExistsError('Choose a new download directory; no overwrite or partial retry')
    c = paramiko.SSHClient()
    c.load_system_host_keys()
    c.load_host_keys(str(Path.home() / '.ssh/known_hosts'))
    c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect('sycamore.unc.edu', username='yuhe32', password=os.environ['SYCAMORE_PASSWORD'],
              look_for_keys=False, allow_agent=False, timeout=20, banner_timeout=20, auth_timeout=20)
    try:
        def command(cmd):
            _, out, err = c.exec_command(cmd, timeout=180)
            result, errors = out.read(), err.read().decode(errors='replace')
            if out.channel.recv_exit_status():
                raise RuntimeError(errors)
            return result

        status = json.loads(command('/work/users/y/u/yuhe32/h100env312/bin/python ' +
                                    LAUNCH + '/inspect_stage_c100.py'))
        names = [run + '/' + name for run in RUNS for name in ARTIFACTS]
        archive = command('tar -czf - -C ' + shlex.quote(ROOT) + ' ' +
                          ' '.join(map(shlex.quote, names)))
        raw = dest / 'raw'
        raw.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
            for member in tar.getmembers():
                target = (raw / member.name).resolve()
                if not target.is_relative_to(raw) or not (member.isfile() or member.isdir()):
                    raise ValueError(f'Unexpected archive member: {member.name!r}, type={member.type!r}, link={member.linkname!r}')
            tar.extractall(raw)
        logs = dest / 'slurm_logs'
        logs.mkdir()
        with c.open_sftp() as sftp:
            for task in range(2):
                for suffix in ('out', 'err'):
                    name = f'hist_C100_0929-4606367_{task}.{suffix}'
                    sftp.get('/work/users/y/u/yuhe32/ipo_runs/slurm_logs/' + name, str(logs/name))
        hashes = {p.relative_to(dest).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for folder in (raw, logs) for p in folder.rglob('*') if p.is_file()}
        receipt = dict(downloaded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       remote_root=ROOT, job_id='4606367', files_sha256=hashes,
                       archive_sha256=hashlib.sha256(archive).hexdigest())
        (dest/'server_status.json').write_text(json.dumps(status, indent=2)+'\n')
        (dest/'download_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(dict(files=len(hashes), bytes=len(archive),
            allocated_gpus=status['account']['allocated_gpus'], runs=status['runs']), indent=2))
    finally:
        c.close()


if __name__ == '__main__':
    main()
