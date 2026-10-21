"""Read-only retrieval of reference90 numeric results; never download adapters."""
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

ROOT = '/work/users/y/u/yuhe32/ipo_runs/cyclic_history_reference90_20260929'
LAUNCH = '/work/users/y/u/yuhe32/ipo/diagnostics/history_reference90_20260929'
RUNS = [f'{method}_reference90_a1_b100_s0' for method in ('ipo', 'dpo')]
ARTIFACTS = ['manifest.json', 'metrics.csv', 'support.json', 'tokenization_audit.json', 'snapshots']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dest = args.output.absolute().resolve()
    if dest.exists():
        raise FileExistsError('Choose a new result directory; do not overwrite raw evidence')
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.load_host_keys(str(Path.home() / '.ssh/known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect('sycamore.unc.edu', username='yuhe32', password=os.environ['SYCAMORE_PASSWORD'],
                   look_for_keys=False, allow_agent=False, timeout=20, banner_timeout=20, auth_timeout=20)

    def remote(command):
        _, stdout, stderr = client.exec_command(command, timeout=180)
        data = stdout.read()
        error = stderr.read().decode(errors='replace')
        if stdout.channel.recv_exit_status():
            raise RuntimeError(error)
        return data

    try:
        status = json.loads(remote('/work/users/y/u/yuhe32/h100env312/bin/python ' +
                                   LAUNCH + '/inspect_reference90.py'))
        names = [run + '/' + name for run in RUNS for name in ARTIFACTS]
        archive = remote('tar --hard-dereference -czf - -C ' + shlex.quote(ROOT) + ' ' +
                         ' '.join(map(shlex.quote, names)))
        dest.mkdir(parents=True)
        raw = dest / 'raw'
        raw.mkdir()
        raw = raw.resolve()
        with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
            for member in tar.getmembers():
                target = (raw / member.name).resolve()
                if not target.is_relative_to(raw) or not (member.isfile() or member.isdir()):
                    raise ValueError(f'Unexpected archive member: {member.name!r}, type={member.type!r}, '
                                     f'target={str(target)!r}, root={str(raw)!r}')
            tar.extractall(raw)
        (dest / 'server_status.json').write_text(json.dumps(status, indent=2) + '\n')
        logs = dest / 'slurm_logs'
        logs.mkdir()
        with client.open_sftp() as sftp:
            for task in range(2):
                for suffix in ('out', 'err'):
                    name = f'hist_ref90_0929-4608669_{task}.{suffix}'
                    sftp.get('/work/users/y/u/yuhe32/ipo_runs/slurm_logs/' + name, str(logs / name))
        hashes = {p.relative_to(dest).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for folder in (raw, logs) for p in folder.rglob('*') if p.is_file()}
        receipt = dict(downloaded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       remote_root=ROOT, job_id='4608669', run_ids=RUNS, files_sha256=hashes,
                       archive_sha256=hashlib.sha256(archive).hexdigest())
        (dest / 'download_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(dict(files=len(hashes), compressed_bytes=len(archive),
                              steps={r['run_id']: r.get('last_complete_step') for r in status['runs']})))
    finally:
        client.close()


if __name__ == '__main__':
    main()
