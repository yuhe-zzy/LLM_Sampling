"""Read-only completed-run retrieval; no scheduler changes or model artifacts."""
import argparse
from collections import defaultdict
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
ARTIFACTS = ('manifest.json', 'metrics.csv', 'support.json', 'tokenization_audit.json', 'snapshots')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dest = args.output.absolute().resolve()
    if (dest/'raw').exists() or (dest/'download_receipt.json').exists():
        raise FileExistsError('Do not overwrite an earlier download')
    status = json.loads((dest/'server_status.json').read_text())
    plan = json.loads((HERE/'experiment_plan.json').read_text())
    eligible = {r['run_id'] for r in status['runs']
                if r.get('state') == 'COMPLETED' and r.get('last_complete_step') == 100}
    groups = defaultdict(list)
    mappings = []
    for row in plan['comparisons']:
        if row['status'] == 'NEW' and row['run_id'] not in eligible:
            continue
        remote_root = row.get('existing_root', plan['common']['output_root'])
        actual_id = row.get('existing_run_id', row['run_id'])
        groups[remote_root].append(actual_id)
        mappings.append(dict(logical_run_id=row['run_id'], actual_run_id=actual_id,
                             remote_root=remote_root, reused=row['status'] != 'NEW'))
    raw = dest/'raw'
    raw.mkdir()
    raw = raw.resolve()
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.load_host_keys(str(Path.home()/'.ssh/known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect('sycamore.unc.edu', username='yuhe32', password=os.environ['SYCAMORE_PASSWORD'],
                   look_for_keys=False, allow_agent=False, timeout=20, banner_timeout=20, auth_timeout=20)
    archives = []
    try:
        for remote_root, runs in groups.items():
            names = [run+'/'+name for run in runs for name in ARTIFACTS]
            cmd = 'tar --hard-dereference -czf - -C '+shlex.quote(remote_root)+' '+ ' '.join(map(shlex.quote,names))
            _, stdout, stderr = client.exec_command(cmd, timeout=240)
            archive = stdout.read()
            error = stderr.read().decode(errors='replace')
            if stdout.channel.recv_exit_status():
                raise RuntimeError(error)
            with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tar:
                for member in tar.getmembers():
                    target = (raw/member.name).resolve()
                    if not target.is_relative_to(raw) or not (member.isfile() or member.isdir()):
                        raise ValueError('Unexpected archive member: '+member.name)
                    if target.exists():
                        raise FileExistsError(target)
                tar.extractall(raw)
            archives.append(dict(remote_root=remote_root,runs=runs,bytes=len(archive),
                                 sha256=hashlib.sha256(archive).hexdigest()))
            print(json.dumps(dict(downloaded_runs=len(runs),compressed_bytes=len(archive))),flush=True)
        logs = dest/'slurm_logs'
        logs.mkdir()
        with client.open_sftp() as sftp:
            for row in plan['comparisons']:
                if row['status'] != 'NEW' or row['run_id'] not in eligible:
                    continue
                for suffix in ('out','err'):
                    name = f"hist_trend_1003-4659985_{row['task_index']}.{suffix}"
                    sftp.get('/work/users/y/u/yuhe32/ipo_runs/slurm_logs/'+name,str(logs/name))
        hashes = {p.relative_to(dest).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                  for folder in (raw,logs) for p in folder.rglob('*') if p.is_file()}
        receipt = dict(downloaded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       mappings=mappings,archives=archives,files_sha256=hashes,
                       new_completed_runs=len(eligible),reused_runs=len(mappings)-len(eligible))
        with (dest/'download_receipt.json').open('x') as f:
            json.dump(receipt,f,indent=2)
        print(json.dumps(dict(files=len(hashes),completed_new=len(eligible),total_runs=len(mappings))))
    finally:
        client.close()


if __name__ == '__main__':
    main()
