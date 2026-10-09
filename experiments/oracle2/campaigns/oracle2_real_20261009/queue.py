"""Idempotent phase submission. Empty full-account queue required; never auto-chain phases."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path('/work/users/y/u/yuhe32/ipo/diagnostics/oracle2_real_20261009')
PYTHON = '/work/users/y/u/yuhe32/h100env312/bin/python'
RESOURCES = {
    'audit': dict(gpus=3, tasks=1, memory='256G', time='12:00:00'),
    'baseline': dict(gpus=1, tasks=1, memory='80G', time='03:00:00'),
    'train': dict(gpus=1, tasks=6, memory='80G', time='5-00:00:00'),
    'generate': dict(gpus=1, tasks=6, memory='80G', time='12:00:00'),
    'wrscore': dict(gpus=3, tasks=1, memory='256G', time='5-00:00:00'),
}


def run(args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def save_new(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2)


def verify(root):
    manifest = json.loads((root / 'deployment.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Deployment changed: {name}')
    return manifest


def owned_snapshot():
    raw = run(['squeue', '-a', '-r', '-h', '-o', '%i|%u|%U|%T|%b|%R'])
    owned = []
    for line in raw.splitlines():
        fields = line.split('|')
        if len(fields) != 6:
            raise ValueError('Unparseable full queue')
        if fields[1] == 'yuhe32' or fields[2] == '448057':
            owned.append(line)
    control = run(['scontrol', '-a', 'show', 'job', '-o'])
    control_owned = [line for line in control.splitlines() if re.search(
        r'\bUserId=(?:yuhe32\(\d+\)|[^ ()]+\(448057\))', line)]
    return dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
                full_queue_rows=len(raw.splitlines()), owned_queue=owned, owned_scontrol=control_owned,
                allocated_gpus=0 if not owned and not control_owned else None,
                hard_account_cap_installed=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--verify', action='store_true')
    p.add_argument('--phase', choices=list(RESOURCES))
    p.add_argument('--submit', action='store_true')
    a = p.parse_args()
    deployment = verify(a.root)
    if a.verify:
        print('VERIFIED', deployment['source_commit'])
        return
    if a.phase is None:
        p.error('--phase required')
    resource = RESOURCES[a.phase]
    source = a.root / 'source/experiments/oracle2'
    if a.phase in ('train', 'generate', 'wrscore'):
        for arm in json.loads((a.root / 'plan.json').read_text())['runs']:
            run([PYTHON, str(source / 'train_oracle2.py'), '--plan', str(a.root / 'plan.json'),
                 '--review', str(a.root / 'audit_review.json'), '--run-id', arm['run_id'],
                 '--source-commit', deployment['source_commit']])
    intent = a.root / f'{a.phase}_submission_intent.json'
    receipt = a.root / f'{a.phase}_submission_receipt.json'
    if intent.exists() or receipt.exists():
        raise FileExistsError('Existing intent/receipt: inspect live scheduler; never blindly retry')
    snapshot = owned_snapshot()
    print(json.dumps(snapshot), flush=True)
    if snapshot['owned_queue'] or snapshot['owned_scontrol']:
        raise RuntimeError('Account not empty. Leave this phase unsubmitted; review dependencies/budget')
    script = source / 'campaigns/oracle2_real_20261009/run_phase.sh'
    command = ['sbatch', '--parsable', '--partition=h100_all', '--account=rc_fanyao_pi',
               f'--gres=gpu:{resource["gpus"]}', '--cpus-per-task=8', f'--mem={resource["memory"]}',
               f'--time={resource["time"]}', '--no-requeue', f'--job-name=oracle2_{a.phase}',
               f'--output={a.root}/logs/{a.phase}-%A_%a.out', f'--error={a.root}/logs/{a.phase}-%A_%a.err']
    if resource['tasks'] > 1:
        command.append(f'--array=0-{resource["tasks"]-1}%{resource["tasks"]}')
    command += [str(script), a.phase]
    if not a.submit:
        print(json.dumps(dict(preview=command)))
        return
    (a.root / 'logs').mkdir(exist_ok=True)
    save_new(intent, dict(source_commit=deployment['source_commit'], command=command, preflight=snapshot,
                          phase=a.phase, maximum_phase_gpus=resource['gpus']*resource['tasks']))
    job_id = run(command)
    if not job_id.split(';')[0].isdigit():
        raise RuntimeError('Ambiguous sbatch result; preserve intent and inspect scheduler before any retry')
    save_new(receipt, dict(job_id=job_id, phase=a.phase, source_commit=deployment['source_commit'],
                           submitted_at_utc=datetime.now(timezone.utc).isoformat(), resources=resource))
    print(json.dumps(dict(submitted_job_id=job_id, phase=a.phase)), flush=True)


if __name__ == '__main__':
    main()
