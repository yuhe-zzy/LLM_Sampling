"""Deploy and queue only the user-approved Stage A; never cancel other jobs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'
OLD = '4593149'
COMMIT = 'f0fe034cc0d8e3bef35b0f5e02806ee81340da76'
OUTPUT = Path('/work/users/y/u/yuhe32/ipo_runs/cyclic_history_calibrated_20260928')
PLAN = SOURCE / 'experiments/cyclic_history/calibration/2026-09-28/experiment_plan.json'
RUNS = ['ipo_ordinary_calibrated_s0', 'ipo_stable_calibrated_s0',
        'dpo_ordinary_calibrated_s0', 'dpo_stable_calibrated_s0']


def command(args):
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f'{args}: {p.stderr}')
    return p.stdout.strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, obj):
    with path.open('x') as stream:
        json.dump(obj, stream, indent=2)
        stream.write('\n')


def verify():
    manifest = json.loads((ROOT / 'deployment.json').read_text())
    for name, expected in manifest['sha256'].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f'Deployed source changed: {name}')
    print('Verified immutable deployment', manifest['commit'], flush=True)


def prepare(expected_archive_hash):
    archive = ROOT / 'source.tar'
    if sha(archive) != expected_archive_hash:
        raise RuntimeError('Source archive hash mismatch')
    SOURCE.mkdir(exist_ok=False)
    with tarfile.open(archive) as tar:
        tar.extractall(SOURCE, filter='data')
    command(['bash', '-n', str(ROOT / 'run_stage_a.sh')])
    command(['bash', '-n', str(SOURCE / 'slurm/cyclic_history_calibrated.sh')])
    sys.path.insert(0, str(SOURCE / 'experiments/cyclic_history'))
    from run_cyclic_history import resolve_config, validate_config
    from history_math import load_panels
    from calibrated_protocol import transform_panels, verify_calibration
    data = '/work/users/y/u/yuhe32/ipo/data/processed/eval_prompt_responses_cyclic_1000.jsonl'
    for name in RUNS:
        cfg = resolve_config(PLAN, name)
        validate_config(cfg)
        panels = load_panels(data, cfg['num_prompts'], cfg['support_seed'], cfg['keep_k'],
                             prompt_ids=cfg['panel_ids'])
        verify_calibration(transform_panels(panels, cfg), cfg)
        print('PASS', name, 'CPU source/data/calibration preflight', flush=True)
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in SOURCE.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    hashes['run_stage_a.sh'] = sha(ROOT / 'run_stage_a.sh')
    hashes['queue_stage_a.py'] = sha(Path(__file__).resolve())
    write_new(ROOT / 'deployment.json', dict(commit=COMMIT, archive_sha256=expected_archive_hash,
              sha256=hashes, run_ids=RUNS, dependency='afterok:' + OLD,
              array='0-3%4', gpu_per_task=1, account_gpu_ceiling=6,
              output_root=str(OUTPUT / 'cyclic_history_calibrated'),
              approved_scope='Stage A only; no Stage B/C submission'))
    verify()


def preflight():
    queue = command(['squeue', '-a', '-r', '-h', '-o', '%i|%u|%U|%T|%b|%R|%j'])
    owned = []
    total = 0
    for row in queue.splitlines():
        parts = [s.strip() for s in row.split('|')]
        if len(parts) < 3 or not (parts[1] == 'yuhe32' or parts[2] == '448057'):
            continue
        jid = parts[0]
        if not re.fullmatch(OLD + r'_[0-5]', jid):
            raise RuntimeError(f'Unexpected owned running/pending job: {row}; do not submit')
        detail = command(['scontrol', '-a', 'show', 'job', '-o', jid])
        if 'UserId=yuhe32(448057)' not in detail:
            raise RuntimeError('Job owner mismatch')
        match = re.search(r'\bAllocTRES=(\S+)', detail)
        if match:
            generic = re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)', match[1])
            total += int(generic[1]) if generic else 0
        owned.append(dict(queue=row, detail=detail))
    if total > 6:
        raise RuntimeError(f'Account already over the approved ceiling: {total}')
    accounting = command(['sacct', '-X', '-j', OLD, '-n', '-P',
                           '--format=JobID,State,ExitCode'])
    states = {row.split('|')[0]: row.split('|')[1].split()[0]
              for row in accounting.splitlines() if re.match(OLD + r'_[0-5]\|', row)}
    if set(states) != {OLD + '_' + str(i) for i in range(6)}:
        raise RuntimeError('Predecessor accounting is incomplete')
    if any(s not in ('RUNNING', 'PENDING', 'COMPLETING', 'COMPLETED') for s in states.values()):
        raise RuntimeError(f'Predecessor failure needs review: {states}')
    if OUTPUT.exists():
        raise RuntimeError('Output directory already exists; possible duplicate campaign')
    return dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                owned_jobs=owned, allocated_gpus=total, predecessor_states=states)


def submit():
    verify()
    if (ROOT / 'submission_intent.json').exists() or (ROOT / 'submission_receipt.json').exists():
        raise RuntimeError('Submission already attempted; inspect receipt/queue, never resubmit blindly')
    state = preflight()
    args = ['sbatch', '--parsable', '--array=0-3%4', '--dependency=afterok:' + OLD,
            '--chdir=' + str(SOURCE), str(ROOT / 'run_stage_a.sh')]
    # Persist intent before the scheduler call: an interrupted connection must
    # not lead to a second array on retry.
    write_new(ROOT / 'submission_intent.json', dict(command=args, preflight=state))
    result = command(args)
    jid = result.split(';')[0]
    if not jid.isdigit():
        raise RuntimeError(f'Unexpected sbatch receipt; inspect before retry: {result}')
    receipt = dict(job_id=jid, submitted_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   dependency='afterok:' + OLD, array='0-3%4', run_ids=RUNS,
                   gpu_per_task=1, commit=COMMIT, scheduler_response=result)
    write_new(ROOT / 'submission_receipt.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', metavar='ARCHIVE_SHA256')
    action.add_argument('--submit', action='store_true')
    action.add_argument('--verify', action='store_true')
    options = parser.parse_args()
    if options.prepare:
        prepare(options.prepare)
    elif options.submit:
        submit()
    else:
        verify()
