"""One-shot guarded deployment/submission of the four approved fixed-target probes."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'source'
PARENTS = Path('/work/users/y/u/yuhe32/ipo_runs/cyclic_history_calibrated_20260928/cyclic_history_calibrated')
OUTPUT = Path('/work/users/y/u/yuhe32/ipo_runs/cyclic_fixed_target_20260928')
RUNS = ['ipo_ordinary_calibrated_s0', 'ipo_stable_calibrated_s0',
        'dpo_ordinary_calibrated_s0', 'dpo_stable_calibrated_s0']


def command(args):
    p = subprocess.run(args, capture_output=True, text=True, check=False)
    if p.returncode:
        raise RuntimeError(f'{args}: {p.stderr}')
    return p.stdout.strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')


def verify():
    deployment = json.loads((ROOT/'deployment.json').read_text())
    for name, digest in deployment['sha256'].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError('Immutable source changed: ' + name)
    print('PASS immutable probe deployment', flush=True)


def prepare(expected_sha):
    if sha(ROOT/'source.tar') != expected_sha:
        raise RuntimeError('Source archive hash mismatch')
    SOURCE.mkdir(exist_ok=False)
    with tarfile.open(ROOT/'source.tar') as tar:
        tar.extractall(SOURCE, filter='data')
    command(['bash', '-n', str(ROOT/'run_probes.sh')])
    sys.path.insert(0, str(SOURCE))
    from run_fixed_target_probe import prepare as prepare_probe
    provenance = {}
    for run in RUNS:
        context = prepare_probe(PARENTS/run)
        provenance[run] = dict(source_job_id=context['manifest']['slurm_job_id'],
            adapter_sha256=sha(context['adapter']/'adapter_model.safetensors'),
            manifest_sha256=sha(PARENTS/run/'manifest.json'),
            step20_sha256=sha(PARENTS/run/'snapshots/step_0020.npz'))
        print('PASS source/checkpoint/target preflight', run, flush=True)
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in SOURCE.rglob('*.py')}
    for name in ('run_probes.sh', 'queue_probes.py'):
        hashes[name] = sha(ROOT/name)
    write_new(ROOT/'deployment.json', dict(sha256=hashes, source_archive_sha256=expected_sha,
              parent_runs=provenance, output_root=str(OUTPUT), run_ids=RUNS,
              gpu_per_task=1, throttle=4, account_ceiling=6,
              approved_scope='Four fixed round-20 targets, 10/30/60 epochs; no Stage B/C'))
    verify()


def queue_snapshot():
    # squeue -u has missed this account on this cluster; inspect all owners/UIDs.
    full = command(['squeue', '-a', '-r', '-h', '-o', '%i|%u|%U|%T|%b|%R|%j'])
    owned, total = [], 0
    for row in full.splitlines():
        fields = [x.strip() for x in row.split('|')]
        if len(fields) < 3 or not (fields[1] == 'yuhe32' or fields[2] == '448057'):
            continue
        detail = command(['scontrol', '-a', 'show', 'job', '-o', fields[0]])
        if 'UserId=yuhe32(448057)' not in detail:
            raise RuntimeError('Unexpected owner identity')
        tres = re.search(r'\bAllocTRES=(\S+)', detail)
        gpu = re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)', tres[1]) if tres else None
        total += int(gpu[1]) if gpu else 0
        owned.append(dict(queue=row, detail=detail))
    return dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                owned_jobs=owned, allocated_gpus=total)


def submit():
    verify()
    if (ROOT/'submission_intent.json').exists() or (ROOT/'submission_receipt.json').exists():
        raise RuntimeError('Submission already attempted: inspect receipt/queue, do not duplicate')
    preflight = queue_snapshot()
    # The current plan assumes an empty account. Abort on new work, including pending jobs.
    if preflight['owned_jobs']:
        raise RuntimeError('Owned running/pending/releasing jobs appeared: review budget first')
    accounting = command(['sacct', '-X', '-j', '4603433', '-n', '-P', '--format=JobID,State,ExitCode'])
    states = {r.split('|')[0]: r.split('|')[1:3] for r in accounting.splitlines()
              if re.match(r'4603433_[0-3]\|', r)}
    if len(states) != 4 or any(s != ['COMPLETED', '0:0'] for s in states.values()):
        raise RuntimeError('Incomplete/failed parent accounting')
    if OUTPUT.exists():
        raise RuntimeError('Probe output already exists; refuse duplicate campaign')
    args = ['sbatch', '--parsable', '--array=0-3%4', '--chdir='+str(SOURCE), str(ROOT/'run_probes.sh')]
    write_new(ROOT/'submission_intent.json', dict(command=args, preflight=preflight, parent_states=states))
    result = command(args)
    jid = result.split(';')[0]
    if not jid.isdigit():
        raise RuntimeError('Ambiguous scheduler response; inspect before retry: ' + result)
    receipt = dict(job_id=jid, submitted_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   run_ids=RUNS, array='0-3%4', gpu_per_task=1, outer_updates=0,
                   scheduler_response=result, output_root=str(OUTPUT))
    write_new(ROOT/'submission_receipt.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    snapshot = queue_snapshot()
    if snapshot['allocated_gpus'] > 6:
        raise RuntimeError('Post-submit total exceeds six; unexpected independent work needs review')
    write_new(ROOT/'queue_verified.json', snapshot)
    print(json.dumps(snapshot, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', metavar='SHA256')
    group.add_argument('--verify', action='store_true')
    group.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    elif args.verify:
        verify()
    else:
        submit()
