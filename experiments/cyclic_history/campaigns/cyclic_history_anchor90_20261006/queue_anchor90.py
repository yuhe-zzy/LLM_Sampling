"""Immutable two-arm deployment; submission requires an empty full-account queue."""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'source'
OUTPUT = Path('/work/users/y/u/yuhe32/ipo_runs/cyclic_history_anchor90_20261006')


def command(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'{args}: {result.stdout}\n{result.stderr}')
    return result.stdout.strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def account_snapshot():
    path = SOURCE/'experiments/cyclic_history/campaigns/cyclic_history_reference90_20260929/queue_reference90.py'
    spec = importlib.util.spec_from_file_location('reviewed_accounting', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.account_snapshot()


def verify():
    deployment = json.loads((ROOT/'deployment.json').read_text())
    for name, digest in deployment['sha256'].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError('Immutable deployment changed: '+name)
    return deployment


def prepare(expected, commit):
    if sha(ROOT/'source.tar')!=expected or not re.fullmatch('[0-9a-f]{40}', commit or ''):
        raise ValueError('Source archive hash or commit invalid')
    if (ROOT/'submission_intent.json').exists():
        raise RuntimeError('Submission already attempted')
    SOURCE.mkdir(exist_ok=False)
    with tarfile.open(ROOT/'source.tar') as archive:
        archive.extractall(SOURCE, filter='data')
    code = SOURCE/'experiments/cyclic_history'
    campaign = code/'campaigns/cyclic_history_anchor90_20261006'
    sys.path[:0] = [str(campaign), str(code)]
    from build_plan import make_plan
    from calibrated_protocol import transform_panels, verify_support_calibration
    from history_math import load_panels
    from run_cyclic_history import resolve_config, validate_config
    plan = json.loads((ROOT/'experiment_plan.json').read_text())
    if plan != make_plan() or len(plan['runs']) != 2:
        raise ValueError('Deployment differs from the approved two-arm plan')
    audit = []
    for row in plan['runs']:
        cfg = resolve_config(ROOT/'experiment_plan.json', row['run_id'])
        validate_config(cfg)
        panels = load_panels(cfg['eval_path'], cfg['num_prompts'], cfg['support_seed'],
                             cfg['keep_k'], prompt_ids=cfg['panel_ids'])
        verify_support_calibration(transform_panels(panels, cfg), cfg)
        audit.append(dict(run_id=cfg['run_id'], config='PASS', support='PASS',
                          reference_coefficients=dict(initial=.9, current=.1, previous=0)))
    command(['bash', '-n', str(ROOT/'run_anchor90.sh')])
    tests = []
    for location, pattern in [(code, 'test_*.py'), (campaign, 'test_plan.py')]:
        result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(location),
                                 '-p', pattern], capture_output=True, text=True)
        output = result.stdout+'\n'+result.stderr
        print(output, flush=True)
        if result.returncode or 'skipped=' in output:
            raise RuntimeError('CPU tests failed or skipped')
        tests.append(output)
    hashes = {p.relative_to(ROOT).as_posix():sha(p) for p in SOURCE.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    for name in ('experiment_plan.json', 'run_anchor90.sh', 'queue_anchor90.py', 'inspect_anchor90.py'):
        hashes[name] = sha(ROOT/name)
    write_new(ROOT/'deployment.json', dict(source_git_commit=commit, source_archive_sha256=expected,
        sha256=hashes, arm_audit=audit, cpu_test_stdout=tests, array='0-1%2',
        gpu_per_task=1, account_ceiling=6, output_root=str(OUTPUT)))
    verify()
    print(json.dumps(dict(prepared_runs=2, validation='PASS')), flush=True)


def submit():
    deployment = verify()
    if any((ROOT/name).exists() for name in ('submission_intent.json', 'submission_receipt.json')):
        raise RuntimeError('Already attempted; inspect evidence, never retry blindly')
    before = account_snapshot()
    if before['owned_jobs'] or before['allocated_gpus']:
        raise RuntimeError('Account is not empty; reconcile running AND pending work first')
    if OUTPUT.exists():
        raise RuntimeError('Output root exists; refuse possible duplicate')
    plan = json.loads((ROOT/'experiment_plan.json').read_text())
    args = ['sbatch', '--parsable', '--array=0-1%2', '--chdir='+str(SOURCE), str(ROOT/'run_anchor90.sh')]
    write_new(ROOT/'submission_intent.json', dict(command=args, account_preflight=before))
    response = command(args)
    job = response.split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Ambiguous scheduler response; do not retry: '+response)
    receipt = dict(job_id=job, submitted_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        run_ids=[r['run_id'] for r in plan['runs']], array='0-1%2', gpu_per_task=1, outer_updates=100,
        scheduler_response=response, output_root=str(OUTPUT), source_git_commit=deployment['source_git_commit'])
    write_new(ROOT/'submission_receipt.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    after = account_snapshot()
    write_new(ROOT/'queue_verified.json', after)
    print(json.dumps(after, indent=2), flush=True)
    if after['allocated_gpus'] > 6:
        raise RuntimeError('Independent work exceeded the full-account budget')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', metavar='ARCHIVE_SHA256')
    group.add_argument('--submit', action='store_true')
    group.add_argument('--verify', action='store_true')
    group.add_argument('--account', action='store_true')
    parser.add_argument('--source-commit')
    args = parser.parse_args()
    if args.prepare: prepare(args.prepare, args.source_commit)
    elif args.submit: submit()
    elif args.account: print(json.dumps(account_snapshot(), indent=2))
    else:
        verify()
        print('PASS immutable anchor90 deployment')
