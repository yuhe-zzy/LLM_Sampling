"""Immutable deployment and at-most-once submission for two approved arms."""
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
RUNS = [f'{m}_reference90_a1_b100_s0' for m in ('ipo','dpo')]
OUTPUT = Path('/work/users/y/u/yuhe32/ipo_runs/cyclic_history_reference90_20260929')


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


def verify():
    deployment = json.loads((ROOT/'deployment.json').read_text())
    for name, digest in deployment['sha256'].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError('Immutable deployment changed: '+name)
    print('PASS immutable reference90 deployment', flush=True)


def prepare(expected, commit):
    if sha(ROOT/'source.tar') != expected or not re.fullmatch('[0-9a-f]{40}', commit or ''):
        raise ValueError('Source archive hash or commit invalid')
    SOURCE.mkdir(exist_ok=False)
    with tarfile.open(ROOT/'source.tar') as tar:
        tar.extractall(SOURCE, filter='data')
    code = SOURCE/'experiments/cyclic_history'
    sys.path.insert(0, str(code))
    from calibrated_protocol import EMPIRICAL_PROTOCOL, transform_panels, verify_support_calibration
    from history_math import load_panels
    from run_cyclic_history import resolve_config, validate_config
    plan_path = ROOT/'experiment_plan.json'
    plan = json.loads(plan_path.read_text())
    if [r['run_id'] for r in plan['runs']] != RUNS:
        raise ValueError('Wrong arms')
    audit = []
    for run in RUNS:
        cfg = resolve_config(plan_path, run)
        validate_config(cfg)
        if (cfg['protocol'],cfg['alpha'],cfg['nu'],cfg['kappa'],cfg['lambda_current'],cfg['seed']) != (
                EMPIRICAL_PROTOCOL,1.,.9,0.,.8,0):
            raise ValueError('Wrong requested intervention')
        if cfg['iters'] != 100 or cfg['epochs_per_iter'] != 10 or cfg['beta_train'] != (
                .2 if cfg['method']=='ipo' else .8):
            raise ValueError('Wrong matched training budget')
        prior = resolve_config(code/'campaigns/cyclic_history_stage_b100_launch_20260929/experiment_plan_b100.json',
                               f"{cfg['method']}_reference_b100_s0")
        changed = {k for k in set(cfg)|set(prior) if cfg.get(k) != prior.get(k)}
        if changed != {'alpha','nu','protocol','plan_status','output_root','run_id',
                       'prediction_role','predictions','calibration_contract_sha256'}:
            raise ValueError(f'Unintended baseline changes: {changed}')
        panels = load_panels(cfg['eval_path'], cfg['num_prompts'], cfg['support_seed'],
                             cfg['keep_k'], prompt_ids=cfg['panel_ids'])
        verify_support_calibration(transform_panels(panels,cfg),cfg)
        audit.append(dict(run_id=run,changed_from_B=sorted(changed),support='PASS'))
    command(['bash','-n',str(ROOT/'run_reference90.sh')])
    tested = subprocess.run([sys.executable,'-m','unittest','discover','-s',str(code),'-p','test_*.py'],
                            capture_output=True,text=True)
    tests = tested.stdout+'\n'+tested.stderr
    print(tests,flush=True)
    if tested.returncode:
        raise RuntimeError('CPU regression tests failed')
    hashes = {p.relative_to(ROOT).as_posix():sha(p) for p in SOURCE.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    for name in ('experiment_plan.json','run_reference90.sh','queue_reference90.py'):
        hashes[name] = sha(ROOT/name)
    write_new(ROOT/'deployment.json',dict(source_git_commit=commit,source_archive_sha256=expected,
        sha256=hashes,arm_audit=audit,cpu_tests='unittest returned success',cpu_test_stdout=tests,
        array='0-1%2',gpu_per_task=1,account_ceiling=6,output_root=str(OUTPUT)))
    verify()
    print(json.dumps(audit),flush=True)


def account_snapshot():
    full = command(['squeue','-a','-r','-h','-o','%i|%u|%U|%T|%b|%R|%j'])
    owned, total = [], 0
    for row in full.splitlines():
        fields = [part.strip() for part in row.split('|')]
        if len(fields)<3 or not (fields[1]=='yuhe32' or fields[2]=='448057'):
            continue
        detail = command(['scontrol','-a','show','job','-o',fields[0]])
        if 'UserId=yuhe32(448057)' not in detail:
            raise RuntimeError('Owner identity mismatch')
        tres = re.search(r'\bAllocTRES=(\S+)',detail)
        gpu = re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres[1]) if tres else None
        count = int(gpu[1]) if gpu else 0
        total += count
        owned.append(dict(queue=row,allocated_gpus=count,detail=detail))
    return dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                allocated_gpus=total,owned_jobs=owned)


def submit():
    verify()
    if any((ROOT/n).exists() for n in ('submission_intent.json','submission_receipt.json')):
        raise RuntimeError('Already attempted: inspect receipt and queue before any action')
    before = account_snapshot()
    if before['owned_jobs']:
        raise RuntimeError('Other owned work appeared: reconcile running and pending capacity first')
    if OUTPUT.exists():
        raise RuntimeError('Output root already exists')
    args = ['sbatch','--parsable','--array=0-1%2','--chdir='+str(SOURCE),str(ROOT/'run_reference90.sh')]
    write_new(ROOT/'submission_intent.json',dict(command=args,account_preflight=before))
    response = command(args)
    job = response.split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Ambiguous sbatch response, do not retry: '+response)
    receipt = dict(job_id=job,submitted_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   run_ids=RUNS,array='0-1%2',gpu_per_task=1,outer_updates=100,
                   scheduler_response=response,output_root=str(OUTPUT),
                   source_git_commit=json.loads((ROOT/'deployment.json').read_text())['source_git_commit'])
    write_new(ROOT/'submission_receipt.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)
    after = account_snapshot()
    write_new(ROOT/'queue_verified.json',after)
    print(json.dumps(after,indent=2),flush=True)
    if after['allocated_gpus']>6:
        raise RuntimeError('Independent submission exceeded account budget')


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',metavar='ARCHIVE_SHA256')
    group.add_argument('--verify',action='store_true')
    group.add_argument('--submit',action='store_true')
    group.add_argument('--account',action='store_true')
    p.add_argument('--source-commit')
    args = p.parse_args()
    if args.prepare:
        prepare(args.prepare,args.source_commit)
    elif args.verify:
        verify()
    elif args.account:
        print(json.dumps(account_snapshot(),indent=2))
    else:
        submit()
