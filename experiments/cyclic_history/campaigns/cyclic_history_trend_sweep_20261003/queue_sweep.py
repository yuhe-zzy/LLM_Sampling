"""Immutable, at-most-once submission of the 44 approved empirical trend runs."""
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
OUTPUT = Path('/work/users/y/u/yuhe32/ipo_runs/cyclic_history_trend_sweep_20261003')


def command(args):
    result = subprocess.run(args,capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(f'{args}: {result.stdout}\n{result.stderr}')
    return result.stdout.strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, data):
    with path.open('x') as handle:
        json.dump(data,handle,indent=2)
        handle.write('\n')


def account_snapshot():
    # Reuse the reviewed full owner/UID accounting, not the unreliable squeue -u filter.
    path=SOURCE/'experiments/cyclic_history/campaigns/cyclic_history_reference90_20260929/queue_reference90.py'
    spec=importlib.util.spec_from_file_location('prior_accounting',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.account_snapshot()


def verify():
    deployment=json.loads((ROOT/'deployment.json').read_text())
    for name,digest in deployment['sha256'].items():
        if sha(ROOT/name)!=digest:
            raise RuntimeError('Immutable deployment changed: '+name)
    return deployment


def audit_reuse(plan):
    ignored={'run_id','output_root','protocol','plan_status','predictions','prediction_role',
             'calibration_contract_sha256'}
    audit=[]
    for reuse in plan['reused_runs']:
        import csv
        import numpy as np
        item=reuse['comparison']
        path=Path(item['existing_root'])/item['existing_run_id']
        manifest=json.loads((path/'manifest.json').read_text())
        if manifest['state']!='COMPLETED' or manifest['last_complete_step']!=100:
            raise RuntimeError('Reuse is not complete: '+str(path))
        actual=manifest['config']
        expected=reuse['expected_config']
        changes={key for key in set(actual)|set(expected) if actual.get(key)!=expected.get(key)}
        if changes-ignored:
            raise ValueError('Unmatched reuse configuration: '+str(changes-ignored))
        with (path/'metrics.csv').open() as handle:
            rows=list(csv.DictReader(handle))
        if [int(row['step']) for row in rows]!=list(range(101)):
            raise ValueError('Incomplete reused metric sequence')
        for step in range(101):
            with np.load(path/'snapshots'/f'step_{step:04d}.npz',allow_pickle=False) as snapshot:
                if any(not np.isfinite(snapshot[key]).all() for key in snapshot.files
                       if np.issubdtype(snapshot[key].dtype,np.number)):
                    raise ValueError('Nonfinite reused snapshot')
        audit.append(dict(run_id=item['run_id'],existing_run_id=item['existing_run_id'],
                          existing_job_id=item['existing_job_id'],matched=True,
                          manifest_sha256=sha(path/'manifest.json'),metrics_sha256=sha(path/'metrics.csv')))
    return audit


def prepare(expected_archive, commit):
    if sha(ROOT/'source.tar')!=expected_archive or not re.fullmatch('[0-9a-f]{40}',commit or ''):
        raise ValueError('Source hash or commit invalid')
    if (ROOT/'submission_intent.json').exists():
        raise RuntimeError('Submission already attempted')
    SOURCE.mkdir(exist_ok=False)
    with tarfile.open(ROOT/'source.tar') as archive:
        archive.extractall(SOURCE,filter='data')
    code=SOURCE/'experiments/cyclic_history'
    sys.path.insert(0,str(code))
    from calibrated_protocol import TREND_PROTOCOL, transform_panels, verify_support_calibration
    from history_math import load_panels
    from run_cyclic_history import resolve_config, validate_config
    spec=importlib.util.spec_from_file_location('frozen_sweep_plan',
        code/'campaigns/cyclic_history_trend_sweep_20261003/build_plan.py')
    builder=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    plan=json.loads((ROOT/'experiment_plan.json').read_text())
    if plan!=builder.make_plan() or len(plan['runs'])!=44 or len(plan['reused_runs'])!=6:
        raise ValueError('Deployment does not match approved grid')
    audit=[]
    for row in plan['runs']:
        cfg=resolve_config(ROOT/'experiment_plan.json',row['run_id'])
        validate_config(cfg)
        if cfg['protocol']!=TREND_PROTOCOL or cfg['iters']!=100 or cfg['seed']!=0:
            raise ValueError('Wrong trend protocol')
        panels=load_panels(cfg['eval_path'],cfg['num_prompts'],cfg['support_seed'],
                           cfg['keep_k'],prompt_ids=cfg['panel_ids'])
        verify_support_calibration(transform_panels(panels,cfg),cfg)
        audit.append(dict(run_id=cfg['run_id'],support='PASS',config='PASS'))
    reused=audit_reuse(plan)
    command(['bash','-n',str(ROOT/'run_sweep.sh')])
    tested=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(code),'-p','test_*.py'],
                          capture_output=True,text=True)
    tests=tested.stdout+'\n'+tested.stderr
    print(tests,flush=True)
    if tested.returncode:
        raise RuntimeError('CPU regressions failed')
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in SOURCE.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts}
    for name in ('experiment_plan.json','queue_sweep.py','run_sweep.sh','inspect_sweep.py'):
        hashes[name]=sha(ROOT/name)
    write_new(ROOT/'deployment.json',dict(source_git_commit=commit,source_archive_sha256=expected_archive,
        sha256=hashes,arm_audit=audit,reused_audit=reused,cpu_test_stdout=tests,
        array='0-43%6',gpu_per_task=1,account_ceiling=6,output_root=str(OUTPUT)))
    verify()
    print(json.dumps(dict(prepared_runs=44,reused_runs=6,validation='PASS')),flush=True)


def submit():
    deployment=verify()
    if any((ROOT/name).exists() for name in ('submission_intent.json','submission_receipt.json')):
        raise RuntimeError('Already attempted; inspect evidence instead of retrying')
    before=account_snapshot()
    if before['owned_jobs'] or before['allocated_gpus']:
        raise RuntimeError('Account is no longer empty; reconcile both running and pending work first')
    if OUTPUT.exists():
        raise RuntimeError('Output root already exists; refuse duplicate campaign')
    plan=json.loads((ROOT/'experiment_plan.json').read_text())
    args=['sbatch','--parsable','--array=0-43%6','--chdir='+str(SOURCE),str(ROOT/'run_sweep.sh')]
    write_new(ROOT/'submission_intent.json',dict(command=args,account_preflight=before))
    response=command(args)
    job=response.split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Ambiguous scheduler response; do not retry: '+response)
    receipt=dict(job_id=job,submitted_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        run_ids=[row['run_id'] for row in plan['runs']],array='0-43%6',gpu_per_task=1,
        outer_updates=100,reused_runs=6,logical_comparisons=50,scheduler_response=response,
        output_root=str(OUTPUT),source_git_commit=deployment['source_git_commit'])
    write_new(ROOT/'submission_receipt.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)
    after=account_snapshot()
    write_new(ROOT/'queue_verified.json',after)
    print(json.dumps(after,indent=2),flush=True)
    if after['allocated_gpus']>6:
        raise RuntimeError('Independent work exceeded the GPU budget')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',metavar='ARCHIVE_SHA256')
    group.add_argument('--submit',action='store_true')
    group.add_argument('--verify',action='store_true')
    group.add_argument('--account',action='store_true')
    parser.add_argument('--source-commit')
    args=parser.parse_args()
    if args.prepare: prepare(args.prepare,args.source_commit)
    elif args.submit: submit()
    elif args.account: print(json.dumps(account_snapshot(),indent=2))
    else:
        verify()
        print('PASS immutable trend deployment')
