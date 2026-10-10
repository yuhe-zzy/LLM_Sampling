"""Guarded one-shot submission of the two approved remaining Stage C100 arms."""
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
PLAN = ROOT/'experiment_plan_c100.json'
RUNS = [f'{method}_mixed_c100_s0' for method in ('ipo','dpo')]
OUTPUT = Path('/work/users/y/u/yuhe32/ipo_runs/cyclic_history_stage_c100_20260929')


def command(args):
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f'{args}: {p.stderr}')
    return p.stdout.strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def verify():
    deployment = json.loads((ROOT/'deployment.json').read_text())
    for name, digest in deployment['sha256'].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError('Immutable deployment changed: '+name)
    print('PASS immutable Stage C100 deployment', flush=True)


def prepare(expected):
    if sha(ROOT/'source.tar') != expected:
        raise ValueError('Source archive hash mismatch')
    SOURCE.mkdir(exist_ok=False)
    with tarfile.open(ROOT/'source.tar') as tar:
        tar.extractall(SOURCE, filter='data')
    command(['bash', '-n', str(ROOT/'run_stage_c100.sh')])
    code = SOURCE/'experiments/cyclic_history'
    sys.path.insert(0, str(code))
    from run_cyclic_history import resolve_config, validate_config
    from calibrated_protocol import transform_panels, verify_calibration
    from history_math import load_panels
    plan = json.loads(PLAN.read_text())
    if [r['run_id'] for r in plan['runs']] != RUNS:
        raise ValueError('Unexpected arm mapping')
    predictions = {}
    for run in RUNS:
        cfg = resolve_config(PLAN, run)
        validate_config(cfg)
        if cfg['iters'] != 100 or cfg['epochs_per_iter'] != 10 or cfg['checkpoint_every'] != 5:
            raise ValueError('Unexpected training/checkpoint budget')
        if (cfg['alpha'], cfg['lambda_current'], cfg['seed']) != (.9, .8, 0):
            raise ValueError('Unexpected common parameters')
        if cfg['beta_train'] != (.2 if cfg['method'] == 'ipo' else .8):
            raise ValueError('Unexpected beta')
        if cfg['scheme'] != 'ordinary' or cfg['nu'] != 0 or cfg['kappa'] != 0:
            raise ValueError('Stage C must use ordinary updates')
        if cfg['orientations'] != [-1, 1, -1, 1, -1, 1]:
            raise ValueError('Unexpected mixed orientation assignment')
        panels = load_panels(cfg['eval_path'], cfg['num_prompts'], cfg['support_seed'],
                             cfg['keep_k'], prompt_ids=cfg['panel_ids'])
        predictions[run] = verify_calibration(transform_panels(panels,cfg),cfg)
        print('PASS', run, 'same support/parameters, 100 outer updates', flush=True)
    write_new(ROOT/'cpu_predictions.json', predictions)
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in SOURCE.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    for name in ('experiment_plan_c100.json','run_stage_c100.sh','queue_stage_c100.py'):
        hashes[name] = sha(ROOT/name)
    write_new(ROOT/'deployment.json', dict(sha256=hashes, source_archive_sha256=expected,
        source_git_commit='f0fe034cc0d8e3bef35b0f5e02806ee81340da76',
        run_ids=RUNS, array='0-1%2',gpu_per_task=1,account_gpu_ceiling=6,
        outer_updates=100,output_root=str(OUTPUT),
        scope='Empirical relative trajectories; no target-fit quality stopping condition'))
    verify()


def account_snapshot():
    full = command(['squeue','-a','-r','-h','-o','%i|%u|%U|%T|%b|%R|%j'])
    owned, total = [], 0
    for row in full.splitlines():
        f = [part.strip() for part in row.split('|')]
        if len(f)<3 or not (f[1]=='yuhe32' or f[2]=='448057'):
            continue
        detail = command(['scontrol','-a','show','job','-o',f[0]])
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
    if any((ROOT/name).exists() for name in ('submission_intent.json','submission_receipt.json')):
        raise RuntimeError('Submission already attempted; inspect receipt/queue, never duplicate')
    before = account_snapshot()
    if before['owned_jobs']:
        raise RuntimeError('Owned running/pending/releasing jobs appeared; reconcile budget first')
    if OUTPUT.exists():
        raise RuntimeError('Output root exists; possible previous launch')
    args = ['sbatch','--parsable','--array=0-1%2','--chdir='+str(SOURCE),str(ROOT/'run_stage_c100.sh')]
    write_new(ROOT/'submission_intent.json',dict(command=args,account_preflight=before))
    response = command(args)
    jid = response.split(';')[0]
    if not jid.isdigit():
        raise RuntimeError('Ambiguous scheduler response; inspect before retry: '+response)
    receipt = dict(job_id=jid,submitted_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   run_ids=RUNS,array='0-1%2',gpu_per_task=1,outer_updates=100,
                   scheduler_response=response,output_root=str(OUTPUT))
    write_new(ROOT/'submission_receipt.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)
    after = account_snapshot()
    write_new(ROOT/'queue_verified.json',after)
    if after['allocated_gpus']>6:
        raise RuntimeError('Unexpected independent work exceeded the total GPU budget')
    print(json.dumps(after,indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',metavar='SHA256')
    group.add_argument('--verify',action='store_true')
    group.add_argument('--submit',action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    elif args.verify:
        verify()
    else:
        submit()
