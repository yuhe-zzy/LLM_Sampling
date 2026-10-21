"""Explicit, receipt-guarded phase advancement with a full-account empty preflight."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import numpy as np

from runtime import (checked, checkpoints, contract, load_json, save_new, scheduler,
                     sha256, utc)

PYTHON = '/work/users/y/u/yuhe32/h100env312/bin/python'


def training_steps(plan, identity):
    steps, anchor, support = [], None, None
    for arm in plan['runs']:
        saved = checkpoints(Path(plan['output_root'])/'training'/arm['run_id'], identity)
        steps.append(len(saved)-1)
        for checkpoint in saved:
            with np.load(checkpoint/'state.npz', allow_pickle=False) as state:
                if set(state.files) != {'initial','current','previous','lengths'}:
                    raise ValueError('Incomplete resumable state')
                if any(state[k].shape != (plan['training']['num_prompts'],4) or not np.isfinite(state[k]).all()
                       for k in state.files):
                    raise ValueError('Nonfinite or malformed resumable state')
                if anchor is None:
                    anchor, support = state['initial'].copy(), state['lengths'].copy()
                if not np.allclose(state['initial'],anchor,atol=.02,rtol=0) or not np.array_equal(state['lengths'],support):
                    raise ValueError('Training initial scores/support do not match across arms')
    return steps


def measurement_complete(directory, identity):
    manifest = load_json(directory/'COMPLETE.json')
    if manifest['identity'] != identity or not manifest.get('files'):
        raise ValueError('Missing measurement completion provenance')
    for name, value in manifest['files'].items():
        if sha256(directory/name) != value:
            raise ValueError('Measurement table changed')


def select_phase(root, plan, identity):
    steps = training_steps(plan, identity)
    probes = plan['continuation']['initial_probe_indices']
    if any(steps[i] < 1 for i in probes):
        return dict(phase='train', indices=[i for i in probes if steps[i]<1], updates=1, stage='startup_probe')
    if any(steps[i] < 2 for i in probes):
        return dict(phase='train', indices=[i for i in probes if steps[i]<2], updates=1, stage='resume_probe')
    if any(s < plan['training']['iters'] for s in steps):
        return dict(phase='train', indices=[i for i,s in enumerate(steps) if s<plan['training']['iters']],
                    updates=10, stage='training_segment')
    output = Path(plan['output_root'])
    if not (output/'generations/baseline/COMPLETE.json').exists():
        return dict(phase='baseline', indices=[0], updates=0, stage='baseline')
    incomplete = [i for i,r in enumerate(plan['runs'])
                  if not (output/'generations'/r['run_id']/'COMPLETE.json').exists()]
    if incomplete:
        return dict(phase='generate', indices=incomplete, updates=0, stage='generation_segment')
    from generate import collect
    collect(root)
    if not (output/'wr/COMPLETE.json').exists():
        return dict(phase='wrscore', indices=[0], updates=0, stage='wr_segment')
    measurement_complete(output/'wr', identity)
    if not (output/'annotation_pilot/COMPLETE.json').exists():
        return dict(phase='pilot', indices=[0], updates=0, stage='annotation_pilot')
    measurement_complete(output/'annotation_pilot', identity)
    if not (root/'annotation_review.json').exists():
        raise RuntimeError('Pilot ready: human review required before full five-trait annotation')
    if not (output/'annotation/COMPLETE.json').exists():
        return dict(phase='annotate', indices=[0], updates=0, stage='annotation_segment')
    measurement_complete(output/'annotation', identity)
    raise RuntimeError('All GPU stages complete; run CPU analysis and deliver results')


def receipt_jobs(receipt):
    base = receipt['job_id'].split(';')[0]
    return [f'{base}_{i}' for i in receipt['indices']] if receipt['phase'] in ('train','generate') else [base]


def prior_receipts(root, helpers):
    directory = root/'submissions'
    directory.mkdir(exist_ok=True)
    intents = sorted(directory.glob('*_intent.json'))
    for intent in intents:
        receipt_path = intent.with_name(intent.name.replace('_intent.json', '_receipt.json'))
        if not receipt_path.exists():
            raise RuntimeError('Unresolved submission intent: inspect scheduler, never duplicate')
        receipt = load_json(receipt_path)
        jobs = receipt_jobs(receipt)
        raw = helpers.run(['sacct','-X','-n','-P','-j',','.join(jobs),'-o','JobID,State,ExitCode'])
        states = {parts[0]:parts[1:] for line in raw.splitlines() if (parts:=line.split('|')) and len(parts)==3}
        if any(states.get(job) != ['COMPLETED','0:0'] for job in jobs):
            raise RuntimeError(f'Prior segment not successful: {[(j,states.get(j)) for j in jobs]}; diagnose first')
    return len(intents)+1


def startup(root):
    plan, _, identity = contract(root)
    helpers = scheduler()
    snapshot = helpers.owned_snapshot()
    helpers.check_live_budget(snapshot)
    base = os.environ.get('SLURM_ARRAY_JOB_ID', os.environ.get('SLURM_JOB_ID',''))
    receipts = [load_json(p) for p in (root/'submissions').glob('*_receipt.json')]
    match = [r for r in receipts if r['job_id'].split(';')[0] == base]
    # Slurm may start before the submitting client receives its job ID.
    if not match:
        import time
        time.sleep(5)
        receipts = [load_json(p) for p in (root/'submissions').glob('*_receipt.json')]
        match = [r for r in receipts if r['job_id'].split(';')[0] == base]
    if len(match) != 1:
        raise RuntimeError('No unique submission receipt for this allocation')
    allowed = set(receipt_jobs(match[0]))
    if any(row.split('|')[0] not in allowed for row in snapshot['owned_queue']):
        raise RuntimeError('Unrelated queued work; no GPU model load')
    if any(not (helpers.job_aliases(row)&allowed) for row in snapshot['owned_scontrol']):
        raise RuntimeError('Unrelated owned allocation; no GPU model load')
    return plan, identity, match[0]


def inspect(root):
    plan, _, identity = contract(root)
    helpers = scheduler()
    snapshot = helpers.owned_snapshot()
    steps = training_steps(plan, identity)
    outputs = Path(plan['output_root'])
    manifests = {}
    for r in plan['runs']:
        path = outputs/'training'/r['run_id']/'manifest.json'
        if path.exists():
            manifests[r['run_id']] = load_json(path)
    receipts = [load_json(p) for p in sorted((root/'submissions').glob('*_receipt.json'))]
    jobs = [j for receipt in receipts for j in receipt_jobs(receipt)]
    accounting = helpers.run(['sacct','-X','-n','-P','-j',','.join(jobs),'-o','JobID,State,ExitCode,Elapsed,AllocTRES']) if jobs else ''
    errors = []
    progress = {}
    for directory in ['generations/baseline','wr','annotation_pilot','annotation']+[
            'generations/'+r['run_id'] for r in plan['runs']]:
        path = outputs/directory/'progress.json'
        if path.exists():
            progress[directory] = load_json(path)
    for p in (root/'logs').glob('*'):
        if p.is_file():
            # Only execution markers; never treat the configured word "nonfinite" as an error.
            found = re.findall(r'(Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|Non-finite|Nonfinite|TIME LIMIT)',
                               p.read_text(errors='replace'))
            if found:
                errors.append(dict(file=p.name, markers=sorted(set(found))))
    return dict(checked_at=utc(), queue=snapshot, accounting=accounting, errors=errors,
        progress=progress,
        training_steps=dict(zip([r['run_id'] for r in plan['runs']],steps)), training=manifests,
        generation_complete=[r['run_id'] for r in plan['runs'] if (outputs/'generations'/r['run_id']/'COMPLETE.json').exists()],
        wr_complete=(outputs/'wr/COMPLETE.json').exists(),
        annotation_pilot_complete=(outputs/'annotation_pilot/COMPLETE.json').exists(),
        annotation_complete=(outputs/'annotation/COMPLETE.json').exists())


def advance(root, submit=False):
    plan, _, identity = contract(root)
    helpers = scheduler()
    snapshot = helpers.owned_snapshot()
    helpers.check_live_budget(snapshot)
    if snapshot['owned_queue'] or snapshot['owned_scontrol']:
        raise RuntimeError('Account is not empty; no new phase or continuation submitted')
    number = prior_receipts(root, helpers)
    choice = select_phase(root, plan, identity)
    phase = choice['phase']
    gpus = 2 if phase == 'wrscore' else 1
    concurrency = min(2,len(choice['indices']))
    if gpus*concurrency > 4:
        raise ValueError('Account GPU budget exceeded')
    code = root/'source/experiments/oracle1_semantic'
    command = ['sbatch','--parsable','--partition=h100_all','--account=rc_fanyao_pi',
        f'--gres=gpu:{gpus}','--cpus-per-task=8', f'--mem={"256G" if gpus==2 else "96G"}',
        '--time=1-00:00:00','--signal=USR1@3600','--no-requeue',
        f'--job-name=o1q3_{number:04d}_{phase}',
        f'--output={root}/logs/{number:04d}-%A_%a.out', f'--error={root}/logs/{number:04d}-%A_%a.err']
    if phase in ('train','generate'):
        command.append('--array='+','.join(map(str,choice['indices']))+'%2')
    command += [str(code/'run_phase.sh'),str(root),phase,str(choice['updates'])]
    record = dict(**choice, sequence=number, identity=identity, command=command, preflight=snapshot,
                  source_commit=load_json(root/'deployment.json')['source_commit'], created=utc())
    if not submit:
        print(json.dumps(record, indent=2))
        return
    (root/'logs').mkdir(exist_ok=True)
    save_new(root/'submissions'/f'{number:04d}_intent.json', record)
    job = helpers.run(command)
    if not job.split(';')[0].isdigit():
        raise RuntimeError('Ambiguous submission result; preserve intent and investigate')
    save_new(root/'submissions'/f'{number:04d}_receipt.json', dict(**record, job_id=job))
    print(json.dumps(dict(job_id=job, **choice)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--inspect', action='store_true')
    p.add_argument('--startup', action='store_true')
    p.add_argument('--advance', action='store_true')
    p.add_argument('--submit', action='store_true')
    a = p.parse_args()
    if a.inspect:
        print(json.dumps(inspect(a.root), indent=2))
    elif a.startup:
        startup(a.root)
        print('STARTUP_ACCOUNT_AND_IDENTITY_VERIFIED')
    elif a.advance:
        advance(a.root, a.submit)
    else:
        p.error('Choose inspect/startup/advance')
