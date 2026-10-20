"""Bound interim WR sidecar and compatible training lane; never edits the base launch."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

from runtime import (checked, contract, exclusive, load_json, save_new, scheduler,
                     sha256, utc)
from queue_campaign import PYTHON, measurement_complete, receipt_jobs, training_steps
from generate import collect, selected_steps
from measure import wr_directory

STEPS = list(range(0, 81, 10))
ACTIVE = {'PENDING', 'RUNNING', 'COMPLETING', 'CONFIGURING', 'SUSPENDED'}


def receipts(directory):
    result = []
    intents = sorted(directory.glob('*_intent.json'))
    if len(list(directory.glob('*_receipt.json'))) != len(intents):
        raise RuntimeError('Unresolved intent/receipt: inspect, never duplicate')
    for index, path in enumerate(intents, 1):
        if path.name != f'{index:04d}_intent.json':
            raise ValueError('Non-contiguous submission ledger')
        intent = load_json(path)
        receipt = load_json(path.with_name(path.name.replace('_intent', '_receipt')))
        if {k:v for k,v in receipt.items() if k != 'job_id'} != intent:
            raise ValueError('Intent/receipt mismatch')
        if not re.fullmatch(r'\d+(;[\w.-]+)?', receipt['job_id']):
            raise ValueError('Invalid receipt job ID')
        result.append(receipt)
    return result


def freeze(root, base, commit):
    plan, _, identity = contract(base)
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Full extension commit required')
    selected_steps(plan, STEPS)
    legacy = receipts(base/'submissions')
    if not legacy or any(r['phase'] != 'train' for r in legacy):
        raise ValueError('Interim launch requires a training-only legacy ledger')
    hashes = {}
    for arm in plan['runs']:
        for step in STEPS:
            path = Path(plan['output_root'])/'training'/arm['run_id']/f'step_{step:04d}'
            checked(path, identity)
            hashes[f'{arm["run_id"]}/step_{step:04d}'] = sha256(path/'COMMIT.json')
    files = {str(p.relative_to(root)):sha256(p) for p in sorted((root/'source').rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts}
    if not files:
        raise ValueError('No deployed extension source')
    value = dict(created=utc(), source_commit=commit, base_root=str(base), identity=identity,
        base_deployment_sha256=sha256(base/'deployment.json'), steps=STEPS,
        files_sha256=files, checkpoint_commits=hashes,
        legacy_ledger={p.name:sha256(p) for p in sorted((base/'submissions').glob('*.json'))},
        account_gpu_limit=4, lane_gpu_limit=2,
        training_execution_source=load_json(base/'deployment.json')['source_commit'])
    save_new(root/'extension.json', value)
    print(json.dumps(dict(frozen=utc(), extension_sha256=sha256(root/'extension.json'), identity=identity)))


def verify(root):
    binding = load_json(root/'extension.json')
    base = Path(binding['base_root'])
    if binding['steps'] != STEPS or binding['account_gpu_limit'] != 4 or binding['lane_gpu_limit'] != 2:
        raise ValueError('Changed interim authorization')
    for name, expected in binding['files_sha256'].items():
        if sha256(root/name) != expected:
            raise ValueError('Changed extension source: '+name)
    if sha256(base/'deployment.json') != binding['base_deployment_sha256']:
        raise ValueError('Changed base deployment')
    plan, _, identity = contract(base)
    if identity != binding['identity']:
        raise ValueError('Base identity changed')
    actual = {p.name:sha256(p) for p in (base/'submissions').glob('*.json')}
    if actual != binding['legacy_ledger']:
        raise ValueError('Legacy ledger changed; reconcile before new work')
    for name, expected in binding['checkpoint_commits'].items():
        path = Path(plan['output_root'])/'training'/name
        checked(path, identity)
        if sha256(path/'COMMIT.json') != expected:
            raise ValueError('Bound evaluation checkpoint changed')
    return base, plan, binding


def control_ids(line):
    array = re.search(r'\bArrayJobId=(\d+)', line)
    task = re.search(r'\bArrayTaskId=([^ ]+)', line)
    if array and task:
        result = set()
        for part in task[1].split('%')[0].split(','):
            if not re.fullmatch(r'\d+(-\d+)?', part):
                raise ValueError('Unknown array task range')
            bounds = [int(v) for v in part.split('-')]
            first, last = bounds[0], bounds[-1]
            if first > last or last-first > 1000:
                raise ValueError('Invalid array task range')
            result.update(f'{array[1]}_{i}' for i in range(first,last+1))
        return result
    job = re.search(r'\bJobId=(\d+)\b', line)
    if not job:
        raise ValueError('Missing job ID')
    return {job[1]}


def accounting(helpers, all_receipts):
    jobs = [j for r in all_receipts for j in receipt_jobs(r)]
    if len(set(jobs)) != len(jobs):
        raise ValueError('Duplicate registered job')
    # Slurm otherwise hides lazy pending tasks behind an aggregated array record.
    raw = helpers.run(['sacct','-X','--array','-n','-P','-j',','.join(jobs),'-o','JobID,State,ExitCode'])
    states = {}
    for line in raw.splitlines():
        parts = line.split('|')
        if len(parts) != 3 or parts[0] in states:
            raise ValueError('Ambiguous accounting record')
        states[parts[0]] = parts[1:]
    for job in jobs:
        state, code = states.get(job, ('UNKNOWN','UNKNOWN'))
        if not ((state == 'COMPLETED' and code == '0:0') or (state in ACTIVE and code == '0:0')):
            raise RuntimeError(f'Unsuccessful/unknown job {job}: {state} {code}; diagnose first')
    return states


def lane(receipt):
    return 'training' if receipt['phase']=='train' else 'evaluation'


def check_account(snapshot, all_receipts, states, helpers, new_lane=None):
    helpers.check_live_budget(snapshot)
    by_job = {j:r for r in all_receipts for j in receipt_jobs(r)}
    queued = {row.split('|')[0] for row in snapshot['owned_queue']}
    controlled = set()
    live_lanes = {'training':set(), 'evaluation':set()}
    for job, r in by_job.items():
        if states[job][0] in ACTIVE:
            live_lanes[lane(r)].add(r['job_id'])
    for row in snapshot['owned_scontrol']:
        ids = control_ids(row)
        if not ids <= by_job.keys():
            raise RuntimeError('Unregistered owned allocation/pending job')
        controlled.update(ids)
        matched = [by_job[j] for j in ids]
        if len({r['job_id'] for r in matched}) != 1:
            raise ValueError('Ambiguous array ownership')
        r = matched[0]
        request = re.search(r'\bReqTRES=(\S+)', row)
        gpu = re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',request[1]) if request else None
        if not gpu or int(gpu[1]) != (2 if r['phase']=='wrscore' else 1):
            raise ValueError('Scheduler GPU request differs from registered phase')
        if r['phase'] in ('train','generate'):
            throttle = re.search(r'\bArrayTaskThrottle=(\d+)', row)
            if not throttle or not 1 <= int(throttle[1]) <= 2:
                raise ValueError('Cannot bound pending array concurrency')
        live_lanes[lane(r)].add(r['job_id'])
    if not queued <= by_job.keys() or not queued <= controlled:
        raise RuntimeError('Unregistered work or inconsistent full-account snapshot')
    if any(len(values)>1 for values in live_lanes.values()):
        raise RuntimeError('Multiple allocations in one reserved two-GPU lane')
    if new_lane and live_lanes[new_lane]:
        raise RuntimeError('Requested lane still active or releasing; no duplicate segment')
    # Each lane reserves two GPUs including work that is pending or releasing.
    potential = 2*sum(bool(values) for values in live_lanes.values())+(2 if new_lane else 0)
    if potential > 4:
        raise RuntimeError('Potential full-account allocation exceeds four GPUs')
    return potential


def legacy_ready(base, plan, legacy, states, snapshot):
    """Old pending workers reject sidecars, even pending sidecars. Wait, never patch them."""
    for r in legacy:
        for job in receipt_jobs(r):
            state = states[job][0]
            if state == 'COMPLETED':
                continue
            if state not in ('RUNNING','COMPLETING'):
                raise RuntimeError('Legacy pending/startup guard still active; defer sidecar submission')
            matching = [s for s in snapshot['owned_scontrol'] if job in control_ids(s)]
            if len(matching) != 1:
                raise RuntimeError('Legacy allocation not consistently visible')
            actual = re.search(r'\bJobId=(\d+)\b', matching[0])[1]
            arm = plan['runs'][int(job.split('_')[1])]['run_id']
            manifest = load_json(Path(plan['output_root'])/'training'/arm/'manifest.json')
            if str(manifest['job']) != actual or manifest['state'] not in ('RUNNING','SEGMENT_COMPLETE','COMPLETE'):
                raise RuntimeError('Legacy worker has not verified checkpoint reload yet')


def ledgers(root, base):
    legacy = receipts(base/'submissions')
    extra = receipts(root/'submissions')
    for r in extra:
        if r['extension_sha256'] != sha256(root/'extension.json'):
            raise ValueError('Submission belongs to another extension')
        if r['phase'] not in ('train','baseline','generate','wrscore') or r['lane'] != lane(r):
            raise ValueError('Unknown cooperative phase')
    return legacy, extra


def choice(base, plan, identity, which):
    if which == 'training':
        steps = training_steps(plan, identity)
        indices = [i for i,s in enumerate(steps) if s < plan['training']['iters']]
        if not indices:
            return None
        return dict(phase='train', indices=indices, updates=10)
    output = Path(plan['output_root'])
    def complete(arm):
        directories = [output/'generations'/arm/f'step_{s:04d}' for s in ([0] if arm=='baseline' else STEPS)]
        if any(not (p/'COMPLETE.json').exists() for p in directories):
            return False
        collect(base, steps=STEPS, arms=[arm])
        return True
    if not complete('baseline'):
        return dict(phase='baseline', indices=[0], updates=0)
    indices = [i for i,r in enumerate(plan['runs']) if not complete(r['run_id'])]
    if indices:
        return dict(phase='generate', indices=indices, updates=0)
    target = wr_directory(plan, STEPS)
    if not (target/'COMPLETE.json').exists():
        return dict(phase='wrscore', indices=[0], updates=0)
    measurement_complete(target, identity)
    if load_json(target/'COMPLETE.json')['steps'] != STEPS:
        raise ValueError('WR subset completion scope mismatch')
    return None


def advance(root, which, submit=False):
    base, plan, binding = verify(root)
    helpers = scheduler()
    # The extension serializes its two lanes' submissions, not their GPU execution.
    with exclusive(root/'dispatch'):
        legacy, extra = ledgers(root, base)
        states = accounting(helpers, legacy+extra)
        selected = choice(base, plan, binding['identity'], which)
        if selected is None:
            print(json.dumps(dict(lane=which, complete=True)))
            return
        snapshot = helpers.owned_snapshot()
        potential = check_account(snapshot, legacy+extra, states, helpers, new_lane=which)
        legacy_ready(base, plan, legacy, states, snapshot)
        phase = selected['phase']
        gpus = 2 if phase=='wrscore' else 1
        number = len(extra)+1
        command = ['sbatch','--parsable','--partition=h100_all','--account=rc_fanyao_pi',
            f'--gres=gpu:{gpus}','--cpus-per-task=8',f'--mem={"256G" if gpus==2 else "96G"}',
            '--time=1-00:00:00','--signal=USR1@3600','--no-requeue',
            f'--job-name=o1i80_{number:04d}_{phase}',f'--output={root}/logs/{number:04d}-%A_%a.out',
            f'--error={root}/logs/{number:04d}-%A_%a.err']
        if phase in ('train','generate'):
            command.append('--array='+','.join(map(str,selected['indices']))+'%2')
        command += [str(root/'source/experiments/oracle1_semantic/run_interim.sh'),str(root)]
        record = dict(**selected, lane=which, sequence=number, identity=binding['identity'],
            extension_sha256=sha256(root/'extension.json'), source_commit=binding['source_commit'],
            training_execution_source=binding['training_execution_source'], command=command,
            created=utc(), preflight=snapshot, potential_account_gpus=potential, steps=STEPS)
        if not submit:
            print(json.dumps(record,indent=2))
            return
        (root/'logs').mkdir(exist_ok=True)
        save_new(root/'submissions'/f'{number:04d}_intent.json', record)
        job = helpers.run(command)
        if not re.fullmatch(r'\d+(;[\w.-]+)?', job):
            raise RuntimeError('Ambiguous submission: preserve intent and investigate')
        save_new(root/'submissions'/f'{number:04d}_receipt.json',dict(**record,job_id=job))
        print(json.dumps(dict(job_id=job,**selected)),flush=True)


def worker(root):
    base, plan, binding = verify(root)
    helpers = scheduler()
    job = os.environ.get('SLURM_ARRAY_JOB_ID',os.environ.get('SLURM_JOB_ID',''))
    for attempt in range(6):
        try:
            legacy, extra = ledgers(root, base)
            matched = [r for r in extra if r['job_id'].split(';')[0]==job]
            if len(matched)==1:
                break
        except (FileNotFoundError, RuntimeError):
            if attempt==5:
                raise
        time.sleep(5)
    else:
        raise RuntimeError('No unique receipt for worker')
    receipt = matched[0]
    states = accounting(helpers, legacy+extra)
    snapshot = helpers.owned_snapshot()
    check_account(snapshot, legacy+extra, states, helpers)
    legacy_ready(base, plan, legacy, states, snapshot)
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    if index not in receipt['indices']:
        raise ValueError('Unexpected array index')
    print('COOPERATIVE_ACCOUNT_AND_BOUND_CHECKPOINTS_VERIFIED',flush=True)
    phase = receipt['phase']
    if phase=='train':
        # Run the original computational source in a fresh process, not this checkout's trainer.
        command = [PYTHON,str(base/'source/experiments/oracle1_semantic/train.py'),
            '--root',str(base),'--run-id',plan['runs'][index]['run_id'],
            '--max-updates',str(receipt['updates'])]
        os.execv(PYTHON,command)
    elif phase in ('baseline','generate'):
        from generate import run
        run(base,'baseline' if phase=='baseline' else plan['runs'][index]['run_id'],steps=STEPS)
    elif phase=='wrscore':
        from measure import score_wr
        score_wr(base,steps=STEPS)


def inspect(root):
    base, plan, binding = verify(root)
    helpers = scheduler()
    legacy, extra = ledgers(root, base)
    states = accounting(helpers,legacy+extra)
    snapshot = helpers.owned_snapshot()
    check_account(snapshot,legacy+extra,states,helpers)
    from queue_campaign import inspect as base_inspect
    result = base_inspect(base)
    result.update(extension_checked_at=utc(),extension_sha256=sha256(root/'extension.json'),
                  cooperative_accounting=states,cooperative_queue=snapshot)
    result['extension_errors'] = []
    for path in (root/'logs').glob('*'):
        found = re.findall(r'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|Nonfinite|Non-finite|TIME LIMIT',path.read_text(errors='replace'))
        if found:
            result['extension_errors'].append(dict(file=path.name,markers=sorted(set(found))))
    result['interim_wr_complete']=(wr_directory(plan,STEPS)/'COMPLETE.json').exists()
    result['generation_banks']={}
    for arm in ['baseline']+[r['run_id'] for r in plan['runs']]:
        for step in ([0] if arm=='baseline' else STEPS):
            directory=Path(plan['output_root'])/'generations'/arm/f'step_{step:04d}'
            result['generation_banks'][f'{arm}:{step}']=dict(complete=(directory/'COMPLETE.json').exists(),
                journal_rows=len(list((directory/'rows').glob('[0-9a-f]'*64+'.json'))))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--base-root',type=Path)
    parser.add_argument('--commit')
    parser.add_argument('--lane',choices=['training','evaluation'],default='evaluation')
    parser.add_argument('--submit',action='store_true')
    parser.add_argument('--output',type=Path)
    mode=parser.add_mutually_exclusive_group(required=True)
    for name in ('freeze','verify','advance','worker','inspect','plot'):
        mode.add_argument('--'+name,action='store_true')
    args=parser.parse_args()
    if args.freeze:
        if args.base_root is None or args.commit is None:
            parser.error('Freeze requires base root and source commit')
        freeze(args.root,args.base_root,args.commit)
    elif args.verify:
        _,_,binding=verify(args.root)
        print(json.dumps(binding,indent=2))
    elif args.advance:
        advance(args.root,args.lane,args.submit)
    elif args.worker:
        worker(args.root)
    elif args.inspect:
        print(json.dumps(inspect(args.root),indent=2))
    elif args.plot:
        if args.output is None:
            parser.error('Plot requires a fresh output directory')
        base,_,_=verify(args.root)
        from analyze import plot
        plot(base,args.output,steps=STEPS)
