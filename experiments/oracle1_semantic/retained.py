"""Owner-approved IPO ordinary evaluation only, with immutable historical disposition."""
import argparse
import json
import os
from pathlib import Path
import re
import time

import numpy as np
from runtime import (Journal, checked, contract, digest, exclusive, jsonl, load_json,
                     save_new, scheduler, sha256, utc)
from interim import ACTIVE, control_ids, receipts
from queue_campaign import receipt_jobs, measurement_complete
from generate import bank, expected
from measure import wr_directory

ARM = 'ipo_ordinary'
STEPS = list(range(0, 101, 10))
AUDIT = 'experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/retain_ipo_ordinary_20261021.json'
PHASE_GPUS = {'retained_generate': 1, 'retained_wrscore': 2}


def inventory(directory):
    return {p.name:sha256(p) for p in sorted(directory.glob('*.json'))}


def query(helpers, parents, jobs):
    if len(jobs) != len(set(jobs)):
        raise ValueError('Duplicate registered job')
    if not jobs:
        return {}
    raw = helpers.run(['sacct','-X','--array','-n','-P','-j',','.join(sorted(set(parents))),
                       '-o','JobID,State,ExitCode'])
    result = {}
    for line in raw.splitlines():
        fields = line.split('|')
        if len(fields) != 3 or fields[0] in result:
            raise ValueError('Ambiguous accounting record; do not filter or retry submission')
        result[fields[0]] = fields[1:]
    if set(result) != set(jobs):
        raise ValueError('Unknown/missing scheduler accounting')
    return result


def history(base, prior, helpers, approved):
    original, extra = receipts(base/'submissions'), receipts(prior/'submissions')
    expected_states = {j:['COMPLETED','0:0'] for r in original for j in receipt_jobs(r)}
    if len(expected_states) != approved['accounting']['original_successful_tasks']:
        raise ValueError('Changed original receipt coverage')
    expected_states.update(approved['accounting']['extension'])
    all_receipts = original+extra
    actual = query(helpers, [r['job_id'].split(';')[0] for r in all_receipts],
                   [j for r in all_receipts for j in receipt_jobs(r)])
    if actual != expected_states:
        raise RuntimeError('Historical disposition changed; investigate before new work')
    for r in extra:
        if r['extension_sha256'] != sha256(prior/'extension.json'):
            raise ValueError('Historical receipt binding mismatch')
        for j in receipt_jobs(r):
            # A failed retained task or shared baseline is never exempted.
            if actual[j] != ['COMPLETED','0:0']:
                if r['phase'] not in ('train','generate') or int(j.split('_')[1]) not in range(1,6):
                    raise RuntimeError('Retained prerequisite failed')
                if actual[j] != ['FAILED','1:0']:
                    raise RuntimeError('Unreviewed historical failure')
    return actual


def account(helpers, snapshot, current=None, gpus=0):
    helpers.check_live_budget(snapshot)
    allowed = {current} if current else set()
    queued = {r.split('|')[0] for r in snapshot['owned_queue']}
    controlled = set()
    for row in snapshot['owned_scontrol']:
        ids = control_ids(row)
        if ids != allowed or 'ArrayJobId=' in row:
            raise RuntimeError('Unrelated owned pending/running/releasing work')
        request = re.search(r'\bReqTRES=(\S+)', row)
        count = re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',request[1]) if request else None
        if not count or int(count[1]) != gpus:
            raise ValueError('Unexpected allocation GPU request')
        controlled.update(ids)
    if queued != allowed or controlled != allowed:
        raise RuntimeError('Full-account queue/control disagreement or unrelated work')
    if snapshot['allocated_gpus'] != gpus:
        raise RuntimeError('Unexpected live allocation')
    return gpus


def freeze(root, base, prior, commit):
    if not re.fullmatch('[0-9a-f]{40}',commit):
        raise ValueError('Full immutable source commit required')
    plan, _, identity = contract(base)
    if plan['runs'][0]['run_id'] != ARM or plan['evaluation']['steps'] != STEPS:
        raise ValueError('Unexpected retained plan')
    approved = load_json(root/'source'/AUDIT)
    old = load_json(prior/'extension.json')
    if (old['identity'] != identity or old['base_root'] != str(base)
            or sha256(prior/'extension.json') != approved['source']['binding_sha256']):
        raise ValueError('Unexpected prior extension')
    helpers = scheduler()
    historical = history(base, prior, helpers, approved)
    account(helpers, helpers.owned_snapshot())
    commits = {}
    for step in STEPS:
        p = Path(plan['output_root'])/'training'/ARM/f'step_{step:04d}'
        checked(p, identity)
        commits[p.name] = sha256(p/'COMMIT.json')
    manifest = load_json(Path(plan['output_root'])/'training'/ARM/'manifest.json')
    if manifest['state'] != 'COMPLETE' or manifest['last_complete_step'] != 100:
        raise ValueError('Retained training is not complete100')
    files = {str(p.relative_to(root)).replace('\\','/'):sha256(p)
             for p in sorted((root/'source').rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    if not files:
        raise ValueError('Missing source')
    binding = dict(created=utc(), source_commit=commit, identity=identity, base_root=str(base),
        prior_root=str(prior), arms=[ARM], steps=STEPS, account_gpu_limit=4,
        authorization='Owner October21: finish retained IPO ordinary evaluation and win rate only',
        base_deployment_sha256=sha256(base/'deployment.json'), prior_binding_sha256=sha256(prior/'extension.json'),
        base_ledger=inventory(base/'submissions'), prior_ledger=inventory(prior/'submissions'),
        historical_disposition=historical, checkpoint_commits=commits, files_sha256=files)
    validate_banks(base, plan, identity)
    save_new(root/'retained.json', binding)
    print(json.dumps(dict(frozen=utc(), binding_sha256=sha256(root/'retained.json'))))


def verify(root):
    b = load_json(root/'retained.json')
    if b['arms'] != [ARM] or b['steps'] != STEPS or b['account_gpu_limit'] != 4:
        raise ValueError('Retained scope changed')
    files = {str(p.relative_to(root)).replace('\\','/'):sha256(p)
             for p in (root/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if files != b['files_sha256']:
        raise ValueError('Immutable source changed')
    base, prior = Path(b['base_root']), Path(b['prior_root'])
    if (sha256(base/'deployment.json') != b['base_deployment_sha256']
            or sha256(prior/'extension.json') != b['prior_binding_sha256']
            or inventory(base/'submissions') != b['base_ledger']
            or inventory(prior/'submissions') != b['prior_ledger']):
        raise ValueError('Historical deployment/ledger changed')
    plan, _, identity = contract(base)
    if identity != b['identity'] or plan['runs'][0]['run_id'] != ARM or plan['evaluation']['steps'] != STEPS:
        raise ValueError('Base plan/identity changed')
    for name, value in b['checkpoint_commits'].items():
        p = Path(plan['output_root'])/'training'/ARM/name
        checked(p, identity)
        if sha256(p/'COMMIT.json') != value:
            raise ValueError('Retained checkpoint changed')
    if set(b['checkpoint_commits']) != {f'step_{s:04d}' for s in STEPS}:
        raise ValueError('Incomplete bound checkpoints')
    return base, plan, b


def validate_banks(base, plan, identity):
    panels = jsonl(base/'data/evaluation.jsonl')
    summary = {}
    for arm in ['baseline', ARM]:
        target = Path(plan['output_root'])/'generations'/arm
        if (target/'writer.lock').exists():
            raise RuntimeError('Generation writer lock exists; diagnose, never delete')
        for step in ([0] if arm=='baseline' else STEPS):
            p = target/f'step_{step:04d}'
            adapter_hash = None if arm=='baseline' else sha256(
                Path(plan['output_root'])/'training'/arm/p.name/'COMMIT.json')
            signature = digest(dict(identity=identity, arm=arm, step=step, adapter=adapter_hash,
                                    settings=plan['evaluation']))
            requests = list(expected(plan, panels, arm, step))
            journal = Journal.__new__(Journal)
            journal.root, journal.identity = p/'rows', signature
            paths = set(journal.root.glob('[0-9a-f]'*64+'.json'))
            if journal.root.exists() and load_json(journal.root/'IDENTITY.json') != {'identity':signature}:
                raise ValueError('Journal identity changed')
            rows, missing = [], False
            for request in requests:
                row = journal.get(request['id'],digest(request))
                if row is None:
                    missing = True
                    continue
                if missing or any(row[k] != v for k,v in request.items()):
                    raise ValueError('Invalid response prefix/identity')
                if not isinstance(row['response'],str) or not row['response'].strip():
                    raise ValueError('Empty/malformed response')
                if not np.isfinite([row[k] for k in ('response_tokens','unique_token_ratio','dominant_token_fraction')]).all():
                    raise ValueError('Nonfinite response diagnostics')
                rows.append(row)
                paths.discard(journal.path(request['id']))
            if paths:
                raise ValueError('Unexpected journal responses')
            complete = (p/'COMPLETE.json').exists()
            if complete and bank(p,identity,requests,signature) != rows:
                raise ValueError('Completed bank/journal disagreement')
            if arm == 'baseline' and not complete:
                raise ValueError('Required shared new-Qwen3 baseline incomplete')
            summary[f'{arm}:{step}'] = dict(rows=len(rows),expected=len(requests),complete=complete,
                cap_count=sum(r['hit_max_new_tokens'] for r in rows),
                mean_tokens=float(np.mean([r['response_tokens'] for r in rows])) if rows else None,
                mean_unique_ratio=float(np.mean([r['unique_token_ratio'] for r in rows])) if rows else None,
                max_dominant_fraction=max((r['dominant_token_fraction'] for r in rows),default=None))
    return summary


def new_receipts(root, binding, helpers, current=None):
    rs = receipts(root/'submissions')
    for r in rs:
        if (r['binding_sha256'] != sha256(root/'retained.json') or r['arms'] != [ARM]
                or r['steps'] != STEPS or r['phase'] not in PHASE_GPUS
                or r['identity'] != binding['identity'] or r['source_commit'] != binding['source_commit']):
            raise ValueError('Invalid retained receipt')
    jobs = [r['job_id'].split(';')[0] for r in rs]
    states = query(helpers,jobs,jobs)
    for job, outcome in states.items():
        if job == current:
            if outcome != ['RUNNING','0:0']:
                raise RuntimeError('Worker allocation is not running')
        elif outcome != ['COMPLETED','0:0']:
            raise RuntimeError('New segment not successful; diagnose before continuing: '+job+' '+str(outcome))
    if current and (not jobs or jobs[-1] != current):
        raise RuntimeError('Worker is not the latest unique receipt')
    return rs, states


def completion(plan, identity):
    target = wr_directory(plan,STEPS,[ARM])
    if not (target/'COMPLETE.json').exists():
        return False
    measurement_complete(target,identity)
    value = load_json(target/'COMPLETE.json')
    count = (len(STEPS)+1)*plan['dataset']['splits']['evaluation']*plan['evaluation']['responses_per_prompt']
    if any(value.get(k) != v for k,v in dict(steps=STEPS,arms=[ARM],scope='selected_arms',count=count).items()):
        raise ValueError('Incorrect retained WR completion scope')
    return True


def advance(root, submit=False):
    base, plan, b = verify(root)
    helpers = scheduler()
    with exclusive(root/'dispatch'):
        actual = history(base,Path(b['prior_root']),helpers,load_json(root/'source'/AUDIT))
        if actual != b['historical_disposition']:
            raise ValueError('Bound historical outcome changed')
        rs, _ = new_receipts(root,b,helpers)
        banks = validate_banks(base,plan,b['identity'])
        if (Path(plan['output_root'])/'wr/writer.lock').exists():
            raise RuntimeError('WR writer lock exists; diagnose, never delete')
        phase = 'retained_generate' if any(not v['complete'] for v in banks.values()) else 'retained_wrscore'
        snapshot = helpers.owned_snapshot()
        account(helpers,snapshot)
        if phase=='retained_wrscore' and completion(plan,b['identity']):
            print(json.dumps(dict(complete=True,arms=[ARM],steps=STEPS)))
            return
        number, gpus = len(rs)+1, PHASE_GPUS[phase]
        command = ['sbatch','--parsable','--partition=h100_all','--account=rc_fanyao_pi',
            f'--gres=gpu:{gpus}','--cpus-per-task=8',f'--mem={"256G" if gpus==2 else "96G"}',
            '--time=1-00:00:00','--signal=USR1@3600','--no-requeue',f'--job-name=o1ipo_{number:04d}',
            f'--output={root}/logs/{number:04d}-%j.out',f'--error={root}/logs/{number:04d}-%j.err',
            str(root/'source/experiments/oracle1_semantic/run_retained.sh'),str(root)]
        value = dict(sequence=number,phase=phase,arms=[ARM],steps=STEPS,identity=b['identity'],
            source_commit=b['source_commit'],binding_sha256=sha256(root/'retained.json'),created=utc(),
            command=command,preflight=snapshot,potential_account_gpus=gpus,banks=banks)
        if not submit:
            print(json.dumps(value,indent=2))
            return
        (root/'logs').mkdir(exist_ok=True)
        save_new(root/'submissions'/f'{number:04d}_intent.json',value)
        job = helpers.run(command)
        if not re.fullmatch(r'\d+(;[\w.-]+)?',job):
            raise RuntimeError('Unknown submission outcome; preserve intent, never retry')
        save_new(root/'submissions'/f'{number:04d}_receipt.json',dict(**value,job_id=job))
        print(json.dumps(dict(job_id=job,phase=phase,arms=[ARM])),flush=True)


def worker(root):
    base, plan, b = verify(root)
    job = os.environ.get('SLURM_JOB_ID','')
    if not job.isdigit() or os.environ.get('SLURM_ARRAY_JOB_ID'):
        raise ValueError('Retained worker requires one non-array allocation')
    helpers = scheduler()
    for attempt in range(12):
        # Only wait for receipt publication. Never retry a rejected accounting result.
        paths = list((root/'submissions').glob('*_receipt.json'))
        if any(load_json(p)['job_id'].split(';')[0]==job for p in paths):
            break
        time.sleep(5)
    rs, _ = new_receipts(root,b,helpers,job)
    if history(base,Path(b['prior_root']),helpers,load_json(root/'source'/AUDIT)) != b['historical_disposition']:
        raise ValueError('Historical outcome changed')
    phase = rs[-1]['phase']
    banks = validate_banks(base,plan,b['identity'])
    account(helpers,helpers.owned_snapshot(),job,PHASE_GPUS[phase])
    print('RETAINED_SCOPE_ACCOUNT_CHECKPOINTS_AND_JOURNALS_VERIFIED',flush=True)
    if phase=='retained_generate':
        from generate import run
        run(base,ARM,steps=STEPS)
    else:
        if any(not v['complete'] for v in banks.values()):
            raise ValueError('Generation must complete before WR')
        from measure import score_wr
        score_wr(base,steps=STEPS,arms=[ARM])


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--base-root',type=Path)
    p.add_argument('--prior-root',type=Path)
    p.add_argument('--commit')
    p.add_argument('--submit',action='store_true')
    p.add_argument('--output',type=Path)
    mode = p.add_mutually_exclusive_group(required=True)
    for name in ('freeze','verify','advance','worker','plot'):
        mode.add_argument('--'+name,action='store_true')
    a = p.parse_args()
    if a.freeze:
        if None in (a.base_root,a.prior_root,a.commit):
            p.error('Freeze requires base, prior and commit')
        freeze(a.root,a.base_root,a.prior_root,a.commit)
    elif a.verify:
        base,plan,b = verify(a.root)
        print(json.dumps(dict(checked_at=utc(),binding_sha256=sha256(a.root/'retained.json'),
                             banks=validate_banks(base,plan,b['identity'])),indent=2))
    elif a.advance:
        advance(a.root,a.submit)
    elif a.worker:
        worker(a.root)
    else:
        if a.output is None:
            p.error('Plot requires a fresh output directory')
        base,plan,b = verify(a.root)
        if not completion(plan,b['identity']):
            raise RuntimeError('No complete retained WR result')
        helpers = scheduler()
        new_receipts(a.root,b,helpers)
        account(helpers,helpers.owned_snapshot())
        from analyze import plot
        plot(base,a.output,steps=STEPS,arms=[ARM])
