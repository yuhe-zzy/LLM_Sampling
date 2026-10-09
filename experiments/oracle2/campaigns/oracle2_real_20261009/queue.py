"""Idempotent phase submission, with an explicit baseline-only afterok exception."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path('/work/users/y/u/yuhe32/ipo/diagnostics/oracle2_real_20261009_v4_empirical')
PYTHON = '/work/users/y/u/yuhe32/h100env312/bin/python'
ACCOUNT_GPU_LIMIT = 4
RESOURCES = {
    'audit': dict(gpus=2, tasks=1, concurrency=1, memory='256G', time='12:00:00'),
    'baseline': dict(gpus=1, tasks=1, concurrency=1, memory='80G', time='03:00:00'),
    'train': dict(gpus=1, tasks=6, concurrency=2, memory='80G', time='5-00:00:00'),
    'generate': dict(gpus=1, tasks=6, concurrency=2, memory='80G', time='12:00:00'),
    'wrscore': dict(gpus=2, tasks=1, concurrency=1, memory='256G', time='5-00:00:00'),
}


def maximum_phase_gpus(resource):
    if not 1 <= resource['concurrency'] <= resource['tasks'] or resource['gpus'] < 1:
        raise ValueError('Invalid task/GPU concurrency')
    maximum = resource['concurrency'] * resource['gpus']
    if maximum > ACCOUNT_GPU_LIMIT:
        raise ValueError('Phase exceeds four-GPU account budget')
    return maximum


def job_aliases(line):
    job = re.search(r'\bJobId=(\S+)', line)
    array = re.search(r'\bArrayJobId=(\d+)', line)
    index = re.search(r'\bArrayTaskId=(\d+)\b', line)
    aliases = {job[1]} if job else set()
    if array and index:
        aliases.add(f'{array[1]}_{index[1]}')
    return aliases


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
    all_owned = [line for line in control.splitlines() if re.search(
        r'\bUserId=(?:yuhe32\(\d+\)|[^ ()]+\(448057\))', line)]
    terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}
    control_owned, recent_terminal = [], []
    queued_ids = {row.split('|')[0] for row in owned}
    for line in all_owned:
        match = re.search(r'\bJobState=(\S+)', line)
        job = re.search(r'\bJobId=(\S+)', line)
        finished = match and match[1] in terminal and job and not (job_aliases(line) & queued_ids)
        (recent_terminal if finished else control_owned).append(line)
    allocated = 0
    accounted_ids = set()
    for line in control_owned:
        job = re.search(r'\bJobId=(\S+)', line)
        state = re.search(r'\bJobState=(\S+)', line)
        value = re.search(r'\bAllocTRES=(\S+)', line)
        if not job or job[1] in accounted_ids:
            raise ValueError('Missing or duplicate scontrol job ID')
        accounted_ids.update(job_aliases(line))
        generic = re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)', value[1]) if value else None
        if generic:
            allocated += int(generic[1])
        elif value and 'gres/gpu:' in value[1]:
            allocated = None
            break
        elif not state or (state[1] != 'PENDING' and not value):
            allocated = None
            break
    if any(row.split('|')[3] != 'PENDING' and row.split('|')[0] not in accounted_ids for row in owned):
        allocated = None
    return dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
                full_queue_rows=len(raw.splitlines()), owned_queue=owned, owned_scontrol=control_owned,
                recent_terminal_scontrol=recent_terminal,
                allocated_gpus=allocated, account_gpu_limit=ACCOUNT_GPU_LIMIT,
                hard_account_cap_installed=False)


def check_live_budget(snapshot):
    allocated = snapshot['allocated_gpus']
    if allocated is None or allocated > ACCOUNT_GPU_LIMIT:
        raise RuntimeError('Cannot verify allocation within four-GPU account budget; do not start GPU work')


def require_complete_generation(directory, expected_files):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['state'] != 'COMPLETE' or set(manifest['files']) != set(expected_files):
        raise ValueError('Required generation bank is incomplete')
    for name in expected_files:
        item = manifest['files'][name]
        if item['count'] != expected_files[name] or hashlib.sha256((directory / name).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Generation bank count/hash mismatch')


def phase_prerequisites(root, phase, defer_baseline=False):
    plan = json.loads((root / 'plan.json').read_text())
    output = Path(plan['output_root'])
    if phase == 'audit':
        if (root / 'reuse_candidate_audit.json').exists() or (Path(plan['data_root']) / 'scored/audit.json').exists():
            raise ValueError('Candidate audit already exists; never duplicate scoring')
    if phase == 'baseline' and (output / 'generations/baseline').exists():
        raise FileExistsError('Baseline output already exists; inspect it rather than overwrite')
    if phase == 'train':
        if any((output / arm['run_id']).exists() for arm in plan['runs']):
            raise FileExistsError('Training output already exists; never duplicate completed/partial arms')
        if not defer_baseline:
            require_complete_generation(output / 'generations/baseline',
                {'step_0000.jsonl': plan['dataset']['evaluation'] * plan['evaluation']['responses_per_prompt']})
    if phase == 'generate':
        for arm in plan['runs']:
            manifest = json.loads((output / arm['run_id'] / 'manifest.json').read_text())
            if manifest['state'] != 'COMPLETED' or manifest['last_complete_step'] != plan['training']['iters']:
                raise ValueError('All training arms must finish before generation phase')
    if phase == 'wrscore':
        count = plan['dataset']['evaluation'] * plan['evaluation']['responses_per_prompt']
        require_complete_generation(output / 'generations/baseline', {'step_0000.jsonl': count})
        for arm in plan['runs']:
            require_complete_generation(output / 'generations' / arm['run_id'],
                {f'step_{step:04d}.jsonl': count for step in plan['evaluation']['steps']})


def check_baseline_dependency(snapshot, receipt, accounting):
    job = receipt['job_id'].split(';')[0]
    if not job.isdigit() or receipt['phase'] != 'baseline' or receipt['resources']['gpus'] != 1:
        raise ValueError('Only the recorded one-GPU baseline may precede training')
    rows = [row.split('|') for row in accounting.splitlines() if row.strip()]
    parent = [row for row in rows if row[0] == job]
    if len(parent) != 1 or len(parent[0]) != 3:
        raise ValueError('Cannot verify predecessor accounting')
    state, exit_code = parent[0][1:]
    if state not in ('PENDING', 'RUNNING', 'COMPLETING', 'COMPLETED') or (state == 'COMPLETED' and exit_code != '0:0'):
        raise ValueError('Predecessor failed; never remove dependency or start training')
    if any(row.split('|')[0] != job for row in snapshot['owned_queue']):
        raise RuntimeError('Unrelated queued work; cannot guarantee phase budget')
    if any(job_aliases(row) != {job} for row in snapshot['owned_scontrol']):
        raise RuntimeError('Unrelated owned allocation; cannot guarantee phase budget')
    return job


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--verify', action='store_true')
    p.add_argument('--check-budget', action='store_true')
    p.add_argument('--check-baseline', action='store_true')
    p.add_argument('--after-baseline', action='store_true', help='Train only, afterok of this root\'s recorded baseline')
    p.add_argument('--phase', choices=list(RESOURCES))
    p.add_argument('--submit', action='store_true')
    a = p.parse_args()
    deployment = verify(a.root)
    if a.check_baseline:
        plan = json.loads((a.root / 'plan.json').read_text())
        require_complete_generation(Path(plan['output_root']) / 'generations/baseline',
            {'step_0000.jsonl': plan['dataset']['evaluation'] * plan['evaluation']['responses_per_prompt']})
        print('VERIFIED_COMPLETE_BASELINE')
        return
    if a.check_budget:
        snapshot = owned_snapshot()
        print(json.dumps(snapshot), flush=True)
        check_live_budget(snapshot)
        return
    if a.verify:
        print('VERIFIED', deployment['source_commit'])
        return
    if a.phase is None:
        p.error('--phase required')
    if a.after_baseline and a.phase != 'train':
        p.error('--after-baseline is only valid for train')
    resource = RESOURCES[a.phase]
    maximum = maximum_phase_gpus(resource)
    plan = json.loads((a.root / 'plan.json').read_text())
    if a.phase in ('audit', 'wrscore') and plan['oracle']['scoring_gpus'] != resource['gpus']:
        raise ValueError('Judge GPU request disagrees with frozen plan')
    source = a.root / 'source/experiments/oracle2'
    if a.phase in ('baseline', 'train', 'generate', 'wrscore'):
        for arm in json.loads((a.root / 'plan.json').read_text())['runs']:
            run([PYTHON, str(source / 'train_oracle2.py'), '--plan', str(a.root / 'plan.json'),
                 '--review', str(a.root / 'audit_review.json'), '--run-id', arm['run_id'],
                 '--source-commit', deployment['source_commit']])
    phase_prerequisites(a.root, a.phase, defer_baseline=a.after_baseline)
    intent = a.root / f'{a.phase}_submission_intent.json'
    receipt = a.root / f'{a.phase}_submission_receipt.json'
    if intent.exists() or receipt.exists():
        raise FileExistsError('Existing intent/receipt: inspect live scheduler; never blindly retry')
    snapshot = owned_snapshot()
    print(json.dumps(snapshot), flush=True)
    check_live_budget(snapshot)
    predecessor = None
    if a.after_baseline:
        baseline = json.loads((a.root / 'baseline_submission_receipt.json').read_text())
        if baseline['source_commit'] != deployment['source_commit']:
            raise ValueError('Baseline receipt belongs to another frozen source')
        predecessor = check_baseline_dependency(snapshot, baseline, run(
            ['sacct', '-j', baseline['job_id'].split(';')[0], '-X', '-n', '-P', '-o', 'JobID,State,ExitCode']))
    elif snapshot['owned_queue'] or snapshot['owned_scontrol']:
        raise RuntimeError('Account not empty. Leave this phase unsubmitted; review dependencies/budget')
    script = source / 'campaigns/oracle2_real_20261009/run_phase.sh'
    command = ['sbatch', '--parsable', '--partition=h100_all', '--account=rc_fanyao_pi',
               f'--export=ALL,ORACLE2_LAUNCH_ROOT={a.root}',
               f'--gres=gpu:{resource["gpus"]}', '--cpus-per-task=8', f'--mem={resource["memory"]}',
               f'--time={resource["time"]}', '--no-requeue', f'--job-name=oracle2_{a.phase}',
               f'--output={a.root}/logs/{a.phase}-%A_%a.out', f'--error={a.root}/logs/{a.phase}-%A_%a.err']
    if resource['tasks'] > 1:
        command.append(f'--array=0-{resource["tasks"]-1}%{resource["concurrency"]}')
    if predecessor:
        command.append(f'--dependency=afterok:{predecessor}')
    command += [str(script), a.phase]
    if not a.submit:
        print(json.dumps(dict(preview=command)))
        return
    (a.root / 'logs').mkdir(exist_ok=True)
    save_new(intent, dict(source_commit=deployment['source_commit'], command=command, preflight=snapshot,
                          phase=a.phase, maximum_phase_gpus=maximum, account_gpu_limit=ACCOUNT_GPU_LIMIT))
    job_id = run(command)
    if not job_id.split(';')[0].isdigit():
        raise RuntimeError('Ambiguous sbatch result; preserve intent and inspect scheduler before any retry')
    save_new(receipt, dict(job_id=job_id, phase=a.phase, source_commit=deployment['source_commit'],
                           submitted_at_utc=datetime.now(timezone.utc).isoformat(), resources=resource))
    print(json.dumps(dict(submitted_job_id=job_id, phase=a.phase)), flush=True)


if __name__ == '__main__':
    main()
