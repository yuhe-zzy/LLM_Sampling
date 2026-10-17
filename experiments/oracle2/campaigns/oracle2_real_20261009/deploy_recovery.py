"""Freeze and CPU-validate a recovery launch. Never allocate GPUs or submit jobs."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile


def save_new(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2)


def deploy(root, original, target):
    if (root/'source').exists() or (root/'deployment.json').exists():
        raise FileExistsError('Use a new immutable launch, never replace deployed source')
    with tarfile.open(root/'source_bundle.tar.gz', 'r:gz') as archive:
        for member in archive.getmembers():
            path = (root/member.name).resolve()
            if not path.is_relative_to(root.resolve()) or not member.isfile():
                raise ValueError('Unsafe source archive entry')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as handle:
                handle.write(archive.extractfile(member).read())
    package = json.loads((root/'source_manifest.json').read_text())
    for name, expected in package['files_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Packaged source mismatch')
    for name in ('plan.json', 'model_lock.json', 'audit_review.json'):
        if (root/name).read_bytes() != (original/name).read_bytes():
            raise ValueError('Recovery must reuse exact original plan/lock/review bytes')
    code = root/'source/experiments/oracle2'
    sys.path.insert(0, str(code))
    from generation_recovery import capture, checked_contract
    summary = capture(original, target, ['ipo_ordinary', 'ipo_reference'], root/'generation_recovery.json')
    _, contract = checked_contract(root/'plan.json', root/'generation_recovery.json')
    queue_spec = importlib.util.spec_from_file_location('recovery_queue', code/'campaigns/oracle2_real_20261009/queue.py')
    queue = importlib.util.module_from_spec(queue_spec)
    queue_spec.loader.exec_module(queue)
    commands = [
        [sys.executable, '-m', 'unittest', 'discover', '-s', 'experiments/oracle2', '-p', 'test*.py', '-v'],
        [sys.executable, '-m', 'unittest', 'discover', '-s', 'experiments/cyclic_history', '-p', 'test_*.py', '-v'],
        ['bash', '-n', str(code/'campaigns/oracle2_real_20261009/run_phase.sh')],
    ]
    plan = json.loads((root/'plan.json').read_text())
    commands += [[sys.executable, str(code/'train_oracle2.py'), '--plan', str(root/'plan.json'),
        '--review', str(root/'audit_review.json'), '--run-id', arm['run_id'],
        '--source-commit', package['source_commit']] for arm in plan['runs']]
    results = []
    for command in commands:
        completed = subprocess.run(command, cwd=root/'source', capture_output=True, text=True)
        results.append(dict(command=command, exit_code=completed.returncode,
                            stdout=completed.stdout, stderr=completed.stderr))
        print(completed.stdout, flush=True)
        print(completed.stderr, flush=True)
        if completed.returncode or ('unittest' in command and 'skipped=' in completed.stderr):
            save_new(root/'cpu_validation_failed.json', results)
            raise RuntimeError('CPU validation failed; no recovery submission allowed')
    save_new(root/'cpu_validation.json', results)
    package['files_sha256']['generation_recovery.json'] = hashlib.sha256((root/'generation_recovery.json').read_bytes()).hexdigest()
    package.update(original_launch_root=str(original), original_source_commit=contract['original_source_commit'],
                   purpose='missing generated responses only; unchanged training and science')
    save_new(root/'deployment.json', package)
    queue.verify(root)
    report = dict(state='CPU_VALIDATED_NOT_SUBMITTED', checked_at_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=package['source_commit'], original_source_commit=contract['original_source_commit'],
        recovery_sha256=package['files_sha256']['generation_recovery.json'], banks=summary,
        queue=queue.owned_snapshot(), gpu_inference_tested=False, original_banks_modified=False,
        plan_changed=False, results_output_created=target.exists(), tests=results)
    if target.exists():
        raise FileExistsError('Recovery destination already exists; investigate before submitting')
    save_new(root/'recovery_validation.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'tests'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--original-root', type=Path, required=True)
    p.add_argument('--target-root', type=Path, required=True)
    a = p.parse_args()
    deploy(a.root, a.original_root, a.target_root)
