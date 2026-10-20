"""Run a batch of panel_po_steer.py configurations on the single GPU (2026-10-19 X14; copy of run_batch_ctrl.py).
Usage: python run_batch_ctrl.py <batch.json>   where batch.json = {"common": "...", "runs": {"name": "...args..."}, "parallel": 2}"""
import json, os, shlex, subprocess, sys, time
B = json.load(open(sys.argv[1]))
root = os.path.expanduser('~/x15/runs'); os.makedirs(root, exist_ok=True)
py = os.path.expanduser('~/oracle2_repro/.venv/bin/python'); script = os.path.expanduser('~/steer/panel_po_steer.py')
queue = list(B['runs'].items()); running = []; par = B.get('parallel', 2)
while queue or running:
    while queue and len(running) < par:
        name, a = queue.pop(0); out = os.path.join(root, name)
        if os.path.exists(os.path.join(out, 'run.npz')):
            print('skip', name, flush=True); continue
        os.makedirs(out, exist_ok=True)
        cmd = [py, script] + shlex.split(B['common']) + shlex.split(a) + ['--out', out]
        running.append((name, subprocess.Popen(cmd, stdout=open(os.path.join(out, 'log.txt'), 'w'), stderr=subprocess.STDOUT)))
        print('start', name, time.strftime('%H:%M:%S'), flush=True)
    for item in list(running):
        if item[1].poll() is not None:
            print('end', item[0], item[1].returncode, time.strftime('%H:%M:%S'), flush=True); running.remove(item)
    time.sleep(5)
print('batch done', flush=True)
