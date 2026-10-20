"""Run a batch of panel_po.py configurations concurrently on the single GPU.
Usage: python run_batch.py <batch.json>   where batch.json = {"common": "...args...", "runs": {"name": "...args...", ...}, "parallel": 4}"""
import json, os, shlex, subprocess, sys, time
B = json.load(open(sys.argv[1]))
root = os.path.expanduser('~/v4val/runs'); os.makedirs(root, exist_ok=True)
py = os.path.expanduser('~/oracle2_repro/.venv/bin/python'); script = os.path.expanduser('~/v4val/panel_po.py')
queue = list(B['runs'].items()); running = []
par = B.get('parallel', 4)
while queue or running:
    while queue and len(running) < par:
        name, a = queue.pop(0)
        out = os.path.join(root, name)
        if os.path.exists(os.path.join(out, 'run.npz')):
            print('skip', name, flush=True); continue
        os.makedirs(out, exist_ok=True)
        cmd = [py, script] + shlex.split(B['common']) + shlex.split(a) + ['--out', out]
        log = open(os.path.join(out, 'log.txt'), 'w')
        running.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
        print('start', name, flush=True)
    for item in list(running):
        if item[1].poll() is not None:
            print('end', item[0], item[1].returncode, flush=True); running.remove(item)
    time.sleep(5)
print('batch done', flush=True)
