"""Read-only CPU export of the nine original cyclic runs; no scheduler calls."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np


def digest_support(row):
    values = [row['prompt']] + [row[f'response_{j}'] for j in range(4)]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    manifest = dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),
                    definition='softmax(response-token-average log probability), tau=1; legacy',
                    runs=[])
    with zipfile.ZipFile(args.out,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for method in ('ipo','dpo'):
            for run in sorted((args.root/f'logs_{method}_nonoracle').glob('cyclic*_dataeval')):
                match = re.fullmatch(r'cyclic_a([\d.]+)_l([\d.]+)_b([\d.]+)_dataeval',run.name)
                if not match:
                    continue
                paths = sorted(run.glob('*iter_dumps*/*prompt_metrics.csv'))
                if not paths:
                    continue
                ids, hashes, steps, probabilities, scores = None,None,[],[],[]
                file_hashes, stages = {},set()
                for path in paths:
                    raw = path.read_bytes()
                    file_hashes[path.name] = hashlib.sha256(raw).hexdigest()
                    rows = sorted(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))),
                                  key=lambda r:int(r['prompt_id']))
                    current_ids = [int(r['prompt_id']) for r in rows]
                    current_hashes = [digest_support(r) for r in rows]
                    if len(set(current_ids)) != len(rows) or any(int(r['K']) != 4 for r in rows):
                        raise ValueError('Duplicate prompt or wrong K: '+str(path))
                    if ids is None:
                        ids, hashes = current_ids,current_hashes
                    if ids != current_ids or hashes != current_hashes:
                        raise ValueError('Support changed inside run: '+str(path))
                    t = {int(r['iter']) for r in rows}
                    if len(t) != 1:
                        raise ValueError('Mixed steps')
                    steps.append(t.pop())
                    stages.update(r.get('snapshot_stage','unknown') for r in rows)
                    probabilities.append([[float(r[f'prob_avg_{j}']) for j in range(4)] for r in rows])
                    scores.append([[float(r[f'avg_logprob_{j}']) for j in range(4)] for r in rows])
                prob, avg = np.asarray(probabilities),np.asarray(scores)
                if steps != list(range(len(paths))):
                    raise ValueError('Noncontiguous steps: '+str(run))
                finite = np.isfinite(avg).all(-1) & np.isfinite(prob).all(-1)
                finite &= (prob >= 0).all(-1) & (prob <= 1).all(-1) & (np.abs(prob.sum(-1)-1)<1e-6)
                good_scores = avg[finite]
                softmax_error = None
                if len(good_scores):
                    exp = np.exp(good_scores-good_scores.max(-1,keepdims=True))
                    softmax_error = float(np.max(np.abs(exp/exp.sum(-1,keepdims=True)-prob[finite])))
                    if softmax_error > 1e-6:
                        raise ValueError('Saved probabilities differ from softmax(avg): '+str(run))
                name = method+'_'+run.name
                buffer = io.BytesIO()
                np.savez_compressed(buffer,step=np.array(steps),prompt_id=np.array(ids),
                    support_sha256=np.array(hashes),prob=prob,avg_logprob=avg,valid=finite)
                z.writestr(name+'/history.npz',buffer.getvalue())
                invalid_steps = np.flatnonzero(~finite.all(1))
                record = dict(name=name,method=method,alpha=float(match[1]),lambda_on=float(match[2]),
                    beta=float(match[3]),seed=0,tau=1,source=str(run),dump_files=file_hashes,
                    snapshot_stages=sorted(stages),steps=len(steps),last_step=steps[-1],prompts=len(ids),
                    invalid_prompt_snapshots=int((~finite).sum()),
                    first_invalid_step=int(invalid_steps[0]) if len(invalid_steps) else None,
                    softmax_avg_max_error=softmax_error)
                manifest['runs'].append(record)
                print(json.dumps({k:v for k,v in record.items() if k!='dump_files'}),flush=True)
        if len(manifest['runs']) != 9:
            raise ValueError('Expected exactly the nine original groups')
        z.writestr('manifest.json',json.dumps(manifest,indent=2))
    print('EXPORTED',args.out,args.out.stat().st_size,flush=True)


if __name__=='__main__':
    main()
