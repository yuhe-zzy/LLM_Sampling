"""Frozen phase entry point; called only by the receipt-checked launcher."""
import argparse
import os
from pathlib import Path
from queue_campaign import startup


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--phase', choices=['train','baseline','generate','wrscore','pilot','annotate'], required=True)
    p.add_argument('--max-updates', type=int, default=10)
    a = p.parse_args()
    plan, _, receipt = startup(a.root)
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    if a.phase != receipt['phase'] or index not in receipt['indices'] or a.max_updates != receipt['updates']:
        raise ValueError('Worker differs from submitted intent')
    if a.phase == 'train':
        from train import run
        run(a.root, plan['runs'][index]['run_id'], a.max_updates)
    elif a.phase in ('baseline','generate'):
        from generate import run
        run(a.root, 'baseline' if a.phase=='baseline' else plan['runs'][index]['run_id'])
    elif a.phase == 'wrscore':
        from measure import score_wr
        score_wr(a.root)
    else:
        from measure import annotate
        annotate(a.root, pilot=a.phase=='pilot')


if __name__ == '__main__':
    main()
