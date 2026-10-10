"""Matched-prompt legacy pi trajectories; invalid score rows are never plotted."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

COLORS = ['#0072B2','#D55E00','#009E73','#CC79A7']
COMMON = [(.1,.1,.01),(.5,.5,5.),(.99,.8,1.)]


def key(run):
    return run['alpha'],run['lambda_on'],run['beta']


def read_bundle(path):
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read('manifest.json'))
        runs = []
        for record in manifest['runs']:
            with np.load(io.BytesIO(z.read(record['name']+'/history.npz')),allow_pickle=False) as f:
                data = {k:f[k] for k in f.files}
            runs.append(dict(record,**data))
    return manifest,runs


def select(runs):
    shared = set.intersection(*(set(r['prompt_id'].tolist()) for r in runs))
    ids = np.array(sorted(shared))
    for pid in ids:
        fingerprints = {r['support_sha256'][np.where(r['prompt_id']==pid)[0][0]] for r in runs}
        if len(fingerprints)!=1:
            raise ValueError('Different prompt/responses across runs: '+str(pid))
    baselines = [next(r for r in runs if r['method']==m and key(r)==(.99,.8,1.)) for m in ('ipo','dpo')]
    horizon = min(r['last_step'] for r in baselines)
    scores = []
    for pid in ids:
        values = []
        for r in baselines:
            j = np.where(r['prompt_id']==pid)[0][0]
            valid = r['valid'][:horizon+1,j]
            if not valid.all():
                break
            values.append(float(np.abs(np.diff(r['prob'][:horizon+1,j],axis=0)).sum(-1).mean()/2))
        if len(values)==2:
            scores.append((sum(values)/2,int(pid),*values))
    scores.sort()
    if len(scores)<3:
        raise ValueError('Insufficient complete finite matched baseline prompts')
    selected = []
    for quantile in (.25,.5,.9):
        score,pid,ipo,dpo = scores[round(quantile*(len(scores)-1))]
        selected.append(dict(prompt_id=pid,selection_quantile=quantile,selection_score=score,
            ipo_baseline_tv=ipo,dpo_baseline_tv=dpo))
    return selected,dict(shared_prompts=len(ids),eligible_prompts=len(scores),selection_window=[0,horizon],
        rule='25th/50th/90th ranks of mean IPO+DPO adjacent TV in alpha=.99 lambda=.8 beta=1 baselines only; identical selected prompts in all groups',
        same_prompt_and_response_order_hashes_verified=True)


def figure(runs,method,params,selected,title):
    fig,axes = plt.subplots(len(selected),len(params),figsize=(max(12.9,4.3*len(params)),7.9),squeeze=False,
                            sharex=True,sharey=True)
    xmax = max(r['last_step'] for r in runs if r['method']==method)
    for col,parameters in enumerate(params):
        r = next(r for r in runs if r['method']==method and key(r)==parameters)
        for row,selection in enumerate(selected):
            ax = axes[row,col]
            pid = selection['prompt_id']
            j = np.where(r['prompt_id']==pid)[0][0]
            y = r['prob'][:,j].copy()
            y[~r['valid'][:,j]] = np.nan
            for k,color in enumerate(COLORS):
                ax.plot(r['step'],y[:,k],color=color,lw=1.35)
            if r['last_step']<xmax:
                ax.axvspan(r['last_step']+.5,xmax,color='#ededed')
                ax.text((r['last_step']+xmax)/2,.88,'Not recorded',ha='center',fontsize=8,color='#666666')
            invalid = np.flatnonzero(~r['valid'][:,j])
            if len(invalid):
                ax.scatter(r['step'][invalid],np.full(len(invalid),.97),marker='x',s=12,color='#a30000')
            ax.set(xlim=(0,xmax),ylim=(0,1),yticks=[0,.25,.5,.75,1])
            ax.grid(axis='y',alpha=.2)
            ax.spines[['top','right']].set_visible(False)
            if row==0:
                a,l,b=parameters
                ax.set_title(f'alpha={a:g}, lambda={l:g}, beta={b:g}',fontsize=11,pad=10)
            if col==0:
                ax.set_ylabel(f"Prompt {pid}\nLegacy panel probability",fontsize=10)
            if row==len(selected)-1:
                ax.set_xlabel('Outer iteration (pre-update)',fontsize=10)
    fig.suptitle(title,fontsize=16,y=.99)
    fig.text(.5,.943,'Legacy definition: pi = softmax(response-token-average log probability); not sequence-sum pi',
             ha='center',fontsize=10,color='#444444')
    fig.legend([Line2D([0],[0],color=c,lw=2) for c in COLORS],
               ['Response 1','Response 2','Response 3','Response 4'],ncol=4,
               loc='lower center',bbox_to_anchor=(.5,.004),frameon=False,fontsize=10)
    fig.subplots_adjust(left=.085,right=.99,bottom=.10,top=.865,hspace=.2,wspace=.10)
    return fig


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    manifest,runs=read_bundle(args.snapshot)
    selected,audit=select(runs)
    audit.update(snapshot_sha256=hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
                 selected=selected,runs=manifest['runs'])
    (args.out/'selection_and_validation.json').write_text(json.dumps(audit,indent=2))
    tables,trajectories=[],[]
    with PdfPages(args.out/'all_original_cyclic_pi.pdf') as pdf:
        for method in ('ipo','dpo'):
            extras = [(0.,0.,.01)] + ([(.99,.8,10.)] if method=='ipo' else [])
            for label,params in [('main',COMMON),('additional',extras)]:
                fig=figure(runs,method,params,selected,f'{method.upper()} | Original cyclic experiments')
                fig.savefig(args.out/f'{method}_{label}_pi.png',dpi=180)
                fig.savefig(args.out/f'{method}_{label}_pi.pdf')
                pdf.savefig(fig)
                plt.close(fig)
        fig=figure(runs,'ipo',[(.99,.8,1.),(.99,.8,10.)],selected,
                   'IPO | Fixed alpha=.99, lambda=.8: beta comparison')
        fig.savefig(args.out/'ipo_beta_comparison_pi.png',dpi=180)
        fig.savefig(args.out/'ipo_beta_comparison_pi.pdf')
        pdf.savefig(fig)
        plt.close(fig)
        for r in runs:
            for s in selected:
                j=np.where(r['prompt_id']==s['prompt_id'])[0][0]
                valid=r['valid'][:,j]
                q=r['prob'][:,j]
                adjacent=valid[1:] & valid[:-1]
                tv=np.abs(np.diff(q,axis=0)).sum(-1)/2
                top=q.argmax(-1)
                tables.append(dict(method=r['method'],alpha=r['alpha'],lambda_on=r['lambda_on'],beta=r['beta'],
                    prompt_id=s['prompt_id'],first_iter=0,last_iter=r['last_step'],valid_rows=int(valid.sum()),
                    mean_adjacent_tv=float(tv[adjacent].mean()) if adjacent.any() else None,
                    top1_switches=int(((top[1:]!=top[:-1]) & adjacent).sum()),
                    min_probability=float(q[valid].min()) if valid.any() else None,
                    max_probability=float(q[valid].max()) if valid.any() else None))
                for t in range(len(q)):
                    trajectories.append(dict(method=r['method'],alpha=r['alpha'],lambda_on=r['lambda_on'],beta=r['beta'],
                        prompt_id=s['prompt_id'],iter=int(r['step'][t]),valid=bool(valid[t]),
                        **{f'pi_{k+1}':float(q[t,k]) if valid[t] else '' for k in range(4)}))
    for name,rows in [('selected_prompt_summary.csv',tables),('selected_pi_trajectories.csv',trajectories)]:
        with (args.out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(selected=selected,audit={k:v for k,v in audit.items() if k not in ('runs','selected')}),indent=2))


if __name__=='__main__':
    main()
