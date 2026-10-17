"""CPU-only descriptive generated-trait/WR plots; no inferred cyclic ground truth."""
import argparse
import csv
from pathlib import Path
import numpy as np
from runtime import contract, save_new
from measure import TRAITS, table
from queue_campaign import measurement_complete


def fit_axes(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] != 5 or not np.isfinite(x).all() or len(x)<10:
        raise ValueError('Insufficient complete discovery measurements')
    center, scale = x.mean(axis=0), x.std(axis=0,ddof=1)
    scale = np.where(scale>1e-8,scale,1.)
    _, singular, axes = np.linalg.svd((x-center)/scale,full_matrices=False)
    for row in axes:
        if row[np.argmax(np.abs(row))]<0:
            row *= -1
    variance = singular**2
    return center,scale,axes[:2],variance[:2]/variance.sum() if variance.sum() else np.zeros(2)


def read_csv(path):
    with Path(path).open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def plot(root, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan,_,identity=contract(root)
    output=Path(plan['output_root'])
    destination.mkdir(parents=True,exist_ok=False)
    colors={'ordinary':'#333333','reference':'#2473b5','feedback':'#bd4a53'}
    def export(fig,name):
        fig.savefig(destination/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(destination/(name+'.pdf'),bbox_inches='tight')
        plt.close(fig)
    measurement_complete(output/'wr',identity)
    wr=read_csv(output/'wr/wr_summary.csv')
    table(destination/'wr_summary.csv',wr)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True,constrained_layout=True)
    for ax,method in zip(axes,('ipo','dpo')):
        for scheme,color in colors.items():
            rows=sorted([r for r in wr if r['run_id']==method+'_'+scheme],key=lambda r:int(r['step']))
            t=np.array([int(r['step']) for r in rows])
            y=np.array([float(r['oracle1_expected_win_rate']) for r in rows])
            se=np.array([float(r['oracle1_expected_win_rate_prompt_se']) for r in rows])
            ax.plot(t,y,label=scheme,color=color,marker='o',ms=3)
            ax.fill_between(t,y-1.96*se,y+1.96*se,color=color,alpha=.12)
        ax.set(title=method.upper(),xlabel='Outer step',ylim=(0,1))
        ax.axhline(.5,color='#aaaaaa',lw=.7)
        ax.legend()
    axes[0].set_ylabel('Oracle1 expected win probability vs initial model')
    export(fig,'oracle1_wr')
    if not (output/'annotation/COMPLETE.json').exists():
        save_new(destination/'coverage.json',dict(identity=identity,annotation='not_complete',wr='complete'))
        return
    measurement_complete(output/'annotation',identity)
    features=read_csv(output/'annotation/features.csv')
    by_id={r['id']:np.array([float(r[k]) if r[k] else np.nan for k in TRAITS]) for r in features}
    originals=[r for r in features if r['id'].startswith('train:')]
    prompts=sorted({r['prompt_key'] for r in originals})
    discovery=set(np.random.default_rng(0).permutation(prompts)[:int(.8*len(prompts))])
    discovery_values=[by_id[r['id']] for r in originals if r['prompt_key'] in discovery and np.isfinite(by_id[r['id']]).all()]
    center,scale,axes,variance=fit_axes(discovery_values)
    save_new(destination/'frozen_axes.json',dict(identity=identity,traits=list(TRAITS),
        fit_source='seed0_80percent_training_prompts_original_responses_only',
        discovery_prompt_keys=sorted(discovery),rows=len(discovery_values),center=center.tolist(),
        scale=scale.tolist(),axes=axes.tolist(),explained_variance_ratio=variance.tolist(),
        interpretation='descriptive_PCA_not_preference_cyclic_eigenmodes'))
    from generate import collect
    banks={}
    for row in collect(root):
        if row['run_id']=='baseline':
            continue
        key=(row['run_id'],row['step'],row['prompt_key'])
        banks.setdefault(key,[]).append(by_id[row['id']])
    means={key:np.mean(rows,axis=0) for key,rows in banks.items() if len(rows)==4}
    cohorts=[]
    arms=[r['run_id'] for r in plan['runs']]
    for dim in range(6):
        sets=[]
        for arm in arms:
            for step in plan['evaluation']['steps']:
                sets.append({key[2] for key,v in means.items() if key[:2]==(arm,step)
                    and (np.isfinite(v).all() if dim==5 else np.isfinite(v[dim]))})
        cohorts.append(set.intersection(*sets))
    if any(len(s)<2 for s in cohorts):
        raise ValueError('Insufficient common complete-prompt coverage; no fabricated trajectory')
    summary=[]
    for arm in arms:
        for step in plan['evaluation']['steps']:
            row=dict(run_id=arm,step=step)
            for dim,trait in enumerate(TRAITS):
                x=np.array([means[(arm,step,key)][dim] for key in sorted(cohorts[dim])])
                row.update({trait+'_mean':x.mean(),trait+'_prompt_se':x.std(ddof=1)/np.sqrt(len(x)),trait+'_n':len(x)})
            z=np.array([(means[(arm,step,key)]-center)/scale@axes.T for key in sorted(cohorts[5])])
            row.update(pc1_mean=z[:,0].mean(),pc2_mean=z[:,1].mean(),pca_n=len(z))
            summary.append(row)
    table(destination/'feature_summary.csv',summary)
    fig,plots=plt.subplots(2,5,figsize=(17,7),sharex=True,sharey=True,constrained_layout=True)
    for i,method in enumerate(('ipo','dpo')):
        for j,trait in enumerate(TRAITS):
            ax=plots[i,j]
            for scheme,color in colors.items():
                rows=[r for r in summary if r['run_id']==method+'_'+scheme]
                t=np.array([r['step'] for r in rows])
                y=np.array([r[trait+'_mean'] for r in rows])
                se=np.array([r[trait+'_prompt_se'] for r in rows])
                ax.plot(t,y,label=scheme,color=color,marker='o',ms=2)
                ax.fill_between(t,y-1.96*se,y+1.96*se,color=color,alpha=.12)
            ax.set(title=method.upper()+' / '+trait.replace('_',' '),ylim=(0,10),xlabel='Outer step')
            if j==0:
                ax.set_ylabel('Prompt-balanced trait score')
                ax.legend()
    export(fig,'generated_five_traits')
    fig,plots=plt.subplots(1,2,figsize=(10,4),sharex=True,sharey=True,constrained_layout=True)
    for ax,method in zip(plots,('ipo','dpo')):
        for scheme,color in colors.items():
            rows=[r for r in summary if r['run_id']==method+'_'+scheme]
            ax.plot([r['pc1_mean'] for r in rows],[r['pc2_mean'] for r in rows],color=color,label=scheme,marker='o',ms=3)
            for r in rows:
                ax.annotate(str(r['step']),(r['pc1_mean'],r['pc2_mean']),fontsize=6)
        ax.set(title=method.upper(),xlabel='Frozen discovery PC1',ylabel='Frozen discovery PC2')
        ax.legend()
    export(fig,'descriptive_fixed_pca')
    save_new(destination/'coverage.json',dict(identity=identity,common_prompt_counts={
        k:len(v) for k,v in zip(list(TRAITS)+['pca'],cohorts)},
        limitations=['one training seed','prompt SE is not seed or rater uncertainty',
                     '256-token capped generations','PCA trajectory is not evidence of cyclic preference']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    plot(a.root,a.output)
