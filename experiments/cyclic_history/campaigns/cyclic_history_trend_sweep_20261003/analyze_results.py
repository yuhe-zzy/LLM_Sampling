"""All-prompt empirical pi plots and paired descriptive summaries for the trend sweep."""
import argparse
import csv
import hashlib
import html
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('validated_reference_plot',
    HERE.parent/'cyclic_history_reference90_20260929/plot_results.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
ARMS = ('ordinary','reference_half','reference_max','feedback_half','feedback_one')
ARM_COLORS = ('#333333','#56B4E9','#0072B2','#E69F00','#C44435')
WINDOWS = ((1,25),(26,50),(51,75),(76,100),(51,100))
PROMPTS = prior.PROMPTS
FIELDS = ('tv_mean','probability_temporal_sd','mean_max_probability',
          'leading_response_switches','relative_entropy_mean')


def window_metrics(pi, relative, start, end):
    """TV/switches use transitions t-1 -> t; SD averages four response-wise SDs."""
    if not 1 <= start <= end < len(pi):
        raise ValueError('Window must have every measured state, including its predecessor')
    q = pi[start:end+1]
    tv = .5*np.abs(pi[start:end+1]-pi[start-1:end]).sum(-1)
    switches = pi[start:end+1].argmax(-1) != pi[start-1:end].argmax(-1)
    return dict(tv_mean=tv.mean(0), probability_temporal_sd=q.std(0).mean(-1),
                mean_max_probability=q.max(-1).mean(0), leading_response_switches=switches.sum(0),
                relative_entropy_mean=relative[start:end+1].mean(0))


def label(arm, row):
    if arm == 'ordinary':
        return 'Ordinary'
    if arm.startswith('reference'):
        return f"Reference: nu={row['nu']:g}"
    return f"Feedback: kappa={row['kappa']:g}"


def parameter_label(row):
    return f"alpha={row['alpha']:g}, lambda={row['lambda_current']:g}, beta={row['beta_train']:g}"


def verify_training_snapshots(root, data):
    cfg = data['manifest']['config']
    scores = data['scores']
    for step in range(1,data['last']+1):
        with np.load(root/'snapshots'/f'step_{step:04d}.npz',allow_pickle=False) as z:
            current, previous = scores[step-1], scores[max(step-2,0)]
            ref = (1-cfg['alpha'])*scores[0]+(cfg['alpha']-cfg['nu'])*current+cfg['nu']*previous
            offset = cfg['kappa']*(z['training_feedback_delta']-z['training_previous_feedback_delta'])
            mu = (1-cfg['lambda_current'])/4+cfg['lambda_current']*prior.softmax(current)
            for name, expected in [('training_reference',ref),('training_offset',offset),
                                   ('training_effective_reference',ref+offset),('training_mu',mu)]:
                np.testing.assert_allclose(z[name],expected,atol=1e-10,rtol=1e-11)


def collect(results):
    receipt = json.loads((results/'download_receipt.json').read_text())
    status = json.loads((results/'server_status.json').read_text())
    plan = json.loads((HERE/'experiment_plan.json').read_text())
    deployment = json.loads((HERE/'deployment.json').read_text())
    if status['receipt']['source_git_commit'] != deployment['source_git_commit']:
        raise ValueError('Wrong frozen training commit')
    for name,digest in receipt['files_sha256'].items():
        if hashlib.sha256((results/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Download checksum changed: '+name)
    mappings = {r['logical_run_id']:r for r in receipt['mappings']}
    reused_audit = {r['existing_run_id']:r for r in deployment['reused_audit']}
    accounting = {r['JobID']:r for r in csv.DictReader(status['sacct'].splitlines(),delimiter='|')}
    data, summaries, statuses = {}, [], []
    support = None
    for row in plan['comparisons']:
        key = row['run_id']
        if key not in mappings:
            statuses.append(dict(run_id=key,plotted=False,state='NO_COMPLETED_DOWNLOAD'))
            continue
        root = results/'raw'/mappings[key]['actual_run_id']
        a = prior.load_run(root,receipt)
        m = a['manifest']
        if m['state'] != 'COMPLETED' or a['last'] != 100:
            raise ValueError('Only complete runs may enter full-window comparisons')
        if m['calibration_gate'] != 'PASSED_FRESH_INITIAL_SCORES':
            raise ValueError('Initial-score calibration did not pass')
        cfg = m['config']
        for field in ('method','alpha','lambda_current','beta_train','nu','kappa'):
            if cfg[field] != row[field]:
                raise ValueError('Unmatched '+field+' in '+key)
        if (cfg['seed'],cfg['iters'],cfg['epochs_per_iter']) != (0,100,10):
            raise ValueError('Unexpected training budget')
        if row['status'] == 'NEW':
            job = accounting[f"4659985_{row['task_index']}"]
            if (job['State'],job['ExitCode']) != ('COMPLETED','0:0'):
                raise ValueError('Slurm completion mismatch')
            for name,digest in m['source_sha256'].items():
                if deployment['sha256']['source/experiments/cyclic_history/'+name] != digest:
                    raise ValueError('Training source hash changed')
        else:
            audit = reused_audit[row['existing_run_id']]
            for name in ('manifest','metrics'):
                extension = 'json' if name == 'manifest' else 'csv'
                digest = hashlib.sha256((root/f'{name}.{extension}').read_bytes()).hexdigest()
                if digest != audit[name+'_sha256']:
                    raise ValueError('Reused result changed since submission')
        if support is None:
            support = a['support']
        if support != a['support']:
            raise ValueError('Fixed response identities are not aligned')
        verify_training_snapshots(root,a)
        a['record'] = row
        data[key] = a
        statuses.append(dict(run_id=key,plotted=True,state='COMPLETED',last_step=100,
                             source='new' if row['status']=='NEW' else 'reused_stage_b'))
        for start,end in WINDOWS:
            values = window_metrics(a['pi'],a['relative'],start,end)
            for j,prompt in enumerate(PROMPTS):
                summaries.append(dict(run_id=key,method=row['method'],base_id=row['base_id'],
                    arm=row['arm'],alpha=row['alpha'],lambda_current=row['lambda_current'],
                    beta_train=row['beta_train'],nu=row['nu'],kappa=row['kappa'],
                    prompt_id=prompt,start_step=start,end_step=end,
                    **{k:float(v[j]) for k,v in values.items()}))
    return plan,status,data,summaries,statuses


def response_legend(fig, y=.01):
    fig.legend(handles=[Line2D([],[],color=prior.COLORS[k],ls=prior.STYLES[k],lw=2,
                              label=f'Response {k+1}') for k in range(4)],
               loc='lower center',bbox_to_anchor=(.5,y),ncol=4)


def plot_pi(ax, a, j):
    for k in range(4):
        ax.plot(np.arange(101),a['pi'][:,j,k],color=prior.COLORS[k],ls=prior.STYLES[k],lw=1.2)
    ax.set(xlim=(0,100),ylim=(0,1),xticks=[0,20,40,60,80,100],yticks=[0,.25,.5,.75,1])


def save_png(fig,out,name,pdf):
    fig.savefig(out/(name+'.png'),dpi=150)
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)


def plot_group(out,method,base,group,pdf):
    baseline = group['ordinary']['record']
    params = parameter_label(baseline)
    for arm,a in group.items():
        fig,axes = plt.subplots(2,3,figsize=(13,7.4))
        fig.subplots_adjust(top=.81,bottom=.14,left=.065,right=.985,hspace=.43,wspace=.24)
        fig.suptitle(f"{method.upper()} | {label(arm,a['record'])} | {params}\n"
                     'Measured panel policy pi: four fixed responses, all six prompts',fontsize=14,y=.97)
        for j,ax in enumerate(axes.flat):
            plot_pi(ax,a,j)
            ax.set(title=f'Prompt {PROMPTS[j]}',xlabel='Outer iteration',ylabel='Panel probability')
        response_legend(fig)
        save_png(fig,out,f'{method}_{base}_{arm}_pi',pdf)
    for family,arms in [('reference',('ordinary','reference_half','reference_max')),
                        ('feedback',('ordinary','feedback_half','feedback_one'))]:
        if not all(arm in group for arm in arms):
            continue
        fig,axes = plt.subplots(6,3,figsize=(13,15.4))
        fig.subplots_adjust(top=.90,bottom=.07,left=.08,right=.985,hspace=.25,wspace=.18)
        fig.suptitle(f'{method.upper()} | {family.capitalize()} comparison | {params}\n'
                     'Same prompt in each row; identical response colors and 0-1 axes',fontsize=14,y=.975)
        for j in range(6):
            for col,arm in enumerate(arms):
                ax=axes[j,col]
                plot_pi(ax,group[arm],j)
                if j==0: ax.set_title(label(arm,group[arm]['record']),fontsize=12,pad=10)
                if col==0: ax.set_ylabel(f'Prompt {PROMPTS[j]}\nPanel probability')
                else: ax.set_yticklabels([])
                if j==5: ax.set_xlabel('Outer iteration')
                else: ax.set_xticklabels([])
        response_legend(fig,.012)
        save_png(fig,out,f'{method}_{base}_{family}_comparison',pdf)
    fig,axes=plt.subplots(2,3,figsize=(13,7.4))
    fig.subplots_adjust(top=.81,bottom=.17,left=.075,right=.985,hspace=.43,wspace=.24)
    fig.suptitle(f'{method.upper()} | Relative-sequence entropy | {params}\n'
                 'H(softmax(s(t)-s(0))), natural logarithm; not raw panel entropy',fontsize=14,y=.97)
    for j,ax in enumerate(axes.flat):
        for arm,a in group.items():
            ax.plot(np.arange(101),a['relative'][:,j],color=ARM_COLORS[ARMS.index(arm)],lw=1.4,
                    label=label(arm,a['record']))
        ax.set(title=f'Prompt {PROMPTS[j]}',xlabel='Outer iteration',ylabel='Entropy (nats)',
               xlim=(0,100),ylim=(0,np.log(4)*1.04))
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,.015),fontsize=9)
    save_png(fig,out,f'{method}_{base}_relative_entropy',pdf)


def aggregate_and_pair(rows):
    lookup = {}
    for r in rows:
        key=(r['method'],r['base_id'],r['arm'],r['start_step'],r['end_step'])
        lookup.setdefault(key,[]).append(r)
    means=[]
    for key,subset in lookup.items():
        assert len(subset)==6
        row={k:subset[0][k] for k in subset[0] if k!='prompt_id' and k not in FIELDS}
        row.update({k:float(np.mean([s[k] for s in subset])) for k in FIELDS})
        means.append(row)
    mean_lookup={(r['method'],r['base_id'],r['arm'],r['start_step'],r['end_step']):r for r in means}
    pairs=[]
    for r in means:
        if r['arm']=='ordinary': continue
        key=(r['method'],r['base_id'],'ordinary',r['start_step'],r['end_step'])
        if key not in mean_lookup: continue
        ordinary=mean_lookup[key]
        p={k:r[k] for k in r if k not in FIELDS}
        for field in FIELDS:
            p[field]=r[field]
            p[field+'_ordinary']=ordinary[field]
            p[field+'_difference']=r[field]-ordinary[field]
            p[field+'_percent_change']=100*(r[field]/ordinary[field]-1) if abs(ordinary[field])>1e-12 else None
        pairs.append(p)
    return means,pairs


def overview(out,plan,pairs):
    fig,axes=plt.subplots(2,2,figsize=(13,8.4))
    fig.subplots_adjust(top=.87,bottom=.12,left=.13,right=.91,wspace=.42,hspace=.45)
    base_ids=[r['base_id'] for r in plan['bases']]
    titles=['Center','alpha = .8','alpha = .99','lambda = .5','beta x 1.5']
    for col,field in enumerate(('probability_temporal_sd','tv_mean')):
        matrices=[]
        for method in ('ipo','dpo'):
            matrix=np.full((5,4),np.nan)
            for p in pairs:
                if p['method']==method and (p['start_step'],p['end_step'])==(51,100):
                    matrix[base_ids.index(p['base_id']),ARMS.index(p['arm'])-1]=p[field+'_difference']
            matrices.append(matrix)
        bound=max(float(np.nanmax(np.abs(m))) for m in matrices)
        bound=max(bound,1e-6)
        for row,(method,matrix) in enumerate(zip(('ipo','dpo'),matrices)):
            ax=axes[row,col]
            im=ax.imshow(matrix,cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto')
            for j in range(5):
                for k in range(4):
                    if np.isfinite(matrix[j,k]):
                        color='white' if abs(matrix[j,k])>.62*bound else 'black'
                        ax.text(k,j,f'{matrix[j,k]:+.3f}',ha='center',va='center',color=color,fontsize=10)
            ax.grid(False)
            ax.set(xticks=range(4),xticklabels=['Ref half','Ref max','Feed .5','Feed 1'],
                   yticks=range(5),yticklabels=titles,
                   title=method.upper()+(' | Temporal SD' if col==0 else ' | Adjacent-step TV'))
            fig.colorbar(im,ax=ax,fraction=.043,pad=.03)
    fig.suptitle('History minus matched ordinary | outer states 51-100\n'
                 'Mean over six prompts; blue = smaller, red = larger. Raw probability units.',fontsize=14,y=.975)
    fig.text(.5,.035,'Temporal SD measures spread over time; TV measures movement per update. These are descriptive, not convergence tests.',
             ha='center',fontsize=10)
    save_png(fig,out,'paired_fluctuation_overview',None)


def write_index(out,plan):
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Cyclic trend sweep results</title>',
           '<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;color:#222}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #ccc;padding:10px;text-align:left}a{color:#006ca5}summary{cursor:pointer;padding:12px 0;font-weight:600}h2{margin-top:36px}</style>',
           '<h1>Cyclic history trend sweep</h1><p>50 complete runs: 44 new + 6 reused Stage B. Six fixed prompts, outer states 0-100, seed0.</p>',
           '<p>Pi = softmax(response sequence-sum scores) over four fixed candidates. It is not token probability or a full response-space distribution. Response colors are fixed within each prompt. No smoothing or interpolation.</p>',
           '<p><a href="ipo_complete_figures.pdf">IPO complete PDF</a> | <a href="dpo_complete_figures.pdf">DPO complete PDF</a> | <a href="run_summary.csv">Numeric summary</a> | <a href="paired_summary.csv">Paired differences</a></p>',
           '<img src="paired_fluctuation_overview.png" alt="Paired descriptive fluctuation overview">']
    for method in ('ipo','dpo'):
        parts.append('<h2>'+method.upper()+'</h2><table><tr><th>Base</th><th>Reference vs ordinary</th><th>Feedback vs ordinary</th><th>Relative entropy</th></tr>')
        for base in plan['bases']:
            b=base['base_id']
            beta=(.2 if method=='ipo' else .8)*base['beta_scale']
            title=f"alpha={base['alpha']:g}, lambda={base['lambda_current']:g}, beta={beta:g}"
            parts.append('<tr><td>'+title+'</td>'+''.join('<td><a href="'+method+'_'+b+'_'+name+'.png">'+label+'</a></td>'
                for name,label in [('reference_comparison','All 6 prompts'),('feedback_comparison','All 6 prompts'),('relative_entropy','All 5 arms')])+'</tr>')
        parts.append('</table>')
        for base in plan['bases']:
            b=base['base_id']
            parts.append('<details><summary>'+method.upper()+' / '+html.escape(b)+' / each arm separately</summary>')
            for arm in ARMS:
                name=f'{method}_{b}_{arm}_pi.png'
                parts.append('<h3>'+html.escape(arm)+'</h3><a href="'+name+'"><img loading="lazy" src="'+name+'" alt="'+name+'"></a>')
            parts.append('</details>')
    parts.append('</html>')
    (out/'index.html').write_text('\n'.join(parts),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.absolute().resolve()
    out.mkdir(parents=True,exist_ok=True)
    plan,status,data,rows,statuses=collect(args.results.absolute().resolve())
    if len(data) != 50:
        raise ValueError('This final report requires all 50 complete logical runs')
    means,pairs=aggregate_and_pair(rows)
    prior.write_csv(out/'prompt_summary.csv',rows)
    prior.write_csv(out/'run_summary.csv',means)
    prior.write_csv(out/'paired_summary.csv',pairs)
    prompt_lookup={(r['method'],r['base_id'],r['arm'],r['prompt_id'],r['start_step'],r['end_step']):r for r in rows}
    prompt_pairs=[]
    for r in rows:
        if r['arm']=='ordinary': continue
        ordinary=prompt_lookup[(r['method'],r['base_id'],'ordinary',r['prompt_id'],r['start_step'],r['end_step'])]
        paired=dict(r)
        for field in FIELDS:
            paired[field+'_ordinary']=ordinary[field]
            paired[field+'_difference']=r[field]-ordinary[field]
        prompt_pairs.append(paired)
    prior.write_csv(out/'paired_prompt_summary.csv',prompt_pairs)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'grid.linewidth':.6,
        'legend.frameon':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
    for method in ('ipo','dpo'):
        with PdfPages(out/f'{method}_complete_figures.pdf') as pdf:
            for base in plan['bases']:
                b=base['base_id']
                group={arm:data[f'{method}_{b}_{arm}_b100_s0'] for arm in ARMS
                       if f'{method}_{b}_{arm}_b100_s0' in data}
                if 'ordinary' in group:
                    plot_group(out,method,b,group,pdf)
                    print(f'PLOTTED {method} {b} ({len(group)} arms)',flush=True)
    overview(out,plan,pairs)
    write_index(out,plan)
    validation=dict(runs=len(data),new_runs=sum(r['source']=='new' for r in statuses if r['plotted']),
        reused_runs=sum(r['source']!='new' for r in statuses if r['plotted']),snapshots=sum(a['last']+1 for a in data.values()),
        measured_probability_values=sum(a['pi'].size for a in data.values()),hashes_verified=True,
        source_and_reuse_verified=True,raw_pi_and_relative_entropy_reconstructed=True,
        reference_feedback_and_sampler_equations_verified=True,all_six_supports_aligned=True,
        png_figures=len(list(out.glob('*.png'))),methods=['ipo','dpo'],windows=WINDOWS,
        checked_at_utc=status['account']['checked_at_utc'],allocated_gpus=status['account']['allocated_gpus'])
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (out/'run_status.json').write_text(json.dumps(statuses,indent=2)+'\n')
    # Publish numeric coordinates only; private prompts/responses remain in local raw evidence.
    np.savez_compressed(out/'measured_pi_and_relative_entropy.npz',
        **{key+'__'+field:a[field] for key,a in data.items() for field in ('pi','relative')})
    print(json.dumps(validation,indent=2))


if __name__=='__main__':
    main()
