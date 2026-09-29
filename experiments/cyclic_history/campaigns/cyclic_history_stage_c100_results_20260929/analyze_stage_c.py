"""Audit Stage C against completed B controls and plot every prompt, without raw text exports."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('stage_b_audit',
    HERE.parent/'cyclic_history_stage_b100_results_20260929'/'analyze_stage_b.py')
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
from calibrated_protocol import mode_coordinates, contract_hash
from history_math import support_hash

METHODS = ('ipo', 'dpo')
ARMS = ('ordinary', 'reference', 'feedback', 'mixed')
LABELS = dict(ordinary='Aligned ordinary', reference='Aligned reference',
              feedback='Aligned feedback', mixed='Mixed ordinary')
COLORS = dict(ordinary='#B74349', reference='#16847A', feedback='#416DB1', mixed='#8056A5')
RESPONSE_COLORS = ('#2879B7', '#D68622', '#20946E', '#BD4F76')
PIDS = [54, 251, 612, 737, 867, 945]


def verify_receipt(root):
    receipt = json.loads((root/'download_receipt.json').read_text())
    for name, expected in receipt['files_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
    return len(receipt['files_sha256'])


def canonical_phase(z):
    if np.any(abs(z[0]) < 1e-12):
        raise ValueError('Initial mode too small for phase alignment')
    return z*np.exp(-1j*np.angle(z[0]))[None,:]


def validate_contrast(aligned, mixed):
    assert [p['prompt_id'] for p in mixed['panels']] == PIDS
    assert mixed['cfg']['orientations'] == [-1, 1, -1, 1, -1, 1]
    assert aligned['cfg']['orientations'] == [1]*6
    excluded = {'preference_matrix'}
    for j, (a, m) in enumerate(zip(aligned['panels'], mixed['panels'])):
        assert {k:v for k,v in a.items() if k not in excluded} == {
            k:v for k,v in m.items() if k not in excluded}
        expected = np.array(a['preference_matrix'])
        if mixed['cfg']['orientations'][j] == -1:
            expected = 1-expected
        np.testing.assert_allclose(m['preference_matrix'], expected, rtol=0, atol=1e-14)
    for key in ('method', 'alpha', 'lambda_current', 'beta_train', 'seed', 'scheme', 'nu', 'kappa',
                'epochs_per_iter', 'batch_size', 'grad_accum', 'lr', 'lora_r', 'lora_alpha',
                'lora_dropout', 'max_length', 'pair_mode', 'iters', 'calibration',
                'warmup_ratio', 'dtype', 'optimizer_reset', 'pairs_per_prompt',
                'support_probability', 'pair_law', 'coverage', 'cycle_probability', 'support_seed'):
        assert aligned['cfg'][key] == mixed['cfg'][key], key
    np.testing.assert_array_equal(aligned['data'][0]['sequence_sum_logprob'],
                                  mixed['data'][0]['sequence_sum_logprob'])
    np.testing.assert_array_equal(aligned['data'][0]['response_token_count'],
                                  mixed['data'][0]['response_token_count'])


def enrich(a):
    assert a['manifest']['state'] == 'COMPLETED'
    np.testing.assert_array_equal(a['steps'], np.arange(101))
    assert contract_hash(a['cfg']) == a['cfg']['calibration_contract_sha256']
    assert support_hash(a['panels']) == a['cfg']['transformed_support_sha256']
    a['pi'] = np.stack([d['panel_probability'] for d in a['data']])
    np.testing.assert_allclose(a['pi'].sum(-1), 1, atol=1e-12)
    assert np.all((a['pi'] >= 0) & (a['pi'] <= 1))
    a['tv'] = .5*np.abs(np.diff(a['pi'], axis=0)).sum(-1)
    np.testing.assert_allclose(a['tv'].mean(1), [float(r['tv_mean']) for r in a['metrics'][1:]], atol=1e-12)
    # Each orientation has its OWN eigenmode and fixed point. Never reuse B's
    # basis for C. Primary cross-arm plots below instead use fixed pi contrasts.
    coords = mode_coordinates(a['panels'], a['cfg'], a['predictions'])
    a['z'] = canonical_phase(np.einsum('pk,tpk->tp', coords['left_modes'], a['x']-coords['fixed_logits']))
    saved_z = np.stack([d['cyclic_mode_real']+1j*d['cyclic_mode_imag'] for d in a['data']])
    # Eigenvectors have arbitrary constant complex phase across BLAS platforms.
    np.testing.assert_allclose(a['z'], canonical_phase(saved_z), atol=1e-10)
    for t, d in enumerate(a['data']):
        np.testing.assert_allclose(d['population_fixed_logits'], coords['fixed_logits'], atol=1e-10)
    a['amp'] = abs(a['z'])
    np.testing.assert_allclose(a['amp'], np.stack([d['cyclic_mode_amplitude'] for d in a['data']]), atol=1e-10)
    phase = np.unwrap(np.angle(a['z']), axis=0)
    a['turns'] = (phase-phase[0])/(2*np.pi)
    roles = a['cfg']['calibration']['response_roles']
    a['xy'] = np.stack([np.stack(((a['pi'][:,j,r[0]]-a['pi'][:,j,r[2]])/np.sqrt(2),
                                  (a['pi'][:,j,r[1]]-a['pi'][:,j,r[3]])/np.sqrt(2)), axis=-1)
                         for j,r in enumerate(roles)], axis=1)


def csv_write(out, name, rows):
    with (out/name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tables(data, out):
    runs, prompts, points = [], [], []
    for (method, arm), a in data.items():
        row = dict(method=method, arm=arm, last_complete_step=100, beta=a['cfg']['beta_train'],
            late_mean_tv=float(a['tv'][50:].mean()),
            late_mean_probability_temporal_sd=float(a['pi'][51:].std(axis=0).mean()),
            late_mean_relative_entropy=float(a['entropy'][51:].mean()),
            final_relative_entropy=float(a['entropy'][-1].mean()),
            late_mean_own_mode_amplitude=float(a['amp'][51:].mean()),
            late_mean_operator_residual=float(np.mean([float(r['operator_residual_rms']) for r in a['metrics'][51:]])))
        runs.append(row)
        for j, pid in enumerate(PIDS):
            prompts.append(dict(method=method, arm=arm, prompt_id=pid,
                orientation=a['cfg']['orientations'][j], late_mean_tv=float(a['tv'][50:,j].mean()),
                late_mean_probability_temporal_sd=float(a['pi'][51:,j].std(axis=0).mean()),
                late_mean_relative_entropy=float(a['entropy'][51:,j].mean()),
                final_relative_entropy=float(a['entropy'][-1,j]),
                late_mean_own_mode_amplitude=float(a['amp'][51:,j].mean())))
            for t in a['steps']:
                for i in range(4):
                    points.append(dict(method=method, arm=arm, prompt_id=pid, step=int(t),
                        response_index=i+1, pi=float(a['pi'][t,j,i]),
                        sequence_sum_logprob=float(a['data'][t]['sequence_sum_logprob'][j,i]),
                        relative_sequence_entropy=float(a['entropy'][t,j]),
                        probability_contrast_x=float(a['xy'][t,j,0]), probability_contrast_y=float(a['xy'][t,j,1]),
                        own_mode_amplitude=float(a['amp'][t,j]), own_mode_phase_turns=float(a['turns'][t,j])))
    csv_write(out,'run_summary.csv',runs)
    csv_write(out,'per_prompt_summary.csv',prompts)
    csv_write(out,'pi_trajectories.csv',points)
    (out/'summary.json').write_text(json.dumps(dict(runs=runs, per_prompt=prompts),indent=2)+'\n')
    return runs


def figures(data, out):
    B.style()
    all_pdf = PdfPages(out/'all_stage_c_figures.pdf')
    def save(fig, name):
        fig.savefig(out/(name+'.png'),dpi=160)
        fig.savefig(out/(name+'.pdf'))
        all_pdf.savefig(fig)
        plt.close(fig)
    def setup(rows, cols, size, title, handles):
        fig, axes = plt.subplots(rows,cols,figsize=size,layout='constrained',squeeze=False)
        fig.get_layout_engine().set(rect=(0,.06,1,.91))
        fig.suptitle(title,fontsize=14)
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=len(handles),fontsize=11)
        return fig, axes
    handles = [Line2D([],[],color=COLORS[a],lw=2,label=LABELS[a]) for a in ARMS]
    response_handles = [Line2D([],[],color=c,lw=2,label=f'Response {i+1}') for i,c in enumerate(RESPONSE_COLORS)]
    pair_handles = [handles[0],handles[-1]]
    fig,axes=setup(2,3,(15,8),
        'Completed cyclic LLM experiments | six-prompt means, seed 0\nIPO beta=0.2; DPO beta=0.8; alpha=0.9, lambda=0.8; all states 0-100',handles)
    max_tv = max(a['tv'].mean(1).max() for a in data.values())*1.08
    max_amp = max(a['amp'].mean(1).max() for a in data.values())*1.08
    for r,method in enumerate(METHODS):
        for arm in ARMS:
            a=data[method,arm]
            axes[r,0].plot(a['steps'][1:],a['tv'].mean(1),color=COLORS[arm],lw=1.4)
            axes[r,1].plot(a['steps'],a['entropy'].mean(1),color=COLORS[arm],lw=1.4)
            axes[r,2].plot(a['steps'],a['amp'].mean(1),color=COLORS[arm],lw=1.4)
        for c,(title,ylabel,upper) in enumerate((('Between-round probability change','Mean TV distance',max_tv),
            ('Relative-sequence entropy','Mean H(softmax(s_t - s_0)), nats',np.log(4)*1.04),
            ('Own-orientation cyclic mode','Own fixed point / mode; nats',max_amp))):
            axes[r,c].set(title=method.upper()+' | '+title,xlabel='Outer iteration',ylabel=ylabel,xlim=(0,100),ylim=(0,upper))
    save(fig,'all_arms_overview')
    for method in METHODS:
        base,mixed=data[method,'ordinary'],data[method,'mixed']
        fig,axes=setup(3,4,(18,10.5),method.upper()+' | matched pi trajectories: aligned vs mixed ordinary\nSame response colors and axes; pi = softmax(response sequence-sum log probability)',response_handles)
        for j,pid in enumerate(PIDS):
            for side,a in enumerate((base,mixed)):
                ax=axes[j//2,2*(j%2)+side]
                change='reversed' if mixed['cfg']['orientations'][j]<0 else 'unchanged'
                for i,color in enumerate(RESPONSE_COLORS):
                    ax.plot(a['steps'],a['pi'][:,j,i],color=color,lw=1.1)
                ax.set(title=f'Prompt {pid} | '+('Aligned' if side==0 else f'Mixed ({change})'),
                       xlabel='Outer iteration',ylabel='Panel probability pi',xlim=(0,100),ylim=(0,1))
        save(fig,method+'_aligned_vs_mixed_pi')
        fig,axes=setup(2,3,(13.5,7.8),method.upper()+' | mixed ordinary: all six prompts\nFour fixed responses; raw panel probability, not relative-to-initial probability',response_handles)
        for j,ax in enumerate(axes.flat):
            for i,color in enumerate(RESPONSE_COLORS):
                ax.plot(mixed['steps'],mixed['pi'][:,j,i],color=color,lw=1.3)
            change='reversed' if mixed['cfg']['orientations'][j]<0 else 'unchanged'
            ax.set(title=f'Prompt {PIDS[j]} | {change}',xlabel='Outer iteration',ylabel='Panel probability pi',xlim=(0,100),ylim=(0,1))
        save(fig,method+'_mixed_pi_vs_iter')
        for metric,ylabel,suffix in (('entropy','Relative-sequence entropy (nats)','relative_entropy'),
                                     ('tv','Between-round TV distance','probability_motion')):
            fig,axes=setup(2,3,(13.5,7.8),method.upper()+' | aligned vs mixed ordinary, all prompts\n'+
                ('H(softmax(s_t - s_0)); not raw panel entropy' if metric=='entropy' else 'TV = 0.5 sum |pi_t - pi_(t-1)|; zero means no panel-probability movement'),pair_handles)
            upper=np.log(4)*1.04 if metric=='entropy' else max(base['tv'].max(),mixed['tv'].max())*1.05
            for j,ax in enumerate(axes.flat):
                for arm in ('ordinary','mixed'):
                    a=data[method,arm]
                    ax.plot(a['steps'] if metric=='entropy' else a['steps'][1:],a[metric][:,j],color=COLORS[arm],lw=1.4)
                change='reversed' if mixed['cfg']['orientations'][j]<0 else 'unchanged'
                ax.set(title=f'Prompt {PIDS[j]} | {change}',xlabel='Outer iteration',ylabel=ylabel,xlim=(0,100),ylim=(0,upper))
            save(fig,method+'_'+suffix)
        fig,axes=setup(2,3,(13.5,9),method.upper()+' | measured trajectories in a COMMON probability plane\nFixed role contrasts, NOT an eigenmode or fixed-point error; circle=start, square=end; pale=0-20',pair_handles)
        lim=(-.73,.73)
        for j,ax in enumerate(axes.flat):
            for arm in ('ordinary','mixed'):
                a=data[method,arm]; xy=a['xy'][:,j]
                ax.plot(xy[:21,0],xy[:21,1],color=COLORS[arm],alpha=.3,lw=1)
                ax.plot(xy[20:,0],xy[20:,1],color=COLORS[arm],alpha=.8,lw=1.2)
                ax.plot(xy[-1,0],xy[-1,1],'s',color=COLORS[arm],ms=5)
                for t in (30,60,90):
                    ax.annotate('',xy=xy[t+1],xytext=xy[t],arrowprops=dict(arrowstyle='->',color=COLORS[arm],lw=1.1))
            ax.plot(xy[0,0],xy[0,1],'o',mfc='white',mec='black',ms=5)
            ax.set(title=f'Prompt {PIDS[j]}',xlabel='(pi_role1 - pi_role3) / sqrt(2)',
                   ylabel='(pi_role2 - pi_role4) / sqrt(2)',xlim=lim,ylim=lim,aspect='equal')
        save(fig,method+'_common_probability_trajectories')
    all_pdf.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-b-dir',type=Path,required=True)
    parser.add_argument('--stage-c-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    checked=verify_receipt(args.stage_b_dir)+verify_receipt(args.stage_c_dir)
    data={}
    for method in METHODS:
        for arm in ARMS:
            B.INPUT=args.stage_c_dir if arm=='mixed' else args.stage_b_dir
            run=f'{method}_{arm}_'+('c100_s0' if arm=='mixed' else 'b100_s0')
            data[method,arm]=B.load(run)
            enrich(data[method,arm])
        validate_contrast(data[method,'ordinary'],data[method,'mixed'])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    rows=tables(data,args.output_dir)
    figures(data,args.output_dir)
    (args.output_dir/'validation.json').write_text(json.dumps(dict(checked_download_files=checked,
        snapshot_count=808, all_finite=True, complete_states='0..100', raw_metrics_reconstructed=True,
        source_hashes_match=True, matched_text_initial_scores_token_lengths=True,
        orientation_only_contrast_verified=True, modes_recomputed_per_orientation=True,
        common_probability_projection=True, exported_private_text=False),indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
