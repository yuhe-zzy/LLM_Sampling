"""Audit and plot all six Stage B arms in matched, measured-LLM coordinates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
INPUT = HERE
CODE = HERE.parent.parent
sys.path.insert(0, str(CODE))
from calibrated_protocol import mode_coordinates
from history_math import describe_distribution

METHODS = ('ipo', 'dpo')
ARMS = ('ordinary', 'reference', 'feedback')
COLORS = dict(ordinary='#B74349', reference='#16847A', feedback='#416DB1')
LABELS = dict(ordinary='Ordinary', reference='Reference (nu=0.45)', feedback='Feedback (kappa=0.5)')
RUNS = [f'{m}_{a}_b100_s0' for m in METHODS for a in ARMS]


def load(run):
    root = INPUT / 'raw' / run
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    cfg = manifest['config']
    assert cfg['iters'] == 100
    for name in ('history_math.py', 'population_calibration.py', 'calibrated_protocol.py',
                 'run_cyclic_history.py'):
        assert hashlib.sha256((CODE / name).read_bytes()).hexdigest() == manifest['source_sha256'][name]
    panels = json.loads((root / 'support.json').read_text(encoding='utf-8'))
    predictions = json.loads((root / 'verified_population_predictions.json').read_text(encoding='utf-8'))
    with (root / 'metrics.csv').open() as f:
        metrics = list(csv.DictReader(f))
    steps = np.array([int(r['step']) for r in metrics])
    np.testing.assert_array_equal(steps, np.arange(len(steps)))
    assert steps[-1] == manifest['last_complete_step']
    if manifest['state'] == 'COMPLETED':
        assert steps[-1] == 100
    data = []
    for step in steps:
        with np.load(root / 'snapshots' / f'step_{step:04d}.npz', allow_pickle=False) as z:
            data.append({k: z[k].copy() for k in z.files})
    assert all(np.isfinite(v).all() for d in data for v in d.values())
    initial = data[0]['sequence_sum_logprob']
    for t, snapshot in enumerate(data):
        previous = None if t == 0 else data[t-1]['sequence_sum_logprob']
        fresh = describe_distribution(snapshot['sequence_sum_logprob'], initial, previous)
        for key, value in fresh.items():
            np.testing.assert_allclose(snapshot[key], value, rtol=1e-10, atol=1e-10)
        for key in ('relative_sequence_entropy', 'panel_entropy'):
            np.testing.assert_allclose(float(metrics[t][key + '_mean']), fresh[key].mean(), atol=1e-10)
    return dict(run=run, cfg=cfg, manifest=manifest, panels=panels, predictions=predictions,
                metrics=metrics, data=data, steps=steps,
                x=np.stack([d['centered_logits'] for d in data]),
                entropy=np.stack([d['relative_sequence_entropy'] for d in data]))


def align(all_data):
    common = all_data[RUNS[0]]
    for a in all_data.values():
        assert a['panels'] == common['panels']
        np.testing.assert_allclose(a['data'][0]['sequence_sum_logprob'],
                                   common['data'][0]['sequence_sum_logprob'], atol=1e-10, rtol=0)
    for method in METHODS:
        baseline = all_data[f'{method}_ordinary_b100_s0']
        coords = mode_coordinates(baseline['panels'], baseline['cfg'], baseline['predictions'])
        mode, fixed = coords['left_modes'], coords['fixed_logits']
        initial_z = np.einsum('pk,pk->p', mode, baseline['x'][0] - fixed)
        # Fix one arbitrary eigenvector phase per prompt for ALL three arms.
        mode = mode * np.exp(-1j * np.angle(initial_z))[:, None]
        for arm in ARMS:
            a = all_data[f'{method}_{arm}_b100_s0']
            for snapshot in a['data']:
                np.testing.assert_allclose(snapshot['population_fixed_logits'], fixed, atol=1e-10)
            a['z'] = np.einsum('pk,tpk->tp', mode, a['x'] - fixed)
            a['amp'] = abs(a['z'])
            saved_amp = np.stack([d['cyclic_mode_amplitude'] for d in a['data']])
            np.testing.assert_allclose(a['amp'], saved_amp, atol=1e-10)
            saved_z = np.stack([d['cyclic_mode_real'] + 1j*d['cyclic_mode_imag'] for d in a['data']])
            rotation = a['z'][0] / saved_z[0]
            np.testing.assert_allclose(a['z'], saved_z * rotation, atol=1e-10)
            np.testing.assert_allclose(a['amp'].mean(1),
                [float(r['cyclic_mode_amplitude_mean']) for r in a['metrics']], atol=1e-10)
            phase = np.unwrap(np.angle(a['z']), axis=0)
            a['turns'] = (phase - phase[0]) / (2*np.pi)
            a['phase_increment'] = np.angle(a['z'][1:] * a['z'][:-1].conj())
            a['step_rms'] = np.sqrt(np.mean(np.diff(a['x'], axis=0)**2, axis=(1, 2)))
            np.testing.assert_allclose(a['step_rms'],
                [float(r['centered_step_rms']) for r in a['metrics'][1:]], atol=1e-10)


def write_csv(name, rows):
    with (HERE / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tables(all_data):
    summary, per_prompt, points = [], [], []
    for method in METHODS:
        baseline = all_data[f'{method}_ordinary_b100_s0']
        for arm in ARMS:
            a = all_data[f'{method}_{arm}_b100_s0']
            common_end = min(d['steps'][-1] for d in all_data.values())
            assert common_end == 100, 'The published 51..100 window requires all runs complete'
            late = slice(51, common_end+1)
            end = int(a['steps'][-1])
            row = dict(method=method, arm=arm, run=a['run'], state=a['manifest']['state'],
                last_complete_step=end, beta=a['cfg']['beta_train'],
                amplitude_0=float(a['amp'][0].mean()), amplitude_final=float(a['amp'][-1].mean()),
                amplitude_mean_51_100=float(a['amp'][late].mean()),
                amplitude_change_vs_ordinary_pct=100*float(a['amp'][late].mean()/baseline['amp'][late].mean()-1),
                step_rms_mean_51_100=float(a['step_rms'][50:common_end].mean()),
                entropy_relative_0=float(a['entropy'][0].mean()),
                entropy_relative_final=float(a['entropy'][-1].mean()),
                entropy_relative_mean_51_100=float(a['entropy'][late].mean()),
                min_mode_amplitude=float(a['amp'].min()),
                max_abs_phase_step_radians=float(abs(a['phase_increment']).max()),
                prompt_count=len(a['panels']))
            summary.append(row)
            for j, panel in enumerate(a['panels']):
                per_prompt.append(dict(method=method, arm=arm, prompt_id=panel['prompt_id'],
                    amplitude_initial=float(a['amp'][0,j]), amplitude_final=float(a['amp'][-1,j]),
                    amplitude_mean_51_100=float(a['amp'][late,j].mean()),
                    amplitude_change_vs_ordinary_pct=100*float(a['amp'][late,j].mean()/baseline['amp'][late,j].mean()-1),
                    net_turns_0_100=float(a['turns'][-1,j]),
                    net_turns_20_100=float(a['turns'][-1,j]-a['turns'][20,j]),
                    net_turns_50_100=float(a['turns'][-1,j]-a['turns'][50,j]),
                    positive_phase_fraction_51_100=float(np.mean(a['phase_increment'][50:common_end,j] > 0)),
                    min_mode_amplitude=float(a['amp'][:,j].min()),
                    phase_low_amplitude_fraction=float(np.mean(a['amp'][:,j] < .1)),
                    entropy_relative_final=float(a['entropy'][-1,j]),
                    entropy_relative_mean_51_100=float(a['entropy'][late,j].mean())))
                for t in a['steps']:
                    points.append(dict(method=method, arm=arm, prompt_id=panel['prompt_id'], step=int(t),
                        mode_real=float(a['z'][t,j].real), mode_imag=float(a['z'][t,j].imag),
                        amplitude=float(a['amp'][t,j]), phase_turns=float(a['turns'][t,j]),
                        relative_sequence_entropy=float(a['entropy'][t,j])))
    write_csv('run_summary.csv', summary)
    write_csv('per_prompt_summary.csv', per_prompt)
    write_csv('trajectory_points.csv', points)
    (HERE / 'summary.json').write_text(json.dumps(dict(runs=summary, per_prompt=per_prompt), indent=2)+'\n')
    return summary


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.titleweight':'semibold',
        'axes.grid':True, 'grid.alpha':.18, 'grid.linewidth':.6,
        'legend.frameon':False, 'savefig.facecolor':'white', 'pdf.fonttype':42})


def legend(fig):
    handles = [Line2D([], [], color=COLORS[a], lw=2, label=LABELS[a]) for a in ARMS]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.005), ncol=3, fontsize=11)


def save(fig, name):
    fig.savefig(HERE / (name+'.png'), dpi=170, facecolor='white')
    fig.savefig(HERE / (name+'.pdf'), facecolor='white')
    plt.close(fig)


def figures(all_data):
    style()
    amp_max = max(a['amp'].max() for a in all_data.values()) * 1.08
    mean_amp_max = max(a['amp'].mean(1).max() for a in all_data.values()) * 1.08
    step_max = max(a['step_rms'].max() for a in all_data.values()) * 1.08
    phase_min = min(a['turns'].min() for a in all_data.values()) - .2
    phase_max = max(a['turns'].max() for a in all_data.values()) + .2
    xy_all = np.concatenate([np.r_[a['z'].real.ravel(), a['z'].imag.ravel()] for a in all_data.values()])
    low, high = float(xy_all.min()), float(xy_all.max())
    pad = .07*(high-low)
    limits = (low-pad, high+pad)

    fig, axes = plt.subplots(2, 3, figsize=(14.5,8.0), layout='constrained')
    fig.get_layout_engine().set(rect=(0,.075,1,.90))
    for r, method in enumerate(METHODS):
        for arm in ARMS:
            a = all_data[f'{method}_{arm}_b100_s0']
            for c, (steps, values) in enumerate(((a['steps'], a['amp'].mean(1)),
                    (a['steps'][1:], a['step_rms']), (a['steps'], a['entropy'].mean(1)))):
                axes[r,c].plot(steps, values, color=COLORS[arm], lw=1.65)
        titles = ('Cyclic-mode amplitude', 'Change between rounds', 'Relative-sequence entropy')
        for c in range(3):
            axes[r,c].set(title=method.upper()+' | '+titles[c], xlabel='Outer iteration', xlim=(0,100))
            axes[r,c].set_ylabel(('Mean amplitude (nats)', 'Centered-logit RMS (nats)', 'Mean entropy (nats)')[c])
            axes[r,c].set_ylim(0, (mean_amp_max, step_max, np.log(4)*1.05)[c])
    fig.suptitle('Stage B | six measured LLM runs, outer states 0-100\nSame six prompts and seed; IPO beta=0.2, DPO beta=0.8; alpha=0.9, lambda=0.8', fontsize=14)
    legend(fig)
    save(fig, 'stage_b_overview')

    for method in METHODS:
        fig, axes = plt.subplots(2,3,figsize=(13.5,9.4),layout='constrained')
        fig.get_layout_engine().set(rect=(0,.065,1,.91))
        for j, ax in enumerate(axes.flat):
            for arm in ARMS:
                a = all_data[f'{method}_{arm}_b100_s0']
                z = a['z'][:,j]
                ax.plot(z[:21].real,z[:21].imag,color=COLORS[arm],lw=1.2,alpha=.35)
                ax.plot(z[20:].real,z[20:].imag,color=COLORS[arm],lw=1.4,alpha=.9)
                ax.plot(z[-1].real,z[-1].imag,'s',color=COLORS[arm],ms=5)
                for t in (30,60,90):
                    if t+1 < len(z):
                        ax.annotate('',xy=(z[t+1].real,z[t+1].imag),xytext=(z[t].real,z[t].imag),
                            arrowprops=dict(arrowstyle='->',color=COLORS[arm],lw=1.3))
            ax.plot(z[0].real,z[0].imag,'o',mfc='white',mec='#333333',ms=6)
            ax.plot(0,0,'+',color='#444444',ms=7)
            ax.set(xlim=limits,ylim=limits,aspect='equal',title='Prompt '+str(a['panels'][j]['prompt_id']),
                   xlabel='Real cyclic-mode coordinate',ylabel='Imaginary cyclic-mode coordinate')
        fig.suptitle(method.upper()+' | same-prompt measured trajectories, all six prompts\nCommon per-prompt mode and origin; pale = states 0-20, solid = 20-100; circle = start, square = end',fontsize=13)
        legend(fig)
        save(fig,method+'_all_prompt_trajectories')

        fig,axes=plt.subplots(3,4,figsize=(18,10.4),layout='constrained')
        fig.get_layout_engine().set(rect=(0,.065,1,.92))
        for j in range(6):
            r,c = j//2,2*(j%2)
            for arm in ARMS:
                a=all_data[f'{method}_{arm}_b100_s0']
                axes[r,c].plot(a['steps'],a['amp'][:,j],color=COLORS[arm],lw=1.45)
                axes[r,c+1].plot(a['steps'],a['turns'][:,j],color=COLORS[arm],lw=1.45)
                near_origin = a['amp'][:,j] < .1
                axes[r,c+1].scatter(a['steps'][near_origin],a['turns'][near_origin,j],
                                   s=16,marker='x',color=COLORS[arm],alpha=.8)
            pid=a['panels'][j]['prompt_id']
            axes[r,c].set(title=f'Prompt {pid} | amplitude',ylabel='Amplitude (nats)',ylim=(0,amp_max))
            axes[r,c+1].set(title=f'Prompt {pid} | phase',ylabel='Net unwrapped turns',ylim=(phase_min,phase_max))
            for ax in (axes[r,c],axes[r,c+1]):
                ax.set(xlabel='Outer iteration',xlim=(0,100))
        fig.suptitle(method.upper()+' | cyclic-mode amplitude and phase for every prompt\nShared scales; x marks amplitude <0.1 (phase sensitive near origin); unwrapped phase is not proof of a limit cycle',fontsize=13)
        legend(fig)
        save(fig,method+'_amplitude_phase')

        fig,axes=plt.subplots(2,3,figsize=(13.5,7.6),layout='constrained')
        fig.get_layout_engine().set(rect=(0,.075,1,.91))
        for j,ax in enumerate(axes.flat):
            for arm in ARMS:
                a=all_data[f'{method}_{arm}_b100_s0']
                ax.plot(a['steps'],a['entropy'][:,j],color=COLORS[arm],lw=1.55)
            ax.set(title='Prompt '+str(a['panels'][j]['prompt_id']),xlabel='Outer iteration',
                   ylabel='Relative-sequence entropy (nats)',xlim=(0,100),ylim=(0,np.log(4)*1.05))
        fig.suptitle(method.upper()+' | relative-sequence entropy on the fixed four-response panel\nr_t = softmax(log p_t(y|x) - log p_0(y|x)); H(r_t), not raw panel entropy',fontsize=13)
        legend(fig)
        save(fig,method+'_relative_entropy')


def main():
    global HERE, INPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=INPUT)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    INPUT = args.results_dir.resolve()
    HERE = (args.output_dir or args.results_dir).resolve()
    HERE.mkdir(parents=True, exist_ok=True)
    receipt=json.loads((INPUT/'download_receipt.json').read_text())
    for name, expected in receipt['files_sha256'].items():
        assert hashlib.sha256((INPUT/name).read_bytes()).hexdigest() == expected
    data={r:load(r) for r in RUNS}
    align(data)
    summary=tables(data)
    figures(data)
    (HERE/'validation.json').write_text(json.dumps(dict(
        checked_files=len(receipt['files_sha256']), snapshot_count=sum(len(a['data']) for a in data.values()),
        all_finite=True, metric_reconstruction_passed=True, shared_support_and_initial_scores=True,
        common_mode_and_origin_within_method=True, source_hashes_match=True,
        complete_steps={r:int(a['steps'][-1]) for r,a in data.items()}),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
