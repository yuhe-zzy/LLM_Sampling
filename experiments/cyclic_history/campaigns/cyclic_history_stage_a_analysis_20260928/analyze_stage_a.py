"""Reproducible read-only analysis of all Stage A prompts and outer rounds."""
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
from history_math import build_outer_state
from population_calibration import exact_trajectory
from calibrated_protocol import mode_coordinates

RUNS = ['ipo_ordinary_calibrated_s0', 'ipo_stable_calibrated_s0',
        'dpo_ordinary_calibrated_s0', 'dpo_stable_calibrated_s0']


def load(run):
    root = INPUT / 'raw' / run
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['state'] == 'COMPLETED' and manifest['last_complete_step'] == 30
    for name in ['history_math.py', 'population_calibration.py', 'calibrated_protocol.py',
                 'run_cyclic_history.py']:
        assert hashlib.sha256((CODE / name).read_bytes()).hexdigest() == manifest['source_sha256'][name]
    cfg = manifest['config']
    panels = json.loads((root / 'support.json').read_text(encoding='utf-8'))
    predictions = json.loads((root / 'verified_population_predictions.json').read_text(encoding='utf-8'))
    with (root / 'metrics.csv').open() as f:
        metrics = list(csv.DictReader(f))
    assert [int(r['step']) for r in metrics] == list(range(31))
    data = []
    for step in range(31):
        with np.load(root / 'snapshots' / f'step_{step:04d}.npz', allow_pickle=False) as z:
            data.append({k: z[k].copy() for k in z.files})
    assert all(np.isfinite(v).all() for d in data for v in d.values())
    x = np.stack([d['centered_logits'] for d in data])
    reference = data[0]['sequence_sum_logprob']
    fixed = data[0]['population_fixed_logits']
    mode = mode_coordinates(panels, cfg, predictions)['left_modes']
    z = np.einsum('pk,tpk->tp', mode, x - fixed)
    saved_z = np.stack([d['cyclic_mode_real'] + 1j*d['cyclic_mode_imag'] for d in data])
    # Eigenvectors have an arbitrary constant phase across LAPACK versions.
    alignment = np.sum(z.conj() * saved_z, axis=0) / np.sum(abs(z)**2, axis=0)
    np.testing.assert_allclose(abs(alignment), 1, atol=1e-10)
    mode *= alignment[:, None]
    z *= alignment
    np.testing.assert_allclose(z, saved_z, atol=1e-10)
    exact = np.stack([exact_trajectory(cfg['method'], p['preference_matrix'], reference[j],
                                      reference[j], cfg['alpha'], cfg['lambda_current'],
                                      cfg['beta_train'], steps=200)
                      for j, p in enumerate(panels)], axis=1)
    exact_z = np.einsum('pk,tpk->tp', mode, exact - fixed)
    target = np.stack([d['training_target_logits'] for d in data[1:]])
    matrices = np.array([p['preference_matrix'] for p in panels])
    for t in range(30):
        state = build_outer_state(reference, x[t], x[max(t-1, 0)], matrices, cfg['method'],
                                  cfg['alpha'], cfg['lambda_current'], cfg['beta_train'],
                                  cfg['nu'], cfg['kappa'])
        np.testing.assert_allclose(state['target_logits'], target[t], atol=2e-6)
    desired_step = target - x[:-1]
    actual_step = np.diff(x, axis=0)
    epsilon = x[1:] - target
    np.testing.assert_allclose([float(r['operator_residual_rms']) for r in metrics[1:]],
                               np.sqrt(np.mean(epsilon**2, axis=(1,2))), atol=1e-10)
    np.testing.assert_allclose([float(r['fixed_point_residual_rms']) for r in metrics],
                               np.sqrt(np.mean((x-fixed)**2, axis=(1,2))), atol=1e-10)
    return dict(run=run, cfg=cfg, panels=panels, predictions=predictions,
                metrics=metrics, data=data, x=x, fixed=fixed, z=z, exact=exact,
                exact_z=exact_z, target=target, desired_step=desired_step,
                actual_step=actual_step, epsilon=epsilon, mode=mode)


def summarize(a):
    z, x = a['z'], a['x']
    amp = abs(z)
    phase = np.unwrap(np.angle(z), axis=0)
    error = x - a['fixed']
    desired, actual, eps = a['desired_step'][20:], a['actual_step'][20:], a['epsilon'][20:]
    eta = float(np.sum(actual * desired) / np.sum(desired ** 2))
    eta_residual = float(np.linalg.norm(actual - eta * desired) / np.linalg.norm(actual))
    summary = dict(run=a['run'], beta=a['cfg']['beta_train'],
        amplitude_mean_steps21_30=float(amp[21:].mean()),
        amplitude_mean_steps11_20=float(amp[11:21].mean()),
        fixed_point_rms_steps21_30=float(np.sqrt(np.mean(error[21:]**2))),
        step_rms_steps21_30=float(np.sqrt(np.mean(actual**2))),
        target_error_rms_steps21_30=float(np.sqrt(np.mean(eps**2))),
        desired_step_rms_steps21_30=float(np.sqrt(np.mean(desired**2))),
        target_error_to_intended_step_ratio=float(np.linalg.norm(eps)/np.linalg.norm(desired)),
        persistent_target_error_rms=float(np.sqrt(np.mean(eps.mean(axis=0)**2))),
        exact_fixed_point_rms_step30=float(np.sqrt(np.mean((a['exact'][30]-a['fixed'])**2))),
        observed_fixed_point_rms_step30=float(np.sqrt(np.mean((x[30]-a['fixed'])**2))),
        exact_amplitude_mean_step30=float(abs(a['exact_z'][30]).mean()),
        observed_amplitude_mean_step30=float(amp[30].mean()),
        min_observed_mode_amplitude=float(amp.min()),
        max_observed_wrapped_phase_increment=float(abs(np.angle(z[1:]*z[:-1].conj())).max()),
        pooled_effective_step_fraction=eta, scalar_fit_relative_residual=eta_residual,
        per_prompt=[])
    for j, panel in enumerate(a['panels']):
        dp = np.diff(phase[:, j])
        ephase = np.unwrap(np.angle(a['exact_z'][:, j]))
        summary['per_prompt'].append(dict(prompt_id=panel['prompt_id'],
            amp_initial=float(amp[0,j]), amp_step10=float(amp[10,j]), amp_final=float(amp[30,j]),
            amp_late_mean=float(amp[21:,j].mean()),
            amp_late_to_mid=float(amp[21:,j].mean()/amp[11:21,j].mean()),
            net_turns_0_30=float((phase[30,j]-phase[0,j])/(2*np.pi)),
            net_turns_10_30=float((phase[30,j]-phase[10,j])/(2*np.pi)),
            late_turns_20_30=float((phase[30,j]-phase[20,j])/(2*np.pi)),
            late_positive_phase_fraction=float(np.mean(dp[20:] > 0)),
            late_fixed_point_rms=float(np.sqrt(np.mean(error[21:,j]**2))),
            exact_net_turns_0_30=float((ephase[30]-ephase[0])/(2*np.pi)),
            exact_amp30=float(abs(a['exact_z'][30,j])),
            exact_amp200=float(abs(a['exact_z'][200,j]))))
    return summary


COLORS = {'ordinary': '#B74349', 'stable': '#16847A'}


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.titleweight':'semibold',
        'axes.labelcolor':'#343434', 'text.color':'#252525', 'axes.grid':True,
        'grid.alpha':.2, 'grid.linewidth':.6, 'legend.frameon':False,
        'savefig.facecolor':'white', 'pdf.fonttype':42})


def save(fig, name):
    fig.savefig(HERE / (name+'.png'), dpi=180, bbox_inches='tight')
    fig.savefig(HERE / (name+'.pdf'), bbox_inches='tight')
    plt.close(fig)


def figures(all_data):
    style()
    fig, axes = plt.subplots(2, 3, figsize=(14.3, 8.0), sharex=True, constrained_layout=True)
    for r, method in enumerate(['ipo', 'dpo']):
        for arm in ['ordinary', 'stable']:
            a = all_data[f'{method}_{arm}_calibrated_s0']
            color = COLORS[arm]
            values = [abs(a['z']).mean(1), np.sqrt(np.mean((a['x']-a['fixed'])**2,axis=(1,2))),
                      np.sqrt(np.mean(a['actual_step']**2,axis=(1,2)))]
            exact = [abs(a['exact_z'][:31]).mean(1),
                     np.sqrt(np.mean((a['exact'][:31]-a['fixed'])**2,axis=(1,2))),
                     np.sqrt(np.mean(np.diff(a['exact'][:31],axis=0)**2,axis=(1,2)))]
            for c in range(3):
                steps = np.arange(31) if c<2 else np.arange(1,31)
                axes[r,c].plot(steps,values[c],color=color,lw=2)
                axes[r,c].plot(steps,exact[c],color=color,lw=1.5,ls='--',alpha=.65)
                axes[r,c].set_ylim(bottom=0)
                axes[r,c].set_xlim(0,30)
                axes[r,c].set_xlabel('Outer round')
        for c, title in enumerate(['Cyclic-mode amplitude', 'Distance to own fixed point', 'Change between rounds']):
            axes[r,c].set_title(method.upper()+' | '+title, fontsize=11)
            axes[r,c].set_ylabel('Mean amplitude (nats)' if c==0 else 'Centered-logit RMS (nats)')
    handles = [Line2D([],[],color=COLORS['ordinary'],lw=2,label='Ordinary: IPO beta=.2 / DPO beta=.8'),
               Line2D([],[],color=COLORS['stable'],lw=2,label='Stable control: IPO beta=.4 / DPO beta=1.6'),
               Line2D([],[],color='#444444',lw=2,label='Solid: measured LLM'),
               Line2D([],[],color='#444444',lw=1.5,ls='--',label='Dashed: exact population replay')]
    fig.legend(handles=handles,loc='lower center',
               bbox_to_anchor=(.5,-.07),ncol=2,fontsize=10)
    fig.suptitle('Stage A: measured LLM trajectories versus the calibrated population map\n6 selected prompts, one seed; fixed points differ between beta controls',fontsize=14)
    save(fig,'stage_a_overview')

    for method in ['ipo','dpo']:
        fig, axes = plt.subplots(3,6,figsize=(19,9.2),constrained_layout=True)
        for arm in ['ordinary','stable']:
            a=all_data[f'{method}_{arm}_calibrated_s0']
            color=COLORS[arm]
            phase=np.unwrap(np.angle(a['z']),axis=0)
            turns=(phase-phase[0])/(2*np.pi)
            exact_phase=np.unwrap(np.angle(a['exact_z'][:31]),axis=0)
            exact_turns=(exact_phase-exact_phase[0])/(2*np.pi)
            for j,p in enumerate(a['panels']):
                axes[0,j].plot(abs(a['z'][:,j]),color=color,lw=1.8)
                axes[0,j].plot(abs(a['exact_z'][:31,j]),color=color,ls='--',alpha=.5,lw=1.2)
                axes[1,j].plot(turns[:,j],color=color,lw=1.8)
                axes[1,j].plot(exact_turns[:,j],color=color,ls='--',alpha=.5,lw=1.2)
                z=a['z'][:,j]*np.exp(-1j*np.angle(a['z'][0,j]))
                axes[2,j].plot(z.real,z.imag,color=color,lw=1.5)
                axes[2,j].plot(z[0].real,z[0].imag,'o',mfc='white',mec=color,ms=6)
                axes[2,j].plot(z[-1].real,z[-1].imag,'s',color=color,ms=5)
                for t in [5,15,25]:
                    axes[2,j].annotate('',xy=(z[t+1].real,z[t+1].imag),xytext=(z[t].real,z[t].imag),
                                       arrowprops=dict(arrowstyle='->',color=color,lw=1.4))
                if arm=='ordinary':
                    axes[0,j].set_title('Prompt '+str(p['prompt_id']),fontsize=11)
                    for r in [0,1]:
                        axes[r,j].set_xlim(0,30)
                        axes[r,j].set_xlabel('Outer round')
                    axes[0,j].set_ylim(0,3.4)
                    axes[1,j].set_ylim(-.3,2.6)
                    axes[2,j].axhline(0,color='#999999',lw=.6)
                    axes[2,j].axvline(0,color='#999999',lw=.6)
                    axes[2,j].plot(0,0,'x',color='#222222',ms=6)
                    axes[2,j].set_aspect('equal',adjustable='datalim')
                    axes[2,j].set_xlabel('Real mode coordinate')
        axes[0,0].set_ylabel('Amplitude (nats)')
        axes[1,0].set_ylabel('Net phase turns from round 0')
        axes[2,0].set_ylabel('Imaginary mode coordinate')
        fig.suptitle(method.upper()+': every selected prompt, with no favorable-example filtering\nRed = ordinary; teal = stable control; dashed = exact population. Phase portraits align the initial angle; circle = start, square = round 30.',fontsize=13)
        save(fig,method+'_all_prompt_trajectories')

    fig,axes=plt.subplots(1,2,figsize=(12,4.4),constrained_layout=True)
    for ax,method in zip(axes,['ipo','dpo']):
        for arm in ['ordinary','stable']:
            a=all_data[f'{method}_{arm}_calibrated_s0']
            ax.plot(np.arange(1,31),np.sqrt(np.mean(a['epsilon']**2,axis=(1,2))),
                    color=COLORS[arm],lw=2,label=arm+' fit error')
            ax.plot(np.arange(1,31),np.sqrt(np.mean(a['desired_step']**2,axis=(1,2))),
                    color=COLORS[arm],ls='--',lw=1.5,label=arm+' intended update')
        ax.set(xlabel='Outer round',ylabel='Centered-logit RMS (nats)',title=method.upper(),ylim=(0,None))
        ax.legend(fontsize=8.5)
    fig.suptitle('Inner-fit error is not negligible relative to the intended outer update',fontsize=13)
    save(fig,'stage_a_target_fit')


def export_tables(summary):
    flat=[{k:v for k,v in row.items() if k!='per_prompt'} for row in summary]
    prompts=[dict(run=row['run'],beta=row['beta'],**p) for row in summary for p in row['per_prompt']]
    for name,rows in [('run_summary.csv',flat),('per_prompt_summary.csv',prompts)]:
        with (HERE/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def main():
    global HERE, INPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=INPUT)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    INPUT = args.results_dir.resolve()
    HERE = (args.output_dir or args.results_dir).resolve()
    HERE.mkdir(parents=True, exist_ok=True)
    all_data = {r: load(r) for r in RUNS}
    initial = all_data[RUNS[0]]['x'][0]
    for a in all_data.values():
        np.testing.assert_allclose(a['x'][0], initial, rtol=0, atol=.01)
    summary = [summarize(a) for a in all_data.values()]
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    export_tables(summary)
    figures(all_data)
    print(json.dumps([{k:v for k,v in row.items() if k!='per_prompt'} for row in summary], indent=2))


if __name__ == '__main__':
    main()
