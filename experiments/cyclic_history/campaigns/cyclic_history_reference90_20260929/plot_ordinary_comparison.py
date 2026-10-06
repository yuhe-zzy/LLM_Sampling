"""Descriptive reference90/ordinary comparisons, explicitly not alpha-matched."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

from plot_results import COLORS, STYLES, PROMPTS, load_run


MATCHED = ('method', 'lambda_current', 'beta_train', 'seed', 'support_seed',
           'iters', 'lr', 'epochs_per_iter', 'batch_size', 'grad_accum',
           'max_length', 'lora_r', 'lora_alpha', 'lora_dropout', 'dtype',
           'coverage', 'pair_law', 'support_probability', 'optimizer_reset',
           'panel_ids', 'pair_mode', 'cycle_probability', 'orientations')


def validate_pair(ordinary, reference):
    configs = [a['manifest']['config'] for a in (ordinary, reference)]
    for cfg, expected in zip(configs, ((.9,0.,0.), (1.,.9,0.))):
        if tuple(cfg[k] for k in ('alpha','nu','kappa')) != expected:
            raise ValueError('Unexpected reference weights')
    for key in MATCHED:
        if configs[0][key] != configs[1][key]:
            raise ValueError('Additional unmatched setting: '+key)
    if ordinary['support'] != reference['support']:
        raise ValueError('Prompt/response supports differ')
    np.testing.assert_array_equal(ordinary['scores'][0], reference['scores'][0])
    for a in (ordinary, reference):
        if a['last'] != 100 or a['manifest']['state'] != 'COMPLETED':
            raise ValueError('Complete states 0..100 are required')
        if a['manifest']['calibration_gate'] != 'PASSED_FRESH_INITIAL_SCORES':
            raise ValueError('Fresh initial-score gate did not pass')


def load_pair(ref_root, stage_root, method):
    pair = []
    for root, run in ((stage_root, f'{method}_ordinary_b100_s0'),
                      (ref_root, f'{method}_reference90_a1_b100_s0')):
        receipt = json.loads((root/'download_receipt.json').read_text())
        a = load_run(root/'raw'/run, receipt)
        cfg = a['manifest']['config']
        for t in range(1,101):
            with np.load(root/'raw'/run/'snapshots'/f'step_{t:04d}.npz', allow_pickle=False) as z:
                expected = ((1-cfg['alpha'])*a['scores'][0]
                            +(cfg['alpha']-cfg['nu'])*a['scores'][t-1]
                            +cfg['nu']*a['scores'][max(t-2,0)])
                np.testing.assert_allclose(z['training_reference'], expected, atol=1e-10, rtol=1e-11)
                np.testing.assert_allclose(z['training_effective_reference'], expected, atol=1e-10, rtol=1e-11)
                np.testing.assert_array_equal(z['training_offset'], np.zeros((6,4)))
        pair.append(a)
    validate_pair(*pair)
    return pair


def draw(pair, method, prompts):
    n = len(prompts)
    fig, axes = plt.subplots(n, 2, figsize=(12, 17 if n==6 else 4.8), squeeze=False)
    fig.subplots_adjust(left=.085, right=.985, top=.865 if n==6 else .60,
                        bottom=.075 if n==6 else .22, wspace=.15, hspace=.30)
    beta = pair[0]['manifest']['config']['beta_train']
    subtitle = 'All six prompts' if n==6 else f'Prompt {PROMPTS[prompts[0]]}'
    fig.suptitle(f'{method.upper()} | Ordinary vs 90% previous reference | {subtitle}\n'
                 f'lambda=.8, beta={beta:g}, seed=0 | Measured fixed-panel pi', fontsize=14, y=.985)
    fig.text(.5, .933 if n==6 else .825,
             'Different alpha (.9 vs 1): descriptive comparison, not a nu-only ablation.',
             ha='center', fontsize=11, color='#9c3228')
    for row, j in enumerate(prompts):
        for col, a in enumerate(pair):
            ax = axes[row,col]
            for k in range(4):
                ax.plot(np.arange(101), a['pi'][:,j,k], color=COLORS[k], ls=STYLES[k], lw=1.45)
            ax.set(xlim=(0,100), ylim=(0,1), xticks=[0,20,40,60,80,100], yticks=[0,.25,.5,.75,1])
            if col==0:
                ax.set_ylabel(f'Prompt {PROMPTS[j]}\nPanel probability pi')
            else:
                ax.tick_params(axis='y', labelleft=False)
            if row==n-1:
                ax.set_xlabel('Outer iteration')
            else:
                ax.tick_params(axis='x', labelbottom=False)
            if row==0:
                ax.set_title(('Ordinary | alpha=.9, nu=0\n'+r'$r_t=0.1s_0+0.9s_t$') if col==0 else
                             ('Reference90 | alpha=1, nu=.9\n'+r'$r_t=0s_0+0.1s_t+0.9s_{t-1}$'),
                             fontsize=12, pad=12)
    fig.legend(handles=[Line2D([],[], color=COLORS[k], ls=STYLES[k], lw=2, label=f'Response {k+1}')
                        for k in range(4)], loc='lower center', ncol=4,
               bbox_to_anchor=(.5,.012), fontsize=11)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference90', type=Path, required=True)
    parser.add_argument('--stage-b', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = {m:load_pair(args.reference90.resolve(), args.stage_b.resolve(), m) for m in ('ipo','dpo')}
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.grid':True,
        'grid.alpha':.18, 'legend.frameon':False, 'pdf.fonttype':42, 'savefig.facecolor':'white'})
    with PdfPages(out/'reference90_vs_ordinary.pdf') as pdf:
        for method, pair in data.items():
            for prompts in (list(range(6)), *([j] for j in range(6))):
                fig = draw(pair, method, prompts)
                suffix = 'all_prompts' if len(prompts)==6 else f'prompt_{PROMPTS[prompts[0]]}'
                fig.savefig(out/f'{method}_reference90_vs_ordinary_{suffix}.png', dpi=180)
                pdf.savefig(fig)
                plt.close(fig)
            print('PLOTTED '+method, flush=True)
    validation = dict(runs=4, numeric_snapshots=404, figures=14, pdf_pages=14,
        all_states=[0,100], prompts=PROMPTS, hashes_verified=True,
        identical_support_and_initial_scores=True, training_references_verified=True,
        ordinary_weights=[.1,.9,0], reference90_weights=[0,.1,.9],
        alpha_matched=False, interpretation='Descriptive only; both alpha and nu change',
        configs={m:[{k:a['manifest']['config'][k] for k in
                    ('run_id','method','alpha','nu','kappa','lambda_current','beta_train','seed')}
                    for a in pair] for m,pair in data.items()})
    (out/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')


if __name__=='__main__':
    main()
