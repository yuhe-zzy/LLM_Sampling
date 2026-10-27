"""Eleven-column measured-policy sheets: ordinary bank, then all ten variants."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

import analyze_results as analysis


BASES = ('center', 'alpha08', 'alpha099', 'coverage05', 'beta15')
FAMILIES = {
    'reference': ('reference_half', 'reference_max'),
    'feedback': ('feedback_half', 'feedback_one'),
}
MATCH_FIELDS = ('method', 'base_id', 'alpha', 'lambda_current', 'beta_train')


def select_grid(data, method, family):
    baselines, variants = [], []
    for b, base in enumerate(BASES, start=1):
        ordinary = data[f'{method}_{base}_ordinary_b100_s0']
        if ordinary['record']['arm'] != 'ordinary':
            raise ValueError('Left bank must contain ordinary runs')
        baselines.append(ordinary)
        for arm in FAMILIES[family]:
            run = data[f'{method}_{base}_{arm}_b100_s0']
            if run['record']['arm'] != arm:
                raise ValueError('Variant arm mismatch')
            if any(run['record'][f] != ordinary['record'][f] for f in MATCH_FIELDS):
                raise ValueError('Variant and ordinary parameters do not match')
            variants.append((b, run))
    if len({run['record']['run_id'] for _, run in variants}) != 10:
        raise ValueError('All ten distinct variants are required')
    for run in baselines + [a for _, a in variants]:
        pi = run['pi']
        if pi.shape != (101, 6, 4) or not np.isfinite(pi).all():
            raise ValueError('Every run requires 101 finite states, six prompts, four responses')
        if np.any(pi < 0) or np.any(pi > 1):
            raise ValueError('Invalid panel probability')
        np.testing.assert_allclose(pi.sum(-1), 1, atol=1e-10, rtol=1e-10)
    return baselines, variants


def curves(ax, pi, prompt, small=False):
    for k in range(4):
        ax.plot(np.arange(101), pi[:, prompt, k], color=analysis.prior.COLORS[k],
                ls=analysis.prior.STYLES[k], lw=.95 if small else 1.25)
    ax.set(xlim=(0, 100), ylim=(0, 1))
    ax.set_xticks([0, 50, 100] if small else [0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 1] if small else [0, .25, .5, .75, 1])
    ax.tick_params(labelsize=7 if small else 10, length=2.5)


def draw_sheet(data, method, family):
    baselines, variants = select_grid(data, method, family)
    fig = plt.figure(figsize=(35, 21))
    grid = fig.add_gridspec(6, 11, width_ratios=[1.35] + [1] * 10,
                            left=.045, right=.993, top=.85, bottom=.09,
                            wspace=.22, hspace=.23)
    fig.suptitle(f'{method.upper()} | {family.capitalize()} | Ordinary / v1 / v2 / ... / v10',
                 fontsize=27, y=.983)
    fig.text(.5, .949,
             'Measured pi on four fixed responses | All six prompts | Outer states 0-100 | Seed 0 | No smoothing',
             ha='center', fontsize=17)
    fig.text(.5, .922,
             'Left column: five matched ordinary baselines B1-B5 per prompt. '
             'Each variant header identifies its baseline; there is no universal ordinary run.',
             ha='center', fontsize=15)

    mappings = []
    for prompt in range(6):
        # Five base settings cannot share one ordinary trajectory. Keep them
        # as small multiples inside the single leftmost ordinary column.
        ordinary_grid = grid[prompt, 0].subgridspec(5, 1, hspace=.15)
        for b, baseline in enumerate(baselines):
            ax = fig.add_subplot(ordinary_grid[b, 0])
            curves(ax, baseline['pi'], prompt, small=True)
            ax.set_ylabel(f'B{b+1}', rotation=0, labelpad=17, fontsize=11, va='center')
            if b < 4:
                ax.tick_params(axis='x', bottom=False, labelbottom=False)
            if prompt == 0 and b == 0:
                ax.set_title('Ordinary\nB1-B5 (matched)', fontsize=15, pad=15)
            if prompt == 5 and b == 4:
                ax.set_xlabel('Outer iteration', fontsize=11)
        bounds = grid[prompt, 0].get_position(fig)
        fig.text(.012, (bounds.y0+bounds.y1)/2, f'Prompt {analysis.PROMPTS[prompt]}',
                 rotation=90, va='center', ha='center', fontsize=17, weight='bold')
        for index, (b, run) in enumerate(variants, start=1):
            ax = fig.add_subplot(grid[prompt, index])
            curves(ax, run['pi'], prompt)
            if index != 1:
                ax.tick_params(axis='y', labelleft=False)
            if prompt == 5:
                ax.set_xlabel('Outer iteration', fontsize=11)
            if prompt == 0:
                r = run['record']
                history = f"nu={r['nu']:g}" if family == 'reference' else f"kappa={r['kappa']:g}"
                ax.set_title(f'v{index} | B{b}\nalpha={r["alpha"]:g}, lambda={r["lambda_current"]:g}'
                             f'\nbeta={r["beta_train"]:g}, {history}', fontsize=12, pad=15)
                mappings.append(dict(method=method, family=family, column=f'v{index}',
                    baseline=f'B{b}', run_id=r['run_id'],
                    ordinary_run_id=baselines[b-1]['record']['run_id'],
                    **{key:r[key] for key in ('alpha', 'lambda_current', 'beta_train', 'nu', 'kappa')}))
    fig.legend(handles=[Line2D([], [], color=analysis.prior.COLORS[k],
                             ls=analysis.prior.STYLES[k], lw=2, label=f'Response {k+1}')
                        for k in range(4)], loc='lower center', ncol=4,
               bbox_to_anchor=(.5, .014), fontsize=16)
    fig.text(.5, .047, 'Every vertical axis is panel probability pi, 0-1; '
             'response identities and colors stay fixed within each prompt.', ha='center', fontsize=13)
    return fig, mappings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Reuse the full raw-data, source, support and training-equation audit.
    _, _, data, _, _ = analysis.collect(args.results.resolve())
    if len(data) != 50:
        raise ValueError('All 50 logical runs are required')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.grid':True, 'grid.alpha':.16,
        'grid.linewidth':.6, 'legend.frameon':False, 'pdf.fonttype':42,
        'savefig.facecolor':'white'})
    mappings = []
    with PdfPages(out/'ordinary_and_all_ten_variants.pdf') as pdf:
        for method in ('ipo', 'dpo'):
            for family in FAMILIES:
                fig, rows = draw_sheet(data, method, family)
                name = f'{method}_{family}_ordinary_v1_to_v10'
                fig.savefig(out/(name+'.png'), dpi=180)
                pdf.savefig(fig)
                plt.close(fig)
                mappings.extend(rows)
                print('PLOTTED '+name, flush=True)
    analysis.prior.write_csv(out/'column_mapping.csv', mappings)
    validation = dict(figures=4, pdf_pages=4, columns_per_figure=11,
        variants_per_figure=10, ordinary_baselines_per_prompt=5,
        prompts=list(analysis.PROMPTS), states=[0,100], all_50_runs_revalidated=True,
        layout='one ordinary column with B1-B5 small multiples, then v1-v10',
        source_training_commits=['419565838535a9fa6c5f12af2b919ce8873db4ab',
                                 'f0fe034cc0d8e3bef35b0f5e02806ee81340da76'])
    (out/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')


if __name__ == '__main__':
    main()
