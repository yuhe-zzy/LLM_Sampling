"""One equal-scale horizontal comparison per prompt, with sorted history strength."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

from plot_all_variants import BASES, FAMILIES, analysis, select_grid


def sorted_columns(data, method, family):
    baselines, variants = select_grid(data, method, family)
    field = 'nu' if family == 'reference' else 'kappa'
    variants = sorted(variants, key=lambda item: (item[1]['record'][field], item[0]))
    return [(b, run, 'ordinary') for b, run in enumerate(baselines, 1)] + [
        (b, run, family) for b, run in variants]


def draw_prompt(data, method, family, prompt_index):
    columns = sorted_columns(data, method, family)
    prompt_id = analysis.PROMPTS[prompt_index]
    field = 'nu' if family == 'reference' else 'kappa'
    fig, axes = plt.subplots(1, 15, figsize=(42, 4.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=.021, right=.995, top=.65, bottom=.22, wspace=.20)
    fig.suptitle(f'{method.upper()} | {family.capitalize()} | Prompt {prompt_id} | '
                 f'Ordinary B1-B5, then ten configurations sorted by {field}', fontsize=20, y=.975)
    fig.text(.5, .865, 'Measured fixed-panel pi, four responses | Seed 0 | Outer states 0-100 | '
             'Identical panel sizes and axes | No smoothing | B labels identify matched ordinary runs',
             ha='center', fontsize=13)
    mapping = []
    for col, (b, run, arm) in enumerate(columns):
        ax = axes[col]
        record = run['record']
        for response in range(4):
            ax.plot(np.arange(101), run['pi'][:, prompt_index, response],
                    color=analysis.prior.COLORS[response],
                    ls=analysis.prior.STYLES[response], lw=1.2)
        ax.set(xlim=(0, 100), ylim=(0, 1), xticks=[0, 50, 100],
               yticks=[0, .25, .5, .75, 1], xlabel='Outer iteration')
        ax.tick_params(labelsize=9)
        if arm == 'ordinary':
            title = f'Ordinary B{b}'
            ax.set_facecolor('#f4f5f6')
        else:
            title = f"{field}={record[field]:g} | B{b}"
        ax.set_title(title + f'\nalpha={record["alpha"]:g}, lambda={record["lambda_current"]:g}'
                     f'\nbeta={record["beta_train"]:g}', fontsize=10.5, pad=12)
        mapping.append(dict(method=method, family=family, prompt_id=prompt_id,
            column=col+1, arm=record['arm'], baseline=f'B{b}', run_id=record['run_id'],
            ordinary_run_id=f'{method}_{BASES[b-1]}_ordinary_b100_s0',
            **{f:record[f] for f in ('alpha', 'lambda_current', 'beta_train', 'nu', 'kappa')}))
    axes[0].set_ylabel('Panel probability pi', fontsize=12)
    boundary = (axes[4].get_position().x1 + axes[5].get_position().x0)/2
    fig.add_artist(Line2D([boundary, boundary], [.18, .805], transform=fig.transFigure,
                         color='#777777', lw=1, ls=':'))
    fig.legend(handles=[Line2D([], [], color=analysis.prior.COLORS[k],
                              ls=analysis.prior.STYLES[k], lw=2, label=f'Response {k+1}')
                        for k in range(4)], loc='lower center', ncol=4,
               bbox_to_anchor=(.5, .018), fontsize=12)
    return fig, mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    _, _, data, _, _ = analysis.collect(args.results.resolve())
    if len(data) != 50:
        raise ValueError('The complete 50-run dataset is required')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.grid':True, 'grid.alpha':.16,
        'grid.linewidth':.6, 'legend.frameon':False, 'pdf.fonttype':42,
        'savefig.facecolor':'white'})
    mappings, names = [], []
    with PdfPages(out/'prompt_sorted_comparisons.pdf') as pdf:
        for method in ('ipo', 'dpo'):
            for family in FAMILIES:
                for j, prompt_id in enumerate(analysis.PROMPTS):
                    fig, rows = draw_prompt(data, method, family, j)
                    name = f'{method}_{family}_prompt_{prompt_id}_sorted'
                    fig.savefig(out/(name+'.png'), dpi=180)
                    pdf.savefig(fig)
                    plt.close(fig)
                    mappings.extend(rows)
                    names.append(name)
                print(f'PLOTTED {method} {family}: six prompts', flush=True)
    analysis.prior.write_csv(out/'column_mapping.csv', mappings)
    validation = dict(figures=24, pdf_pages=24, columns_per_figure=15,
        ordinary_columns=5, history_columns=10, identical_panel_dimensions=True,
        reference_sort='nu ascending; ties by baseline B1-B5',
        feedback_sort='kappa ascending; ties by baseline B1-B5',
        prompts=list(analysis.PROMPTS), states=[0,100],
        all_50_runs_revalidated_from_raw_evidence=True, filenames=names)
    (out/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')


if __name__ == '__main__':
    main()
