"""Plot measured finite-panel policy probabilities, without mode projection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
OUT = HERE / 'pi_curves'
METHODS = ('ipo', 'dpo')
ARMS = ('ordinary', 'reference', 'feedback')
COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')
LINESTYLES = ('-', '--', '-.', ':')


def load_raw():
    receipt = json.loads((HERE / 'download_receipt.json').read_text())
    all_data = {}
    panels_common = None
    for method in METHODS:
        for arm in ARMS:
            run = f'{method}_{arm}_b100_s0'
            root = HERE / 'raw' / run
            names = ['manifest.json', 'support.json'] + [f'snapshots/step_{s:04d}.npz' for s in range(101)]
            for name in names:
                key = f'raw/{run}/{name}'
                assert hashlib.sha256((root / name).read_bytes()).hexdigest() == receipt['files_sha256'][key]
            manifest = json.loads((root / 'manifest.json').read_text())
            assert manifest['state'] == 'COMPLETED' and manifest['last_complete_step'] == 100
            cfg = manifest['config']
            panels = json.loads((root / 'support.json').read_text(encoding='utf-8'))
            if panels_common is None:
                panels_common = panels
            assert panels == panels_common
            pi, scores = [], []
            for step in range(101):
                with np.load(root / 'snapshots' / f'step_{step:04d}.npz', allow_pickle=False) as z:
                    s = z['sequence_sum_logprob']
                    q = z['panel_probability']
                    weights = np.exp(s - s.max(-1, keepdims=True))
                    expected = weights / weights.sum(-1, keepdims=True)
                    np.testing.assert_allclose(q, expected, atol=1e-12, rtol=1e-12)
                    np.testing.assert_allclose(q.sum(-1), 1.0, atol=1e-12, rtol=0)
                    assert q.shape == (6, 4) and np.isfinite(q).all()
                    assert np.all((q >= 0) & (q <= 1))
                    pi.append(q.copy())
                    scores.append(s.copy())
            all_data[(method, arm)] = dict(pi=np.stack(pi), scores=np.stack(scores), cfg=cfg)
    return all_data, panels_common


def load_table(path):
    """Replot the public numeric export without requiring private panel text."""
    ids = [54, 251, 612, 737, 867, 945]
    all_data = {(method, arm): dict(pi=np.full((101, 6, 4), np.nan),
                                  scores=np.full((101, 6, 4), np.nan),
                                  cfg={'beta_train': .2 if method == 'ipo' else .8})
                for method in METHODS for arm in ARMS}
    seen = set()
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = row['method'], row['arm']
            t, p, k = int(row['outer_iteration']), int(row['prompt_id']), int(row['response_index'])
            if key not in all_data or p not in ids or not 0 <= t <= 100 or not 1 <= k <= 4:
                raise ValueError('Unexpected method/arm/prompt/state/response')
            index = (*key, t, p, k)
            if index in seen:
                raise ValueError('Duplicate probability record')
            seen.add(index)
            all_data[key]['pi'][t, ids.index(p), k-1] = float(row['pi_panel'])
            all_data[key]['scores'][t, ids.index(p), k-1] = float(row['sequence_sum_logprob'])
    if len(seen) != 6*101*6*4:
        raise ValueError('The public plot requires all six complete 0..100 trajectories')
    return all_data, [{'prompt_id': p} for p in ids]


def validate_probabilities(all_data):
    for a in all_data.values():
        q, s = a['pi'], a['scores']
        if not np.isfinite(q).all() or not np.isfinite(s).all() or not np.all((q >= 0) & (q <= 1)):
            raise ValueError('Non-finite or out-of-range probability data')
        weights = np.exp(s - s.max(-1, keepdims=True))
        np.testing.assert_allclose(q, weights/weights.sum(-1, keepdims=True), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(q.sum(-1), 1, atol=1e-12, rtol=0)
    initial = all_data[('ipo', 'ordinary')]['pi'][0]
    for a in all_data.values():
        np.testing.assert_allclose(a['pi'][0], initial, atol=1e-12, rtol=0)


def main():
    global HERE, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=HERE)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--table', type=Path, help='Public pi_trajectories.csv; no raw data needed')
    parser.add_argument('--include-text', action='store_true', help='Export private text mapping for local use only')
    args = parser.parse_args()
    if args.table and args.include_text:
        parser.error('The public table does not contain response text')
    HERE = args.results_dir.resolve()
    OUT = (args.output_dir or HERE/'pi_curves').resolve()
    all_data, panels_common = load_table(args.table) if args.table else load_raw()
    validate_probabilities(all_data)
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'semibold',
        'axes.grid': True, 'grid.alpha': .18, 'grid.linewidth': .6,
        'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    exported = []
    with PdfPages(OUT / 'stage_b_all_pi_curves.pdf') as combined:
        for (method, arm), a in all_data.items():
            fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), layout='constrained')
            fig.get_layout_engine().set(rect=(0, .073, 1, .89))
            for j, ax in enumerate(axes.flat):
                for k in range(4):
                    ax.plot(np.arange(101), a['pi'][:, j, k], color=COLORS[k],
                            ls=LINESTYLES[k], lw=1.8)
                ax.set(title=f"Prompt {panels_common[j]['prompt_id']}",
                       xlabel='Outer iteration', ylabel='Panel policy probability',
                       xlim=(0, 100), ylim=(0, 1))
                ax.set_xticks(np.arange(0, 101, 20))
                ax.set_yticks(np.linspace(0, 1, 6))
            intervention = dict(ordinary='No history correction',
                                reference='Lagged reference: nu=0.45',
                                feedback='Feedback extrapolation: kappa=0.5')[arm]
            fig.suptitle(f'{method.upper()} | {arm.capitalize()} | Policy probabilities over iterations\n'
                         f"alpha=0.9, lambda=0.8, beta={a['cfg']['beta_train']:g}, seed=0 | {intervention}\n"
                         'Sequence-sum probabilities normalized over the same four candidate responses',
                         fontsize=13)
            fig.legend(handles=[Line2D([], [], color=COLORS[k], ls=LINESTYLES[k], lw=2,
                                      label=f'Response {k+1}') for k in range(4)],
                       loc='lower center', bbox_to_anchor=(.5, .015), ncol=4, fontsize=11)
            name = f'{method}_{arm}_pi_vs_iter'
            fig.savefig(OUT / f'{name}.png', dpi=180)
            fig.savefig(OUT / f'{name}.pdf')
            combined.savefig(fig)
            plt.close(fig)
            exported.append(name)

    with (OUT / 'pi_trajectories.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'arm', 'prompt_id', 'outer_iteration',
                                              'response_index', 'pi_panel', 'sequence_sum_logprob'])
        writer.writeheader()
        for (method, arm), a in all_data.items():
            for t in range(101):
                for j, p in enumerate(panels_common):
                    for k in range(4):
                        writer.writerow(dict(method=method, arm=arm, prompt_id=p['prompt_id'],
                            outer_iteration=t, response_index=k+1,
                            pi_panel=float(a['pi'][t,j,k]), sequence_sum_logprob=float(a['scores'][t,j,k])))
    if args.include_text:
        mapping = [{'prompt_id': p['prompt_id'], 'prompt': p['prompt'],
                    'responses': {str(k+1): text for k,text in enumerate(p['responses'])}}
                   for p in panels_common]
        (OUT / 'response_mapping.json').write_text(json.dumps(mapping, indent=2, ensure_ascii=False)+'\n',
                                                   encoding='utf-8')
    report = ['# Stage B: pi versus outer iteration', '',
              'Each of the six figures is one method/arm, with all six prompts and four response curves.',
              'The color and line style of each response index are fixed across all arms and methods.', '',
              'pi_panel(i,t) = exp(s_i(t)) / sum_j exp(s_j(t)), where s is the response sequence-sum',
              'log-probability including EOS. This is the policy CONDITIONAL on the four-response panel.',
              'It is not the full response-space probability, the relative-to-initial distribution,',
              'or the sampling mixture mu = 0.2*uniform + 0.8*pi_panel.', '',
              'All states 0..100 are measured LLM outputs. There is no smoothing, interpolation,',
              'mode projection or theoretical rollout in these figures. Every panel sums to one.', '',
              'The combined PDF has six pages; matching per-arm PNG/PDF files are also provided.',
              'Response indices refer to the original fixed support order, not cyclic roles.',
              'Private prompt/response text is not included in the public export.', '']
    (OUT / 'README.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    validation = dict(runs=6, prompts_per_run=6, states_per_run=101, responses_per_prompt=4,
                      probability_points=6*6*101*4, all_finite_and_in_0_1=True,
                      unit_sums_verified=True, softmax_reconstruction_passed=True,
                      identical_initial_pi=True, raw_support_and_hashes_verified=not bool(args.table),
                      input_mode='public_numeric_csv' if args.table else 'private_raw_snapshots',
                      figures=exported)
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
