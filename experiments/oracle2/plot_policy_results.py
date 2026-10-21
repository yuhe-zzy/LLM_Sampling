"""Plot measured Oracle2 finite-panel policy; never substitute panel scores for WR."""
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
from matplotlib.lines import Line2D
import numpy as np

METHODS = ('ipo', 'dpo')
ARMS = ('ordinary', 'reference', 'feedback')
COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')
STYLES = ('-', '--', '-.', ':')
ARM_COLORS = ('#555555', '#0072B2', '#D55E00')
GROUPS = ('cyclic', 'transitive', 'ambiguous')


def softmax(x):
    w = np.exp(x - x.max(axis=-1, keepdims=True))
    return w / w.sum(axis=-1, keepdims=True)


def entropy(p):
    return -(p * np.log(np.maximum(p, np.finfo(float).tiny))).sum(axis=-1)


def validate(data):
    initial = None
    for run, a in data.items():
        s, p = a['sequence_sum_logprob'], a['panel_probability']
        if s.shape != (101, 500, 4) or p.shape != s.shape:
            raise ValueError(f'Incomplete grid in {run}')
        if not all(np.isfinite(v).all() for v in a.values()):
            raise ValueError('Nonfinite numeric export')
        if np.any((p < 0) | (p > 1)):
            raise ValueError('Probability outside [0, 1]')
        np.testing.assert_array_equal(a['steps'], np.arange(101))
        np.testing.assert_allclose(p, softmax(s), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(p.sum(-1), 1, atol=1e-12, rtol=0)
        np.testing.assert_allclose(a['panel_entropy'], entropy(p), atol=1e-12)
        np.testing.assert_allclose(a['relative_sequence_entropy'],
                                   entropy(softmax(s - s[0])), atol=1e-12)
        tv = np.concatenate((np.zeros((1, 500)), .5 * np.abs(np.diff(p, axis=0)).sum(-1)))
        np.testing.assert_allclose(a['tv'], tv, atol=1e-12)
        if initial is None:
            initial = s[0]
        else:
            np.testing.assert_array_equal(initial, s[0])


def choose_prompts(metadata):
    """Choose by frozen metadata only, never trajectory size or visual desirability."""
    selected = []
    for group, count in (('cyclic', 1), ('transitive', 3), ('ambiguous', 2)):
        available = sorted((p for p in metadata if p['group'] == group),
                           key=lambda p: (int(p['prompt_id']), p['prompt_key']))
        selected.extend(p['panel_index'] for p in available[:count])
    return selected


def read_export(path):
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read('export_manifest.json'))
        for name, expected in manifest['files_sha256'].items():
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise ValueError(f'Export hash mismatch: {name}')
        metadata = json.loads(z.read('prompt_metadata.json'))
        data = {}
        for method in METHODS:
            for arm in ARMS:
                run = f'{method}_{arm}'
                with np.load(io.BytesIO(z.read(f'{run}/policy_numeric.npz')), allow_pickle=False) as n:
                    data[run] = {k: n[k].copy() for k in n.files}
        wr = list(csv.DictReader(io.StringIO(z.read('wr_summary.csv').decode()))) if 'wr_summary.csv' in z.namelist() else []
    validate(data)
    assert [p['panel_index'] for p in metadata] == list(range(500))
    return data, metadata, manifest, wr


def save(fig, output, name):
    fig.savefig(output / f'{name}.png', dpi=160, facecolor='white')
    fig.savefig(output / f'{name}.pdf', facecolor='white')
    plt.close(fig)


def policy_figure(data, metadata, selected, method, output, relative=False):
    fig, axes = plt.subplots(len(selected), 3, figsize=(14, 2.05 * len(selected) + 1.5),
                             sharex=True, sharey=True)
    for row, idx in enumerate(selected):
        panel = metadata[idx]
        for col, arm in enumerate(ARMS):
            a = data[f'{method}_{arm}']
            probabilities = (softmax(a['sequence_sum_logprob'] - a['sequence_sum_logprob'][0])
                             if relative else a['panel_probability'])
            ax = axes[row, col]
            for k in range(4):
                ax.plot(a['steps'], probabilities[:, idx, k], color=COLORS[k],
                        ls=STYLES[k], lw=1.3)
            if row == 0:
                ax.set_title(('Ordinary', 'Reference: nu=0.45', 'Feedback: kappa=0.5')[col], fontsize=12)
            if col == 0:
                ax.set_ylabel(f"Prompt {panel['prompt_id']}\n{panel['group']}\n" +
                              ('Relative weight' if relative else 'Panel policy probability'), fontsize=9)
            ax.set(xlim=(0, 100), ylim=(-.02, 1.02), xticks=[0, 20, 40, 60, 80, 100], yticks=[0, .5, 1])
            if row == len(selected) - 1:
                ax.set_xlabel('Outer step (measured states 0..100)')
    metric = 'Relative-to-initial weights (NOT raw policy)' if relative else 'Policy over four fixed candidate responses'
    fig.suptitle(f'{method.upper()} | Oracle2 | {metric}\n'
                 f"alpha=0.9, lambda=0.8, beta={.2 if method == 'ipo' else .8}, seed=0",
                 fontsize=14, y=.988)
    handles = [Line2D([], [], color=COLORS[k], ls=STYLES[k], label=f'Candidate {k+1}', lw=2) for k in range(4)]
    fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(.5, .012))
    fig.subplots_adjust(left=.11, right=.985, bottom=.075, top=.92, hspace=.20, wspace=.10)
    save(fig, output, f'{method}_' + ('relative_weights' if relative else 'policy_vs_step'))


def overview(data, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for row, method in enumerate(METHODS):
        for color, arm in zip(ARM_COLORS, ARMS):
            a = data[f'{method}_{arm}']
            axes[row, 0].plot(a['steps'][1:], a['tv'][1:].mean(1), color=color, label=arm)
            axes[row, 1].plot(a['steps'], a['relative_sequence_entropy'].mean(1), color=color)
        axes[row, 0].set_ylabel(f'{method.upper()} | mean adjacent-step TV')
        axes[row, 1].set_ylabel(f'{method.upper()} | relative entropy (nats)')
        axes[row, 0].set_ylim(0, max(a['tv'][1:].mean(1).max() for a in data.values()) * 1.08)
        axes[row, 1].set_ylim(0, 1.43)
        for ax in axes[row]:
            ax.set_xlim(0, 100)
            ax.set_xlabel('Outer step')
    axes[0, 0].legend(ncol=3, fontsize=9)
    fig.suptitle('All 500 training prompts | equal prompt weights | no smoothing\n'
                 'Left: change in raw panel policy; right: entropy of softmax(s_t - s_0)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .93))
    save(fig, output, 'all_prompt_stability')


def heatmap(data, metadata, output):
    order = sorted(range(500), key=lambda i: (GROUPS.index(metadata[i]['group']), int(metadata[i]['prompt_id'])))
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharex=True, sharey=True, layout='constrained')
    for row, method in enumerate(METHODS):
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            im = ax.imshow(data[f'{method}_{arm}']['tv'][1:, order].T, aspect='auto', origin='upper',
                           cmap='magma', vmin=0, vmax=1, extent=(.5, 100.5, 499.5, -.5), interpolation='nearest')
            ax.set_title(f'{method.upper()} | {arm}')
            ax.set_xlabel('Outer step')
            if col == 0:
                ax.set_ylabel('Prompt index in fixed group/ID order')
            for boundary in (1, 437):
                ax.axhline(boundary-.5, color='#009E73', lw=.75)
            ax.grid(False)
    fig.colorbar(im, ax=axes, label='Adjacent-step total variation (0..1)', shrink=.8)
    fig.suptitle('All 500 prompts, no filtering | cyclic: 1; transitive: 436; ambiguous: 63\n'
                 'Same prompt order and scale in every panel. Ambiguous is not a non-ST label.', fontsize=13)
    save(fig, output, 'all_prompt_change_heatmap')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(data, metadata, output):
    per_prompt, summary = [], []
    for run, a in data.items():
        p = a['panel_probability']
        early, late = a['tv'][1:21].mean(0), a['tv'][81:101].mean(0)
        spread = p[80:101].std(0).mean(-1)
        for i, panel in enumerate(metadata):
            per_prompt.append(dict(run_id=run, prompt_id=panel['prompt_id'], group=panel['group'],
                early_TV_mean=float(early[i]), late_TV_mean=float(late[i]),
                late_probability_temporal_SD=float(spread[i]),
                final_max_panel_probability=float(p[-1, i].max()),
                final_relative_entropy=float(a['relative_sequence_entropy'][-1, i])))
        for group in ('all',) + GROUPS:
            subset = np.array([i for i, m in enumerate(metadata) if group == 'all' or m['group'] == group])
            summary.append(dict(run_id=run, group=group, prompts=len(subset),
                early_TV_mean=float(early[subset].mean()), late_TV_mean=float(late[subset].mean()),
                late_probability_temporal_SD=float(spread[subset].mean()),
                final_mean_max_panel_probability=float(p[-1, subset].max(-1).mean()),
                final_relative_entropy=float(a['relative_sequence_entropy'][-1, subset].mean())))
    write_csv(output / 'per_prompt_summary.csv', per_prompt)
    write_csv(output / 'stability_summary.csv', summary)
    return summary


def plot_wr(rows, output):
    if not rows:
        (output / 'winning_rate_status.json').write_text(json.dumps(dict(
            state='NOT_AVAILABLE', plotted=False,
            reason='Open-generation Oracle2 scoring has not produced wr_summary.csv; no surrogate or zero filled.'), indent=2))
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, method in zip(axes, METHODS):
        for color, arm in zip(ARM_COLORS, ARMS):
            subset = sorted((r for r in rows if r['run_id'] == f'{method}_{arm}' and r['group'] == 'all'),
                            key=lambda r: int(r['step']))
            assert [int(r['step']) for r in subset] == list(range(0, 101, 10))
            assert all(int(r['prompt_count']) == 200 for r in subset)
            values = np.array([float(r['oracle2_expected_win_rate']) for r in subset])
            assert np.isfinite(values).all() and np.all((values >= 0) & (values <= 1))
            ax.plot(range(0, 101, 10), values, marker='o', color=color, label=arm)
        ax.set(title=method.upper(), xlabel='Outer step', ylim=(0, 1), xlim=(0, 100))
        ax.axhline(.5, color='#999999', ls=':', lw=1)
    axes[0].set_ylabel('Oracle2 expected win rate vs initial-policy generations')
    axes[0].legend()
    fig.tight_layout()
    save(fig, output, 'generated_winning_rate')
    write_csv(output / 'wr_summary.csv', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data, metadata, manifest, wr = read_export(args.archive)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': .18, 'legend.frameon': False, 'pdf.fonttype': 42})
    selected = choose_prompts(metadata)
    for method in METHODS:
        policy_figure(data, metadata, selected, method, args.output)
        policy_figure(data, metadata, selected, method, args.output, relative=True)
    overview(data, args.output)
    heatmap(data, metadata, args.output)
    summary = summarize(data, metadata, args.output)
    plot_wr(wr, args.output)
    trajectory = []
    for run, a in data.items():
        for step in range(101):
            for idx in selected:
                for k in range(4):
                    trajectory.append(dict(run_id=run, step=step, prompt_id=metadata[idx]['prompt_id'],
                        group=metadata[idx]['group'], candidate=k+1,
                        panel_probability=float(a['panel_probability'][step, idx, k]),
                        sequence_sum_logprob=float(a['sequence_sum_logprob'][step, idx, k])))
    write_csv(args.output / 'selected_trajectories.csv', trajectory)
    record = dict(execution_source_commit=manifest['execution_source_commit'],
        archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        runs=6, states=101, prompts=500, candidates=4, probability_points=1212000,
        all_export_hashes_verified=True, softmax_and_TV_reconstructed=True,
        raw_and_relative_entropy_verified=True, same_initial_scores=True,
        selection_rule='lowest prompt IDs within 1 cyclic, 3 transitive, 2 ambiguous; no trajectory-based selection',
        selected_prompts=[metadata[i] for i in selected], wr_available=bool(wr))
    (args.output / 'validation.json').write_text(json.dumps(record, indent=2))
    # Compact, text-free probabilities cover every prompt, not just the displayed examples.
    np.savez_compressed(args.output / 'all_prompt_probabilities.npz',
                        **{run: a['panel_probability'] for run, a in data.items()})
    (args.output / 'prompt_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(dict(selected=[metadata[i]['prompt_id'] for i in selected],
                          wr_available=bool(wr), summary=summary), indent=2))


if __name__ == '__main__':
    main()
