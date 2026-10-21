"""Validate and plot empirical reference90 results without population projections."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')
STYLES = ('-', '--', '-.', ':')
PROMPTS = [54, 251, 612, 737, 867, 945]
SOURCE = '2e73527d3508d0da85b90f86654d21d61ea40adc'


def softmax(scores):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite scores')
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def entropy(probabilities):
    return -(probabilities * np.log(np.maximum(probabilities, np.finfo(float).tiny))).sum(-1)


def load_run(root, receipt):
    def checked(name):
        path = root / name
        key = f'raw/{root.name}/{name}'
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['files_sha256'][key]:
            raise ValueError(f'Hash mismatch: {key}')
        return path

    manifest = json.loads(checked('manifest.json').read_text())
    support = json.loads(checked('support.json').read_text(encoding='utf-8'))
    with checked('metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    last = manifest['last_complete_step']
    if [int(row['step']) for row in rows] != list(range(last + 1)):
        raise ValueError('Metrics are not contiguous through manifest complete step')
    if [panel['prompt_id'] for panel in support] != PROMPTS:
        raise ValueError('Unexpected panel order')
    scores, snapshots = [], []
    for step in range(last + 1):
        with np.load(checked(f'snapshots/step_{step:04d}.npz'), allow_pickle=False) as data:
            snap = {k: data[k].copy() for k in data.files}
        for name, value in snap.items():
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise ValueError(f'Nonfinite {root.name}/{step}/{name}')
        scores.append(snap['sequence_sum_logprob'])
        snapshots.append(snap)
    scores = np.stack(scores)
    if scores.shape != (last + 1, 6, 4):
        raise ValueError('Unexpected score shape')
    pi = softmax(scores)
    relative = entropy(softmax(scores - scores[0]))
    raw_entropy = entropy(pi)
    tv = np.concatenate([np.zeros((1, 6)), .5 * np.abs(np.diff(pi, axis=0)).sum(-1)])
    for t, (snap, row) in enumerate(zip(snapshots, rows)):
        for name, expected in [('panel_probability', pi[t]), ('panel_entropy', raw_entropy[t]),
                               ('relative_sequence_entropy', relative[t]), ('tv', tv[t])]:
            np.testing.assert_allclose(snap[name], expected, atol=1e-11, rtol=1e-11)
        for name, expected in [('panel_entropy_mean', raw_entropy[t].mean()),
                               ('relative_sequence_entropy_mean', relative[t].mean()),
                               ('tv_mean', tv[t].mean())]:
            np.testing.assert_allclose(float(row[name]), expected, atol=1e-11, rtol=1e-11)
        np.testing.assert_array_equal(snap['response_token_count'], snapshots[0]['response_token_count'])
        if t and manifest['config']['alpha'] == 1:
            expected_ref = .1 * scores[t-1] + .9 * scores[max(t-2, 0)]
            np.testing.assert_allclose(snap['training_reference'], expected_ref, atol=1e-11, rtol=1e-11)
            np.testing.assert_allclose(snap['training_effective_reference'], expected_ref, atol=1e-11, rtol=1e-11)
            np.testing.assert_array_equal(snap['training_offset'], np.zeros((6, 4)))
    return dict(manifest=manifest, support=support, scores=scores, pi=pi,
                relative=relative, raw_entropy=raw_entropy, tv=tv, last=last)


def summaries(data, method, arm):
    window = slice(51, 101)
    if data['last'] < 100:
        raise ValueError('Fixed 51..100 descriptive window requires complete data')
    result = []
    for j, prompt in enumerate(PROMPTS):
        q = data['pi'][:, j]
        winners = q[window].argmax(-1)
        result.append(dict(method=method, arm=arm, prompt_id=prompt, last_step=data['last'],
            tv_mean_51_100=float(data['tv'][window, j].mean()),
            probability_temporal_sd_51_100=float(q[window].std(axis=0).mean()),
            mean_max_probability_51_100=float(q[window].max(-1).mean()),
            fraction_max_probability_ge_095_51_100=float((q[window].max(-1) >= .95).mean()),
            leading_response_switches_51_100=int(np.count_nonzero(np.diff(winners))),
            relative_entropy_mean_51_100=float(data['relative'][window, j].mean()),
            relative_entropy_100=float(data['relative'][-1, j]),
            raw_panel_entropy_100=float(data['raw_entropy'][-1, j]),
            final_leading_response=int(q[-1].argmax())+1,
            final_max_probability=float(q[-1].max())))
    return result


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage-b', type=Path)
    args = parser.parse_args()
    out = args.output.absolute()
    out.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((args.results/'download_receipt.json').read_text())
    status = json.loads((args.results/'server_status.json').read_text())
    deployment = json.loads((Path(__file__).parent/'deployment.json').read_text())
    assert status['receipt']['source_git_commit'] == SOURCE
    assert deployment['source_git_commit'] == SOURCE
    data, rows = {}, []
    for method, beta in [('ipo', .2), ('dpo', .8)]:
        a = load_run(args.results/'raw'/f'{method}_reference90_a1_b100_s0', receipt)
        cfg = a['manifest']['config']
        for key, value in dict(alpha=1., nu=.9, kappa=0., lambda_current=.8,
                               beta_train=beta, seed=0, iters=100, method=method).items():
            assert cfg[key] == value, (key, cfg[key], value)
        assert a['manifest']['calibration_gate'] == 'PASSED_FRESH_INITIAL_SCORES'
        assert a['manifest']['population_prediction'] == 'NOT_COMPUTED_FULL_REFRESH_EMPIRICAL'
        assert a['manifest']['state'] == 'COMPLETED' and a['last'] == 100
        for name, digest in a['manifest']['source_sha256'].items():
            assert deployment['sha256'][f'source/experiments/cyclic_history/{name}'] == digest
        assert a['support'] == data.get('ipo', a)['support']
        np.testing.assert_array_equal(a['scores'][0], data.get('ipo', a)['scores'][0])
        data[method] = a
        rows.extend(summaries(a, method, 'reference90_alpha1'))
    if args.stage_b:
        old_receipt = json.loads((args.stage_b/'download_receipt.json').read_text())
        for method in ('ipo', 'dpo'):
            for arm in ('ordinary', 'reference', 'feedback'):
                old = load_run(args.stage_b/'raw'/f'{method}_{arm}_b100_s0', old_receipt)
                assert old['support'] == data[method]['support']
                np.testing.assert_array_equal(old['scores'][0], data[method]['scores'][0])
                rows.extend(summaries(old, method, f'stage_b_{arm}_alpha09'))
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10.5,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.titleweight':'semibold',
        'axes.grid':True, 'grid.alpha':.18, 'grid.linewidth':.6, 'legend.frameon':False,
        'pdf.fonttype':42, 'savefig.facecolor':'white'})
    with PdfPages(out/'reference90_all_figures.pdf') as pdf:
        for method, a in data.items():
            fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), layout='constrained')
            fig.get_layout_engine().set(rect=(0, .073, 1, .89))
            for j, ax in enumerate(axes.flat):
                for k in range(4):
                    ax.plot(np.arange(a['last']+1), a['pi'][:, j, k], color=COLORS[k], ls=STYLES[k], lw=1.8)
                ax.set(title=f'Prompt {PROMPTS[j]}', xlabel='Outer iteration',
                       ylabel='Panel policy probability', xlim=(0,100), ylim=(0,1))
                ax.set_xticks(np.arange(0,101,20))
                ax.set_yticks(np.linspace(0,1,6))
            fig.suptitle(f'{method.upper()} | 90% previous reference | Measured policy probabilities\n'
                f"alpha=1, nu=0.9, lambda=0.8, beta={a['manifest']['config']['beta_train']:g}, seed=0\n"
                r'$r_t=0s_0+0.1s_t+0.9s_{t-1}$ | Sequence-sum, normalized within four responses', fontsize=13)
            fig.legend(handles=[Line2D([],[],color=COLORS[k],ls=STYLES[k],lw=2,label=f'Response {k+1}')
                                 for k in range(4)], loc='lower center', bbox_to_anchor=(.5,.015), ncol=4)
            fig.savefig(out/f'{method}_reference90_pi_vs_iter.png', dpi=180)
            fig.savefig(out/f'{method}_reference90_pi_vs_iter.pdf')
            pdf.savefig(fig)
            plt.close(fig)
        fig, axes = plt.subplots(2,3,figsize=(14,7.7),layout='constrained')
        fig.get_layout_engine().set(rect=(0,.07,1,.91))
        for j, ax in enumerate(axes.flat):
            for method, color in [('ipo',COLORS[0]),('dpo',COLORS[1])]:
                ax.plot(np.arange(101),data[method]['relative'][:,j],color=color,lw=1.8,label=method.upper())
            ax.set(title=f'Prompt {PROMPTS[j]}',xlabel='Outer iteration',ylabel='Relative entropy (nats)',
                   xlim=(0,100),ylim=(0,1.42))
        fig.suptitle('90% previous reference | Relative-sequence entropy\n'
                     r'$H(\mathrm{softmax}(s_t-s_0))$; not raw panel entropy',fontsize=13)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.01),ncol=2)
        fig.savefig(out/'reference90_relative_entropy.png',dpi=180)
        fig.savefig(out/'reference90_relative_entropy.pdf')
        pdf.savefig(fig)
        plt.close(fig)
    points = [dict(method=method, prompt_id=prompt, outer_iteration=t, response_index=k+1,
                   pi_panel=float(a['pi'][t,j,k]), relative_sequence_entropy=float(a['relative'][t,j]))
              for method,a in data.items() for t in range(a['last']+1)
              for j,prompt in enumerate(PROMPTS) for k in range(4)]
    write_csv(out/'pi_trajectories.csv', points)
    write_csv(out/'per_prompt_summary.csv', rows)
    aggregate = []
    fields = [key for key in rows[0] if key not in ('method','arm','prompt_id','last_step','final_leading_response')]
    for method,arm in sorted({(r['method'],r['arm']) for r in rows}):
        selected = [r for r in rows if r['method']==method and r['arm']==arm]
        aggregate.append(dict(method=method,arm=arm,**{key:float(np.mean([r[key] for r in selected])) for key in fields}))
    write_csv(out/'run_summary.csv',aggregate)
    validation = dict(source_commit=SOURCE, job_id='4608669', numeric_snapshots=202,
        all_states=list(range(101)), prompt_ids=PROMPTS, reference_weights=[0,.1,.9],
        all_finite=True, hashes_verified=True, softmax_and_both_entropies_verified=True,
        manifest_source_hashes_match_frozen_deployment=True,
        training_reference_verified=True, identical_initial_scores_and_support=True,
        stage_b_context_loaded=bool(args.stage_b), interpolation=False, population_mode_projection=False,
        source_files_sha256={k:v for k,v in receipt['files_sha256'].items() if k.startswith('raw/')})
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(aggregate,indent=2))


if __name__ == '__main__':
    main()
