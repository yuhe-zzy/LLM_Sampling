# Oracle2 policy trajectories: 2026-10-13

## Scope and availability

All six training arms completed measured outer states 0..100. These are
actual Qwen2.5-1.5B sequence-sum scores on the same 500 training prompts,
four fixed real responses per prompt, seed0. Execution source remains
`23401de58b27ef2f7b1ea645551f37602a3c1bb8`; the plotting commit is not the
training source. No training, generation or scoring job was submitted,
cancelled or modified for this analysis.

There is **no generated-response winning-rate curve yet**. The frozen
`wr_results/wr_summary.csv` did not exist at export, and no WR scoring job
had been submitted. Partial generated response banks are not WR scores.
At2026-10-13T06:15:34Z, sacct confirmed generate4794738_0/1 TIMEOUT after
12:00:26. Each preserved complete states0..70 and partial state80 with728/561
rows respectively. Their on-disk RUNNING manifests are stale after scheduler
termination. Tasks2/3 started with24-hour limits,4/5 remained pending,
full account2H100s. No OOM/Traceback/nonfinite was found. No recovery was
submitted and no complete or partial bank was overwritten. This generation
failure does not invalidate the six complete training/policy trajectories.
See `winning_rate_status.json` and
[scheduler evidence](../records/attempt4/generate_timeout_and_policy_results_20261013.json).

## What each plot means

- `ipo_policy_vs_step` and `dpo_policy_vs_step`: rows are the same six prompts;
  columns are ordinary, reference nu=.45, feedback kappa=.5. Candidate colors
  and line styles never change across arms. The x-axis is outer step, not
  optimizer step. The y-axis is pi_panel(i,t)=exp(s_i(t))/sum_j exp(s_j(t)),
  conditional on the four-response panel. Scores include response EOS, not
  prompt or padding tokens. This is not the full response-space probability,
  the sampling mixture, token-average likelihood, or a winning rate.
- `ipo_relative_weights` / `dpo_relative_weights`: explicitly different
  diagnostic softmax(s_t-s_0). These are NOT raw policy trajectories and may
  move when raw panel probabilities are extremely concentrated.
- `all_prompt_stability`: left is mean adjacent-step total variation,
  .5*sum_i|pi_panel(i,t)-pi_panel(i,t-1)|. Right is relative-sequence entropy
  H(softmax(s_t-s_0)) in nats. All 500 prompts receive equal weight.
  Low relative entropy alone does not establish temporal stability.
- `all_prompt_change_heatmap`: all 500 prompts in identical fixed group/ID
  order across all arms, no filtering; color is adjacent-step TV on 0..1.

No smoothing, synthetic rollouts, interpolation, or fixed-point fitting is
used. All plots have PNG and PDF counterparts.

## Selection and interpretation

The overview rows were chosen by metadata, not observed trajectory:
the sole cyclic prompt416; the three smallest transitive IDs4,5,7;
the two smallest ambiguous IDs6,11. This is an illustrative fixed subset,
not a representative estimate; the aggregate plots and per-prompt CSV cover
all500 prompts. Training groups are cyclic1/transitive436/ambiguous63.
Ambiguous is not evidence of non-ST. Evaluation contains no cyclic prompts;
a future cyclic WR subgroup must remain missing, never zero.

The cyclic prompt416 has candidate token counts222,167,122,2 (including EOS).
Initial sequence-sum log-probabilities are approximately
[-233.605,-246.849,-187.380,-26.873]. The fourth candidate consequently already
has panel probability indistinguishable from1 at step0. Its raw policy stays
visually flat in all arms. This does not show history-induced stabilization
or global model convergence; score/length imbalance limits this case study.
No model, mixture, temperature, truncation, support or threshold was changed.

## Descriptive findings

Single seed, no significance or convergence claims. Over all500 prompts:

| Objective/arm | Mean TV steps1..20 | Mean TV steps81..100 | Temporal SD states80..100 |
| --- | ---: | ---: | ---: |
| IPO ordinary | .023943 | .013426 | .012888 |
| IPO reference | .022510 | .015258 | .010657 |
| IPO feedback | .025243 | .013065 | .012521 |
| DPO ordinary | .046819 | .024145 | .023742 |
| DPO reference | .042088 | .026925 | .018290 |
| DPO feedback | .047950 | .025022 | .026441 |

Temporal SD is computed per candidate across the specified states, then
averaged over four candidates and prompts (population SD, ddof=0).
All arms have smaller late than early mean TV, but this is not a stopping
criterion or proof of convergence. Reference has smaller late temporal SD
than ordinary, yet larger late adjacent-step TV in both objectives. Thus
smaller excursions and smaller step-to-step jitter are different conclusions.
Feedback has no uniform improvement over ordinary across objectives.
Prompt5 illustrates reduced excursions with reference; prompt7 under DPO
is a counterexample to blanket stabilization. One seed cannot establish a
reliable causal or generalization claim.

## Files and reproduction

`selected_trajectories.csv` contains all measured scores and probabilities for
the six displayed prompts. `all_prompt_probabilities.npz` contains all
1,212,000 panel probabilities, keyed by run; each array has shape101x500x4.
The prompt axis is `panel_index` in `prompt_metadata.json`, not prompt ID.
`per_prompt_summary.csv` retains every prompt, including counterexamples;
`stability_summary.csv` includes all/group aggregates. `validation.json`
records export hash, execution source, selection, and passed reconstruction
checks. No private prompt/response text or weights are published.

Read-only remote export (a new destination, never inside a frozen source):

```sh
python experiments/oracle2/export_policy_results.py --root LAUNCH_ROOT --destination NEW_EXPORT.zip
```

Local figure rebuild from the text-free numeric export:

```sh
python experiments/oracle2/plot_policy_results.py --archive NEW_EXPORT.zip --output NEW_FIGURES
python -m unittest discover -s experiments/oracle2 -p test_policy_results.py -v
```

The full21MB numeric export remains local in
`oracle2_results_20261013/policy_numeric_export.zip`; the smaller8.8MB
all-prompt probability archive is published for inspection. The export recorded
all606 original snapshot hashes and verified finite fields, all probability normalizations,
softmax reconstruction, both entropy definitions, TV, and matched initials.
An absent WR summary creates only a missing-status record, never a zero curve.
