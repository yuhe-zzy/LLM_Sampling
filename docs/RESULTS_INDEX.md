# Experiment results atlas

Publication review: 2026-10-06. This is a navigation and provenance index,
not a live scheduler dashboard. Read each batch's own timestamp and metric
definitions before comparing results. Current code does not retroactively
become the code used by historical jobs.

## Start by scientific question

| Question / batch | Evidence entry point | Scope and caution |
|---|---|---|
| Original cyclic parameter trajectories | [Legacy cyclic analysis](../experiments/legacy_cyclic_analysis/README.md) | Nine historical runs; legacy token-average panel probabilities; selected prompts and selection rule disclosed |
| Does population calibration transfer to the LLM? | [Stage A report](../experiments/cyclic_history/campaigns/cyclic_history_stage_a_analysis_20260928/REPORT.md) | Four arms, outer 0..30; target-fit discrepancies retained |
| Does longer inner training fit the same target? | [Fixed-target probes](../experiments/cyclic_history/campaigns/cyclic_fixed_target_launch_20260928/README.md) | Four probes, 60 inner epochs; no new outer updates |
| Ordinary vs reference vs feedback | [Stage B report](../experiments/cyclic_history/campaigns/cyclic_history_stage_b100_results_20260929/REPORT.md) | Six complete runs, outer 0..100; all six prompts |
| Mixed cycle orientations across prompts | [Stage C report](../experiments/cyclic_history/campaigns/cyclic_history_stage_c100_results_20260929/REPORT.md) | Two complete controls, compared with reused Stage B ordinary runs |
| Aggressive lag: reference weights (0,.1,.9) | [Reference90 results](../experiments/cyclic_history/campaigns/cyclic_history_reference90_20260929/RESULTS_2026-10-03.md) | Two complete runs; alpha and nu both differ from Stage B reference |
| Ten reference and ten feedback settings per objective | [Trend sweep](../experiments/cyclic_history/campaigns/cyclic_history_trend_sweep_20261003/README.md) | 50 logical runs = 44 new + 6 reused; five distinct matched ordinary bases |
| Initial anchor: reference weights (.9,.1,0) | [Anchor90 launch record](../experiments/cyclic_history/campaigns/cyclic_history_anchor90_20261006/README.md) | Two submitted runs; last recorded check was pending, not completion; no fresh server check in this publication |
| Oracle1: sequence-sum entropy and generated-response WR | [September 11 snapshot](../experiments/oracle1/campaigns/sequencesum_snapshot_20260911/README.md) | 12 complete, 1 partial, 3 absent grid cells in that snapshot; not a final all-16 result |
| Collaborator Hodge diagnostics | [Hodge entry point](../experiments/hodge_diagnostics/README.md) | Preserve collaborator definitions and distinction between plans and execution |
| Oracle2: mixed reward-model judge | [Design review](ORACLE2_DESIGN_REVIEW_20261006.md) | Proposed only; no results, model deployment or training submitted |

## Useful figure collections

- [Stage B: six pi plots in one PDF](../experiments/cyclic_history/campaigns/cyclic_history_stage_b100_results_20260929/pi_curves/stage_b_all_pi_curves.pdf).
- [Stage C: all comparisons](../experiments/cyclic_history/campaigns/cyclic_history_stage_c100_results_20260929/all_stage_c_figures.pdf).
- [Trend sweep: IPO figure book](../experiments/cyclic_history/campaigns/cyclic_history_trend_sweep_20261003/results_20261004/ipo_complete_figures.pdf)
  and [DPO figure book](../experiments/cyclic_history/campaigns/cyclic_history_trend_sweep_20261003/results_20261004/dpo_complete_figures.pdf).
- [Trend sweep: prompt-sorted comparisons](../experiments/cyclic_history/campaigns/cyclic_history_trend_sweep_20261003/prompt_sorted_20261006/).
  These show five matched ordinary panels plus ten intervention panels, not
  one universal ordinary baseline. Repeated nu/kappa can have different alpha,
  lambda or beta.
- [Oracle1 snapshot gallery](../experiments/oracle1/campaigns/sequencesum_snapshot_20260911/README.md#figures).

## How to read the numeric evidence

For the trend sweep, `results_20261004/run_summary.csv` and
`paired_summary.csv` contain group summaries; `prompt_summary.csv` and
`paired_prompt_summary.csv` retain all six prompts and exceptions. The
campaign's `experiment_plan.json` maps all 50 logical configurations to 44
new tasks and six reused runs. Do not count reused runs as new independent
replications. Other batches keep their own plans and manifests alongside
their analysis code.

Panel pi, relative-sequence entropy, generation WR and legacy token-average
metrics are separate quantities. No current synthetic fixed-panel figure is
an unrestricted-generation WR measurement. Lower temporal SD can coexist
with larger adjacent-step TV. Neither is by itself a convergence proof.

## Publication structure and policy

Keep one authoritative directory per campaign. Existing campaign paths and
immutable deployed training sources stay unchanged. New campaign packages
should contain these logical components (old layouts need not be moved):

```text
experiments/<family>/campaigns/<campaign_id>/
  README.md                 question, findings, limits, dated coverage
  experiment_plan.json      configurations, matched controls and reuse
  provenance.json           actual training commit/hashes, data IDs, jobs
  metrics/                  compact anonymous numeric evidence
  figures/                  preview PNGs and vector PDFs
  analysis/                 plotting code and focused tests
  publication_manifest.json file sizes and SHA256 checksums
```

The atlas links these packages instead of duplicating their results at the
root. Within each batch, show complete, partial, missing and failed runs
explicitly. Keep all prompts/counterexamples or document a selection rule.
Use timestamped snapshots for incomplete batches; do not overwrite them
with later data or present historical queue labels as current status.

Publish reviewed code/configs/tests, sanitized provenance, small numeric
intermediates, figures and conclusions. Exclude credentials, private raw
prompt/response text, model weights/adapters/checkpoints, caches, raw logs and
large dumps/archives. Do not loosen `.gitignore` globally. For withheld data,
provide dataset IDs/revisions, acquisition instructions and support hashes;
state which analyses require the private local artifacts. Unknown historical
training provenance must remain unknown, not be filled with current HEAD.

## Remaining publication gaps

- Oracle1 here is the latest locally identified consolidated snapshot, dated
  September 11, not a fresh server retrieval or certification of final state.
  A later final archive must reconcile missing/partial runs and their actual
  retry provenance before publication.
- The separate legacy sampling-sweep analysis is not yet published as a
  reviewed package. Its local report records a DPO on-policy NaN failure after
  iteration 81; subsequent uniform fallback values must not be called stable
  training. This is not the nine-run original cyclic batch above.
- No new non-oracle transitive result package was validated in this review;
  availability of runnable code does not establish published result coverage.
- This update does not claim all raw intermediate data are publicly released.
