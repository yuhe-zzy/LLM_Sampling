# Experiment results atlas

Publication review: 2026-10-13; retained Qwen3 WR added2026-10-21.
This is a navigation and provenance index,
not a live scheduler dashboard. Read each batch's own timestamp and metric
definitions before comparing results. Current code does not retroactively
become the code used by historical jobs.

Read [current project context](../doc/INDEX.md) and
[publication coverage](RESULT_PUBLICATION_20261013.md) for known gaps and
exclusions. The [artifact inventory](result_artifacts_20261013.json) lists
public PNG/PDF/CSV paths, sizes and SHA256 hashes. Historical previews remain
separate from preferred consolidated results.
The October21 Qwen3 package has its own publication manifest; it is not part
of the historical October13 inventory.

## Start by scientific question

| Question / batch | Evidence entry point | Scope and caution |
|---|---|---|
| Original cyclic parameter trajectories | [Legacy cyclic analysis](../experiments/legacy_cyclic_analysis/README.md) | Nine historical runs; legacy token-average panel probabilities; selected prompts and selection rule disclosed |
| Uniform exploration on fixed cyclic pairs | [Historical sampling ablation](../experiments/legacy_sampling_analysis/snapshot_20260911/README.md) | Ten seed0 runs; legacy token-average; DPO rho0 invalid from82, never interpreted as stability |
| Does population calibration transfer to the LLM? | [Stage A report](../experiments/cyclic_history/campaigns/cyclic_history_stage_a_analysis_20260928/REPORT.md) | Four arms, outer 0..30; target-fit discrepancies retained |
| Does longer inner training fit the same target? | [Fixed-target probes](../experiments/cyclic_history/campaigns/cyclic_fixed_target_launch_20260928/README.md) | Four probes, 60 inner epochs; no new outer updates |
| Ordinary vs reference vs feedback | [Stage B report](../experiments/cyclic_history/campaigns/cyclic_history_stage_b100_results_20260929/REPORT.md) | Six complete runs, outer 0..100; all six prompts |
| Mixed cycle orientations across prompts | [Stage C report](../experiments/cyclic_history/campaigns/cyclic_history_stage_c100_results_20260929/REPORT.md) | Two complete controls, compared with reused Stage B ordinary runs |
| Aggressive lag: reference weights (0,.1,.9) | [Reference90 results](../experiments/cyclic_history/campaigns/cyclic_history_reference90_20260929/RESULTS_2026-10-03.md) | Two complete runs; alpha and nu both differ from Stage B reference |
| Ten reference and ten feedback settings per objective | [Trend sweep](../experiments/cyclic_history/campaigns/cyclic_history_trend_sweep_20261003/README.md) | 50 logical runs = 44 new + 6 reused; five distinct matched ordinary bases |
| Initial anchor: reference weights (.9,.1,0) | [Anchor90 launch record](../experiments/cyclic_history/campaigns/cyclic_history_anchor90_20261006/README.md) | October9 preflight recorded job completion; dedicated validated trajectory export is not yet published; launch-time pending label is historical |
| Oracle1: sequence-sum entropy and generated-response WR | [September 11 snapshot](../experiments/oracle1/campaigns/sequencesum_snapshot_20260911/README.md) | 12 complete, 1 partial, 3 absent grid cells in that snapshot; not a final all-16 result |
| NEW Oracle1/Qwen3-4B: IPO ordinary generated-response WR | [Completed0..100 report](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/ipo_only_wr_results_20261021/README.md) | One retained arm/seed,11 checkpoints,200 eval prompts,9600 scored answers including shared baseline;47.569% to47.536%, no clear gain; prompt SE only, cap256, other five arms withdrawn |
| Oracle1: earlier intermediate exports | [Historical archive](../experiments/oracle1/campaigns/historical_local_snapshots/README.md) | Dated local previews, not new results or independent replications; entropy definitions kept separate |
| Collaborator Hodge diagnostics | [Hodge entry point](../experiments/hodge_diagnostics/README.md) | Preserve collaborator definitions and distinction between plans and execution |
| Oracle2: mixed reward-model judge | [Policy results](../experiments/oracle2/campaigns/oracle2_real_20261009/policy_results_20261013/README.md) | Six complete states0..100 arms, all500-prompt metrics; no generated-response WR yet. [Execution/recovery](../experiments/oracle2/campaigns/oracle2_real_20261009/README.md) and [earlier design review](ORACLE2_DESIGN_REVIEW_20261006.md) retained |
| Oracle2: average Hodge residual across prompts | [All500-panel averages](../experiments/oracle2/campaigns/oracle2_real_20261009/average_hodge_20261013/README.md) | Mean residual vs mean norm, identity/logit and averaging order distinguished; original candidate indices, not a validated shared semantic space |

## Useful figure collections

- [NEW Oracle1/Qwen3 retained IPO ordinary WR PDF](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/ipo_only_wr_results_20261021/oracle1_wr.pdf) and [numeric summary](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/ipo_only_wr_results_20261021/wr_summary.csv).
- [Stage B: six pi plots in one PDF](../experiments/cyclic_history/campaigns/cyclic_history_stage_b100_results_20260929/pi_curves/stage_b_all_pi_curves.pdf).
- [Stage C: all comparisons](../experiments/cyclic_history/campaigns/cyclic_history_stage_c100_results_20260929/all_stage_c_figures.pdf).
- [Trend sweep: IPO figure book](../experiments/cyclic_history/campaigns/cyclic_history_trend_sweep_20261003/results_20261004/ipo_complete_figures.pdf)
  and [DPO figure book](../experiments/cyclic_history/campaigns/cyclic_history_trend_sweep_20261003/results_20261004/dpo_complete_figures.pdf).
- [Trend sweep: prompt-sorted comparisons](../experiments/cyclic_history/campaigns/cyclic_history_trend_sweep_20261003/prompt_sorted_20261006/).
  These show five matched ordinary panels plus ten intervention panels, not
  one universal ordinary baseline. Repeated nu/kappa can have different alpha,
  lambda or beta.
- [Oracle1 snapshot gallery](../experiments/oracle1/campaigns/sequencesum_snapshot_20260911/README.md#figures).
- [Oracle2 policy/relative-weight/stability figures](../experiments/oracle2/campaigns/oracle2_real_20261009/policy_results_20261013/README.md).
- [Historical sampling figures and all-prompt tables](../experiments/legacy_sampling_analysis/snapshot_20260911/README.md).

## How to read the numeric evidence

For the trend sweep, `results_20261004/run_summary.csv` and
`paired_summary.csv` contain group summaries; `prompt_summary.csv` and
`paired_prompt_summary.csv` retain all six prompts and exceptions. The
campaign's `experiment_plan.json` maps all 50 logical configurations to 44
new tasks and six reused runs. Do not count reused runs as new independent
replications. Other batches keep their own plans and manifests alongside
their analysis code.

October13 adds compact numeric `metrics_original/<run>.csv` alongside existing
A/probe/B/C/Reference90/trend packages. The probe coordinate is block, not outer
step. These are original numeric exports, with row counts and hashes in the
[import manifest](result_import_manifest_20261013.json); no response text is included.

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
- The separate legacy sampling-sweep analysis is now published with the DPO
  on-policy NaN failure after iteration81 clearly marked. Its invalid uniform
  fallback is not stable training. This is not the nine-run original cyclic batch.
- Anchor90 still needs a dedicated validated trajectory export; Oracle2 WR
  still awaits completion of generation, recovery and scoring.
- No new non-oracle transitive result package was validated in this review;
  availability of runnable code does not establish published result coverage.
- This update does not claim all raw intermediate data are publicly released.
