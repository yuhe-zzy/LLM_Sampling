# Historical non-oracle reproduction

The two runners here are retained as evidence-compatible sources, not the
recommended defaults for new sequence-sum experiments:

- scripts/legacy/run_ipo.py: average-token IPO, dynamic inner reference.
- scripts/legacy/run_dpo.py: sequence-sum DPO training, dynamic inner reference;
  sampling and displayed panel probabilities still use average-token scores.

Both accept fixed transitive pairs or fixed cyclic pairs. Data, optimizer,
model version, precision, seeds, support selection and run horizon are part
of the protocol. The root scripts/run_ipo.py and scripts/run_dpo.py instead
select the corrected common core.

## Historical cyclic grid

Use configs/cyclic_legacy.json and the dry-run launcher. It specifies ten runs:

| Method | alpha | lambda_pair | beta_train | pairs/round | seed |
|---|---:|---|---:|---:|---:|
| IPO | .99 | 1, .75, .5, .25, 0 | 10 | 1000 | 0 |
| DPO | .99 | 1, .75, .5, .25, 0 | 1 | 500 | 0 |

Other settings: tau=1, mix_eps=0, clipping [0,1e6], two pair proposals per
training prompt, 500 diagnostic prompts, four fixed dataset responses,
150 pre-update snapshots 0..149. The corresponding training loop also updates
after the last saved pre-update snapshot. This is not the new core's
evaluation-only final-state convention.

The existence of lambda_pair=0 in this historical cyclic sampling ablation
does not imply any lambda_base=0 scalar-oracle experiment. Nor does it add a
lambda_current=0 arm to the new history pilot.

## Known invalid interval

Archived DPO lambda_pair=1 first reports a nonfinite loss at round 81; all
500 panels have nonfinite raw scores at snapshots 82..149. The old softmax
helper falls back to uniform probabilities. Mask these snapshots instead of
interpreting them as stability or uniform-policy convergence.

The exploratory DPO summary window was 30..81, selected after identifying
the numerical failure; IPO summaries used 30..149. Disclose these different
windows and report valid counts. A 150-state plot with a flat fallback tail
is misleading even if Slurm reported COMPLETED.

Example prompt IDs were selected near baseline TV percentiles 50/90/99:
IPO 94/333/882; DPO 89/367/947. Use the same prompts and response ordering
across each method's sampling settings. These are descriptive examples,
not independently replicated training runs.

For a new causal history comparison, use all three newly matched arms in
experiments/cyclic_history/. Comparing one of them with the legacy baseline
would confound reference freezing, precision, dropout, accumulation, pair
budget, likelihood reduction and sampling law.
