# Original cyclic pi trajectories: September 29, 2026

Read-only reanalysis of the nine original `logs_{ipo,dpo}_nonoracle/cyclic*_dataeval`
runs. No training was submitted, stopped or changed. These are not the later
sampling sweep and not the calibrated Stage A/B/C or reference90 campaigns.

## What is plotted

Horizontal axis: original outer iteration, pre-update snapshots. Vertical axis:
the legacy fixed-panel distribution `pi_avg = softmax(avg_response_logprob)`
(tau=1), with four colored responses. This is not sequence-sum pi, relative
sequence pi, unrestricted generation probability, or a winning rate. Plotting
the original data does not retroactively change the historical training loss.
Straight segments connect recorded iterations; no smoothing or extrapolation.

Columns of the main figures: `(alpha,lambda,beta)` = `(.1,.1,.01)`, `(.5,.5,5)`,
`(.99,.8,1)`. Rows: prompts **814,439,543**, identical across all plots.
IPO main runs have states 0..99, DPO main runs 0..149. The additional plots
include both `(0,0,.01)` runs, recorded only through 49, and IPO `(.99,.8,10)`
through 99. Grey areas explicitly mark unrecorded iterations.

## Selection and validation

Selection is fixed by mean adjacent total variation in the IPO and DPO
`(.99,.8,1)` baselines over the shared window 0..99: average the two method
scores per prompt, sort (with prompt ID breaking ties), take ranks nearest
the 25th, 50th and 90th percentiles. All 500 prompts were eligible. Do not
describe these three as a random sample or an unbiased population estimate.

All 950 dump files, comprising **475,000 prompt snapshots**, were read on
the server. Every prompt/response ordered support hash was constant within
each run and matched across all nine runs. All stored probabilities and
average log probabilities were finite. Saved probabilities exactly matched
softmax(avg) in this export. This does not establish absence of repetition
in open generation, nor successful convergence. This batch is distinct from
the later sampling-sweep DPO baseline with NaNs after iteration 81.

Numeric archive SHA256:
`1d0a5b95301264402dea37c5fd3b5d30e6c6a5b85a5480afb4da0eb4b9abd639`.
The full numeric archive stays local, not in Git. Public outputs contain only
figures, numeric selected trajectories and hashes, no raw prompt/response text.

## Observations, not causal or convergence claims

- In these three prompts, `(.5,.5,5)` yields much smaller probability motion
  than the other main settings. Because all three parameters change between
  columns, the contrast cannot isolate the effect of alpha, lambda or beta.
- IPO `(.99,.8,1)` quickly concentrates for 814 and 439; 543 has pronounced
  later switching. Concentration and ongoing redistribution coexist across
  prompts; neither is a proof of a periodic orbit or open-generation collapse.
- DPO `(.99,.8,1)` shows substantial late motion for the selected prompts,
  particularly 439 and 543. Do not infer a universal IPO-vs-DPO ordering from
  a one-seed historical implementation with differing training budgets.
- The separate IPO beta comparison holds alpha=.99 and lambda=.8 fixed in
  the labels and uses identical supports. For prompt 543, beta=10 has lower
  mean adjacent TV (.06875 versus .14747) and fewer top-1 switches (7 vs 16)
  over 0..99. For 814 and 439, beta=10 instead moves more than the near-frozen
  concentrated beta=1 trajectories. Thus the effect is not universal. A
  strict causal beta-only interpretation additionally needs full historical
  optimizer/source provenance alignment, beyond the support audit here.

## Files and reproduction

- `figures/ipo_main_pi.png`, `figures/dpo_main_pi.png`: matched 3x3 main grids.
- `figures/*_additional_pi.png`: remaining historical parameters.
- `figures/ipo_beta_comparison_pi.png`: matched IPO beta=1/10 panels.
- `figures/all_original_cyclic_pi.pdf`: all five pages; individual PDFs also saved.
- `figures/selected_prompt_summary.csv`: actual-horizon metrics for 27 panels.
- `figures/selected_pi_trajectories.csv`: numeric plotted data; invalid values
  would be blank, never replaced with uniform distributions.
- `figures/selection_and_validation.json`: selection, provenance and file hashes.

`export_original.py --root <ipo_runs> --out <new.zip>` is a CPU-only exporter
with exclusive creation and immutable support checks. `plot_original.py
--snapshot <new.zip> --out <new_directory>` recreates the figures. Four local
regression tests cover selection, support mismatch, finite filtering and
omitting invalid fallback rows. All five final figure layouts were visually
checked (the unchanged main-grid layout was checked on the first render).
