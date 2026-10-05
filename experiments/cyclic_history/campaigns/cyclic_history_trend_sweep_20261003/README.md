# Cyclic history trend sweep, 2026-10-03

## Authorization and scope

User explicitly approved Reference **10 configurations per objective** and
Feedback **10 configurations per objective**, each with matched ordinary
baselines. IPO and DPO both run. Seed0, the same six calibrated aligned panels,
100 outer updates (states 0..100), one H100 per task, and at most six allocated
GPUs over the full account remain fixed. This is an empirical trend study;
exact theory realization or convergence is not an acceptance gate.

There are 50 logical runs: 20 Reference, 20 Feedback and 10 ordinary controls.
Six already-completed Stage B runs are reused, leaving **44 new tasks**.
No existing job is restarted or overwritten. Prior full-refresh Reference90
results are contextual evidence only; all new sweep arms retain alpha<1.

## Exact grid

| Base | alpha | lambda_current | IPO beta_train | DPO beta_train | Change from center |
|---|---:|---:|---:|---:|---|
| center | .9 | .8 | .2 | .8 | Existing center |
| alpha08 | .8 | .8 | .2 | .8 | Refresh only |
| alpha099 | .99 | .8 | .2 | .8 | Refresh only |
| coverage05 | .9 | .5 | .2 | .8 | Coverage mixture only |
| beta15 | .9 | .8 | .3 | 1.2 | Training beta only, times 1.5 |

Each base and objective has these five arms:

1. Ordinary: nu=0, kappa=0.
2. Reference half: nu=alpha/2, kappa=0.
3. Reference maximal: nu=alpha, kappa=0.
4. Feedback moderate: nu=0, kappa=.5.
5. Feedback stronger: nu=0, kappa=1.

Reference is `(1-alpha)*s0 + (alpha-nu)*st + nu*s_previous`, in sequence-score
coordinates. Its weights are .1/.45/.45 and .1/0/.9 at the center; .2/.4/.4
and .2/0/.8 at alpha=.8; .01/.495/.495 and .01/0/.99 at alpha=.99.
Thus increasing lag at each base keeps the initial anchor unchanged.

Feedback targets `r_t + (1+kappa)*delta_t - kappa*delta_previous`, implemented
via an effective score reference and positive losses. IPO uses the identity
payoff increment; DPO uses the actual weighted BT optimizer increment, not
an entrywise-logit surrogate. This is not the manuscript's signed two-sampler
implementation. No Reference/Feedback combination arm is added.

Coverage is `mu=(1-lambda)*uniform_4 + lambda*pi_panel`, not temporal smoothing.
Beta is the training-loss coefficient, not the manuscript's update-gain beta.
All scores and losses use response sequence sums including EOS.

## Matched controls and reuse

The center ordinary/reference-half/feedback-moderate arms for IPO and DPO
reuse `4605560_0..5`, source f0fe034. Each reuse must have complete 0..100
metrics, finite snapshots and identical mathematical/training configuration
apart from protocol/prediction metadata and output identifiers. No 30-round
Stage A control is treated as a complete 100-round run. The new beta values
.3/1.2 differ from the previously completed 30-round .4/1.6 controls.

`comparison_grid.csv` maps every intervention to its exact shared baseline,
existing run or new array index. Comparing IPO with DPO is not a matched-beta
claim. Compare arms within method/base first, then compare their relative trends.

## Runtime and validation

Only a separately named empirical partial-refresh protocol is added. The
calibrated-v2 spectral gates and empirical full-refresh restrictions retain
their original behavior. In the new protocol, fresh initial Qwen scores,
support hashes, configuration hashes, tokenizer/EOS lengths, finite outputs,
gradient checks and cuDNN-SDPA exclusion remain mandatory. Losses, optimizers,
inner epochs and initialization are unchanged. No fixed-point radius is
preassigned or required to make a run count as a result.

Frozen source, configs and launcher hashes are checked by each task. CPU tests
and all 44 support checks plus six reuse audits must pass before submission.
The launcher refuses an existing intent/receipt or output root. Submission
requires an empty full owner/UID queue and uses array `0-43%6`. Pending jobs
are included in this preflight. No account-wide hard scheduler cap is claimed.

Remote launch: `/work/users/y/u/yuhe32/ipo/diagnostics/history_trend_sweep_20261003`.
Outputs: `/work/users/y/u/yuhe32/ipo_runs/cyclic_history_trend_sweep_20261003`.
Submitted **4659985_0..43%6** at **2026-10-03T07:34:15Z**, with frozen training
commit `419565838535a9fa6c5f12af2b919ce8873db4ab`. Deployment passed all 76 CPU
tests (no skips), all 44 support/config checks and all six completed-run reuse
audits. The full-account preflight was empty. At **07:34:41Z**, all 44 tasks
were PENDING(Resources), zero GPUs allocated and no other owned work. This is
a timestamped snapshot, not completion or a hard account-wide cap. Fresh
GPU initial-score validation is still pending and will run in every task.
The intent, receipt, deployment hashes and startup snapshot are saved alongside
this document. Never duplicate this submission, including while it is pending.

Prior runs suggest roughly 2.3-2.5 GPU-hours each: approximately 100-110 new
GPU-hours, or 17-20 hours at uninterrupted six-way allocation. Queue delays
and changed parameter runtimes can increase this; it is not a completion ETA.

## Planned descriptive analysis

Retain all six prompts and all states 0..100. Plot same-prompt pi trajectories
with shared response colors and fixed 0..1 axes. Report per-prompt and mean
adjacent TV, probability temporal standard deviation, maximum panel probability,
leading-response switches, and separately labeled relative-sequence entropy
`H(softmax(s_t-s0))`. Predefine windows 1..25, 26..50, 51..75, 76..100 and the
late summary 51..100 to see trends without favorable endpoint selection.
Report paired differences versus the matching ordinary baseline and retain
exceptions. No smooth/interpolated observations, fabricated winning rate,
or inference of repeated-token generation collapse from panel concentration.
No follow-up runs or automatic resubmission are authorized by this grid.

## Complete results: 2026-10-04 America/Chicago

Read-only inspection at **2026-10-04 20:19 CDT** (2026-10-05T01:19:08Z)
verified all **44 new tasks COMPLETED, exit 0:0**, each with all states 0..100
and fresh initial-score calibration passed. The full owner/UID account queue
was empty, with zero allocated GPUs. No OOM, Traceback or nonfinite failure was
found. New tasks used 101.416 GPU-hours in total, excluding the reused controls.
No job was submitted, cancelled, requeued or modified during this inspection.

Including the six reused Stage B runs, all **50 logical runs / 5,050 snapshots**
are complete. Download hashes, frozen source hashes, unchanged reuse manifests
and metrics, identical support identities, finite numbers and contiguous steps
were checked. Every recorded pi, raw entropy, relative entropy and adjacent TV
was independently reconstructed. Reference weights, feedback offsets and
sampling mixtures were also checked against all saved update snapshots.

Start at [the complete figure index](results_20261004/index.html). The two
40-page PDF books contain all arms and all prompts; PNGs allow individual
inspection. There are 50 individual six-prompt pi figures, 20 same-prompt
ordinary/history comparisons, ten separately labeled relative-entropy figures
and one descriptive overview. No private prompt/response text, adapters, weights
or raw dumps are published. The numeric NPZ contains only measured pi and
relative entropy. Raw evidence remains in the local results directory.

### Reading the figures

- Each pi curve is `softmax(s_t)` over the same four fixed response sequences;
  it is not token probability, full response-space mass or the sampling mixture.
- Four response colors and line styles are fixed across all arms. Comparisons
  place the same prompt in the same row; all axes span 0..100 and 0..1.
- Relative entropy is `H(softmax(s_t-s_0))` in nats, not `H(softmax(s_t))`.
- Temporal SD is the mean of four response-wise standard deviations over time.
  Adjacent TV is `0.5*sum_i abs(pi_i(t)-pi_i(t-1))`. The first describes spread
  over a window; the second describes movement at each update. They need not
  change in the same direction. Neither alone proves convergence.
- Summary windows were specified before inspection: 1..25, 26..50, 51..75,
  76..100, and 51..100. TV and winner switches include transitions ending at
  every t in the window, including its first step.

### Descriptive observations, states 51..100

Across the 40 intervention/base/objective comparisons, mean temporal SD over
six prompts is lower than the matched ordinary baseline in all 40. This does
not mean every prompt improves: SD decreases in 53/60 prompt comparisons for
Reference half, 52/60 for Reference max, 45/60 for Feedback .5, and 54/60 for
Feedback 1. All individual exceptions are retained in the plots and CSVs.

Maximal lag (nu=alpha) increases mean adjacent TV in all ten base/objective
comparisons; only 2/60 individual prompt TV comparisons decrease. Thus a
smaller broad excursion can coexist with stronger short-period jitter.
At the center, Reference max lowers temporal SD by 28.6% (IPO) / 28.3% (DPO),
but increases adjacent TV by 34.2% / 46.5%.

The strongest descriptive Feedback contrast is at alpha=.99. With kappa=1,
mean temporal SD is lower by 73.3% (IPO) / 80.9% (DPO), and adjacent TV is
lower by 59.3% / 64.8%, relative to each matched ordinary baseline. Other
base settings have smaller or mixed TV changes. These are single-seed,
fixed-panel LLM observations, not theoretical trajectories or full-generation
collapse diagnoses. The full grid, not just this favorable contrast, is shown.

`run_summary.csv` and `paired_summary.csv` contain six-prompt means for every
window; `prompt_summary.csv` and `paired_prompt_summary.csv` retain each prompt.
The analysis tests check constant and alternating policies, window boundaries,
and exact base/method pairing. All four tests pass. Both PDF books were rendered
for visual inspection. Training sources remain frozen at 4195658 (new) and
f0fe034 (reused); adding analysis and results does not change those sources.
