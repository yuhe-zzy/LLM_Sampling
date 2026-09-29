# Stage C: completed mixed-orientation controls

## Completion and validation

Read-only Sycamore inspection on 2026-09-29 at 13:44 UTC confirmed both
4606367_0 (IPO mixed) and 4606367_1 (DPO mixed) COMPLETED, exit 0:0.
Elapsed times were 2:17:06 and 2:17:53, one H100 per task. Both contain
101 contiguous states 0..100. There were no OOM/Traceback/nonfinite matches.
The full owner/UID account queue was empty, allocated GPUs zero. No job was
submitted, restarted, cancelled or changed during retrieval/analysis.

The download retained 216 hash-recorded raw/log files outside Git. Together
with the existing B download, analysis verified 864 file hashes and all 808
finite snapshots, reconstructed the metric fields, and checked frozen source
hashes. All runs used training revision f0fe034, not this analysis revision.
Text, response order, initial scores, token counts and training settings match
the corresponding B ordinary control. Preference matrices differ only by the
predeclared orientation vector [-1,+1,-1,+1,-1,+1].

All arms: alpha=.9, lambda_current=.8, seed0, six prompts, four responses each.
IPO beta=.2; DPO beta=.8. Stage C uses ordinary updates, not a history scheme.
Reversed prompts: 54/612/867. Unchanged prompts: 251/737/945.
Every prompt and every recorded state is included without smoothing.

## Actual observations

Descriptive late window: states 51..100, averaged over all six prompts.
TV uses the 50 transitions ending at states 51..100; temporal SD uses the
50 probabilities per response at those states, population SD (ddof=0).

| Method | Orientation | Mean adjacent-state TV | Mean per-response temporal SD | Mean relative entropy |
|---|---|---:|---:|---:|
| IPO | Aligned ordinary | 0.07614 | 0.09653 | 0.83303 |
| IPO | Mixed ordinary | 0.09205 | 0.12907 | 0.76468 |
| DPO | Aligned ordinary | 0.07785 | 0.09999 | 0.82793 |
| DPO | Mixed ordinary | 0.10198 | 0.14870 | 0.77579 |

Mixed orientations did NOT suppress the measured probability motion. Mean
TV increased about 20.9% (IPO) and 31.0% (DPO); mean temporal SD increased
33.7% and 48.7%. Both measures increased in 5/6 prompts for each objective;
prompt 945 decreased and remains in every plot. These are descriptive
within-seed effects, not significance estimates.

Prompts 251 and 737 have unchanged preference matrices yet display different
trajectories in the mixed training run. This is consistent with cross-prompt
effects in a shared model, but it does not identify a gradient-interference
mechanism or exclude optimization/numerical sensitivity. No independent-panel
training control or feature/gradient coherence measurement was performed.

Endpoint entropy alone is misleading: mixed H_rel at state 100 is 0.88523
(IPO) / 0.92931 (DPO), higher than aligned ordinary 0.85669 / 0.78349, while
its late-window mean is lower. All arms retain nonzero probability motion.
Neither entropy nor a single endpoint establishes convergence, text collapse,
quality, a periodic limit cycle, or an asymptotic behavior.

The all-arms overview also retains B reference/feedback. Their previously
reported reduction in cyclic-mode amplitude is not necessarily an equal
reduction in adjacent-state TV. Motion speed, temporal spread, and mode
amplitude are different measurements. Mixed is an orientation contrast,
not a fourth history stabilizer at the same preference matrix.

## Reading the figures

- `*_aligned_vs_mixed_pi`: six same-prompt pairs, aligned left and mixed right.
  Colors denote the same response index across runs, not cycle role or method.
  pi = softmax(s_t) for response sequence sums including EOS, not full response
  space probability, and not mu = .2 uniform + .8 pi.
- `*_mixed_pi_vs_iter`: six prompt panels for the newly completed arm alone.
- `*_relative_entropy`: H(softmax(s_t - s_0)), in nats; initial value log(4).
  Never substituted for raw panel entropy H(pi_t).
- `*_probability_motion`: TV = .5 sum_i |pi_t,i - pi_(t-1),i|. This uses a
  common probability definition without theoretical coordinate dependence.
- `*_common_probability_trajectories`: projection onto two fixed role
  contrasts ((pi_role1-pi_role3)/sqrt(2), (pi_role2-pi_role4)/sqrt(2)). Roles
  are identical across arms; the origin is not an inferred fixed point.
  This is a common probability plane, NOT an eigenmode projection. One of
  the three independent probability dimensions is omitted. Circle=start,
  square=end, pale=0..20, stronger=20..100, arrows show selected actual steps.
- `all_arms_overview`: rows=IPO/DPO, columns=mean TV, relative entropy,
  own-orientation mode amplitude. For the LAST column, each orientation has
  its own population fixed point and normalized left eigenmode, independently
  recomputed and checked against its stored snapshots. Do not interpret these
  as distances from the same point. Primary orientation comparisons use pi/TV.

Complex eigenvector phase is arbitrary across BLAS platforms. Verification
allows one constant unit rotation per prompt, fixed by the initial coordinate;
it does not change amplitude or rotate each iteration separately.

## Reproduction and publication

From the repository root, with the existing private downloads available:

```bash
python experiments/cyclic_history/campaigns/cyclic_history_stage_c100_results_20260929/analyze_stage_c.py \
  --stage-b-dir /path/to/cyclic_history_stage_b100_results_20260929 \
  --stage-c-dir /path/to/cyclic_history_stage_c100_results_20260929 \
  --output-dir /path/to/new-stage-c-plots
```

`fetch_results.py --output-dir NEW_DIRECTORY` performs read-only retrieval,
requires authorized SSH/VPN and SYCAMORE_PASSWORD in the caller environment,
and never writes a password to disk. It does not download model checkpoints.
The output directory must be new or empty; existing downloads are preserved.

Public outputs: PNG/PDF, combined PDF, run/per-prompt summary CSV, numeric
pi_trajectories.csv, summary.json and validation.json. Raw text/support files,
snapshots, logs and full account listings stay local. Public tables include
all eight B/C runs. Full hash/metric reconstruction requires the private
downloads; it is not claimed to work from the public CSV alone.

Five new analysis regression tests passed. The local cyclic suite ran 64
tests: 53 passed, 11 skipped for unavailable Linux Bash/Torch. Training code
was unchanged; these local checks are not a claim of a new full GPU test.

This closes the approved A/probe/B/C execution sequence. The larger two-week
roadmap remains a proposal. No further seeds or experiments were started.
