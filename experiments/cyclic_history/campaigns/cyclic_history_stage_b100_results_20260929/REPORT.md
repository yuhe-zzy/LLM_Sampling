# Stage B: completed six-arm LLM comparison

## Execution and provenance

Array **4605560**, tasks 0..5, finished successfully (COMPLETED, exit 0:0).
All six runs contain the complete outer states **0..100**, with 101 metric rows
and 101 numeric snapshots per run. Actual wall times were 2:17:05 to 2:19:56.
The read-only account check on September 29, 2026 UTC found no owned running or
pending jobs and zero allocated GPUs. No OOM/Traceback/numerical errors were
found. No experiment was submitted, canceled, requeued, or modified.

Raw inputs are in `raw/`; the 12 original Slurm logs are in `slurm_logs/`.
`download_receipt.json` contains hashes for 648 downloaded files.
`server_status.json` preserves full queue/accounting and per-run status.
`validation.json` records checks of all 606 finite snapshots, source hashes,
identical support/initial scores, reconstructed metrics and shared coordinates.

Common parameters: alpha=0.9, lambda_current=0.8, seed=0, four responses for
each of the same six prompts (54, 251, 612, 737, 867, 945), sequence-sum scoring.
IPO beta=0.2; DPO beta=0.8. Each method has ordinary, lagged reference (nu=0.45)
and full-preference feedback extrapolation (kappa=0.5). This feedback arm is
not the manuscript's signed two-sampler implementation. The inner budget,
initial model and training support are matched within each method.

## Main observations

The table uses an explicitly descriptive **second-half window, states 51..100**.
Numbers are averaged over all six prompts and all 50 states, without selecting
favorable prompts. Percentages are relative to the same method's ordinary arm.

| Method | Arm | Mean cyclic amplitude, 51..100 | Change vs ordinary | Relative entropy at 100 |
|---|---|---:|---:|---:|
| IPO | Ordinary | 0.5588 | -- | 0.8567 |
| IPO | Reference | 0.4162 | -25.5% | 0.9110 |
| IPO | Feedback | 0.4470 | -20.0% | 0.8271 |
| DPO | Ordinary | 0.5746 | -- | 0.7835 |
| DPO | Reference | 0.4433 | -22.8% | 0.8481 |
| DPO | Feedback | 0.4659 | -18.9% | 0.8434 |

- Both history variants reduce second-half amplitude in **5 of 6 prompts**
  for each method. This supports relative attenuation, not universal dominance.
- Exceptions matter: reference increases prompt 612 amplitude by 0.8% (IPO)
  and 16.9% (DPO). Feedback increases prompt 54 amplitude by 8.3% (IPO) and
  3.3% (DPO). All these prompts remain in the figures and tables.
- Prompt 251 shows clear repeated rotation in the ordinary arm; history
  trajectories generally occupy a smaller region. The full six-prompt plots
  show substantial heterogeneity rather than six identical circles.
- **Ordinary also contracts strongly**, from mean amplitude about 1.99 to
  about 0.56 (IPO) / 0.57 (DPO) in the second half. Thus these LLM experiments
  do not realize a clean contrast of divergent ordinary versus convergent
  history dynamics. All three arms continue moving at late iterations.
- Mean per-round centered-logit RMS during 51..100 is IPO 0.2398 / 0.2259 /
  0.2231, DPO 0.2493 / 0.2280 / 0.2286 (ordinary/reference/feedback). The
  finite nonzero updates and visible oscillations do not establish fixed-point
  convergence, even when entropy fluctuates within a narrow range.
- Endpoint rankings differ from time-window rankings. For example, IPO
  reference amplitude at state 100 is 0.5729 versus ordinary 0.5009. Do not
  substitute a favorable endpoint for the full trajectories.
- Relative-sequence entropy initially decreases and subsequently fluctuates
  around a broadly stable range; it is not monotonically decreasing to zero.
  Reference has higher second-half mean relative entropy than ordinary in
  both methods. Entropy alone does not establish convergence or text quality.

Suggested empirical wording: "On the selected cyclic response panels, history
updates attenuate the late-stage cyclic-mode amplitude relative to matched
ordinary updates, while residual motion and prompt-dependent exceptions remain."
This is a one-seed, six-prompt mechanism study, not a statistical significance
claim, neural limit-cycle proof, or representative generation-quality result.

## Coordinate and entropy definitions

For prompt x with four fixed responses, let s_t contain the response-token
sequence log-probability sums, including EOS. No token-length averaging is used.

- Centered logits: u_t = s_t - mean(s_t).
- Mode coordinate: z_t = w^T (u_t - u_star), where w is the normalized complex
  left eigenvector for the ordinary population map at its calibrated fixed
  point. The ordinary/reference/feedback arms of a method share this same w
  and u_star. The theoretical fixed point is only the coordinate origin, not
  a simulated trajectory or an assertion that the LLM converged there.
- One constant complex rotation per prompt makes z_0 real and positive,
  identically across the three arms. Plot axes use common limits and equal
  aspect ratios; amplitude is |z_t|, not distance traveled or text diversity.
- Phase is unwrap(arg(z_t)), reported in turns relative to state 0. Near-zero
  amplitudes make angle unstable, and discrete phase increments can approach
  pi. Crosses mark |z_t|<0.1 in phase plots as a descriptive warning only.
  No observations are dropped; turn counts are not robust counts of proven
  limit cycles or evidence of complete stabilization.
- Relative distribution: r_t = softmax(s_t - s_0).
- Relative-sequence entropy: H_rel(t) = -sum_i r_t(i) log r_t(i), in nats.
  H_rel(0)=log(4)=1.38629 by construction. It is not full-vocabulary entropy,
  raw panel entropy H(softmax(s_t)), or token-normalized legacy entropy.

We recomputed both stored entropy fields and plotted only the relative metric
under that name. No winning-rate or open-generation data were collected in
this campaign; no such outcomes are inferred from the fixed panels.

## Figures and reproducibility

Every figure is exported as matching PNG and vector PDF:

- `stage_b_overview`: six-prompt mean amplitude, update RMS, relative entropy.
- `ipo_all_prompt_trajectories`, `dpo_all_prompt_trajectories`: all six prompts,
  all three arms; pale paths show 0..20 and solid paths show 20..100.
- `ipo_amplitude_phase`, `dpo_amplitude_phase`: all prompt-wise amplitude and
  phase curves over 0..100, with near-origin phase warnings.
- `ipo_relative_entropy`, `dpo_relative_entropy`: all prompt-wise relative
  entropy curves, on identical 0..log(4) scales (with a small display margin).

Tables: `run_summary.csv`, `per_prompt_summary.csv`, `trajectory_points.csv`.
`summary.json` holds the same aggregate and per-prompt summaries.
`analyze_stage_b.py` reproduces validation, tables and all figures from the
downloaded numeric inputs and the hash-verified training mathematics.
No smoothing or interpolation is applied. Old experiment files are untouched.
