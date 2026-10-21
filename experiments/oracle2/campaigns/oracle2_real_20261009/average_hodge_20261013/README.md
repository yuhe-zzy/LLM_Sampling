# Average Oracle2 Hodge decomposition

Computed 2026-10-13 from all 500 training panels, not the six illustrated
prompts. All six IPO/DPO arms share these ground-truth matrices, so there are
500 matrices, not 3,000 independent observations. This is static preference
analysis, not a new experiment, training trajectory, or generated-response WR.

## Inputs and coordinates

- Source: [published numeric metadata](../policy_results_20261013/prompt_metadata.json).
- Groups: 1 robust cyclic, 436 transitive, 63 ambiguous, using the frozen rule.
- Equal prompt weights; four candidates and six uniformly weighted edges per panel.
- Original candidate order is unchanged. The repository's
  [panel builder](../../../../../scripts/build_pairs.py) sorts responses by the
  source scalar `u` descending; [Oracle2 preparation](../../../prepare_data.py)
  preserves that order. This is not alignment by detail, tone, or latent semantics.
  The original historical builder's exact score-field arguments were not
  independently re-audited in this analysis.
- The published JSON parses identically to the original numeric export metadata.
  Its byte encoding differs, so byte hashes must not be equated. The export's
  metadata SHA256 is `4806a9b05b7555cf00882d851d8f78cdba935c3ccd32c3f11b045b3a3bf01c65`.
  The output records both the actual local input byte hash and a canonical JSON hash.
- No response text, weights, adapters, or credentials are included.

## Definition

For identity flow, `A_x = P_x - .5`. For logit flow,
`A_x = log(P_x/(1-P_x))`. Diagonals are zero. Reuse the existing
[Hodge core](../../../../hodge_diagnostics/hodge/core.py):

```text
u_x = A_x 1 / 4
G_x[i,j] = u_x[i] - u_x[j]
C_x = A_x - G_x
bar A = mean_x A_x; bar G = mean_x G_x; bar C = mean_x C_x
```

The complete-graph projection is linear, so decomposing `bar A` equals
averaging these components. L2 here is the matrix spectral norm. For an exact
skew, zero-row-sum K4 residual it also equals the unique-edge Euclidean norm;
the Frobenius norm is sqrt(2) times as large. Gradient/cyclic orthogonality is
in the edge/Frobenius inner product, not a spectral-norm energy decomposition.
This follows the usual [HodgeRank formulation](https://arxiv.org/abs/0811.1067).

## Actual averages

```text
bar P =
 0.500000000  0.562452519  0.643434642  0.734721456
 0.437547481  0.500000000  0.592624839  0.700633508
 0.356565358  0.407375161  0.500000000  0.621631351
 0.265278544  0.299366492  0.378368649  0.500000000

Identity bar u = [0.110152154, 0.057701457, -0.028607033, -0.139246579]

Identity bar G =
 0.000000000  0.052450697  0.138759187  0.249398733
-0.052450697  0.000000000  0.086308490  0.196948036
-0.138759187 -0.086308490  0.000000000  0.110639546
-0.249398733 -0.196948036 -0.110639546  0.000000000

Identity bar C =
 0.000000000  0.010001822  0.004675455 -0.014677277
-0.010001822  0.000000000  0.006316350  0.003685472
-0.004675455 -0.006316350  0.000000000  0.010991805
 0.014677277 -0.003685472 -0.010991805  0.000000000
```

All matrices and all 500 per-prompt norm/energy summaries for both links are
in [summary.json](summary.json), calculated without rounding input probabilities.

| Quantity | Identity: mean(P-.5) | Logit: mean(logit(P)) |
|---|---:|---:|
| Spectral norm of average cyclic component | 0.022618964 | 0.052292637 |
| Average of per-prompt cyclic spectral norms | 0.221815660 | 1.145352104 |
| RMS of per-prompt cyclic spectral norms | 0.246250175 | 1.562145525 |
| Norm of mean / mean of norms | 10.1972% | 4.5656% |
| Frobenius norm of average cyclic component | 0.031988045 | 0.073952956 |
| Cyclic energy share of average flow | 0.357286% | 0.010474% |
| Mean of per-prompt cyclic energy shares | 6.844496% | 3.858542% |

The aggregate energy share is `||bar C||_F^2 / ||bar A||_F^2`.
It is NOT the average of per-prompt energy shares. Nor is the retained-norm
ratio a percentage of cyclic prompts or a significance test.

**Noncommuting logit:** applying logit to `bar P` instead gives cyclic
spectral norm **0.057420385** and cyclic energy share **0.128827%**.
That answers "decompose the average probability matrix using logit", not
"average the per-prompt logit Hodge components". The two operations agree
for identity only. Logit is an additional diagnostic, not the actual DPO
training update, and its absolute scale is not comparable to identity.

Observed preferences are strictly inside (0,1), so no clipping is needed.
The original logit flow has a maximum per-panel reciprocity/skew roundoff
error 1.1142e-7, amplified by probabilities close to zero/one. Mean-flow skew
error is 3.8008e-10. No symmetrization was silently applied. Independent edge
least-squares residuals differ from the core by at most 2.8644e-8; all reported
checks pass the disclosed 1e-6 absolute tolerance. Identity errors are below
1.2e-15. Full validation errors are retained in the JSON.

## What is established

The signed cyclic residual is much weaker after equal averaging in these
candidate-index coordinates. The mean preference matrix has the strict
majority ordering 1 > 2 > 3 > 4 and no majority triangle, despite its nonzero
cardinal Hodge residual. This is descriptive evidence of cancellation and a
gradient-dominated average, not evidence that all individual panels are BT,
not a test of neural convergence, and not a bound on every semantic projection.
No permutation null, bootstrap, or held-out semantic test has been run here.

## Shared semantic directions: proposed next analysis, not executed

The user's question concerns dataset-wide output behavior, rather than
individual response IDs. A consistent feature map `phi(prompt,response)` is
needed first. Fixed HelpSteer attributes such as verbosity/complexity provide
interpretable starting coordinates; tone or other semantic features require
a separately checked extraction rule. Dataset attributes remain diagnostics,
not replacements for the frozen Oracle2 ground truth.

For each arm, a direct fixed-panel trajectory is
`z_t = mean_x sum_i (pi_t(i|x)-pi_0(i|x)) phi(x,y_i)`.
This measures changes in expected shared features across prompts, not open
generation behavior. The latter requires features on generated responses.
Use identical features, weights, axes and step windows across all six arms.
Keep raw policy and relative-sequence weights separate.

For an exploratory latent cyclic plane, the collaborator already implements
the shared-feature compression idea in
[hodge/coherence.py](../../../../hodge_diagnostics/hodge/coherence.py).
For feature matrices E_x and fixed reference-policy Jacobians J_x, form
`S = mean E_x^T J_x C_x J_x E_x` and
`M = mean E_x^T J_x E_x`. On the support of M, whiten S with the square root
of M's pseudoinverse and inspect its leading two-dimensional skew mode.
This is a proxy under a shared-feature policy, not a measurement of the full
LLM's parameter Jacobian. Choice of uniform versus initial-policy J_x must
be stated; saturation of the actual initial policies may suppress coupling.

PCA of vectorized cyclic matrices maximizes second-moment variation, which
can remain large even if signs cancel. It does not maximize a coherent signed
mean. A single real scalar has `v^T S v=0`; a nontrivial cyclic operator needs
at least a two-dimensional plane (although a scalar time series can oscillate).
For one common orthonormal Q on existing candidate coordinates,
`mean Q^T C_x Q = Q^T bar C Q` and its norm cannot exceed `||bar C||_2`.
Thus rotating the same misaligned indices cannot recover already cancelled
signal. Prompt-dependent alignment must come from a consistent shared feature
rule, not free per-prompt sign flips chosen to manufacture a large mean.

Fit features/planes on a discovery/calibration split, freeze the directions,
then evaluate on separate prompts and all arms. Otherwise maximization on the
same data is exploratory only. A useful plane need not be interpretable, but
it must not be called a specific semantic axis without validation. No observed
cycle does not establish that training noise is the cause; weak residuals,
feature misalignment, saturated policy Jacobians, stable update parameters or
neural approximation can also explain the absence.

## Reproduce

From the repository root:

```powershell
python experiments/oracle2/aggregate_hodge.py --metadata experiments/oracle2/campaigns/oracle2_real_20261009/policy_results_20261013/prompt_metadata.json --out experiments/oracle2/campaigns/oracle2_real_20261009/average_hodge_20261013/summary.json
python -m unittest discover -s experiments/oracle2 -p test_aggregate_hodge.py -v
```

Seven focused CPU unit tests cover cancellation versus magnitude, potential
flows, logit noncommutation, endpoint refusal, common relabeling invariance,
invalid input and all500-panel regression. This is not a full repository test
suite or a GPU test. No jobs, frozen sources, scientific settings or monitor
schedule changed.
