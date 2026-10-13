# Scientific context and interpretation

## Confirmed definitions

Current runnable IPO/DPO experiments use sequence-sum response log likelihood,
including the protocol's EOS, not token-average training. Historical archived
token-average figures retain their old label; publishing them does not restore
an old training mode or make them sequence-sum results.

- Raw panel policy: softmax(s_t) over the fixed candidate responses. It is not
  normalized over every possible generated response.
- Relative-sequence distribution: softmax(s_t-s_0). Its entropy is distinct
  from raw panel entropy and historical length-normalized entropy.
- History reference: (1-alpha)*s_0+(alpha-nu)*s_t+nu*s_(t-1).
  Alpha belongs to the reference refresh, not an independent policy mixture.
  Ordinary uses nu=0; initial-anchor control alpha=.1 yields(.9,.1,0).
- History sampling lambda_current and older static/oracle lambda_on do not
  share one meaning. Read [protocols](../docs/PROTOCOLS.md) before comparisons.
- Generated-response WR requires an external comparison rule. The synthetic
  cyclic fixed-panel tournament is not an unrestricted-generation judge.

## Approved Oracle2 question and settings

Empirical stability trends, not a demand to eliminate oscillations or prove
convergence. Oracle1 is Nemotron's scalar reward judge. Oracle2 mixes .6 of
Nemotron BT probabilities and .4 of Skywork BT probabilities, each temperature1;
it does not mix raw rewards. Fixed real candidates train the model; open
generations are compared with an independent shared pretraining baseline.

Disjoint calibration/train/evaluation panels100/500/200, four candidates each.
Audited robust cyclic counts0/1/0, transitive81/436/160, ambiguous19/63/40.
The owner explicitly accepted the sparse cyclic case on2026-10-09. Do not
retune data, mixture, temperature or cycle thresholds to manufacture cycles.
Six arms: IPO/DPO ordinary/reference/feedback, seed0, alpha=.9,lambda_current=.8;
IPO beta=.2, DPO beta=.8, reference nu=.45, feedback kappa=.5; states0..100.

## Checked limitations, not favorable-result filters

The sole cyclic training prompt416 starts with nearly unit raw panel mass on
a two-token candidate. Its flat raw trajectory is not history stabilization.
Ambiguous does not mean non-ST; no cyclic evaluation subgroup exists, so its
WR is missing, not0. Single-seed/all-prompt findings are descriptive.
Lower temporal SD can coexist with higher adjacent-step TV. Neither alone
proves convergence or absence of periodicity. Preserve counterexamples.

Matrix inspection2026-10-13: ground-truth P[i,j] is the Oracle2 BT mixture
probability that fixed candidate i beats j, not policy probability; all six
arms share it. Prompt416 has a robust1>2>4>1 subtriangle, while candidate3
beats each other candidate (one-based indices). Its weakest cycle edge is
.5204165987, only .0004165987 above the fixed .52 cutoff. A cyclic
subtriangle and a Condorcet winner coexist; this does not predict that neural
training must cycle. The displayed ambiguous prompts6/11 have no majority
triangle and include near-tie edges. Group labels do not certify or reject
strong stochastic transitivity. Values were checked against the hash-validated
export and published policy_results_20261013/prompt_metadata.json.

Hodge norm convention checked2026-10-13: reuse hodge_diagnostics/hodge/core.py
on the complete uniform graph. Identity flow is A=P-.5, logit flow is
A=log(P/(1-P)); u=A1/K, G_ij=u_i-u_j, C=A-G. C_norm2 is the matrix spectral
norm, not Frobenius. For these K4 skew zero-row-sum residuals, the unique-edge
Euclidean norm equals spectral norm and Frobenius is sqrt(2) times it.
All six displayed panels have nonzero residuals for both links, despite their
different ordinal labels. Logit diagnostics are not the actual DPO optimizer.
See the dated note for numeric values and independent projection validation.

## Cross-prompt averages and latent directions

User clarification2026-10-13: the scientific target is shared output-direction
behavior across prompts (detail, tone or latent semantics), not just individual
response trajectories. First requested diagnostic is the mean preference/Hodge
matrix. [All500 training-panel results](../experiments/oracle2/campaigns/oracle2_real_20261009/average_hodge_20261013/README.md)
are computed in original candidate-index coordinates with equal prompt weights.
Identity norm(mean C)=.022618963780 versus mean(norm C)=.221815660192;
aggregate cyclic energy share=.357286%, versus mean individual share6.844496%.
Mean logit-flow norm(mean C)=.052292636504, mean(norm C)=1.145352103763;
aggregate share=.010474%. Logit of mean P is different: norm=.057420385418,
share=.128827%. No clipping. Full matrices and independent numerical checks
are saved; seven focused CPU tests pass, not the entire repository suite.

The builder orders candidates by source scalar u, and Oracle2 preserves their
order; these are not shared semantic labels. Exact historical score-field
arguments have not been re-audited. Weak signed average cyclic signal in these
coordinates does not prove that semantic cycling is absent or noise dominates.
PCA variance is not coherent mean cyclic strength. A fixed shared orthonormal
projection cannot recover cancellation in the full mean matrix. Semantic
alignment must follow a common feature rule, not independent sign choices.

Proposal, not executed: define shared response features, inspect dataset-mean
expected-feature trajectories across all six arms, and optionally discover a
two-dimensional cyclic plane using the collaborator's shared-feature compression
in hodge/coherence.py. Fit directions on discovery prompts and freeze before
held-out comparison. Initial-policy saturation and the difference between a
feature-policy proxy and the actual LLM Jacobian must remain explicit. A scalar
trajectory can oscillate, but one real scalar axis cannot represent a nonzero
skew cyclic operator. No new GPU job or scientific setting was authorized here.

### Proposed operational protocol, not yet executed

2026-10-13 follow-up asks how to make the shared-direction question concrete
with existing data. The numeric export actually contains static response token
counts and panel probabilities of shape101x500x4 for each of six arms. The five
original HelpSteer attributes are not in this numeric export; join them from
the raw source by exact prompt/response identity and audit duplicates/missing
matches before claiming all500 feature panels. Dataset documentation confirms
helpfulness, correctness, coherence, complexity, verbosity scores0..4; none
directly measures tone. Proposed first feature set: those five scores and log
token count. Attributes are diagnostics, never a substitute for Oracle2 P.

Define f_xi as the fixed feature vector, with within-prompt centering and common
scales established on discovery prompts; retain native-unit plots as well.
Shared change trajectory is z_t=mean_x sum_i(pi_txi-pi_0xi)*f_xi. Use raw panel
probabilities, not relative-sequence weights. Reweighting static responses is
a fixed-panel output diagnostic, not measured open-generation semantics.

Proposed continuous metrics (no universal convergence cutoff): per-coordinate
excursion range; total variation L=sum_t|z_(t+1)-z_t|; backtracking B=L-|z_end-z_start|.
Monotone scalar drift has B=0; reversals make B positive. B alone also responds
to stochastic jitter and does not establish periodicity. Report absolute units,
not just normalized ratios that can inflate nearly-flat curves. Moving10-update
mean increment magnitude and predeclared early0..20 versus late80..100 windows
describe late movement; no post-hoc best window or convergence-time claim.
Compare the six arms on identical features/axes/windows, within each objective.
Also compare magnitude of mean prompt increment versus mean of individual
magnitudes to reveal cancellation; these are not equivalent global metrics.

Optional latent plane: fit only to static C/features, not to whichever training
trajectory looks cyclic. Reuse the shared-feature S/M construction described
above, initially with uniform J and identity P-.5 as the primary diagnostic;
initial-policy J and logit are separately labeled sensitivity analyses. A
leading skew plane maximizes projected cyclic coupling under the declared
feature metric, not arbitrary unnormalized feature scale. It is not itself a
preference probability matrix or the true neural Jacobian. A plane restricted
to these six features says nothing exhaustive about all language semantics.

For the first post-hoc analysis, propose a fixed seed0 prompt split250/250:
fit scales/plane on one half and evaluate the signed coupling in the frozen
plane and trajectories on the other, without refitting its direction. These
are held out from axis selection, NOT held out from model training, since all
500 prompts trained the LLM. Use within-prompt feature-label permutation as
an alignment-null diagnostic, keeping the validation plane fixed; report its
exchangeability assumption. Paired prompt bootstrap can describe uncertainty
across sampled prompts but cannot supply missing independent training seeds.
No semantic features, latent plane, permutations or new metrics have been
computed yet. No new jobs, judging, embeddings or scientific changes approved.

## Proposal versus execution

The collaborator's Hodge notes and two-week roadmap are hypotheses/plans unless
linked to an actual launch receipt and validated result. Theory trajectories
must not be labeled LLM measurements. Training provenance is the frozen commit
recorded per campaign, never the current documentation/plotting commit.
