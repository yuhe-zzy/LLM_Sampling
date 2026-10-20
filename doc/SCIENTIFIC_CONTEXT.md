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

## Current Oracle1 rationale and scope

Discussion clarified2026-10-19 against the owner's October13/15 endpoint
correction: ask whether actual generated responses change back and forth along
shared interpretable directions during iterative preference training, and how
history interventions change these trajectories and Oracle1 WR. Fixed-candidate
probabilities and static preference cycles alone do not answer that question.
See the [active protocol](../experiments/oracle1_semantic/README.md).

The current design measures new responses on the same200 held-out prompts with
the same decoder and frozen five-trait rater. Equal-prompt feature summaries
estimate generated behavior under this evaluation distribution, not all language
behavior. Use distributions and prompt-level evidence alongside means; opposing
prompt changes can cancel. Frozen discovery-only PCA avoids choosing axes to
make observed trajectories look rotational. WR against the new initial4B model
is a separate single-judge quality endpoint, not a stability measurement.

History changes the iteration itself. With alpha=.9 and nu=.45, reference uses
.1*s0+.45*st+.45*s(t-1). Its two-round average cancels a purely alternating
input perturbation, but this algebra does not prove stability or faster settling
of the coupled neural training process. Feedback instead extrapolates the
update: delta_t+.5*(delta_t-delta_(t-1)). It can change tracking or overshoot;
it is not an averaging rule or a guaranteed damping intervention.

The active single-BT Oracle1 ground truth has zero exact logit cyclic residual.
It supports a transitive control and descriptive generated-behavior comparisons,
not a test that intrinsic cyclic preferences caused any observed oscillation.
DPO's exact delta is centered reward/beta and constant across sampling weights;
its feedback correction is zero. Retain this requested null control. IPO's
delta can depend on the changing sampler. Qwen3-4B is a new policy choice, not
evidence that model size causes or reveals cycles; its template/post-training
also differ from the old policy.

Lower motion together with preserved/improved WR would support useful empirical
stabilization within the measured scope. Lower motion alone can also reflect
stalled learning or collapsed outputs. Distinguish excursion, adjacent changes,
drift and repeated reversals. Ten-step generation can miss faster motion; one
training seed and five operational traits do not establish universal convergence
or robust seed-level effects. This explanation changes no settings, thresholds,
analysis authorization or job state, and claims no new empirical result.

## Historical Oracle2 question and settings

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

### Fan's shared latent estimator proposal, reviewed2026-10-14

Owner requested explanation only of fan/latent-estimation-2026-10-13 at7e152a2.
Read the agent documents, reference notes and both HTML report texts; no report
PDF rendering, proof certification, new computation or branch merge performed.
The [execution clarifications](../experiments/hodge_diagnostics/plans/latent-estimation-2026-10-13/EXECUTION_CLARIFICATIONS.md)
override shorthand remaining in the draft/human reports. This is not a result.

Fit pooled within-prompt comparisons using common response features psi:
flow_hat(i,j)=u^T(psi_i-psi_j)+psi_i^T B psi_j, B skew. The shared coefficient
operator is not a probability matrix or the arithmetic average of aligned P.
For model-induced panel flows, C_hat=Psi_centered B Psi_centered^T; observed
Hodge C can differ because of fit error. This is a declared modeling assumption,
not the only possible shared estimator. Coefficient energy shares depend on
feature scaling/origin convention, so require predictive and panel-flow checks.

S0 uses the exact private500x4 fixed bank, no guessed reconstruction. S1 uses
frozen embeddings/PCA8,16 first and new0..10 judged style features second;
existing HelpSteer attributes are an optional control, not a substitute for
new style labels. Paid annotation channel/budget and new resources need owner
confirmation. Mean/scales/PCA/fit/plane use discovery250 prompts only (seed0),
with250 held out from estimator selection, not from original model training.

Owner follow-up2026-10-14 asks whether style labels come from the oracle. The
proposal leaves the structured style-annotation judge/model unspecified. It
is a feature extractor role, distinct from preference ground truth and WR.
Existing scalar Nemotron/Skywork rewards or mixed pairwise probabilities do
not directly supply the requested multidimensional style scores. No model,
paid annotation batch or new judging action was selected/authorized here.

Follow-up: style labels are response feature vectors, not pairwise preferences;
do not request/retune annotations to manufacture cycles. Freeze the rubric and
validate shared structure independently. Majority cycles (A>B>C>A) are distinct
from non-gradient Hodge residuals under a specified link. For example, probabilities
Pab=.7, Pbc=.7, Pac=.8 have an acyclic majority order, but identity-flow triangle
sum .2+.2-.3=.1 is nonzero (illustration, not experimental data). The single
robust cyclic train label does not mean all other Hodge residuals vanish.
References: Jiang et al., Statistical ranking and combinatorial Hodge theory,
https://arxiv.org/abs/0811.1067; the existing complete-graph Hodge implementation.
Nonzero fitted B alone is not measured shared cycling; require observed-data
and held-out checks and allow a null result. Static Hodge signal need not induce
training oscillation, and jitter does not identify cyclic preferences as its
cause. Empirical history/stability comparisons do not require majority cycles.
No new cycle-enriched dataset, annotator tuning or scientific change authorized.

### Oracle comparison and feature literature, discussed2026-10-14

Owner challenges whether Oracle2 remains scientifically distinct if cycles are
not needed. Clarification: descriptive history/stability comparisons can use
either oracle; a claim about cyclic-preference mechanisms needs actual measured
non-potential signal under the declared link and appropriate controls. Do not
equate no manufactured cycles with no need for cycle evidence. Oracle1 has
P_N=sigmoid(delta r_N), so exact logit flow is additive. Oracle2 mixes .6 P_N+
.4 P_S and need not admit a single BT score, but non-BT signal is not guaranteed
on a given bank. It can also change ranking/strength without meaningful shared
cycling; these effects are not isolated by comparing unmatched old campaigns.
Identity Hodge residual can be nonzero for a scalar BT judge, so logit checks
are useful for separating mixture nonadditivity from link effects, alongside
the identity diagnostics relevant to the intended analysis. S5 can compare all
three stored score components on identical panels/features without new training;
this discussion does not authorize running it. Shared feature misspecification
can still yield nonzero fitted B for an exactly additive observed flow.

Primary sources inspected (paper text, not rendered-layout review):
- HelpSteer, NAACL2024, Appendix A.1: five0..4 attributes; helpfulness,
  correctness and coherence are quality-oriented, complexity/verbosity give
  expression-related axes. Verbosity is prompt-relative, not simply length.
  https://aclanthology.org/2024.naacl-long.185.pdf
- Pavlick and Tetreault, TACL2016: perceived formality,7-point scale -3..3,
  five human ratings and agreement analysis; not a ready validated LLM rubric
  for our response domain. https://aclanthology.org/Q16-1005/
- Danescu-Niculescu-Mizil et al., ACL2013: request politeness on a continuous
  perceived-politeness scale, with multiple annotators; adaptation from requests
  to assistant responses requires checks. https://aclanthology.org/P13-1025/
- APM,2026 preprint, Appendix D/E/Table6: LLM-judged1..10 response principles
  include elaborate, acknowledges uncertainty, casual, gentle and multiple
  perspectives. It screens low-variation/redundant attributes. Borrowing these
  features does not authorize its synthetic preference mapping or judge model.
  https://arxiv.org/html/2605.21063v1

Proposed compact attributes: verbosity/elaboration, linguistic complexity,
formality, politeness, uncertainty expression and multiple perspectives; keep
token length/formatting as separate deterministic diagnostics. This is an
adapted candidate set, not a universal standard or accepted change to Fan's
plan. Do not double-count concise/detailed as presumed independent dimensions
or treat expressed uncertainty as factual error/calibration. Higher style scores
do not mean better quality. Existing HelpSteer labels remain optional controls
after exact identity join, not newly judged style annotations. Preserve/report
native scales or explicitly document a0..10 adaptation; discovery-only scaling.
Suggested pilot/reliability checks precede paid full annotation and any cycle
selection; no model, budget, dimensions, new labels or experiments approved.

Owner correction2026-10-14: HelpSteer is our existing candidate-source dataset,
not a newly discovered replacement dataset. Its exact response-level attribute
labels can be reused after identity/order validation; the numeric export alone
does not contain them. Shared-feature/Hodge analysis does not inherently require
a new LLM judge. Fan's S4 uses saved log-probs on the same fixed responses at
each step, so one set of fixed features suffices. Existing labels can support
an explicitly named reduced feature analysis, but do not fulfill Fan's separate
proposal for newly judged politeness/formality/uncertainty style attributes.

Open-generation style analysis needs features for the new response texts; old
response labels cannot be transferred merely because the prompt matches. These
could be obtained by humans, a validated attribute predictor, deterministic
metrics for suitable attributes, or an LLM; a new LLM is convenient, not logically
required. A different labeled dataset can train/calibrate a predictor but cannot
directly provide labels for our unannotated responses. Pairwise preference labels
alone likewise do not supply multidimensional features. Oracle2's already-defined
open-generation WR judging is separate and does not need style annotations.
No new evaluation, annotations, simplified feature analysis or data join executed.

S2 fits identity or a declared logit estimator separately for Nemotron, Skywork
and their mixed Oracle2. Preserve user naming: Oracle1=Nemotron, Oracle2=mix,
not Skywork. Pure shared fitting first; no prompt-specific residual/rho sweep.
S3 weights cyclic coupling by panel softmax geometry, actual initial policy
primary and uniform secondary; leading skew mode gives a2D plane, not a single
PCA axis. Require held-out observed-flow fit, potential-only baseline, fixed-plane
observed-Hodge alignment,500-draw candidate-conjugation/orientation-flip nulls
and transitive surrogates. Prompt bootstrap, not independent-edge bootstrap.
The local linear-feature beta frontier is a diagnostic proxy, not an LLM theorem.

S4 regresses within-panel-centered saved sequence-sum logits on shared features
to estimate theta_t, then uses the S3 plane for all arms. Report both initial-
fixed and checkpoint-moving policy metrics, fit explained share/residuals,
radius/phase. This differs from the earlier E_pi[features] proposal and measures
fixed-panel behavior, not newly generated output semantics. S5 compares judges
and links; a nonzero fitted B for a scalar BT judge can be feature misspecification
even though its exact per-panel logit Hodge residual is zero. No rotation cannot
identify training noise/capacity as its cause. No aligned result has been computed.

### Primary endpoint correction: actual generated outputs,2026-10-15

User requirement: test whether actual model outputs exhibit shared semantic or
style oscillation across training checkpoints, not merely movement of fixed
response probabilities within individual prompts. Earlier fixed-panel label
reuse discussion remains technically valid, but is insufficient for this goal.
Fan S0-S3 may discover common preference-related directions; his saved-logit S4
is an auxiliary fixed-panel diagnostic, not the final open-generation endpoint.
Finding no motion does not identify training noise/capacity as its cause.

Proposed measurement, not executed or approved for implementation: on the same
evaluation prompts at each checkpoint, sample under a fixed decoding protocol
and extract a multidimensional feature vector phi(prompt,response). Traits
are simultaneous scores, not mutually exclusive response classes. Keep quality,
style and Oracle2 pairwise WR separate. For N prompts and K responses each,
m_t=(1/N) sum_x (1/K) sum_k phi(x,y_t,x,k), with equal prompt weights and no
additional likelihood reweighting of generated samples. This estimates output
features under this prompt distribution and decoder, not universal model behavior.
New responses need their own features; original HelpSteer labels cannot directly
label them. A structured LLM is one practical measurement tool, not logically
mandatory: validated predictors or frozen embeddings are alternatives.

Freeze feature scales/extractor and any learned directions across arms/time;
calibrate original human labels versus generated-text predictions if mixed.
Do not choose/refit axes on visually appealing trajectories. A common latent
projection of m_t is not automatically Fan's theta_t coefficient estimator or
its theoretical dynamical operator. Validate the intended connection explicitly.
Proposed figures: per-feature means/uncertainty, fixed-plane trajectories, and
feature distributions/quantiles (means can hide opposing groups). Distinguish
drift, isolated reversals, sampling/annotation jitter, damped swings and repeated
rotation. A nonzero Hodge component alone is not measured dynamic oscillation.
Use repeated outputs and prompt-level paired uncertainty checks; responses from
one prompt are not independent prompts. Seed0 evidence does not establish
between-training-seed robustness. No numeric acceptance threshold selected.

Checked plan_two_gpu.json:200 evaluation prompts,4 responses each, states
0,10,...,100, temperature.8, top_p.95, max_new_tokens256. Existing validated WR
generation banks could be reused without duplicate generation, subject to a
fresh completeness/provenance check; current server completion is unknown.
These are11 observed checkpoints, not101. Fixed-panel numeric snapshots do not
supply unsaved model states or generated answers. Ten-step sampling can miss
faster motion; denser checkpoint generation requires separate planning/approval.
No new annotation, generation, GPU work, timer or scientific config was changed.

### Concrete generated-output proposal,2026-10-15

Owner requested a concrete reorganization of Fan's plan, explicitly no running.
See [proposed study](../experiments/hodge_diagnostics/plans/generated-output-dynamics-2026-10-15/PLAN.md).
Proposed local rater is Qwen/Qwen3-32B BF16 non-thinking, with seven simultaneous
0..10 response-only traits: elaboration, formality, politeness, hedging, structure,
exemplification and linguistic complexity. New original/generated answers use
the same rater/scales. Prompt-relative appropriateness and WR remain separate.
Proposed secondary encoder is Qwen3-Embedding-0.6B, fixed1024 output then
discovery-only PCA8/16. Official cards checked; no local availability, memory,
annotation quality or immutable revision verified and no model downloaded.

Primary plane proposal is the leading2D skew block of the discovery-fitted B
under standardized features, mixed-Oracle2 soft-logit fit; identity/components
are predeclared secondary analyses. Fan's initial-panel S_eff and theta regression
are retained as auxiliary, not equated with generated-feature means. This is
an explicit analytical change, not an implementation or overwrite of Fan's plan.
Freeze the plane before any trajectory selection; hold out250 estimator panels
and the200 evaluation prompts. Include observed-Hodge alignment/nulls and
potential-only/single-BT controls, and allow all raw trait dynamics even if the
plane lacks shared-cycle evidence. No static-signal gate for descriptive plots.

Re-read local inspection2026-10-14T05:01Z:48 complete banks=38,400 generated
rows, plus800 baseline. IPO ordinary/reference complete0..70, feedback and DPO
ordinary0..100, DPO reference/feedback0..40; partial prefixes remain incomplete.
This is dated evidence, not live progress. Step0 cap rate43.5% at256 tokens is
a substantial limitation: report capped-output behavior and length/cap diagnostics,
not unrestricted-answer dynamics. Full main labeling would be55,600 rows before
cache reuse; pilot adds200 original calibration responses. No GPU-hour claim
without measured pilot throughput. No paid endpoint, runtime or resources approved.
Raw text remains private; proposed oneGPU sequential annotation/extraction must
still respect the accountFOUR ceiling and protected phases after new approval.

### Attribute references and dimension choice,2026-10-15

Owner asks whether the seven traits have references and whether three would
perform better. Rechecked HelpSteer, formality/politeness primary papers and
APM2026 Appendix E/Tables6,8. Individual traits have related precedents, not
validation of this exact seven-item0..10 instrument or Qwen3-32B annotation.
Structure/example scores are proposed operational choices. APM supports explicit
elaboration/uncertainty traits and redundancy screening, not our dimension count
or cyclic-dynamics conclusions. Exact citations/boundaries added to the plan.

Keep raw trait count, effective rank and the final2D plane separate. For d traits,
skew B has d(d-1)/2 coefficients plus d potential coefficients:3 traits give3/6,
7 give21/28. Three may estimate more stably but omit relevant directions; seven
may carry more information and more noise. A rank2 seven-trait fit can retain
all measured attributes while limiting the number of coupled modes. Attribute
count does not alter already-trained policy behavior. No evidence of an optimal
count yet; neither a larger fitted B norm nor more visible rotation selects it.

Suggested, unapproved comparison: compact elaboration/formality/hedging view,
seven-trait rank2 model and full seven-trait sensitivity after pilot reliability
checks. Any dimension selection uses discovery-only grouped CV, keeping final
validation and generated dynamics untouched. No automatic replacement of the
primary plan and no annotation/analysis execution. Negative outcomes remain valid.

### Five-trait scope and proposed Oracle1/Qwen3 change,2026-10-15

Owner chooses five dimensions and reaffirms FOUR total accountGPUs. Recommended
five simultaneous0..10 traits are elaboration, linguistic complexity, formality,
politeness and hedging. Structure/exemplification leave the primary rubric;
format/length diagnostics remain. Count accepted, specific rubric/model still
proposed. Historical three/seven options are not additional required experiments.
Five-feature B has10 free skew coefficients plus5 potential coefficients; no
claim this dimension is optimal or produces a stronger scientific result.

Proposed labeler remains frozen Qwen/Qwen3-32B BF16 non-thinking, unvalidated.
The actual campaign model lock confirms Qwen2.5-1.5B, not3B, and Oracle1 is
nvidia/Llama-3.1-Nemotron-70B-Reward-HF. Official Qwen cards/blog confirm4B dense
and30B-A3B MoE are different sizes. Recommend Qwen/Qwen3-4B non-thinking as a
new policy candidate. Existing adapters/generated texts cannot become4B results;
new training/generation/baseline are needed if approved. Post-training/template
differences mean this is not a controlled size-only comparison. No model was
downloaded or tested, and no campaign settings were changed.

Clarify Oracle1-only WR versus replacing training preferences. Oracle1 suffices
for a declared single-judge WR endpoint and descriptive style dynamics, not an
equivalent mixed-oracle mechanism. Exact single-BT logit flow is r_i-r_j and
per-prompt cyclic residual zero. Identity link, probability aggregation across
heterogeneous prompts and restricted feature compression can have residuals;
do not conflate them with intrinsic within-prompt non-BT preference. A nonzero
fitted B can reflect feature/nonlinear/context misspecification. For Oracle1-only
training, lead with raw five-trait/generated trajectories and frozen descriptive
axes; preference-cycle claims require separate evidence. The inherited Oracle2
fit remains a proposal for old-data analysis, not silently replaced by Oracle1.

Tentative resources:4B LoRA/generation oneH100/task initially two concurrent,
32B labeler oneH100, Oracle1 twoH100s, phases sequential. These are targets,
not measured guarantees; previous Nemotron memory evidence covers its audited
candidate bank only. Full-account queue/receipt guards still required, no
unrelated pending/running work, no automatic timer restart. Planning only.

### Training preference source clarification,2026-10-15

Owner asks what would supply training preferences if not Oracle1 alone. The
existing alternative is the already implemented Oracle2, not a new annotator:
P_ij=.6*sigmoid(r_Ni-r_Nj)+.4*sigmoid(r_Si-r_Sj), native temperatures1.
It still includes Oracle1/Nemotron; it is not Skywork alone, an average of raw
rewards, or the proposed five-dimensional labels. This fixed candidate P is
the base preference input for the IPO/DPO schemes. Qwen3-32B's proposed role
is frozen attribute measurement, not training preference generation.

Retaining Oracle2 training and reporting Oracle1-only WR is a possible separate
measurement choice: it measures Nemotron preference, not Oracle2 WR or direct
validation of the mixed training objective. The matching Oracle2 training/WR
protocol remains unchanged for the existing campaign. A future all-Oracle1
campaign is also a valid descriptive dynamics study with the previously stated
mechanistic limits; no switch has been selected by this clarification question.
No model download, scientific configuration, analysis or GPU action authorized.

## Proposal versus execution

The collaborator's Hodge notes and two-week roadmap are hypotheses/plans unless
linked to an actual launch receipt and validated result. Theory trajectories
must not be labeled LLM measurements. Training provenance is the frozen commit
recorded per campaign, never the current documentation/plotting commit.
