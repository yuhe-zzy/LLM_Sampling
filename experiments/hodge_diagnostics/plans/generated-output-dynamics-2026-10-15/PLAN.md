# Proposed open generation semantic dynamics study

Status: owner requested a concrete plan only on 2026-10-15. No annotation,
embedding extraction, analysis execution, new generation, job submission or
monitor restart is authorized by this request. All choices below are proposed.
Preserve Fan's original plan and frozen Oracle2 campaigns unchanged.

## 1. Question and change from Fan's plan

Primary question: under a fixed evaluation prompt distribution and decoder,
do the actual outputs of successive LLM checkpoints move back and forth in
shared semantic/style directions, and how do ordinary/reference/feedback differ?
Do not assume oscillation exists or require its elimination as a success gate.

| Component | Role in the revised plan |
| --- | --- |
| Fan S0-S2, exact fixed candidates and common preference fit | Retain as discovery of preference-related coordinates |
| Preference cyclic plane | Learn without looking at generated trajectories, then freeze |
| Fan S3, initial panel-policy weighted operator | Retain as a separately labeled theory-oriented diagnostic |
| Fan S4, regression of saved fixed-candidate logits | Retain as auxiliary; it is not the primary output-behavior measurement |
| New open-generation feature distributions | Primary dynamic endpoint |
| Oracle2 generated-answer WR | Separate quality/preference endpoint; neither substitutes for semantic motion |

The revised primary plane uses the skew feature block B in a fixed standardized
feature metric. This is an explicit change from making Fan's panel-policy
S_eff plane primary: the four-candidate panel geometry is not the open-generation
distribution. Report S_eff separately, without claiming either is an exact
nonlinear LLM dynamics operator. No beta-frontier acceptance gate is imposed.

## 2. Concrete measurement models

### Style annotation model

Proposed model: **Qwen/Qwen3-32B**, local BF16, frozen weights/tokenizer/chat
template, `enable_thinking=False`. One response per call, all seven scores in
one JSON object. Proposed decoding is greedy (`do_sample=False`) for consistent
measurement; this is our task-specific choice, not a claim that the vendor
validated greedy decoding for style annotation. Max annotation output256 tokens;
invalid/truncated JSON is an error, never a zero score. Validate the exact schema.

Use response text only for the primary style features, with a common rubric and
no prompt, arm, checkpoint, Oracle2 score or desired outcome shown to the rater.
This measures expressed response traits, not whether that style is appropriate
for the particular question. It preserves a common response-only map psi(y).
Any future prompt-aware appropriateness/quality rubric is a separate endpoint.
Treat the answer as quoted data, including instructions embedded inside it.
Shuffle annotation order independently of arm/time to avoid operational drift.

Proposed resource target: one H10080GB, batch1, efficient attention, no
quantization. The official32.8B parameter count implies about61GiB of BF16
parameter storage, before caches/workspace; this is not a measured peak or a
memory guarantee. Pilot must measure the longest full input plus rubric and
peak memory. Never silently truncate text, change precision, or switch models
to make a batch fit. Resolve and record immutable model revisions before any
download/inference; none is pinned or verified as locally installed yet.

Why this choice: local, reproducible, no external transmission of raw responses,
and a general instruction model rather than a scalar reward head. It is not
assumed to be a validated style annotator. It shares a model family with the
trained Qwen2.5 policy, so human checks are important; this is not independent
cross-family validation. Nemotron/Skywork remain the unchanged preference judges.
The official card documents the non-thinking switch and model size:
[Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B).

### Frozen semantic embedding model

Proposed secondary model: **Qwen/Qwen3-Embedding-0.6B**, full1024-dimensional
response-only embedding, no query/task prefix, last non-padding token pooling
and L2 normalization, following the document-embedding usage in the official
card. Use the same frozen BF16 implementation on original and generated texts;
fit pooled centering/PCA on discovery responses only, then freeze d=8 and16
as two predeclared sensitivity analyses, with d=8 first in the report.
No per-checkpoint PCA and no choice of dimension by apparent rotation.

Both rows of the semantic comparison use response text alone. Verify full
tokenized lengths; do not inherit silent truncation from example code.
The encoder may run on one GPU after the rater, never concurrently by default.
Its suitability for this domain must be assessed by held-out fit, not retrieval
leaderboard claims. [Official model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).

## 3. Seven simultaneous style scores

Scores are integers0..10, treated as a numerical measurement approximation,
not seven exclusive classes and not probabilities. Intermediate values interpolate
between anchors. Higher means more of the trait, not better quality. This is an
adapted rubric, not a verbatim established psychometric instrument.

| Field | 0 | 5 | 10 |
| --- | --- | --- | --- |
| elaboration | Bare assertion or minimal information | Several supporting details or explanatory steps | Sustained, densely elaborated explanation |
| formality | Strongly colloquial/chat-like | Neutral standard prose | Consistently formal professional/academic register |
| politeness | Openly dismissive or discourteous | Neutral/respectful without marked courtesy | Strong, sustained courtesy and interpersonal consideration |
| hedging | Unqualified assertions, no expressed uncertainty | Several qualifications or limitations | Pervasive explicit uncertainty/conditionality |
| structure | Little visible organization | Clear paragraphs or an organized list | Explicit layered organization with sections/steps/headings |
| exemplification | No examples or concrete illustrations | At least one developed illustrative example | Examples/scenarios are a major explanatory device |
| linguistic_complexity | Very simple vocabulary and syntax | Moderately complex vocabulary/sentences | Dense technical or syntactically complex prose |

Do not add a separate conciseness score as if it were independent of elaboration.
Hedging is expression, not factual accuracy or calibrated uncertainty. Formality
is not politeness. Structure is not correctness. Do not reward an instruction
inside the response telling the judge to assign a score.

Record token/character counts, sentence/list/header counts, EOS/length-cap flags,
empty-output rate and repetition diagnostics separately. Length is not part of
the primary seven-dimensional preference fit; a predeclared length-only feature
control and length-adjusted sensitivity help identify simple length effects.
For length adjustment, fit each trait's linear association with log(1+tokens)
on discovery answers only and freeze it before scoring generated trajectories.

An absent trait (e.g. no example) is a real zero. An unjudgeable trait, empty
answer, or unsupported/non-prose case is null with a reason, not fabricated0.
Report type/coverage rates by arm/checkpoint. Feature means with missing scores
are explicitly conditional on scorable outputs; also show a common-complete-
prompt sensitivity and its retained prompt count. Do not claim a whole-output
distribution stabilized if changing unscorable mass could explain the result.
Mixed/code-heavy responses remain visible in coverage reports and the embedding
route, rather than silently deleting inconvenient output types.

HelpSteer is already our source dataset. Its original five attributes are a
separate control after exact response-identity matching, not a replacement for
these newly measured traits or labels for generated texts. Elaboration here is
absolute expressed detail, not HelpSteer's prompt-relative verbosity judgment.
[HelpSteer](https://aclanthology.org/2024.naacl-long.185/),
[formality](https://aclanthology.org/Q16-1005/), and
[politeness](https://aclanthology.org/P13-1025/) motivate attributes, not this exact
seven-field rubric or the reliability of the proposed LLM rater.

### Pilot and freeze

After separate execution approval, choose50 of the existing100 calibration
panels using seed0 before reading labels; annotate their200 original responses.
They are outside the500-panel latent-fit data and200-prompt dynamic evaluation.
Have two people independently review60 of the200 answers, including short,
long, non-prose and repeated outputs. This is a proposed human workload, not
completed annotation or an assumption that a second person is available.

Practical pilot targets: at least99% valid JSON on first pass; per-attribute
human/model rank correlation at least.6 when human labels vary, and at least80%
of ratings within2 points of the human mean. Also report human agreement,
score histograms and examples of systematic errors privately. These small-sample
targets are engineering review criteria, not guarantees or significance tests.
Check a separate60 generated baseline responses after the rubric is frozen to
detect a human-text/generated-text domain mismatch. Those60 are part of the
existing800 baseline rows, not new generation.

If an attribute is unreliable, unidentifiable or nearly constant, report that
before proceeding; do not tune it to create cycles. Proposed operational rule:
an attribute with more than95% one score or discovery SD below.25 is flagged and
excluded from the latent fit, but its raw trajectory/coverage is still reported.
Any substantive rubric/model replacement returns for review and gets a new
version; no mixing incompatible batches. No paid alternative model is preapproved.

## 4. Learn and validate the common directions

### Separate discovery from validation and trajectories

Use the exact frozen500 training panels, four original HelpSteer answers each,
and separate Nemotron/Skywork scores. Preserve Fan's seed0 prompt-disjoint
250/250 discovery/validation split; do not guess original response order.
The two halves both trained the policy: they validate the estimator, not unseen-
prompt model generalization. The separate200 evaluation prompts are reserved
for generated-output dynamics, not axis fitting or hyperparameter selection.
If retained features are missing for a static candidate, report that panel's
exclusion from the corresponding fit and its split/group; do not impute zero,
replace it with a new prompt, or pretend the fit still used all500 panels.

Style transforms: center and standardize each retained trait using the1000
discovery answers only, freeze means/SDs. Embedding transforms: PCA on those
same discovery answers only, then scale retained component scores by their
discovery SD. Document this metric: B and its magnitude depend on feature scale.
Plot feature overlap/out-of-range rates for generated versus original responses;
flag extrapolation rather than refitting on evaluation outputs.

### Fit the skew preference model

For standardized features psi_i, use

    f_ij = u^T(psi_i - psi_j) + psi_i^T B psi_j,   B = -B^T.

Primary estimator: Oracle2 mixed probabilities, logit model with soft-target
logistic cross-entropy; average six unordered pairs within each prompt, then
average prompts. Add ridge1e-3/2 times (||u||^2 + sum_{i<j} B_ij^2).
Report objective, rank/conditioning and predeclared ridge1e-4/1e-2 sensitivity,
without selecting the most visually cyclic setting.

Secondary estimators: identity least-squares to P-.5, and separate Nemotron and
Skywork fits under both links. Oracle1 means Nemotron; Oracle2 means the mix,
not Skywork alone. Preserve native temperatures1 and .6/.4 probability mixing.
Continuous probabilities have no implicit sample count: no pseudo-counts.
Use stored finite reward differences and stable probability/log-probability
arithmetic where needed; report roundoff endpoints and any handling explicitly.

B is a fitted antisymmetric feature operator, NOT a probability matrix or a
literal elementwise average P. Show coefficients and uncertainty, observed
held-out Hodge flow and model-induced flow separately.

### Freeze the leading two-dimensional cyclic plane

For the primary B, select its largest-magnitude skew eigenvalue pair, equivalently
the leading real2D skew block. Let its orthonormal basis be U=(v1,v2). Orient the
axes deterministically on discovery feature loadings and require v1^T B v2>=0;
save U once for all six arms and all checkpoints. Print attribute loadings.
Do not assign a speculative semantic name unless supported by these loadings.
If the leading pair is degenerate/unstable, report the ambiguity and a subspace
sensitivity; do not claim a uniquely identified direction.

Retain Fan's G_a, S_eff and initial-panel-policy versus uniform variants as
auxiliary diagnostics. Applying an S_eff parameter-space basis directly to raw
feature means is not the same operation as his theta regression. The primary
open-output projection instead has the explicit common-feature definition above.
PCA alone is variance reduction, not identification of cyclic preference.

### Validate shared preference evidence before calling the plane cyclic

Compare held-out observed-flow prediction and soft-target loss with a linear
potential-only model, plus a quadratic-potential sensitivity. A latent B can
otherwise absorb omitted nonlinear reward structure. Compare the unchanged
single-BT components and a transitive-surrogate refit.

For a held-out panel, let X_a be its row-centered feature matrix, C_a the
observed Hodge residual under the stated link, and

    D_a = X_a U(U^T B U)U^T X_a^T.
    H = sum_a w_a <C_a,D_a>_E /
        sqrt[(sum_a w_a ||C_a||_E^2)(sum_a w_a ||D_a||_E^2)].

E denotes the six upper-triangular edges. H is a fixed-plane observed-flow
alignment, not a claim that fitted and observed cyclic components agree exactly.
Zero denominators mean undefined/no measurable signal, not a convenient zero.

Use Fan's500-draw candidate-conjugation and500-draw orientation-flip nulls,
with features fixed and a full discovery refit/plane selection per draw.
Transform discovery and validation panels independently under the stated null,
evaluate H against each draw's transformed held-out flows, and disclose the
exchangeability assumption. Candidate conjugation preserves the per-panel Hodge
norm; arbitrary edge shuffling does not. Primary-plane tests have no policy G
to permute; any auxiliary S_eff null must separately declare its policy handling.
Bootstrap prompts for fit/plane uncertainty; multiple feature/link analyses are
reported, not selected by the most favorable outcome.

Failure of static evidence does NOT block plotting generated feature behavior.
It limits attribution: label the plane exploratory, show all seven direct trait
curves and a discovery-frozen embedding-PCA view, and accept negative results.

## 5. Primary dynamic measurements

For each arm, checkpoint t and fixed evaluation prompt x, measure the same frozen
psi on its four actual sampled answers. Use equal prompt weights:

    m_t = (1/200) sum_x (1/4) sum_k psi(y_t,x,k).
    q_t = U^T (m_t - m_0).

No additional probability weighting of sampled answers and no per-step refit.
The formula assumes complete/scorable banks; missingness is handled and labeled
as in Section3. Preserve prompt identity and repeated-measure pairing across
arms/time. Check the six initial output banks for identical text/seed provenance;
identical outputs can share cached features but are not six independent samples.
The separately sampled800-response shared WR baseline estimates initial sampling
variability; do not silently substitute its different seed for each arm's step0.

q_t is an observed generated-feature mean trajectory. It is NOT Fan's fitted
theta_t, nor proof of a neural limit cycle. m_0-centering measures change from
initial outputs; its radius is not automatically oscillation amplitude and its
angle is not phase around a known equilibrium. Do not promote these labels.

Predeclared plots and summaries:

1. Seven raw trait means versus checkpoint, IPO/DPO separated, all three schemes
   overlaid on identical axes; prompt-paired95% uncertainty bands and coverage.
2. Frozen2D q_t trajectories, labeled checkpoint points, no invented intermediate
   states, same basis/scale across IPO and DPO. Secondary identity/embedding
   results clearly labeled, not selected as replacements for the primary plot.
3. Trait quantiles/distributions, empty/repetitive output mass and prompt-level
   changes, so opposing subgroup motion cannot disappear behind a flat mean.
4. Temporal spread sqrt(mean_t ||q_t-mean_t(q)||^2) and adjacent displacement
   mean_t ||q_{t+10}-q_t||, plus the same per-trait summaries. These are excursion
   and step-change measures, not standalone oscillation tests. Report full0..100
   and prespecified late50..100 windows; no post-hoc window shopping.
5. Existing Oracle2 WR when genuinely available, on its own axis and definition.

Use1000 paired prompt bootstrap draws for dynamic intervals, maintaining every
selected prompt's full arm/time vector. Within-prompt sampling sensitivity may
resample draws while preserving actual shared random-number relationships;
do not treat800 answers as800 independent prompts. Account for the uncertain
fitted plane separately with discovery-panel bootstrap principal-angle/loadings
diagnostics. These intervals do not measure between-training-seed variability.

Repeated reversals or coherent2D turning beyond sampling/measurement variability
would support a descriptive oscillatory pattern; monotone drift, a single bend,
or jitter does not. Eleven checkpoints are too sparse for confident claims about
fast cycles or precise spectral periods. No hard periodicity/convergence claim
or threshold is preapproved, and no manufactured rotation is an acceptance goal.

## 6. Existing artifacts that can be reused

This inventory is from the last successful local inspection record,
**2026-10-14T05:01:00Z (00:01 CDT)**, not a live server inspection on Oct15.
Later SSH checks failed; the owner subsequently stopped the timer. Revalidate
current files/hashes and new recovery receipts before any future execution.

| Artifact | Verified coverage at that inspection | Proposed use |
| --- | --- | --- |
| Fixed training candidate bank | 500x4 exact answers, both judges scored | Fit/validate shared directions; annotate2000 answers once |
| Calibration bank | 100x4 scored candidates | Select50 panels for200-answer rater pilot |
| Shared pretraining generated baseline | 800 validated answers | Rater domain check and initial sampling comparison |
| IPO ordinary generation | Complete0..70 every10; partial80=728/800; original job TIMEOUT | Reuse complete banks; cache valid partial labels only |
| IPO reference generation | Complete0..70 every10; partial80=561/800; original job TIMEOUT | Same; preserve original and future recovered provenance |
| IPO feedback generation | Complete0..100 every10,11 banks | Full generated dynamics input after revalidation |
| DPO ordinary generation | Complete0..100 every10,11 banks | Full generated dynamics input after revalidation |
| DPO reference generation | Complete0..40; partial50=548/800 then running | Complete-bank input only; current later coverage unknown |
| DPO feedback generation | Complete0..40; partial50=461/800 then running | Complete-bank input only; current later coverage unknown |
| Training metrics and snapshots | Six arms, states0..100,606 snapshots | Auxiliary Fan S4 and probability diagnostics, NOT generated text |

This record contains48 complete arm/checkpoint banks, or38,400 generated rows,
plus800 baseline rows. It does not prove all66 banks are now complete. At that
snapshot the common six-arm comparison grid was0..40 every10; do not compare
full-window summaries from different endpoints. Valid partial prefixes can be
cached for annotation, but cannot stand in for the full200-prompt mean because
generation order makes them a nonrandom subset. No interpolation to100.

Original generation location:
`/work/users/y/u/yuhe32/ipo_runs/oracle2_real_20261009_v3_2gpu/generations/<arm>/step_XXXX.jsonl`.
Original candidate location: v3 private_data/scored/train.jsonl; use the exact
audited file, not a reconstruction from public HelpSteer ordering.
Future recovered IPO ordinary/reference banks must be resolved using the v5
generation_recovery.json contract, not by directory-name guessing. Recovery
preparation is not a submission/completion receipt. Complete banks from a
TIMEOUT job remain useful data when individually verified; that does not bypass
the independent full-campaign WR submission guards.

### Important cap and noise limitations

Frozen generation settings: raw prompt format, temperature.8, top_p.95,
max_new_tokens256, four draws per prompt, checkpoint states0,10,...,100. Keep
these settings and original seed identities. Checkpoint seeds depend on step,
prompt and draw, not arm; this permits matched-arm comparisons but does not
make different time points use identical generation randomness.

At the verified step0 banks,43.5% hit the256-token maximum without EOS.
Thus the main estimand is the distribution of these capped outputs, not complete
unrestricted answers. Preserve and label capped text; do not silently regenerate,
extend or drop it. Plot length/cap rates with style curves and report length-
adjusted sensitivity. EOS-only views are selected subpopulations, not unbiased
substitutes. A change in truncation/length composition can mimic a style change.
No101-step claim can be made from11 generated banks or101 numeric snapshots.

## 7. Workload and proposed execution order

Full row counts before exact-text cache reuse:

- Six arms x11 checkpoints x200 prompts x4 draws =52,800 generated observations.
- 2,000 training candidates +800 shared baseline =55,600 main annotation rows.
- 200 additional calibration pilot rows =55,800 total before human review and
  any explicitly approved reliability repetitions. The60 baseline checks are
  already included, not extra generation.

One annotation call returns all seven fields. Cache by exact response bytes,
model revision, tokenizer/template, rubric and inference settings, then join
scores back to every original observation; never deduplicate statistical weights.
Embedding extraction covers the same relevant texts and remains a separate
feature route. These counts are a workload ceiling before cache reuse, not a
validated count of distinct answers available now.

After separate approval, proposed order: read-only inventory/locks; pilot and
resource/quality review; annotate originals and validated generations; fit and
freeze directions on discovery; validate static fit; report all available
generated-output curves and diagnostics. Original/embedding feature extraction
can be scheduled before trajectory plotting, but no positive static-cycle gate
is required for the descriptive output analysis.

Use one GPU at a time for new annotation/embedding work, never exceeding the
account-wideFOUR limit and never overlapping existing protected campaign phases
or unrelated work. Full account/receipt guards must be checked before allocation.
Do not change existing recovery/WR order. No training, additional seeds or new
policy generations are included in this proposal. Missing banks are handled
under the previously prepared recovery protocol only after explicit continuation
and fresh state checks, not by this planning request.

GPU hours are not known: after pilot, estimate unique uncached rows divided by
measured responses/second, include loading/validation overhead, and obtain the
owner's runtime ceiling before the full batch. Do not promise a fixed completion
time based solely on model size. No model weights were downloaded this turn.

## 8. Outputs and execution hold

Proposed deliverables: frozen feature/model/rubric records; prompt split and
data hash manifests; private keyed feature tables; fitted u/B and plane loadings;
held-out/null/uncertainty summaries; all-arm style/distribution/latent plots;
length/cap/missingness diagnostics; optional Fan S4/S_eff appendix; README with
the observed scope and counterexamples. Share code, aggregate numeric results
and figures, never raw private texts, rater payloads, weights or adapters.

Only this planning document and canonical discussion context are updated now.
Approval is still needed to download/deploy models, perform the pilot/human
review, run the annotation/embedding/analysis batch, or resume any remote phase.

## Evidence and related documents

- [Fan plan](../latent-estimation-2026-10-13/README.md) and its execution
  clarifications remain unchanged; this proposal explicitly changes the primary
  dynamic endpoint and plane role, not the frozen experiment.
- [Oracle2 plan](../../../oracle2/campaigns/oracle2_real_20261009/plan_two_gpu.json).
- [Recovery protocol](../../../oracle2/campaigns/oracle2_real_20261009/RECOVERY_20261013.md).
- [Recorded two-generator completion](../../../oracle2/campaigns/oracle2_real_20261009/records/attempt5/two_generators_complete_20261013T2300.json).
- Local private later inspection: outer workspace
  `oracle2_real_launch_20261009/attempt5/heartbeat_20261014T0500.json`; numeric
  summaries only used here. No live allocation or progress inferred from it.
