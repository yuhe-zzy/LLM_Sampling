# Execution clarifications — take precedence over the original drafts

2026-10-13. Fan requested both embedding and newly judged interpretable features.
This note resolves implementation/interpretation ambiguities before Yu He runs
the plan. No results from this aligned analysis are claimed yet.

## 1. Freeze inputs and scope

Use the existing500-panel train bank and preserve exact candidate order. Freeze
prompt-disjoint seed0 discovery/validation halves (250/250) before inspecting
signal. Both halves already participated in training the LLM: this is validation
of feature/preference estimation and axis selection, not unseen-prompt model
generalization. Reuse the same splits across feature sets and links.

First round is pure latent fitting with small conditioning ridge; NO rho-scan,
NO new latent oracle, NO new training, NO modifications to frozen campaigns.
Two-oracle components are necessary for S5 and cannot be inferred from Pmix.
Baseline static S2/S3 should precede optional S4 checkpoint behavior analysis.

## 2. Feature choices and costs

Embedding: use an available frozen sentence encoder, record model revision,
normalization, pooling, precision and truncation/chunking. Long answers must not
be silently truncated. Fit pooled mean/PCA on discovery responses only, then
freeze the transform for validation. Report d=8 and16. Zero annotation fee
does not mean zero CPU time or permission for an additional GPU job.

Interpretable: one structured judge annotation per response (2000 responses)
for concise/detailed/confident-or-hedged/polite/structured/formal/example-giving
attributes on a common0–10 rubric; token length is deterministic. Provide exact
rubric, model, missing-value policy and cost/resource estimate; obtain the
owner-confirmed ceiling/channel before a paid batch or new resource allocation.
Existing five HelpSteer labels are an optional separately named cheap control,
not a replacement for newly judged style attributes. Do not overwrite oracle
preference labels with either feature set.

Freeze feature scales using discovery data and report them. The coefficient
energy share ||A_cyc||_F²/||A||_F² depends on feature scaling and the intercept
convention; it is descriptive, not an invariant cross-feature comparison.
Also report predictive fit and exposed panel-flow energies on held-out panels.

## 3. S2 fit and continuous probabilities

Identity: least squares to Pij-1/2. Logit: explicitly distinguish least squares
to logit(Pij) from logistic cross-entropy with soft targets Pij. The handoff's
logistic model can be the primary estimator; disclose objective, regularizer
normalization and numerical tolerance. Ridge1e-3 is a conditioning choice,
not permission to hide rank deficiency; report rank/conditioning and a small
sensitivity check if it affects the result.

Continuous oracle probabilities have no implicit sample count. Do not apply
the draft's0.5 pseudo-count unless an effective count model is supplied.
Interior probabilities admit direct logits; report endpoint/roundoff handling.

## 4. S3 nulls and held-out checks

Arbitrary shuffling of six edge labels does NOT preserve a panel's Hodge norm.
For the stated norm-preserving alignment null use candidate conjugation
A_a -> Q_a A_a Q_a^T while feature rows remain fixed; this preserves ||C_a||.
Specify whether policy probabilities are permuted or held fixed, and the
corresponding null question. Do not claim it preserves all geometry unless the
policy is transformed consistently. Random orientation flips use A_a -> +/-A_a
(soft probabilities P -> P or1-P) with a separately documented fixed geometry.

Freeze discovery feature transforms and repeat the declared fit/plane-selection
pipeline within each null draw. Evaluate the null statistic on held-out panels;
use500 draws each, record RNG seeds and95th percentiles. A geometry-only held-out
omega computed using the same discovery-fitted A does not by itself validate
held-out preferences. Include held-out observed-flow prediction, a potential-only
baseline, and a fixed-plane observed-Hodge alignment statistic, with its exact
formula documented before the test. Plane orientation/sign should be fixed on
discovery data, not selected again on the validation trajectory.

Bootstrap prompts, not the3000 edges as independent observations. State whether
the interval conditions on a frozen fit/plane or refits the complete estimator.
Treat dimensions/feature choices as exploration or disclose multiplicity.

## 5. Attribution and dynamics limits

Single BT scores give additive per-panel logit flow and zero exact per-panel
Hodge residual. A nonzero fitted latent cycle can nevertheless arise from
feature misspecification: it does NOT prove an oracle is nontransitive.
Use per-panel additive/logit checks, potential-only prediction and transitive
surrogate refits before attributing cycle to link mismatch or oracle mixture.
Identity residuals of BT probabilities need not be zero, but need not always
be strictly positive either. Report what is measured rather than this shorthand.

beta*=sqrt(1-alpha²)/(lambda*omega) is the realizable linear-feature local
frontier. Uniform/initial-policy evaluations are diagnostic proxies, not a
certificate at a verified nonlinear LLM fixed point. Record actual alpha,
lambda, link and effective paper beta from the frozen run config/implementation;
the DPO0.1 example in the draft is not a statement of this campaign's settings.
Compare the run gain ratio beta/beta*, not exact membership in a singleton beta
range. Separate (a) measurable shared signal from (b) gain reaching the proxy
frontier. Failure of either condition is not proof of no semantic cycle.

S4 phase/radius patterns are descriptive unless a declared periodicity/noise
test supports them. Do not label jitter as a cycle or call fixed-panel features
open-generation behavior. Keep raw sequence-sum policy and relative-reference
weights separate.

## 6. Return early if inputs are missing

The local public raw HelpSteer data matched all500 prompt keys and the original
raw-source hash. A default reconstruction did not reproduce the frozen candidate
source hash, so candidate ordering was NOT guessed. Yu He should use the exact
existing private train bank on his host. Report embedding availability and judge
budget before proceeding; missing files are not permission to create new data.
