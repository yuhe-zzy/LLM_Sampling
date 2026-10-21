# Handoff to Yu He — aligned latent-preference aggregation (feasibility check), 2026-10-13

From: Fan (drafted with Claude; decisions by Fan). Scope: the P0 item of the 10-13 meeting ("define common latent coordinates and a legal aggregation, compute the aggregate cyclic magnitude, decide whether the current data can show a common cyclic signal"), plus the checkpoint projection that turns it into the per-checkpoint metric of Idea A. Everything here is CPU-only diagnostics on data and checkpoints you already have. No new GPU training, no change to the training protocol, no publication.

Background notes (read Sections 0–3 of each; proofs optional): `references/latent_preference_framework.md` (Props L1–L2, Thm L3), `references/partial_sharing_theorem.md`, and the plan `draft-v2.md` in this folder. The 09-25 diagnostic `hodge_diagnostics/reports/2026-09-25/REPORT.md` §"coherence" and `hodge_diagnostics/hodge/coherence.py` already implement the one-hot special case of what follows (model identity / length rank as the shared feature); you extend it to continuous features.

## 1. The objects (keep these three apart)

1. **Latent preference** Ā ∈ Skew_m: a fixed antisymmetric matrix on feature space, φ(y) = (ψ(y), 1), ψ ∈ R^{m−1} prompt-independent attributes. Flow between any two responses: φ(y)ᵀĀφ(y′). Its intercept row is the latent potential (a linear score), its attribute block Ā_cyc is the **shared cycle**. Each prompt's Hodge data are compressions: C_a = Φ̃_a Ā_cyc Φ̃_aᵀ, u_a = Φ̃_a(ū_lat + Ā_cyc ψ̄_a). Only Ā_cyc is invariant to shifting the attribute origin; it is the thing we want.
2. **Per-round estimated panel matrices**: P_ij on the responses the checkpoint generated. Not used in this handoff except for S0's fixed panels.
3. **Model latent state** θ^{(t)}: the checkpoint's logits regressed onto the shared features (S4). This is the Idea A metric.

Aggregating raw response-index matrices across prompts is not legal; the latent fit below is the legal aggregation (it is the least-squares pullback of every panel's flow to Skew_m).

## 2. Steps

### S0. Freeze the data table
Export from the current 500-panel run: `(prompt_id, i, j, P1_ij, P2_ij, Pmix_ij, mix_weights, response_text_i, response_text_j)` for all 6 pairs of each 4-response panel, with the two oracles' probabilities **separately** and the mixed one. Record oracle identities, mixing rule (probability vs hard label), run/config IDs, and which checkpoint generated the responses (the responses are fixed HelpSteer text per the meeting; confirm). Compute flows under both links: identity A = P − 1/2 (IPO) and logit A = log(P/(1−P)) with 0.5 pseudo-counts (DPO). Deliverable: one parquet/CSV + a README line per column.

### S1. Features (two sets, embedding first)
- (b) **Embedding, first**: sentence embedding of each response (any frozen encoder you already have on the server; record the model), pooled PCA to d ∈ {8, 16}; center on the pooled mean. Zero annotation cost.
- (a) **Interpretable, second**: one judge call per response (2000 responses) scoring continuous 0–10 on about 6–8 style attributes: concise, detailed, confident/hedged, polite, structured (lists/headers), formal, example-giving, plus length in tokens. Record the judge model and prompt verbatim. Center on the pooled mean.
- Also keep the cheap set (length, length rank, sentence count) so S2 can be compared with the 09-25 one-hot result.
Feature vectors must vary continuously across prompts on the same attribute; pure one-hot role labels plus intercept confound cycle and potential (Prop L2(a), corrected).

### S2. Fit Ā (pure latent, no prompt-specific residual this round)
For each feature set × link × {oracle 1, oracle 2, mixed}: logistic regression (logit link) or least squares on flows (identity link) with design row vec(skew(φ_iφ_jᵀ)) and target the label/flow; parameters = Skew_m, m(m−1)/2 unknowns; small ridge (1e-3) only for conditioning. Report: Ā_cyc, latent cyclic share ‖Ā_cyc‖_F²/‖Ā‖_F², design rank vs m(m−1)/2, and per-prompt exposed share ‖C_a‖_F²/‖A_a‖_F² with C_a = Φ̃_aĀ_cycΦ̃_aᵀ. The ρ-scan (shared + prompt-specific residual with ridge ρ, the partial-sharing construction) is **second round**, not now.

### S3. The shared cyclic direction and its null
At two policies — uniform on each panel, and the initial checkpoint's actual panel probabilities π_a (sequence-sum, as logged) — compute
G = Σ_a w_a Φ̃_aᵀ J_a Φ̃_a, S = Σ_a w_a G_a Ā_cyc G_a (G_a = Φ̃_aᵀ J_a Φ̃_a, J_a = diag(π_a) − π_aπ_aᵀ), S_eff = G^{-1/2} S G^{-1/2}.
Report ω̃_1 = largest |Im eig(S_eff)|, its 2-D invariant plane (two vectors in attribute space; print loadings on the interpretable attributes when set (a) is available), and the latent frontier β* = √(1−α²)/(λ ω̃_1) for the α, λ of the run; compare with the run's effective β (DPO β_train = 0.1 ⇒ paper β = 10; IPO β = 1/τ).
Nulls and validation (mandatory, the meeting's "do not select noise" condition):
- within-prompt label shuffle (keeps every |C_a| and G; destroys alignment) — 500 draws, report the 95th percentile of ω̃_1;
- random orientation flip per prompt (A_a → ±A_a) — same;
- transitive surrogate (single BT score per panel pushed through the same link, then refit) — the ω̃_1 a purely transitive order produces by discretization/link alone (09-25 used this control);
- cross-fitting: split prompts in halves; fit Ā_cyc and the plane on one half, evaluate ω̃ and cyclic share on the other; report held-out values with a prompt-level bootstrap CI.
**Decision rule** (stated precisely): primary policy = the initial checkpoint's actual panel probabilities (uniform reported as secondary); nulls and ω̃_1 both evaluated on the held-out half of the cross-fit (500 draws each, 95th percentile); β range per link = the β values actually used in the run (DPO: paper β = 1/β_train; IPO: β = 1/τ), which you list in the report. A measurable common cyclic signal exists if held-out ω̃_1 exceeds both null 95th percentiles and the transitive-surrogate value, and β* lies inside that β range. Otherwise the current two-oracle construction does not supply a visible common cycle (conditional reading from the meeting).

### S4. Checkpoint projection (the Idea A metric)
For every saved checkpoint t and the same 500 panels, take the sequence-sum log-probabilities of the 4 fixed responses, center within panel (x̃_a^{(t)}), and solve θ^{(t)} = argmin_θ Σ_a w_a ‖Φ̃_a θ − x̃_a^{(t)}‖²_{J_a}. Report both variants, as Fan decided: J_a from the initial checkpoint (fixed metric) and J_a from checkpoint t (moving metric). Project θ^{(t)} onto the S3 plane → (p_t, q_t); plot radius and phase over t, plus the explained share Σ‖Φ̃_aθ‖²_{J_a} / Σ‖x̃_a‖²_{J_a}. Three outcomes are distinguishable: rotation (phase advances monotonically, radius bounded), polarization (radius grows, phase freezes), noise/stationary (small radius, random phase). No re-scoring is needed, only panel log-probs, which the pipeline already logs (PROTOCOLS.md: sequence-sum panel probabilities and panel mass per round). Do this for both links' Ā and both feature sets.

### S5. Where the cycle comes from
Compare Ā_cyc from oracle 1, oracle 2, and the mixture, under both links. A single Bradley–Terry oracle has zero residual under the logit link and a positive one under the identity link (09-25 report, summary item 1), so: (i) if a single oracle shows Ā_cyc ≠ 0 under identity but ≈ 0 under logit, the source is link mismatch (a transitive BT oracle read through the IPO link); (ii) if a single oracle shows Ā_cyc ≠ 0 under logit, that oracle is not BT-transitive in the feature coordinates (its own scoring is nonlinear/inconsistent); (iii) if only the mixture shows Ā_cyc ≠ 0 under logit, the source is heterogeneity between the two oracles (probability mixing is not logit-additive). Also run a transitive-surrogate control (fit a single BT score per panel, push it through the same link, refit Ā): any Ā_cyc it produces is discretization/link artefact, not a shared cycle. This decides which source the paper should name. [corrected 2026-10-13 after the text-report check]

## 3. Deliverables
- `outputs/latent/`: S0 table; S2 tables (one row per feature set × link × oracle); S3 table with ω̃_1, nulls, held-out CI, β*, loadings; S4 plots (radius/phase vs t, explained share) for each variant; S5 comparison table. A short `REPORT.md` with the decision-rule verdict and the exact commands/configs.
- Code as a module next to `hodge/coherence.py` (e.g. `hodge/latent.py`): reuse `core.flow`, `core.hodge`, `core.softmax_jacobian`; a `permutation_null`-style function for both nulls; tests on a synthetic Ā (recover Ā_cyc to 1e-8 from 60 prompts × 1 pair with dense features, and the K=2 ⇒ C_a=0 identity).
- Please keep the existing six GPU runs and the per-step win-rate task untouched; they are parallel, not replaced.

## 4. What this does not authorize
No new training runs; no latent cyclic oracle design (only discussed if S3 is negative); no ρ-scan this round; no manuscript edits. Numbers from this handoff are diagnostics of data and checkpoints, not neural stability results.

## 5. Questions to send back early
- Which encoder/judge is available on the server without new access? Any constraint on calling an external judge for 2000 responses?
- Confirm that the 500 panels use fixed HelpSteer responses (not regenerated per round) and that panel log-probs exist for every checkpoint.
- The α, λ, β_train of the run whose checkpoints will be projected.


---
Shared-copy note (2026-10-13): read `EXECUTION_CLARIFICATIONS.md` before execution. It takes precedence for statistical tests, attribution and scope. This is a candidate research plan/background, not a new training authorization or certified manuscript theorem. Source-local companion scripts mentioned in background are not included in this plan bundle.
