# Sources: Aligned latent-preference aggregation: a feasibility check

One line per claim or number shown in the report: the claim, where it comes from, and its status: **confirmed** (primary source read; plan statements are confirmed as statements of the plan, not as facts about data), **single-source** (participant report in the meeting minutes, not verified against code or data), or **inference** (ours, added in this report). All files accessed 2026-10-13. Paths relative to the report folder `latent-estimation-plan/text-report/`.

Short names: **H** = `../../HANDOFF_TO_YU_HE_2026-10-13.md`; **D** = `../../draft-v2.md`; **L** = `../../references/latent_preference_framework.md`; **PS** = `../../references/partial_sharing_theorem.md`; **R** = `../../../../reports/2026-09-25/REPORT.md`; **M** = private project meeting minutes dated2026-10-13 (not included in this share bundle).

## Scope, authorship, context

| Claim | Source | Status |
|---|---|---|
| From Fan, drafted with Claude, decisions by Fan; date 2026-10-13 | H header | confirmed |
| Scope = the P0 item of the 10-13 meeting plus the checkpoint projection (Idea A metric) | H header; M §7 todo row 2 | confirmed |
| CPU-only diagnostics on data and checkpoints Yu He already has; no new GPU training, no training-protocol change, no publication | H header; D §4, §7 | confirmed |
| Background notes to read: Sections 0–3 of the latent-preference note (Props L1–L2, Thm L3), the partial-sharing note, draft-v2 | H header | confirmed |
| 09-25 coherence analysis and `hodge/coherence.py` implement the one-hot special case (model identity / length rank); extend to continuous features | H header; D §4 | confirmed |
| Current experiment: HelpSteer prompts, four responses each, original attribute ratings not used; labels from two newly designed oracles, mixed (equal weight and 0.6/0.4 tried/discussed); 1 explicit "cyclic matrix" among 500 prompts | M §1; D §4 ("会议报告，未核验") | single-source |
| Printed ‖C‖_2: matrices without explicit cycles can have large C; largest C groups do not show stronger oscillation; no reliable full numeric table | M §1 | single-source |
| Two explanations Fan proposed (noise vs cancellation under shared parameters) | M §1 | confirmed (as minutes) |
| Checkpoints saved every step; per-step win rate not computed this round | M §4; D §4 | single-source |
| Oracle identities, mixing rule, full parameters and run IDs not fixed in the transcript | M §1 | confirmed (as minutes) |
| Meeting quotes (Chinese) in §1, §2, §3 of the report | M §1 heading, §2 (实验提议 item 2; operational metric paragraph), §3 (conditional reading; 尚未解决的实施点), §7 todo P0 rows | confirmed |
| Meeting caveat: explicit cycle, Hodge C ≠ 0, large ‖C‖_2 and unstable outer dynamics are different levels; "identity=IPO, logit=DPO" is a spoken shorthand | M §1 "技术表述的保留项" | confirmed |
| Yu He is "usually Speaker 1 in Otter", identified from context | M front matter | confirmed (as minutes) |

## Objects and definitions

| Claim | Source | Status |
|---|---|---|
| Three objects kept apart: latent preference Ā; per-round estimated panel matrices P_ij; model latent state θ^(t) | H §1; M §2 ("不能把它们当作同一个东西") | confirmed |
| φ(y) = (ψ(y), 1), ψ ∈ R^{m−1} prompt-independent attributes; flow φ(y)ᵀĀφ(y′) | H §1; D §1; L §1 | confirmed |
| Intercept row = latent potential ū_lat (a linear score); attribute block = latent cycle Ā_cyc | H §1; L §1 | confirmed |
| C_a = Φ̃_aĀ_cycΦ̃_aᵀ; u_a = Φ̃_a(ū_lat + Ā_cycψ̄_a); A_a = Φ_aĀΦ_aᵀ | H §1; L Prop L1 | confirmed |
| K_a = 2 ⇒ C_a = 0 | L Prop L1(iii); D §1 | confirmed |
| Gauge: ψ → ψ + t maps (ū_lat, Ā_cyc) → (ū_lat − Ā_cyc t, Ā_cyc), all A_a unchanged; only Ā_cyc gauge-invariant; panel centroid = latent potential | D §1; L Remark (b) | confirmed |
| Raw response-index matrices cannot be averaged; the latent fit is the least-squares pullback of every panel's flow to Skew_m | H §1; D §1; M §2 | confirmed |
| Prompt-specific part E_a; "addition" on the flow scale Ψ(P) − Ψ(1/2), not on probabilities | D §1 | confirmed |
| Random panels in a 3-D attribute space, pure latent cycle: mean cyclic share 0.000 (K=2), 0.304 (K=3), 0.509 (K=4), 0.667 (K=6), 200 panels each; summary rounds to 30%, 51%, 67% | L §5(2), L §0 item 2 | confirmed |
| Shares fall when the attribute distribution has a nonzero mean | L §5(2) | confirmed |

## Estimation (S2, Prop L2, ρ)

| Claim | Source | Status |
|---|---|---|
| Model P(i ≻ j | a) = σ(⟨Ā, M_ij⟩), M_ij = (φ_iφ_jᵀ − φ_jφ_iᵀ)/2; m(m−1)/2 parameters; logistic (logit) or least squares (identity) | D §2; L Prop L2; H S2 | confirmed |
| Ā identified iff span{M_ij} = Skew_m; otherwise only its restriction; consistency needs E[vec M vec Mᵀ] ≻ 0 | L Prop L2 | confirmed |
| One-hot roles + intercept: each observation = Ā_rs + ū_r − ū_s, cycle and potential confounded; features must vary continuously across prompts on the same attribute | L Prop L2(a); H S1; D §2 | confirmed |
| Dense features, 60 prompts × 1 pair: design rank 6 of 6, Ā (4×4) recovered with max error 2.0·10⁻¹⁶ | L §5(1) | confirmed |
| Role-plus-intercept toy: design rank 3 of 6 | L §5(1) | confirmed |
| Ridge 1e-3 only for conditioning | H S2 | confirmed |
| Full model σ(φ_iᵀĀφ_j + ε_{a,ij}), ε skew per prompt, group-lasso/ridge penalty = partial-sharing ρ; ρ → ∞ = pure latent model; ρ-scan second round | D §2, §6 item 3; H S2, §4 | confirmed |
| Partial-sharing policy class x_a = s + t_a, penalty (ρ/2)Σ_a w_a‖t_a‖²; ρ = 0 isolated, ρ → ∞ pooled common-role model (X1); coupling (J_a + ρΠ)⁻¹J_a | PS §0 items 1–2, §1 | confirmed |
| Latent version: features [φ_attr; 1_group(a) ⊗ φ_attr] with ridge on the second block; adds ρP to the Hessian | L §0 item 6; L §4 Remark (a) | confirmed |
| Parameter counts: d = 8 ⇒ m = 9 ⇒ 36; d = 16 ⇒ m = 17 ⇒ 136; interpretable set with 7–9 attribute coordinates ⇒ m = 8–10 ⇒ 28–45 unknowns | derived from H S1–S2 (m = d + 1) | inference (derived) |
| 500 panels × 6 pairs = 3,000 labelled pairs per oracle | derived from H S0 | inference (derived) |
| 2000 responses (= 500 × 4) | H S1 | confirmed |
| 1,500 pairs per cross-fitting half (half of 3,000) | derived | inference (derived) |
| Range 0.014–0.097 of the four coherent ω̃ values (tl;dr) | derived from R table | inference (derived) |

## Shared cyclic direction, frontier, nulls (S3, Thm L3)

| Claim | Source | Status |
|---|---|---|
| G = Σ w_a Φ̃_aᵀJ_aΦ̃_a; S = Σ w_a G_a Ā_cyc G_a; J_a = diag(π_a) − π_aπ_aᵀ; S_eff = G^{-1/2}SG^{-1/2} | H S3; L Thm L3(i)–(ii) | confirmed |
| Multipliers α ± igω̃_j; stable iff α² + g²ω̃_max² < 1 (g = βλ); with common Fisher S = GĀ_cycG | L Thm L3(ii)–(iii); PS §1 (g = βλ) | confirmed |
| Frontier β* = √(1−α²)/(λω̃_1); latent frontier α² + (βλω̃_1)² < 1 | H S3; D §3 | confirmed |
| Not PCA: principal rotation plane of a skew matrix; weighted by J_a ("an attribute counts only if it varies on many prompts and the policy still has mass there") | D §3 item 2 | confirmed |
| Two policies: uniform per panel; initial checkpoint's π_a (sequence-sum, as logged) | H S3 | confirmed |
| DPO β_train = 0.1 ⇒ paper β = 10; IPO β = 1/τ | H S3; D §4 | confirmed |
| Nulls: within-prompt label shuffle (keeps every |C_a| and G); random orientation flip A_a → ±A_a (expected Ā_cyc = 0, X4 random-direction model); 500 draws each; 95th percentile | H S3; D §3 | confirmed |
| Cross-fitting on prompt halves; held-out ω̃ and cyclic share; prompt-level bootstrap CI | H S3; D §3 | confirmed |
| Decision rule (corrected handoff): primary policy = initial checkpoint's π_a (uniform secondary); ω̃_1 and nulls on the held-out half (500 draws, 95th pct.); β range per link = β values the run used (DPO β = 1/β_train, IPO β = 1/τ); signal if held-out ω̃_1 > both null 95th pcts and the transitive-surrogate value, and β* in that range | H S3 (re-read 2026-10-13 after correction) | confirmed |
| Transitive surrogate as third S3 null (single BT score per panel, same link, refit) | H S3 (corrected) | confirmed |
| Latent toy: S_eff eigenvalues ±0.0797i and 0; α = 0.9, λ = 1 ⇒ β* = 5.47 | L §5(3) | confirmed |
| Latent toy dynamics: bounded nonconvergent for 8 ≤ β ≤ 16; concentration at β = 24 although u_a = 0 | L §5(3) | confirmed |
| Latent cycle necessary not sufficient; nonvanishing aggregate induced potential → concentration | L §0 item 5; Thm L3(iv) | confirmed |
| 09-25 ω̃ = max |Im eig(G⁺S)| with one-hot features; same spectrum as S_eff when G ≻ 0 | R coherence formula; similarity is ours | confirmed / inference (similarity remark) |

## Background numbers from the 09-25 report (coherence section), quoted exactly

| Claim | Source | Status |
|---|---|---|
| UltraFeedback 63,966 (model): per-prompt ω RMS 0.181; coherent ω̃ 0.018; ratio 0.10; shuffle null 95th 0.003; transitive surrogate 0.053; soft BT 0.0016 | R coherence table | confirmed (report read; not recomputed) |
| HelpSteer K = 4, 6,692 (length rank): 0.103; 0.014; 0.14; 0.002; 0.063; 0.0015 | R coherence table | confirmed (not recomputed) |
| MT-Bench human, 160 (model): 0.364; 0.066; 0.18; 0.039; 0.104; – | R coherence table | confirmed (not recomputed) |
| MT-Bench GPT-4, 160 (model): 0.237; 0.097; 0.41; 0.024; 0.119; – | R coherence table | confirmed (not recomputed) |
| Uniform policies, logit link, 0.5 pseudo-votes; null shuffles labels within each prompt | R coherence section | confirmed |
| Alignment beyond chance, shuffle-null p = 0.002 on every dataset; aligned part 10–41% of per-prompt magnitude; transitive surrogate reproduces and exceeds it; soft BT shrinks it tenfold | R Summary item 7; coherence bullets | confirmed |
| Exact dynamics of the maximally shared policy converge in every configuration, real and shuffled labels | R Summary item 7; coherence bullets | confirmed |
| A single BT score gives exactly zero residual under logit and a small positive one under identity; two sources (link mismatch, heterogeneity) | R Summary item 1; D §4 S5 | confirmed |
| Hard labels on a mean score through the IPO identity link give share 0.10 on every strictly ordered four-response panel; make DPO logit targets infinite | R Summary item 6 | confirmed |
| Coherence analysis used only identity features; other features not excluded | R Caveats | confirmed |

## Steps, deliverables, decisions, boundaries

| Claim | Source | Status |
|---|---|---|
| S0 columns `(prompt_id, i, j, P1_ij, P2_ij, Pmix_ij, mix_weights, response_text_i, response_text_j)`, 6 pairs per panel, metadata list, two links, 0.5 pseudo-counts; deliverable parquet/CSV + README | H S0 | confirmed |
| S1 embedding first, PCA d ∈ {8, 16}, centered; interpretable 6–8 attributes 0–10 + length; judge and prompt verbatim; cheap set (length, length rank, sentence count) | H S1; D §6 item 1 | confirmed |
| Draft-v2 S1 suggested Nemotron or a GPT-class judge and 5–8 attributes | D §4 S1 | confirmed |
| S2 report list (Ā_cyc, latent cyclic share, design rank, exposed share) | H S2 | confirmed |
| S3 report list (ω̃_1, invariant plane, loadings, β*, comparison with run β) | H S3 | confirmed |
| S4 regression; fixed and moving J_a; (p_t, q_t), radius, phase, explained share; three outcomes; only panel log-probs needed (PROTOCOLS.md); both links × both feature sets | H S4; D §4 S4, §6 item 2 | confirmed |
| Draft also reports the S4 residual share ‖x̃_a − Φ̃_aθ‖ | D §4 S4 | confirmed |
| S5 logic (corrected handoff): single BT oracle zero residual under logit, positive under identity; (i) single-oracle Ā_cyc ≠ 0 under identity, ≈ 0 under logit ⇒ link mismatch; (ii) single-oracle Ā_cyc ≠ 0 under logit ⇒ oracle not BT-transitive in feature coordinates; (iii) mixture-only under logit ⇒ heterogeneity (mixing not logit-additive); transitive-surrogate control | H S5 (corrected 2026-10-13) | confirmed |
| Deliverables `outputs/latent/`, `REPORT.md`; code `hodge/latent.py` reusing `core.flow`, `core.hodge`, `core.softmax_jacobian`; `permutation_null`-style; tests (Ā_cyc to 1e-8 from 60 prompts × 1 pair; K = 2 ⇒ C_a = 0) | H §3 | confirmed |
| Keep six GPU runs and per-step win-rate task untouched | H §3 | confirmed |
| Not authorized: training runs, latent cyclic oracle (only if S3 negative), ρ-scan, manuscript edits; numbers are diagnostics | H §4; D §6 item 4 | confirmed |
| Fan's five decisions | D §6 | confirmed |
| Not in plan: no 09-25 result as this plan's result; Ā estimation error left for later | D §7 | confirmed |
| Decision tree: signal + rotation → Section 7 strongest empirical layer; signal + stationary → capacity (small ρ), LoRA overlap, shared-LoRA vs per-group adapters; no signal → latent cyclic oracle or HelpSteer as "concentration without oscillation" negative control | D §5 | confirmed |
| Questions to send back early (encoder/judge; fixed responses and log-probs; α, λ, β_train) | H §5 | confirmed |

## Our additions (confirmed by the coordinator 2026-10-13 unless noted)

| Claim | Status |
|---|---|
| Rationale sentence in report §4: a nonlinear transitive score can leak into a fitted bilinear Ā_cyc; shuffle, flip and cross-fit do not remove it; the transitive surrogate measures it | inference (ours), now part of the plan via H S3/S5 |
| Former "points to check" (a) S5 wording and (c) decision-rule inputs: resolved by the corrected handoff and removed from the report | — |
| Report-level wording, captions, labels, derived numbers, mixture split pill | inference (editorial), confirmed |
