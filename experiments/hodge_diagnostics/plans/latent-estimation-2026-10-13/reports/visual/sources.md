# Sources and fact sheet: "One skew form, many prompts" (handoff deck)

Deck: `index.html` (single file: `report.html`). Built 2026-10-13 by Claude (Opus 5.5) for Fan Yao; reader Yu He first, Fan second. All sources read on **2026-10-13**; [H] re-read after its 14:01 correction (S3 nulls, S3 decision rule, S5). Paths are relative to the `Iterative_PsiPO` workspace.

Sources
- [H] `paper/reviews/2026-10-08-cross-prompt-coupling-theorems/latent-estimation-plan/HANDOFF_TO_YU_HE_2026-10-13.md` (authoritative where sources differ)
- [D] `paper/reviews/2026-10-08-cross-prompt-coupling-theorems/latent-estimation-plan/draft-v2.md` (in Chinese; Fan's decisions in §6)
- [F] `paper/reviews/2026-10-08-cross-prompt-coupling-theorems/latent-preference/latent_preference_framework.md` (Props L1–L2, Thm L3, §5 numerics; refereed)
- [R] `hodge_diagnostics/reports/2026-09-25/REPORT.md`, section "Cross-prompt coherence under parameter sharing" (lines ~210–254)

Marks: **confirmed** = stated in two or more sources, or a primary statement (theorem / table) of the source that owns it; **single-source** = stated in one planning or meeting-derived source only; **inference** = my own reading or a design choice of the deck.

| # | Claim / number | Source location | Mark | Where shown |
|---|---|---|---|---|
| 1 | Three objects kept apart: fixed latent Ā; per-round estimated panel matrices (only S0's fixed panels used); model latent state θ^(t) | H §1; D §4 S0 | confirmed | slide 2 |
| 2 | Averaging raw response-index matrices across prompts is not legal; the latent fit is the legal aggregation (least-squares pullback to Skew_m) | H §1; D §1 | confirmed | slide 2 |
| 3 | Everything is CPU-only diagnostics on existing data and checkpoints; no new GPU training, no protocol change | H preamble; D §4, §7 | confirmed | slides 2, 6 |
| 4 | Ā ∈ Skew_m on φ(y) = (ψ(y), 1); flow φ(y)ᵀĀφ(y′); intercept row = latent potential (linear score); attribute block Ā_cyc = shared cycle | H §1.1; D §1; F §1 | confirmed | slides 2, 3 |
| 5 | C_a = Φ̃_a Ā_cyc Φ̃_aᵀ depends only on Ā_cyc; K_a = 2 ⇒ C_a = 0 | F Prop L1(iii–iv); H §1.1 | confirmed | slide 3 |
| 6 | Only Ā_cyc is invariant to shifting the attribute origin (gauge) | H §1.1; D §1; F Prop L1 Remark (b) | confirmed | slide 3 |
| 7 | Mean exposed share of a pure latent cycle: 0 (K=2), 0.304 (K=3), 0.509 (K=4), 0.667 (K=6); 200 panels each, random standard-normal 3-D attributes | F §5(2) | single-source (synthetic check) | slide 3 bars, shown as 30%, 51%, 0%, 67% |
| 8 | Matrix entries a, b, c, u₁…u₃ on slide 3 | — | inference (illustration of skew structure) | slide 3, labelled "Illustrative matrix entries" |
| 9 | Estimation: skew-bilinear BT, design row vec(skew(φ_iφ_jᵀ)), m(m−1)/2 unknowns, logistic (logit link) or least squares (identity link), ridge 1e-3 for conditioning only | H S2; D §2; F Prop L2 | confirmed | slide 4 |
| 10 | Continuous features: 60 prompts × 1 pair, dense features, m = 4: design rank 6 of 6, Ā recovered to 2.0·10⁻¹⁶ | F §5(1) | single-source (synthetic check) | slide 4 |
| 11 | One-hot roles + intercept: rank 3 of 6 (3 prompts, 3 role pairs); each observation = Ā_rs + ū_r − ū_s, cycle confounded with potential | F Prop L2(a) [ref], §5(1); H S1 | confirmed | slide 4 |
| 12 | S1 features: embedding first (frozen encoder, pooled PCA d ∈ {8,16}); interpretable second (one judge call per response, 0–10 on ~6–8 style attributes + length); cheap length set for comparison with 09-25; centred on pooled mean | H S1; D §4 S1, §6.1 | confirmed | slides 4, 6 |
| 13 | S_eff = G^{-1/2} S G^{-1/2}; ω̃₁ = largest |Im eig|; its 2-D invariant plane is the shared cyclic direction; weighted by J_a | H S3; D §3; F Thm L3(ii) | confirmed | slide 5 |
| 14 | β* = √(1−α²)/(λω̃₁); compare with DPO β_train = 0.1 ⇒ paper β = 10; IPO β = 1/τ | H S3; D §4 S3 | confirmed | slide 5 |
| 15 | Synthetic: ω̃ = 0.0797 at α = 0.9, λ = 1 ⇒ β* = 5.47 | F §5(3) | single-source (synthetic check) | slide 5 note |
| 16 | Three nulls: within-prompt label shuffle (keeps |C_a| and G, destroys alignment) and per-prompt orientation flip, 500 draws each, 95th pct.; transitive surrogate (single BT score per panel through the same link, refit; the 09-25 control); cross-fitting on prompt halves, prompt-bootstrap CI | H S3 (corrected); D §3 lists shuffle/flip only; R uses the surrogate | confirmed for shuffle/flip/cross-fit; single-source (H, corrected) for the surrogate as an S3 null | slide 5 (right panel schematic), slide 6 S3 card |
| 17 | Histogram shapes and the held-out marker position on slide 5 | — | inference (schematic) | slide 5, labelled "schematic, not data" |
| 18 | S3 at two policies; primary = initial checkpoint's panel probabilities π_a, uniform secondary | H S3 decision rule (corrected) | single-source | slides 5, 6, 9 |
| 19 | S0–S5 inputs, outputs, checks (pipeline panel text) | H §2 S0–S5 (corrected version), §3 | single-source (handoff), consistent with D §4 | slide 6 interaction |
| 19b | S5 attribution: a single BT oracle has zero residual under logit, positive under identity (R summary item 1); single oracle with Ā_cyc under identity but not logit ⇒ link mismatch; single oracle with Ā_cyc under logit ⇒ not BT-transitive in feature coordinates; only the mixture under logit ⇒ heterogeneity; transitive-surrogate refit as artefact control | H S5 (corrected); D §4 S5 (corrected); R Summary 1 | confirmed | slide 6 S5 card |
| 20 | Deliverables: outputs/latent/, REPORT.md, hodge/latent.py next to hodge/coherence.py, tests (recover Ā_cyc to 1e-8 from 60 prompts × 1 pair; K=2 ⇒ C_a=0); six GPU runs and win-rate task untouched | H §3 | single-source | slide 6 note |
| 21 | S4: θ^(t) = J_a-weighted regression of centred sequence-sum log-probs; project onto S3 plane (p_t, q_t); radius, phase, explained share; three outcomes rotation / polarization / noise-stationary; no re-scoring needed | H S4; D §4 S4 | confirmed | slide 7 |
| 22 | Phase-plane and radius/phase sparklines | — | inference (schematic) | slide 7, labelled "Schematic sketches, not data" |
| 23 | 09-25 one-hot coherence table: UltraFeedback 63,966 (model) 0.181 / 0.018 / 0.10 / 0.003 / 0.053; HelpSteer K=4 6,692 (length rank) 0.103 / 0.014 / 0.14 / 0.002 / 0.063; MT-Bench human 160 (model) 0.364 / 0.066 / 0.18 / 0.039 / 0.104; MT-Bench GPT-4 160 (model) 0.237 / 0.097 / 0.41 / 0.024 / 0.119 (per-prompt ω RMS / coherent ω̃ / ratio / shuffle null 95th / transitive surrogate) | R table | confirmed (primary table) | slide 8 bars (one scale, 1300 units per 1.0) |
| 24 | Alignment real (above null) but small and explained by discretization; exact shared dynamics did not cycle; coherence in other features not excluded | R bullets and Caveats; D §4 | confirmed | slide 8 take and note |
| 25 | Decision rule: primary policy = initial checkpoint's π_a; nulls and ω̃₁ on the held-out half (500 draws, 95th pct.); β range per link = the run's actual β (DPO β = 1/β_train, IPO β = 1/τ); signal exists iff held-out ω̃₁ exceeds both null 95th percentiles and the transitive-surrogate value and β* lies in that range; otherwise no visible common cycle in the current two-oracle construction (conditional reading) | H S3 (corrected); D §4 S3 still states the older two-null rule | single-source (H, authoritative; D not yet updated) | slide 9 |
| 26 | Not authorized: no new training runs; no latent cyclic oracle (only if S3 is negative); no ρ-scan this round; no manuscript edits; numbers are diagnostics, not neural stability results | H §4; D §6.3–6.4, §7 | confirmed | slide 9 |
| 27 | Fan's four decisions: both feature sets, embedding first; S4 both J_a variants; round 1 pure latent fit; both links (identity/IPO, logit/DPO) | D §6.1–6.3, 6.5; H S1, S2, S4 | confirmed | slide 9 |
| 28 | If S3 is negative: read HelpSteer-type data as a negative control (concentration without oscillation) or move to a latent cyclic oracle | D §5 | single-source | slide 9 note |
| 29 | Questions to send back early (encoder/judge access; fixed HelpSteer panels with per-checkpoint log-probs; α, λ, β_train) | H §5 | single-source | slide 10 |
| 30 | Current experiment state (500 HelpSteer prompts × 4, two-oracle BT mixture) | D §4 says "meeting report, not verified" | single-source, unverified | slide 2 only as "500" via H S0; not otherwise asserted |

Notes on simplification
- The brief called the pipeline "five steps S0–S5"; the handoff has six (S0, S1, S2, S3, S4, S5). The deck says six.
- Draft-v2 §6 lists five numbered items; item 4 (latent cyclic oracle) is a non-authorization, so the deck shows the other four as "Fan's decisions" and item 4 under "Not in this round".
- The S5 logic (link mismatch / non-BT oracle / heterogeneity, plus the surrogate control) appears only in the interactive panel of slide 6.
- Discrepancy: draft-v2 §4 S3 still gives the older rule (two nulls, "experiment's β range"); the deck follows the corrected handoff.
