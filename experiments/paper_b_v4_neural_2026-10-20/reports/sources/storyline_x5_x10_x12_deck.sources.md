# Fact sheet: storyline deck (2026-10-19)

The deck tells the proposed JMLR storyline for Paper B to a reader who does not know the paper. Every number on a slide traces to a row here. Paths are relative to the Iterative_PsiPO root.

**Status values**
- confirmed: read from a scored output or recomputed.
- ours: our derivation, checked numerically.
- schematic: an illustration, not data.
- my reading: the storyline itself.

**Source abbreviations**
- V4 = `reviews/2026-10-17-v4-neural-validation/visual/sources.md` (v4 deck fact sheet)
- CT = `reviews/2026-10-18-controls-neural/shared/sources.md`
- X19 = `reviews/2026-10-18-dpo-inner-neural/shared/sources.md`
- ST = `reviews/2026-10-19-jmlr-storyline/shared/sources.md`
- Data for the figures: `visual/build/data.json` (from `build/prep.py`).

| id | slide | claim | value | source | status |
|---|---|---|---|---|---|
| D1 | 3, 5 | The running example | panel "Suggest a name for a pet golden retriever." with Golden Max / Golden Maxime / Golden Poochie / Golden Prodigy; Qwen2.5-0.5B-Instruct | `reviews/2026-10-17-v4-neural-validation/shared/examples_series.json` | confirmed |
| D2 | 4 | DPO's β = 0.1 is the paper's β = 10 | the paper's β is the inverse of the KL coefficient | `jmlr_main.tex`, §5 ("A DPO-style coefficient β_DPO = 0.1 corresponds to β = 10") | confirmed |
| D3 | 4 | Recipes set α differently | fixed SFT anchors; per-round replacement; EMA | paper §2 (Related Work, 133d64c) | confirmed (paper) |
| D4 | 5 | Three behaviours of one prompt | full refresh with hard reward-model labels (H3_a1_b1, α = 1, β = 1): one name takes over. α = 0.8, β = 1.6 with soft labels (H1_a08_b16): settles. Planted cycle at α = 0.2, β = 9.6 (H6_B12_a02): keeps rotating; the large step, not the small α, causes it (D9c). | examples_series.json (labels and verdicts as recorded there) | confirmed |
| D5 | 6 | The score/cycle split | Hodge decomposition after the link. One reward model gives scores only; juries, annotator mixtures and the identity link create a cycle part. | paper §4, Prop. `cyclic-sources` | confirmed (paper); picture schematic |
| D6 | 7 | The one-line recursion | x_{t+1} = αx_t + (1−α)r + βu + β(1−λ)Cπ_0 + βλCπ_t; the constant β(1−λ)Cπ_0 is folded into the score push on the slide | paper eq. (master recursion), Theorem 1 | confirmed |
| D7 | 8 | Same B, same endpoint | Final entropy 0.782 / 0.780 / 0.784 / 0.797 for (α, β) = (0, 8) / (0.5, 4) / (0.8, 1.6) / (0.9, 0.8), all B = 8. 1.085 for (0.5, 1), B = 2. Largest pairwise KL among the four B = 8 runs: 0.0076. | V4 C-H1_*, C-pair | confirmed |
| D8 | 9 | Interactive three-answer example | scores (Δ, 0, −Δ), cycle 1.5·R, λ = 1, uniform reference; Δ = 0.3 (majority cycle), 0.9, 1.2 (Condorcet winner for Δ > 0.75) | paper Figure 1 (fig:outcomes) | schematic; computed live |
| D9 | 10 | Oscillating share of 32 cycle panels at B = 12 | α = 0.2: 1.00 (predicted 1.00). 0.5: 0.94 (0.91). 0.8: 0.00 (0.00). 0.9: 0.00 (0.00). | V4 G-H6_* | confirmed |
| D9b | 10 | Oscillating share at a fixed step β = 4 | α = 0.5 (B = 8): 0.00 (predicted 0.00). α = 0.9 (B = 40): 1.00 (1.00). α = 0.95 (B = 80): 1.00 (1.00). | V4 G-H5_* | confirmed |
| D9c | 5, 10 | Why the α = 0.2 panel rotates | At B = 12, β = (1 − α)·12 = 9.6. The rotation per round is γ = (1 − α)γ_B ≈ 0.8 × 1.96 ≈ 1.6 > 1, so α² + γ² > 1. At α = 0.9, γ ≈ 0.2 and α² + γ² ≈ 0.85. | paper Thm 12, Prop. 14; D10 | derived |
| D10 | 10 | Stability threshold | α_loc = (γ² − 1)/(γ² + 1); median over the balanced panels 0.586 (range 0.43–0.62) | `reviews/2026-10-18-jmlr-review/theory/onset_predictions.json` | confirmed |
| D11 | 10 | Credit for the edge | α² + γ² < 1 extends the 2×2 EWA condition of Pangallo et al. 2022 | `reviews/2026-10-18-jmlr-review/lit_positioning.md` C1 | single (audit) |
| D12 | 11 | Price of ordinary refresh | best α = γ²/(1+γ²); radius γ/√(1+γ²); rounds per e-fold 2/ln(1 + 1/γ²) ≈ 2γ² + 1. Grid: 8.96 at γ = 2, 129 at γ = 8. | paper Prop. 14; ST R1, R2 | confirmed |
| D13 | 12 | Lagged leash | reference π_{t−1}^α π_ref^{1−α}, best at ν = α; rounds per e-fold 1/ln(1/s*), s* = (√(1+γ²) − 1)/γ, ≈ γ. Grid: 2.18 at γ = 2, 8.05 at γ = 8 (16× fewer); lagged sampling ≈ 2γ (15.8 at γ = 8). | paper Theorem 22; ST R1, R2 | ours |
| D14 | 12 | Credit | the gap mirrors GDA vs extragradient / optimism / negative momentum in games: Mokhtari et al. 2020; Azizian et al. 2020; Ibrahim et al. 2020; Gidel et al. 2019 | ST L1–L4 | recalled |
| D15 | 13 | Planted panels, B = 12 | Rounds to median KL < 0.001: 19 (ordinary α = 0.8) vs 7 (lag α = 0.4). Rounds to KL < 0.01: 11 vs 4. Trained model, ordinary α = 0.8: KL < 0.01 at round 17, floor about 0.002. γ median 1.96. | ST R4–R8 | confirmed (population); trained model confirmed |
| D16 | 14 | Interactive race | closed forms of D12, D13 with a step cap; regimes max(B/β̄, 2γ²) vs max(2B/β̄, γ) | ST R3 | ours |
| D17 | 15 | Knob verdicts | H4 hard-label DPO; E1 judge bias (α′ = α + βε); E2 entropy bonus; E3′ refresh, then anchor; E4 replay; E5 raise α, keep B; X1 DPO on cyclic labels; X10 proposed | V4 F-*, CT E1–E5, X19 X1c | confirmed |
| D18 | 16 | Boundary | E3: 16 of 64 panels end more than KL 0.1 from the target. E6: KL 0.052 (weight average) vs 0.016 (random recent checkpoint). X9: decisive panels oscillating 1/32 → 20/32 from k = 40 to k = 1. | CT E3-dr2, E6; X19 X9b | confirmed |
| D19 | 17 | Evidence grid | H1–H4, 1.5B, 512 pairs pass; H5–H6 pass after logged measure changes; E5, E3′, E1, E2, E4, X1 pass; E3, E6, X9, 32 pairs fail; X10 proposed | V4, CT, X19 | confirmed |
| D20 | 17 | Natural feedback | 1 of 130 jury panels lacks a Condorcet winner; 0 of 1,200 SPPO panels has a majority three-cycle | V4 J1, J2 | confirmed |
| D21 | 2, 18, 19 | The storyline, title, outline, decisions; X10 cost about 0.7 VM h, about $5; X5 + X6 about 4 VM h | — | ST; `reviews/2026-10-19-jmlr-storyline/text/report.html` | my reading; costs approximate |

## Update 2026-10-19 (evening): mixed criteria

Two slides were added after the evidence slide (new slides 18 and 19); the paper and decision slides are now 20 and 21. Source abbreviation: MX = `reviews/2026-10-19-mixed-criteria-cycles/shared/sources.md`, the mixed-criteria report's fact sheet. The generator reads the numbers from that study's `shared/facts.json`, `analysis/onset_scaling.json` and `realdata/results/probe.json`.

| id | slide | claim | value | source | status |
|---|---|---|---|---|---|
| D22 | 18 | Three transitive judges in your rotation pooled three ways (m = 1) | each cycle edge 0.53 (uniform mix), 0.73 (owner routing, c = 1), 0.88 (salience routing, reverse direction) | MX T-M4 | ours |
| D23 | 18 | Largest DPO β that cycles, fixed reference, three answers, uniform reference, λ = 1 | β_DPO < 4(p̄ − ½)/√3; at m = 1: 0.063 (c = 0) to 0.53 (c = 1); dots checked 16 of 16 | MX T-M5b, T-M5c | ours |
| D24 | 18 | Covariance identity and order | pooled cycle = Σ_k 3 Cov(w_k, A^k) for transitive sources; slopes 0.98 (pair-dependent) and 2.86 (uniform) | MX T-M1, T-M2 | ours |
| D25 | 19 | Prompts whose per-criterion pairs contain a directed cycle (best vs worst / random pair) | HelpSteer quality only 0.06% / 0.09%; + concision 1.42% / 3.03%; + simplicity 3.14% / 5.56%; UltraFeedback 0.55% / 1.40% | MX R3–R6 | confirmed |
| D26 | 19 | Training panels pooled, worst rule per row | measured criteria: no Condorcet winner 1.5%, predicted oscillation 1.5%; rubric contexts (130 panels): 0%, 0%; style panels (25): 0%, 0% | MX P0c, P0d, P1d, P1i | confirmed |
| D27 | 19, 21 | X11 decision and cost | training stage not run by the preregistered rule; about 40 VM minutes, $5 | MX P1j | confirmed |
| D28 | 17 | Natural-feedback box | adds "mixing criteria rarely changes that (next two slides)" | D25, D26 | my reading |
| D29 | 20, 21 | Paper outline and fourth decision | Section 4 gains a pooled-sources proposition; decision 4 asks whether to add it and the mixed-data rates | mixed-criteria report §7 | my reading |
