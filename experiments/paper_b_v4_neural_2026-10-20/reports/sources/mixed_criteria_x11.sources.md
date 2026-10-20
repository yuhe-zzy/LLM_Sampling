# Fact sheet: mixed-criteria cycles (2026-10-19)

Every number in the text report and in the deck slides traces to a row here. Paths are relative to the Iterative_PsiPO root.

**Status values**
- confirmed: read from a script output or recomputed.
- ours: our derivation, checked numerically.
- inference: our reading.

**Source abbreviations**
- TM = `reviews/2026-10-19-mixed-criteria-cycles/analysis/theory_mix.json` (script `theory_mix.py`)
- ON = `.../analysis/onset_scaling.json`
- CW = `.../analysis/cw_baseline.json`
- PR = `.../realdata/results/probe.json` (scripts `realdata/scripts/`; provenance in `realdata/README.md`)
- S0 = `experiments/2026-10-19-mixed-criteria/prereg/population_predictions.json` and `prereg/stage0_no_condorcet_winner.json`
- S1 = `.../prereg/stage1_predictions.json`
- S1b = `.../prereg/stage1b_predictions.json`
- PRE = `.../prereg/PREREG.md`
- FA = `reviews/2026-10-19-mixed-criteria-cycles/shared/facts.json` (written by `shared/gen.py`)

## Theory

| id | claim | value | source | status |
|---|---|---|---|---|
| T-M1 | Covariance identity (identity link): curl_T(Ā) = Σ_k w̄_k curl_T(A^k) + 3 Σ_k Cov_T(w_k, A^k) | max abs error 3.1e-16 over 200 random cases with 2–5 intransitive sources | TM `M1_max_abs_error` | ours |
| T-M2 | Order of the cycle in the judges' margin scale m (three BT judges, logit link) | log–log slope 2.86 under pair-independent weights, 0.98 under pair-dependent weights; the cubic Jensen formula leaves a residual of slope 4.86 | TM `M2` | ours |
| T-M2b | Cubic coefficient | logit(Σ w σ(D)) = Σ w D + (1/12)[(Σ w D)³ − Σ w D³] + O(m⁵) | derivation in `theory.md` M2 | ours |
| T-M3 | Random static assignment: E‖C‖² = ‖C(mean)‖² + ((n−2)/n) Σ_e D_eᵀ Σ_w D_e | variance formula within 1.5% of 20,000 simulated assignments in 5 cases | TM `M3` | ours |
| T-M4 | Latin-square judges (scores (m, 0, −m) rotated), pooled edge on each cycle edge | m = 0.5 / 1 / 2: soft 0.505 / 0.527 / 0.593; owner routing 0.622 / 0.731 / 0.881; salience routing 0.731 / 0.881 / 0.982 (reverse direction) | TM `M4_latin_square`; toy `analysis/toy_mix.json` for soft | ours |
| T-M5a | Local modulus at the DPO fixed point vs simulated oscillation | agreement 99 of 99 cases (3 recipes × 3 margins × 11 couplings) | TM `dynamics_owner_routing` | ours |
| T-M5b | Onset step at a uniform reference: β* = √3/(4(p̄ − ½)) | settles at 0.9β* and cycles at 1.1β* in 16 of 16 cases | ON | ours |
| T-M5c | Largest β_DPO that cycles at m = 1 | soft 0.063; c = 0.1: 0.110; c = 0.25: 0.180; c = 0.5: 0.298 | ON | ours |
| T-M5d | Largest β_DPO that cycles at m = 2 | soft 0.215 | ON | ours |
| T-M5e | Smallest coupling (grid 0.1) with local modulus > 1 at B = 10, SFT-like start (0.5, 0.3, 0.2) | fixed reference: m = 0.5: 0.4; m = 1: 0.1; m = 2: 0. α = 0.5, β = 5: 0.7; 0.3; 0. | TM rows | ours |
| T-M5f | Same, slow refresh α = 0.9, β = 1 | m = 0.5: none (largest modulus 0.942); m = 1: 0.8 (modulus 1.001; amplitude 0.06 at 0.8, 0.244 at 1); m = 2: 0.4 (1.020) | TM rows | ours |
| T-M5g | Rotation order in the example (m = 1, c = 0.5, fixed reference) | the top answer goes concise → creative → rigorous → concise, period 3 rounds; the edge is 0.629 | FA `F5` | ours |
| T-iid | No Condorcet winner, four answers, i.i.d. Gaussian criteria, scale 1 | 3 criteria: soft 0.9%, salience 3.9%, filtered 3.1%, static random 33.5%. 5 criteria: 1.0%, 9.6%, 5.9%, 41.1%. | CW | ours (simulation, 4,000 draws) |
| T-2crit | With two criteria, hard salience routing equals the sign of the summed gaps (transitive) | the region where the larger-magnitude component is positive is the half-plane Δ1 + Δ2 > 0 | derivation | ours |

## Real-data probe (HelpSteer v1, UltraFeedback)

| id | claim | value | source | status |
|---|---|---|---|---|
| R1 | Units | HelpSteer 6,690 prompts with 4 distinct responses; UltraFeedback 54,340 | PR `sets.*.n`; README "Data counts" | confirmed |
| R2 | Sanity | C0 and C1 give 0 directed cycles in every set | PR | confirmed |
| R3 | HS-S (helpfulness, correctness, coherence, concision), share of prompts with a directed cycle | C2 0.00% (0); C3 1.42% [1.15, 1.73] (95); C4 2.53% [2.15, 2.92]; C5 3.03% [2.66, 3.47] | PR | confirmed |
| R4 | UF (4 aspects) | C2 0.02% (11); C3 0.55% [0.50, 0.61] (301); C4 1.20% [1.11, 1.29]; C5 1.40% [1.30, 1.51] | PR | confirmed |
| R5 | HS-Q (quality only) | C3 0.06%, C4 0.09%, C5 0.09% | PR | confirmed |
| R6 | HS-S2 (+ simplicity) | C3 3.14%, C4 4.78%, C5 5.56% | PR | confirmed |
| R7 | Contradictions (the same pair labelled both ways), C3 | HS-S 27.8%; UF 8.4% | PR | confirmed |
| R8 | Agreement between criteria (mean Kendall τ-b) | HelpSteer quality criteria 0.61–0.76; concision against them −0.29 to −0.38; UF aspects 0.48–0.58 | PR `kendall_tau_b`; README | confirmed |
| R9 | C3 sparsity | about half of C3 prompts end with at most 2 distinct pairs (HS-S 51.6%, UF 49.4%); among prompts with ≥ 3 pairs, C3 cycles 2.93% (HS-S), 1.09% (UF) | PR; README | confirmed |
| R10 | Seed average of C3 (seeds 0–19) | HS-S 1.39% (1.18–1.64); UF 0.58% | PR | confirmed |
| R11 | Downloads | HelpSteer train/validation jsonl.gz (15.9 MB, 0.8 MB); UltraFeedback parquet shards (185 MB, 138 MB); sha256 in the README; raw data kept in the session scratchpad only | `realdata/README.md` | confirmed |

## Panels (X11 stages 0, 1, 1b)

| id | claim | value | source | status |
|---|---|---|---|---|
| P0a | Stage 0 criteria on the 130 v4 panels: quality (Skywork-8B), concision (−characters), originality (−base log-probability); mean Kendall τ | Q–C 0.19, Q–N −0.09, C–N −0.03 | S0 `criteria_T` | confirmed |
| P0b | Stage 0 mean logit cyclic share | soft 0.017, salience 0.079, filtered 0.122; share above 0.1: 1.5%, 26.9%, 50.8% | S0 | confirmed |
| P0c | Stage 0, no Condorcet winner / majority three-cycle | soft 0% / 1.5%; salience 1.5% / 1.5%; filtered 1.5% / 3.1% | S0 | confirmed |
| P0d | Stage 0 predicted oscillation (population DPO, recipes A/B/C) | at most 1.5% in every cell; a single criterion gives 0–0.8% | S0 | confirmed |
| P0e | Reward-model jury as three sources (stage 0) | salience 0.044 mean cyclic share, no three-cycle, 0% oscillation | S0 | confirmed |
| P1a | Stage 1: three rubric contexts appended to the prompt, scored by Skywork-8B (main), Skywork-V2-1.7B, GRM-3B | VM 12:55–13:07 EDT; plain scores reproduce the panel file (max diff 0, 0, 0.125) | PRE log; S1 `plain_check` | confirmed |
| P1b | Main judge, Kendall τ between plain and each context | rigor 0.84, concision 0.78, creativity 0.44; between contexts 0.46–0.79 | S1 `kendall` | confirmed |
| P1c | Panels whose three contexts pick three different top answers | 7 / 3 / 8 of 130 (8B / 1.7B / 3B) | PRE log | confirmed |
| P1d | Stage 1 pooling, main judge | mean cyclic share 0.001 / 0.035 / 0.037 (soft / salience / filtered); no panel without a Condorcet winner; 0% predicted oscillation under every recipe | S1 | confirmed |
| P1e | Decision | stage 2 not run on these panels (fewer than 8 panels predicted to oscillate) | PRE log | confirmed |

| P1f | Stage 1b: style panels (plain, rigorous, creative, concise answers written by the policy; plain prompt kept for training) | 25 of 600 prompts kept: 470 below the probability floor 0.01, 71 without a valid answer in some style, 34 with a repeated answer; 4 would pass 0.05 | PRE log; `data/style_panels.json` stats | confirmed |
| P1g | Stage 1b, main-judge Kendall τ between contexts | rigor–creativity 0.16, creativity–concision 0.17, rigor–concision 0.72 | S1b `kendall` | confirmed |
| P1h | Stage 1b, each context's top answer is its own style | rigor 16%, creativity 48%, concision 32% (chance 25%); all three on 0 of 25 (other judges 0 and 1 of 25) | S1b `specialists` | confirmed |
| P1i | Stage 1b pooling, main judge | mean cyclic share 0.002 / 0.060 / 0.069; no panel without a Condorcet winner (all judges); 0% predicted oscillation | S1b | confirmed |
| P1j | Decision and cost | stage 2 not run; X11 VM use about 40 minutes in two passes (12:55–13:07 and 13:12–13:36 EDT), about $5 | PRE log | confirmed |
