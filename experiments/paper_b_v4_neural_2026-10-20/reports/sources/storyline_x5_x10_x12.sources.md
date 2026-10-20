# Fact sheet: JMLR storyline (2026-10-19)

Paths are relative to the Iterative_PsiPO root.

**Status values**
- confirmed: recomputed, or read from the primary file.
- ours: our own derivation.
- recalled: a citation's existence was checked on 2026-10-19, but its theorem was not re-read.
- reading: my judgement.

**Abbreviations**
- A = `reviews/2026-10-19-jmlr-storyline/analysis/`
- F = `reviews/2026-10-19-jmlr-storyline/shared/facts.json`

## Where the current story stands

| id | claim | value | source | status |
|---|---|---|---|---|
| S1 | Current contributions | Five parallel items: reduction; margins; stability frontier and fixed-target price; practical choices including history-aware refresh "trade stability against adaptation"; neural test. | `paper/mirror/overleaf/JMLR-B-v4/jmlr_main.tex` (contributions paragraph, 133d64c) | confirmed |
| S2 | Inherited parts | The B-only target is classical (Galla–Farmer 2013, Coucheney et al. 2015). The frontier extends the 2×2 EWA condition (Pangallo et al. 2022). The first KL certificate is MMD Thm 3.4. The endpoint is the QRE limit (McKelvey–Palfrey) and maximal lottery (Fishburn). | `reviews/2026-10-18-jmlr-review/lit_positioning.md` (b), (d) | single (audit) |
| S3 | New parts per the audit | Replay multipliers, judge bias as refresh, the reduction with a separate arithmetic sampler and geometric reference, the preregistered tests. | same, headline 5 | single (audit) |
| S4 | Natural feedback rarely cycles | 1 of 130 jury panels lacks a Condorcet winner; 0 of 1,200 SPPO panels have a majority three-cycle. | v4 deck fact sheet J1, J2 (`reviews/2026-10-17-v4-neural-validation/visual/sources.md`) | confirmed |
| S5 | Neural verdicts so far | H0–H4 pass; H5–H6 pass after two logged measure changes; 32-pair test failed, 512-pair passed; 1.5B passed. Controls: E1, E2, E4, E5 pass; E3, E6 fail; E3′ pass. X1 pass; X9 fail. | v4, controls and X1/X9 fact sheets | confirmed |
| S6 | Current framing of history | Thm `lagged-reference-stability` is titled "stability gain paid in adaptation speed". The Discussion says "a better tradeoff between speed and stability, not the only way to reach a target stably". | `jmlr_main.tex` l. 2037–2060 and the Discussion | confirmed |
| S7 | Lagged reference at maximal lag | ν = α gives the reference π_{t−1}^α π_ref^{1−α}. Every mode with γ ≤ 2√α has spectral radius exactly √α; for 2√α < γ < 1+α it is (γ + √(γ² − 4α))/2. | `jmlr_main.tex`, Thm `lagged-reference-stability` part 3 and eq. (lagged-reference) | confirmed |
| S8 | Ordinary refresh at a fixed target | Spectral radius √(α² + (1−α)²γ_B²) ≥ γ_B/√(1+γ_B²), with equality at α = γ_B²/(1+γ_B²); about 2γ_B² log(1/ε) rounds. | `jmlr_main.tex`, Prop `fixed-target` part 3 | confirmed |
| S9 | T5 | With C = 0 and a step cap β̄, reaching sharpness B needs ≥ ⌈B/β̄⌉ rounds; refresh-then-anchor attains it. | `reviews/2026-10-18-jmlr-review/theory/theorems.md` T5 | ours (refereed) |

## The new result: rounds to reach a fixed target

| id | claim | value | source | status |
|---|---|---|---|---|
| R1 | Best rounds per e-fold, no step cap | Ordinary vs lagged reference vs lagged sampling: 2.89 / 1.17 / 1.72 at γ_B = 1; 8.96 / 2.18 / 3.32 at 2; 33.1 / 4.18 / 7.82 at 4; 129 / 8.05 / 15.8 at 8; 513 / 19.0 / 28.8 at 16; 2,053 / 38.6 / 63.3 at 32. The best lag is always ν = α. | A/accel_modes.json (`accel_modes.py`) | confirmed (grid) |
| R2 | Closed forms | Ordinary 2/ln(1 + 1/γ²) ≈ 2γ² + 1. Lagged reference 1/ln(1/s*), s* = (√(1+γ²) − 1)/γ ≈ 1 − 1/γ, hence ≈ γ. Ratio → 2γ. | derived from S7, S8 | ours |
| R3 | With a step cap β ≤ hB | Ordinary ≈ max(1/h, 2γ²); lagged ≈ max(2/h, γ). Examples: γ = 2, h = 1: 8.96 vs 2.09; γ = 2, h = 0.05: 21.8 vs 39.0; γ = 8, h = 0.1: 129 vs 19.0. | A/accel_cap.json | confirmed (grid); approximation ours |
| R4 | Cyclic gain of the planted panels at B = 12 | Balanced: median 1.96, range 1.59–2.07. Decisive: median 0.40, max 1.72. | A/accel_panels.json | confirmed |
| R5 | Nonlinear population, planted balanced panels, B = 12: first round with median KL < 1e-3 to stay | Ordinary α = 0.7 / 0.75 / 0.8 / 0.85 / 0.9 / 0.95: 23 / 20 / 19 / 22 / 31 / 59. Lagged reference α = ν = 0.35 / 0.4 / 0.45 / 0.5 / 0.6: 10 / 7 / 8 / 9 / 11; α ≤ 0.3 never. Lagged sampling (α = 0.7, κ = 0.5): 9. 90th percentile: ordinary best 24 (α = 0.75, 0.8); lagged 9 (α = 0.4, 0.45). | A/accel_panels.json | confirmed |
| R6 | Same, KL < 1e-2 (median, 90th percentile) | Ordinary α = 0.8: 11, 14. Ordinary α = 0.9: 19, 23. Lagged α = 0.4: 4, 7. Lagged α = 0.45: 5, 6. Lagged sampling α = 0.7: 6, 6. | A/accel_thresholds.json | confirmed |
| R7 | Median KL at rounds 0 / 5 / 10 / 20 | Ordinary α = 0.8: 0.666 / 0.053 / 0.011 / 0.00079. Lagged α = 0.4: 0.666 / 0.0035 / 5.2e-5 / 3.8e-9. | F curves | confirmed |
| R8 | Trained model, ordinary refresh, B = 12 | H6_B12_a08: median KL first stays below 1e-2 at round 17 (90th percentile at round 37), with a floor of about 0.0017 over the last 20 rounds. H6_B12_a09: 33, floor 0.0027. Neither reaches 1e-3. | v4 runs, computed 2026-10-19 | confirmed |
| R9 | The training script supports the lag | `--nu` sets r_t = (α − ν)s_t + ν s_{t−1} + (1 − α)s_ref | `experiments/2026-10-17-controls/vm/panel_po_ctrl.py` l. 18, 50, 187 | confirmed |

## Literature for the acceleration credit

| id | reference | status |
|---|---|---|
| L1 | Ibrahim, Azizian, Gidel, Mitliagkas. "Linear Lower Bounds and Conditioning of Differentiable Games." ICML 2020, PMLR 119:4583–4593: lower bounds matching linear rates in games, via the spectrum of the update operator. | recalled (PMLR page) |
| L2 | Mokhtari, Ozdaglar, Pattathil. "A Unified Analysis of Extra-gradient and Optimistic Gradient Methods for Saddle Point Problems: Proximal Point Approach." AISTATS 2020, PMLR 108:1497–1507. | recalled; the exact κ-dependence was not re-read |
| L3 | Azizian, Mitliagkas, Lacoste-Julien, Gidel. "A Tight and Unified Analysis of Gradient-Based Methods for a Whole Spectrum of Differentiable Games." AISTATS 2020, PMLR 108:2863–2873. | recalled |
| L4 | Gidel et al. 2019, negative momentum (already `gidel2019negative` in the bib) | in bib |
