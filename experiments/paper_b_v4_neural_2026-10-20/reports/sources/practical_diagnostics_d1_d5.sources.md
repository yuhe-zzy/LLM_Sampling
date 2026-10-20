# Sources: Diagnosing concentration and cycling without the preference (2026-10-19)

Paths are relative to the Iterative_PsiPO root. `AN` = `reviews/2026-10-19-practical-diagnostics/analysis/`. `MS` = `paper/mirror/overleaf/JMLR-B-v4/jmlr_main.tex` at 46fd577.

Status:
- **confirmed**: read from the file, or recomputed by the script named;
- **derived**: arithmetic on confirmed numbers;
- **inference**: ours.

No new training and no VM time; every number is recomputed from saved runs or saved data.

## Theory already in the manuscript

| id | claim | source | status |
|---|---|---|---|
| T1 | A judge whose label log-odds get + ε(x_i − x_j) of the current policy turns MRS-ΨPO into the same recursion with refresh α′ = α + βε. α′ ≥ 1 means no stable finite target; with C = 0, x_{t+1} = α′x_t + c. The same bias computed from the reference leaves α unchanged. | MS prop:controls part 1 (§9.7, lines ~1105–1110) | confirmed |
| T2 | A popularity penalty is ε = −η. Under full refresh it gives the finite sharpness 1/η, with the reference factor 0. | MS text after prop:controls | confirmed |
| T3 | Target sharpness B = β/(1 − α). Ordinary refresh at fixed B is locally stable iff α² + (1 − α)²γ_B² < 1, i.e. α > α_loc = (γ_B² − 1)/(γ_B² + 1) when γ_B > 1. | MS prop:target, prop:fixed-target, tab:fixed-target-comparison | confirmed (α_loc algebra derived) |
| T4 | Local multipliers at the target are α ± iβλω_ℓ, with ω_ℓ the eigenvalues of CJ*. | MS thm:local-spectrum | confirmed |
| T5 | Pooled sources: curl = Σ_k w̄_k curl(A^k) + 3Σ_k Cov_T(w_k, A^k). First order under pair-dependent assignment, third order under a uniform mixture. Only on-policy sources feed back. | MS prop:pooled | confirmed |
| T6 | Deterministic labels under the logit link leave no finite target at any refresh level. Soft labels, smoothing or the identity link restore one. | MS §8 "Choose the link"; §9.3 | confirmed |
| T7 | Step limit: a target of sharpness B needs at least ⌈B/β̄⌉ rounds; refresh-then-anchor attains it. | MS §8 "Reach the target in the fewest rounds" | confirmed |
| T8 | With independent noise on the K(K−1)/2 edges (any variances), the expected cyclic energy of the noise is (K − 2)/K times the sum of the edge variances. The Hodge projection onto cyclic flows is orthogonal with trace (K − 1)(K − 2)/2, and its diagonal is constant over the edges of the complete graph by symmetry. | derivation; used in `AN/d3b_d4b.py` | derived |

## Earlier experiments cited as remedies

| id | claim | source | status |
|---|---|---|---|
| E1 | End entropies 0.638 / 0.300 / 0.061 at ε = 0 / 0.3 / 0.45 (predicted 0.638 / 0.301 / 0.066). ε = 0.55 collapses to 0.007. Frozen judge 0.525 (predicted 0.527). α = 0.5, β = 1. | MS §9.7 "Judge bias and entropy bonus"; `experiments/2026-10-17-controls/prereg/PREREG.md` | confirmed |
| E2 | Popularity penalties η = 0.125 / 0.25: end entropies 0.186 / 0.416, against anchors B = 8 / 4 at 0.195 / 0.404. KL from the initial model 1.60 vs 1.26, and 1.38 vs 0.93. | MS §9.7 | confirmed |
| E3 | A per-round KL controller under full refresh ends at entropy 6 × 10⁻⁶ after 60 rounds. | MS §9.7 "Drift controllers" | confirmed |
| E4 | Replay ρ = 0.2: oscillating share of balanced planted panels 94% → 0% at α = 0.5, and 0% → 59% at α = 0.8. | MS §9.7 "Replay and adaptive refresh" | confirmed |
| E5 | The adaptive refresh rule moved α from 0.2 to 0.91 with no oscillating panel and median KL 0.0009 to the target. | MS §9.7 | confirmed |
| E6 | At α = 0.2: last iterate KL 0.479, policy average 0.008, weight average 0.052. Serving a random one of the last 30 checkpoints realizes the policy average. | MS §9.7 "Deploying an average" | confirmed |
| X5 | Refresh-then-anchor reached median KL 0.001 at rounds 15 and 4 (step caps 0.5 and 2). The constant anchor took 33 and 8. | MS §9.7; `experiments/2026-10-19-x10-x5-x12/results/evaluation_x10_x5.json` | confirmed |
| H4 | Hard-label DPO: panel entropy 0.007 after one round, and its top-two log-odds keep growing despite the anchor α = 0.8. | MS §9.3 | confirmed |
| H6 | Balanced planted panels at B = 12 oscillating in the last 30 rounds: 100% (32/32) at α = 0.2, 94% (30/32) at 0.5, 0% at 0.8 and 0.9. | `AN/f1b_shares.json` (computed by `shared/gen.py` with the v4 oscillation rule) | confirmed |
| H6g | Median balanced panel: α_loc = 0.59, so γ_B ≈ 1.97. | MS §9.7 ("threshold α = 0.59 of the median panel"); γ from T3 | derived |
| P1 | Pairwise LLM judge (X12): Qwen2.5-7B-Instruct on the 130 v4 panels, both orders. 4 of 130 panels (3.1%) have no Condorcet winner. | `experiments/2026-10-19-x10-x5-x12/data/llm_judge.json`; title-routes fact sheet F11 | confirmed |
| P2 | HelpSteer (v1): 6,690 prompts with 4 distinct responses; criteria helpfulness, correctness, coherence, complexity, verbosity. UltraFeedback: 54,340 units with aspects instruction following, honesty, truthfulness, helpfulness. | `reviews/2026-10-19-mixed-criteria-cycles/text/sources.md` R1; units file fields | confirmed |

## New measurements (this report)

### D1, effective refresh from the probe trajectories
Script `AN/d1_d2_trajectory.py`, output `AN/d1_d2_trajectory.json`. 24 runs; estimates are medians over panels.

| run | objective, labels | α | β | true α′ | estimate | error |
|---|---|---|---|---|---|---|
| E1 ε = 0 | DPO soft | 0.5 | 1 | 0.5 | 0.546 | +0.046 |
| E1 ε = 0.3 | DPO soft | 0.5 | 1 | 0.8 | 0.813 | +0.013 |
| E1 ε = 0.45 | DPO soft | 0.5 | 1 | 0.95 | 0.946 | −0.004 |
| E1 ε = 0.55 | DPO soft | 0.5 | 1 | 1.05 | 1.031 | −0.019 |
| E1 frozen judge | DPO soft | 0.5 | 1 | 0.5 | 0.599 | +0.099 |
| E2 η = 0.125 | DPO soft | 1 | 1 | 0.875 | 0.870 | −0.005 |
| E2 η = 0.25 | DPO soft | 1 | 1 | 0.75 | 0.772 | +0.022 |
| E2 anchor B = 8 | DPO soft | 0.875 | 1 | 0.875 | 0.873 | −0.002 |
| E2 anchor B = 4 | DPO soft | 0.75 | 1 | 0.75 | 0.774 | +0.024 |
| X5 anchor β̄ = 0.5 | DPO soft | 0.9375 | 0.5 | 0.9375 | 0.935 | −0.002 |
| X5 anchor β̄ = 2 | DPO soft | 0.75 | 2 | 0.75 | 0.766 | +0.016 |
| H1 α = 0, β = 8 | IPO soft | 0 | 8 | 0 | 0.234 | +0.234 |
| H1 α = 0.5, β = 4 | IPO soft | 0.5 | 4 | 0.5 | 0.599 | +0.099 |
| H1 α = 0.8, β = 1.6 | IPO soft | 0.8 | 1.6 | 0.8 | 0.799 | −0.001 |
| H1 α = 0.9, β = 0.8 | IPO soft | 0.9 | 0.8 | 0.9 | 0.849 | −0.051 |
| H1 α = 0.5, β = 1 | IPO soft | 0.5 | 1 | 0.5 | 0.662 | +0.162 |
| H2 α = 0.8 | IPO soft | 0.8 | 1 | 0.8 | 0.797 | −0.003 |
| H2 α = 0.9 | IPO soft | 0.9 | 1 | 0.9 | 0.807 | −0.093 |
| H2 α = 0.95 | IPO soft | 0.95 | 1 | 0.95 | 0.926 | −0.024 |
| H3 full refresh | IPO hard | 1 | 1 | none | 0.989 | – |
| H4 hard-label DPO | DPO hard | 0.8 | 1 | none | 0.907 | – |
| H4 soft-label DPO | DPO soft | 0.8 | 1 | 0.8 | 0.812 | +0.012 |
| H4 hard-label IPO | IPO hard | 0.8 | 1 | 0.8 | 0.841 | +0.041 |
| E3 per-round KL controller | DPO soft | 1 | 1 | none | 1.197 | – |

Summaries (derived):
- **DPO, 12 runs with a numerical α′.** |error| ≤ 0.025 on 10, ≤ 0.05 on 11; the largest is the frozen judge, +0.099.
- **IPO, 9 runs.** |error| ≤ 0.05 on 4 and ≤ 0.1 on 7; the largest are +0.234 (α = 0, β = 8) and +0.162 (α = 0.5, β = 1).
- **Finite-target runs.** The largest estimate is 0.946 (truth 0.95).
- **No-target runs.** Estimates are 0.989, 1.197 and 1.031 (ε = 0.55, truth 1.05). Hard-label DPO gives 0.907.

**Projection variant** (`est_proj` in the same JSON): it is biased low on IPO, for example H2 α = 0.9 gives 0.613. It is not used. Confirmed.

**Drift check** (`AN/d5_d1drift.py`, `AN/d5_d1drift.json`): median increment ratio over rounds 1–5 against rounds 6–15.

| run | rounds 1–5 | rounds 6–15 |
|---|---|---|
| hard-label DPO | 0.929 | 0.979 |
| hard-label IPO | 0.792 | 0.947 |
| E1 ε = 0.3 | 0.799 | 0.892 |
| H2 α = 0.95 | 0.875 | 1.008 |

The ratio rises in finite-target runs too, so the check does not separate hard-label DPO. Confirmed (negative).

### D2, rotation from early rounds
Scripts `AN/d1_d2_trajectory.py` and `AN/d2_extra.py`; outputs `AN/d1_d2_trajectory.json`, `AN/d2_extra.json` and `AN/d2_features.npz`.

**Data.** 9 runs × 64 panels = 576 panels:
- H5: β = 4, α = 0.5, 0.9, 0.95;
- H6: B = 12, α = 0.2, 0.5, 0.8, 0.9;
- X1 DPO: B = 4 at α = 0.8; B = 2 at α = 0.3.

25.5% of panels are oscillating, by the v4 rule: detrended amplitude over the last 30 rounds > 0.0536. H5 and H6 run 80 rounds; X1 runs 120.

**Pooled AUC.** ρ̂ 0.901; |Im| 0.834; ρ̂ + |Im| 0.922.

**Thresholds, sensitivity and specificity.**

| rule | sensitivity | specificity |
|---|---|---|
| ρ̂ ≥ 0.80 | 0.946 | 0.676 |
| ρ̂ ≥ 0.85 | 0.864 | 0.790 |
| ρ̂ ≥ 0.90 | 0.714 | 0.893 |
| \|Im\| ≥ 0.5 | 0.660 | 0.765 |

**Leave one run out.**

| rule | thresholds chosen | held-out sensitivity | held-out specificity | accuracy |
|---|---|---|---|---|
| ρ̂ | 0.833–0.841 | 0.912 | 0.744 | 0.786 |
| ρ̂ + \|Im\| | – | 0.782 | 0.772 | 0.774 |
| \|Im\| | – | 0.633 | 0.699 | 0.682 |

**Per run** (oscillating share; median ρ̂ oscillating / stable):

| run | oscillating | median ρ̂ (osc. / stable) |
|---|---|---|
| H5 α = 0.5 | 0.00 | – / 0.583 |
| H5 α = 0.9 | 0.50 | 0.936 / 0.884 |
| H5 α = 0.95 | 0.50 | 0.963 / 0.922 |
| H6 α = 0.2 | 0.5625 | 0.915 / 0.535 |
| H6 α = 0.5 | 0.484 | 0.925 / 0.548 |
| H6 α = 0.8 | 0 | – / 0.714 |
| H6 α = 0.9 | 0 | – / 0.770 |
| X1 B = 4, α = 0.8 | 0.047 | 0.865 / 0.770 |
| X1 B = 2, α = 0.3 | 0.203 | 0.869 / 0.689 |

**Median |Im|, oscillating / stable.**
- H6 α = 0.2: 0.913 / 0.400.
- H6 α = 0.5: 0.816 / 0.370.
- X1 B = 2: 0.828 / 0.652.

**Example panel 0** ("golden retriever"), H6. Confirmed.
- α = 0.2: ρ̂ 0.94, |Im| 0.94, oscillates.
- α = 0.8: ρ̂ 0.65, |Im| 0.26, settles.

### D3, census of the cyclic gain
Scripts `AN/d3_d4_data.py` (hard census) and `AN/d3b_d4b.py` (gain census); outputs `AN/d3_d4_data.json`, `AN/d3b_d4b.json`.

**Population**, X12 judge probabilities at B = 10:
- γ_B > 1 on 2.3% of panels (3 of 130), with γ = 1.37, 2.37 and 2.37;
- median γ_B 0.082; 90th percentile 0.473;
- largest α_loc 0.699;
- no Condorcet winner on 3.1%.

**Hard census**, share of prompts without a Condorcet winner. Certification uses Hoeffding with δ = 0.05.

| judge calls per prompt | no Condorcet winner | certified winner |
|---|---|---|
| 12 | 0.559 | 0 |
| 36 | 0.488 | 0 |
| 120 | 0.415 | 0.392 |
| 360 | 0.357 | 0.455 |
| 1,200 | 0.307 | 0.497 |

**Gain census**, noise-corrected, 5 replicates.

| judge calls per prompt | flagged share | sensitivity | false-positive share | mean largest α_loc |
|---|---|---|---|---|
| 12 | 0.042 | 0.933 | 0.020 | 0.757 |
| 36 | 0.026 | 1.0 | 0.003 | 0.720 |
| 120 | 0.025 | 1.0 | 0.002 | 0.710 |
| 360 | 0.023 | 1.0 | 0 | 0.707 |
| 1,200 | 0.023 | 1.0 | 0 | 0.699 |

**Mean cyclic energy** (population 0.1176):

| judge calls per prompt | raw | corrected |
|---|---|---|
| 36 | 0.1295 | 0.1232 |
| 120 | 0.1206 | 0.1179 |

At 12 calls the correction is zero, because the variance estimate p̂(1 − p̂) vanishes when n = 1. Confirmed.

### D4, assignment-permutation test
Scripts `AN/d3_d4_data.py` (real data) and `AN/d3b_d4b.py` (synthetic); outputs `AN/d3_d4_data.json`, `AN/d3b_d4b.json`.

**Method.** Each criterion picks its best-versus-worst pair (C3). The observed share of prompts with a directed 3- or 4-cycle is compared with the share after randomly reassigning the criteria among the same pairs: 10 permutations per prompt, and a 95% CI from a 500-draw bootstrap over prompts.

| set | n | observed | permuted | difference [95% CI] |
|---|---|---|---|---|
| HS-Q (helpfulness, correctness, coherence) | 6,690 | 0.090% | 0.024% | 0.066 pp [0.010, 0.135] |
| HS-S (+ −verbosity) | 6,690 | 1.55% | 0.63% | 0.92 pp [0.65, 1.17] |
| HS-S2 (+ −complexity) | 6,690 | 3.00% | 1.40% | 1.60 pp [1.24, 1.97] |
| UF, first 15,000 valid units, 4 aspects | 15,000 | 0.70% | 0.58% | 0.12 pp [0.002, 0.25] |
| Synthetic specialists, noise 0.3 | 2,000 | 94.3% | 47.3% | 46.9 pp [46.1, 47.8] |
| Synthetic specialists, noise 1.0 | 2,000 | 13.5% | 7.9% | 5.6 pp [4.8, 6.5] |
| Synthetic independent criteria | 2,000 | 2.8% | 2.7% | 0.09 pp [−0.37, 0.50] |

**First synthetic control** (`AN/d3_d4_data.json`): a fourth, weak answer made every criterion's worst pick the same answer, so no triangle could form. It gave 0 / 0 and was replaced by the controls above, whose fourth answer is in the middle. Confirmed.

### D5, judge-bias regression (simulated judge on real trajectories)
**Main estimator.** Conditional likelihood (`AN/d5_conditional.py`, output `AN/d5_conditional.json`).
- **Judge.** The E1 judge: log-odds (r_i − r_j)/3.168 + ε(s_i − s_j). The scores are current, or s_0 for the frozen judge.
- **Trajectories.** s_t from the trained E1 runs.
- **Probe and replicates.** Rounds 0 and 1 are queried; 300 replicates.
- **Columns.** α + βε̂, mean ± sd.

| judge | true α′ | 1,536 calls (64 × n 2) | 3,072 (64 × n 4) | 3,072 (16 × n 16) | 12,288 (64 × n 16) | 49,152 (64 × n 64) |
|---|---|---|---|---|---|---|
| ε = 0 | 0.5 | 0.498 ± 0.105 | 0.499 ± 0.072 | 0.515 ± 0.083 | 0.504 ± 0.037 | 0.500 ± 0.019 |
| ε = 0.3 | 0.8 | 0.797 ± 0.127 | 0.809 ± 0.080 | 0.800 ± 0.087 | 0.801 ± 0.042 | 0.802 ± 0.019 |
| ε = 0.45 | 0.95 | 0.946 ± 0.144 | 0.955 ± 0.098 | 0.957 ± 0.101 | 0.948 ± 0.052 | 0.950 ± 0.026 |
| ε = 0.55 | 1.05 | 1.058 ± 0.156 | 1.056 ± 0.110 | 1.045 ± 0.110 | 1.051 ± 0.052 | 1.047 ± 0.029 |
| frozen, ε = 0.55 | 0.5 | 0.493 ± 0.111 | 0.497 ± 0.074 | 0.499 ± 0.067 | 0.501 ± 0.035 | 0.501 ± 0.018 |

**Fixed-effects logistic fit** (one intercept per pair; `AN/d5_d1drift.py`, `AN/d5_d1drift.json`), 64 panels. Biased upward at small n (incidental-parameter bias):

| n | ε = 0 | ε = 0.3 | ε = 0.45 | ε = 0.55 | frozen |
|---|---|---|---|---|---|
| 4 | 0.558 | 0.882 | 1.048 | 1.165 | 0.596 |
| 16 | 0.513 | 0.815 | 0.975 | 1.081 | 0.527 |
| 64 | 0.505 | 0.806 | 0.956 | 1.056 | 0.508 |

**Naive least squares** on empirical log-odds changes (`AN/d5_ols_check.py`, `AN/d5_ols_check.json`; n = 32, 36,864 calls): ε̂ = 0.001, 0.065, 0.072, 0.072 against the true 0, 0.3, 0.45, 0.55; frozen −0.001. At round 0, 16–20% of the labels lie beyond 0.05 or 0.95. Confirmed.

## Inferences (ours)

| id | claim | status |
|---|---|---|
| I1 | The IPO errors in D1 have two plausible sources. Rounds that do not reach their optimum push the ratio up: effective contraction 1 − κ(1 − α). Increments that fall below the 0.05 floor within a few rounds bias it down. We have not separated them. | inference |
| I2 | The frozen-judge error (+0.099) is unexplained. A possible cause is incomplete DPO rounds when labels are near 0 or 1. | inference |
| I3 | The naive D5 estimator fails because the pairs that move most already have labels near 0 or 1, and their empirical log-odds cannot move. | inference |
| I4 | In these runs the probe panels were the training panels. Held-out probes are untested. | inference (limit) |
| I5 | The D2 threshold 0.83 is calibrated on these runs: the leave-one-run-out folds pick 0.833–0.841. It sits below the theoretical value 1 because rounds 1–12 are far from the target. | confirmed folds; reason inferred |
| I6 | X13 cost: about 1 VM hour at $6.98/h, about $7. | estimate |
