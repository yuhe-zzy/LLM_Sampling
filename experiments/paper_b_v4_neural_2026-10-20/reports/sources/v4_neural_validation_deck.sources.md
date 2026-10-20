# Fact sheet: neural validation of the JMLR-B-v4 theory (2026-10-16/17)

Status: **confirmed** = computed from the saved run files (`results/runs*/<run>/run.npz`) by the named script; **frozen** = a prediction file written before the runs it predicts (timestamps in `prereg/PREREG.md`); **logged** = from the timestamped log in `prereg/PREREG.md`; **single** = an external dataset or paper, read but not re-derived. All paths are relative to `experiments/2026-10-16-v4-validation/`.

## Setup
| # | Fact | Source | Status |
|---|---|---|---|
| A1 | Policy Qwen2.5-0.5B-Instruct (full fine-tuning, fp32 weights, bf16 autocast); replication on Qwen2.5-1.5B-Instruct | `vm/panel_po.py`, `vm/batch*.json` | confirmed |
| A2 | 600 prompts tried (65 templated short-answer, 100 NoveltyBench curated, 435 NoveltyBench WildChat), suffix "answer with at most six words"; 48 samples at temperature 1; panel = 4 most probable distinct complete answers; kept if every initial panel probability >= 0.05; 130 panels kept (0.5B), 390 (1.5B) | `vm/build_panels.py`, `data/panels_q05.json`, `data/panels_q15.json` | confirmed |
| A3 | Reward models: Skywork-Reward-Llama-3.1-8B-v0.2 (oracle), Skywork-Reward-V2-Qwen3-1.7B, GRM-Llama3.2-3B (jury) | `data/panels_q05.json` | confirmed |
| A4 | Soft labels sigma((r_i - r_j)/T), T = 3.168 (half the std of within-panel reward gaps, 0.5B); T = 2.671 (1.5B); hard labels 1[r_i > r_j] | `prereg/predictions_batch1.json`, `prereg/predictions_batch4.json` | frozen |
| A5 | Round: reference alpha*s_t + (1-alpha)*s_0; sampler (1-lambda)*uniform + lambda*softmax(s_t); pairwise IPO square loss or DPO logistic loss with pair weights mu_i mu_j; 40 Adam steps, lr 2e-5, fresh optimizer each round | `vm/panel_po.py` | confirmed |
| A6 | Population predictions: exact MRS recursion on the same panels and oracles | `analysis/population.py` | confirmed |
| A8 | Training runs: 2 pilots + 19 main (batches 1-2, `results/runs/`), 10 sampled-pair (batches 3/3b, `results/runs_sampled/`), 6 at 1.5B (batch 4, `results/runs_q15/`): 37 in total | `results/` directory listing | confirmed |
| A7 | VM: one H100 NVL 94 GB, $6.98/h; about 13 h of uptime (start 2026-10-16 about 17:20 UTC, deallocated 2026-10-17 about 06:35 UTC); about $90 | VM notes, session log | logged (approximate) |

## H0 fidelity
| # | Fact | Value | Status |
|---|---|---|---|
| B0 | median over main runs of the per-round residual (nats) | 0.092 | confirmed (`analysis/evaluate.py`) |
| B0b | pilot (lr 1e-5): residual median 0.14 (q90 0.29); payoff form 0.44 | 0.14 / 0.44 | logged |

## H1 same target at B = 8 (IPO, soft labels, lambda = 0.8, 30 rounds)
| # | Run | Final entropy | Target entropy | KL final to target | Status |
|---|---|---|---|---|---|
| C-H1_a0_b8 | H1_a0_b8 | 0.782 | 0.782 | 0.0008 | confirmed |
| C-H1_a05_b4 | H1_a05_b4 | 0.780 | 0.782 | 0.0011 | confirmed |
| C-H1_a08_b16 | H1_a08_b16 | 0.784 | 0.782 | 0.0019 | confirmed |
| C-H1_a09_b08 | H1_a09_b08 | 0.797 | 0.782 | 0.0067 | confirmed |
| C-H1_a05_b1 | H1_a05_b1 | 1.085 | 1.087 | 0.0013 | confirmed |
| C-pair | max pairwise KL among the four B = 8 runs | 0.0076 |  |  | confirmed |
| C-ctl | KL of B = 2 control to the B = 8 runs (range) | 0.207 | 0.227 |  | confirmed |
| C-pred | frozen prediction: B = 8 target entropy (panel-file scores) / KL(B=8 target, B=2 target) | 0.792 | 0.223 |  | frozen |

## H2 plateau set by B (IPO, soft, beta = 1)
| # | Run | Final entropy | Target entropy | Entropy at M_t = 4 / 8 | Status |
|---|---|---|---|---|---|
| D-H1_a05_b1 | H1_a05_b1 | 1.085 | 1.087 | n/a | confirmed |
| D-H2_a0.8 | H2_a0.8 | 0.940 | 0.943 | 1.002 / nan | confirmed |
| D-H2_a0.9 | H2_a0.9 | 0.691 | 0.689 | 1.006 / 0.779 | confirmed |
| D-H2_a0.95 | H2_a0.95 | 0.415 | 0.410 | 0.996 / 0.779 | confirmed |

## H3 full refresh (IPO, hard labels, lambda = 1, 40 rounds)
| # | Fact | Value | Status |
|---|---|---|---|
| E1 | share of panels whose top answer is the Condorcet winner at the end | 1.00 | confirmed |
| E2 | share with Condorcet-winner mass >= 0.95 at the end | 1.00 | confirmed |
| E3 | mean Condorcet-winner mass at t = 0, 5, 10, 20, 40 | 0.2732, 0.7042, 0.9490, 0.9995, 1.0000 | confirmed |
| E4 | mean entropy at t = 0, 5, 10, 20, 40 | 1.1396, 0.6883, 0.1871, 0.0042, 0.0000 | confirmed |

## H4 link (alpha = 0.8, beta = 1, lambda = 0.8, 60 rounds)
| # | Fact | Value | Status |
|---|---|---|---|
| F-H4_dpo_soft | H4_dpo_soft: final entropy / target / KL | 0.388 / 0.385 / 0.0005 | confirmed |
| F-H4_ipo_hard | H4_ipo_hard: final entropy / target / KL | 0.677 / 0.679 / 0.0010 | confirmed |
| F-hard | hard-label DPO entropy at t = 1, 5 | 0.0073, 0.0000 | confirmed |
| F-gap | hard-label DPO median top-two log-odds gap at t = 1, 10, 30, 60 (nats) | 10, 89, 385, 508 | logged (computed 2026-10-16) |

## H5/H6 planted cycles (IPO, lambda = 1, 80 rounds, 64 panels: 32 balanced, 32 decisive)
Oscillating = detrended standard deviation of an answer probability over the last 30 rounds above 0.0536 (4 x the noise floor 0.0134 of the converged batch-1 runs).

| # | Config | Balanced: neural / population share | Balanced amplitude neural / population | Decisive share neural / population | Status |
|---|---|---|---|---|---|
| G-H5_b4_a05 | H5_b4_a05 | 0.00 / 0.00 | 0.017 / 0.000 | 0.00 / 0.00 | confirmed |
| G-H5_b4_a09 | H5_b4_a09 | 1.00 / 1.00 | 0.306 / 0.307 | 0.00 / 0.00 | confirmed |
| G-H5_b4_a095 | H5_b4_a095 | 1.00 / 1.00 | 0.363 / 0.362 | 0.00 / 0.00 | confirmed |
| G-H6_B12_a02 | H6_B12_a02 | 1.00 / 1.00 | 0.317 / 0.317 | 0.12 / 0.16 | confirmed |
| G-H6_B12_a05 | H6_B12_a05 | 0.94 / 0.91 | 0.181 / 0.185 | 0.03 / 0.00 | confirmed |
| G-H6_B12_a08 | H6_B12_a08 | 0.00 / 0.00 | 0.023 / 0.000 | 0.00 / 0.00 | confirmed |
| G-H6_B12_a09 | H6_B12_a09 | 0.00 / 0.00 | 0.025 / 0.001 | 0.00 / 0.00 | confirmed |

| # | Fact | Value | Status |
|---|---|---|---|
| G-q | Planted construction: q = 0.3, e = 0.3, balanced Delta = 0.05 (majority cycle, maximal lottery support 3), decisive Delta = 0.2 (Condorcet winner) | `analysis/make_planted.py` | confirmed |
| G-amend1 | Amendment 1 (after 2 of 7 runs): top-switch share misfires on near-tied stable targets (41% at alpha = 0.5 though converged, amplitude 0.06) | `prereg/PREREG.md` | logged |
| G-amend2 | Amendment 2: logit amplitude flags decisive panels at beta = 4, alpha = 0.9/0.95 (78%, 100%) though top mass 0.989/1.000; cause: unconstrained logits of near-zero-probability answers; fix: measure on probabilities | `prereg/PREREG.md` | logged |

## Batch 3/3b sampled pairs (0.5B, 64 panels)
| # | Fact | Value | Status |
|---|---|---|---|
| H-S1_dpo_a0_b8 | S1_dpo_a0_b8 (n = 32): KL time-avg to target / final KL / fluctuation KL | 0.6383 / 1.0718 / 0.3815 | confirmed |
| H-S1_dpo_a08_b16 | S1_dpo_a08_b16 (n = 32): KL time-avg to target / final KL / fluctuation KL | 0.0946 / 0.0820 / 0.0045 | confirmed |
| H-S1n512_dpo_a0_b8 | S1n512_dpo_a0_b8 (n = 512): KL time-avg to target / final KL / fluctuation KL | 0.0086 / 0.0515 / 0.0089 | confirmed |
| H-S1n512_dpo_a08_b16 | S1n512_dpo_a08_b16 (n = 512): KL time-avg to target / final KL / fluctuation KL | 0.0015 / 0.0032 / 0.0009 | confirmed |
| H-ratio | fluctuation ratio alpha = 0 vs 0.8 at n = 512 (predicted 9) | 10.4 | confirmed (exact; 9.9 after rounding) |
| H-S3_B12_a02 | S3_B12_a02: balanced amplitude neural / population; decisive neural | 0.323 / 0.317; 0.115 | confirmed |
| H-S3_B12_a09 | S3_B12_a09: balanced amplitude neural / population; decisive neural | 0.063 / 0.001; 0.021 | confirmed |
| H-S3n512_B12_a02 | S3n512_B12_a02: balanced amplitude neural / population; decisive neural | 0.326 / 0.317; 0.036 | confirmed |
| H-S3n512_B12_a09 | S3n512_B12_a09: balanced amplitude neural / population; decisive neural | 0.028 / 0.001; 0.011 | confirmed |
| H-S2 | S2 (n = 32): Bernoulli DPO final entropy / target; hard DPO entropy t = 1 | 0.328 / 0.331; 0.023 | confirmed (`results/eval_sampled.json`) |

## Batch 4 1.5B (64 panels, expected labels)
| # | Fact | Value | Status |
|---|---|---|---|
| I1 | R1 KL final to target alpha = 0 / 0.8; pairwise | 0.0005 / 0.0015; 0.0020 | confirmed |
| I2 | R2 soft DPO final entropy / target; hard DPO entropy t = 1 | 0.382 / 0.381; 0.0082 | confirmed |
| I3 | R3 balanced amplitude alpha = 0.2 neural / population; decisive; alpha = 0.9 balanced | 0.307 / 0.306; 0.0095; 0.027 | confirmed |
| I4 | frozen predictions: R1 target entropy, R2 soft target entropy, R3 alpha = 0.2 balanced unstable share and population amplitude | 0.818, 0.364, 0.97, 0.311 | frozen |

## Post-hoc checks of preregistered sub-predictions (computed 2026-10-17 from saved runs, `analysis/check_subpredictions.py`)
| # | Fact | Value | Status |
|---|---|---|---|
| K-rate-H1_a05_b4 | H1_a05_b4: per-round contraction of the median distance to the target while it exceeds 0.5 (neural / population; alpha = 0.5) | 0.519 / 0.484 over 2 rounds; neural distance floor at the end 0.13 | confirmed (post hoc) |
| K-rate-H1_a08_b16 | H1_a08_b16: per-round contraction of the median distance to the target while it exceeds 0.5 (neural / population; alpha = 0.8) | 0.815 / 0.794 over 8 rounds; neural distance floor at the end 0.16 | confirmed (post hoc) |
| K-rate-H1_a09_b08 | H1_a09_b08: per-round contraction of the median distance to the target while it exceeds 0.5 (neural / population; alpha = 0.9) | 0.907 / 0.897 over 16 rounds; neural distance floor at the end 0.28 | confirmed (post hoc) |
| K-sw-H1_a05_b1 | H1_a05_b1 (B = 2): ordered pairs inside the sandwich, target / neural final (of 1560); band width median; neural log-odds deviation from target median / q99 | 1560 / 704; 0.09; 0.06 / 0.24 | confirmed (post hoc) |
| K-sw-H2_a0.8 | H2_a0.8 (B = 5): ordered pairs inside the sandwich, target / neural final (of 1560); band width median; neural log-odds deviation from target median / q99 | 1560 / 726; 0.24; 0.10 / 0.46 | confirmed (post hoc) |
| K-sw-H2_a0.9 | H2_a0.9 (B = 10): ordered pairs inside the sandwich, target / neural final (of 1560); band width median; neural log-odds deviation from target median / q99 | 1560 / 788; 0.47; 0.15 / 0.73 | confirmed (post hoc) |
| K-sw-H2_a0.95 | H2_a0.95 (B = 20): ordered pairs inside the sandwich, target / neural final (of 1560); band width median; neural log-odds deviation from target median / q99 | 1560 / 728; 0.95; 0.18 / 0.96 | confirmed (post hoc) |
| K-cor8 | H3 panels on which every worst-case margin of the Condorcet winner is positive (cor:dominance applies) | 0 of 130 | confirmed (post hoc) |
| K-resid | per-round residual median in the oscillating runs H5 beta = 4, alpha = 0.9 / 0.95 and H6 alpha = 0.2 (q90) | 0.18 / 0.44 / 0.12 (q90 1.14 / 1.97 / 0.36) | confirmed |

| # | Fact | Value | Status |
|---|---|---|---|
| K-notrun | Not run: jury-oracle arm of H5 (only 1 of 130 jury panels lacks a Condorcet winner, so the planted control (iv) replaced it); target against the maximal lottery as B grows; actual DPO on non-realizable (cyclic) labels (Fan's review comment R46) | `prereg/PREREG.md`, v4 R46 | logged |
| K-design | H1 design revised after the pilot and before any H1 run: B = 4 vs B = 1 became B = 8 vs B = 2 (larger effect) and soft-label T fixed at 3.168 | `prereg/PREREG.md` | logged |


## Panels: examples and a data-quality check (`results/eval_panels_examples.json`, computed 2026-10-17)
| # | Fact | Value | Status |
|---|---|---|---|
| L1-q05 | q05: panels whose training-time initial probability (re-tokenized answer text, fp32) has some answer below 0.05 / below 0.01; panels with a log-prob shift > 1 nat between builder and training scores (of 130) | 13 / 6; 8 | confirmed |
| L1-q05_subset64 | q05_subset64: panels whose training-time initial probability (re-tokenized answer text, fp32) has some answer below 0.05 / below 0.01; panels with a log-prob shift > 1 nat between builder and training scores (of 64) | 8 / 5; 6 | confirmed |
| L1-q15_subset64 | q15_subset64: panels whose training-time initial probability (re-tokenized answer text, fp32) has some answer below 0.05 / below 0.01; panels with a log-prob shift > 1 nat between builder and training scores (of 64) | 6 / 1; 2 | confirmed |
| L1-cause | cause: the panel builder scored the sampled token ids; training re-tokenizes the stripped answer text (`vm/build_panels.py` lines 96-110, `vm/panel_po.py` lines 94-99); predictions and evaluations use the training-time scores s_0 | - | confirmed (code read) |
| L2-0 | panel 0: "Suggest a name for a pet golden retriever." answers ['Golden Max', 'Golden Maxime', 'Golden Poochie', 'Golden Prodigy'] | p0 [0.29, 0.44, 0.2, 0.06]; Skywork reward [-1.9, -4.9, -5.0, -4.1]; final H1 alpha = 0, beta = 8: [0.71, 0.18, 0.07, 0.04]; final H3 full refresh: [1.0, 0.0, 0.0, 0.0] | confirmed |
| L2-27 | panel 27: "Name one person of Indian Origin." answers ['Mahatma Gandhi', 'J.K. Rowling', 'Indira Gandhi', 'Gandhi'] | p0 [0.34, 0.28, 0.22, 0.15]; Skywork reward [-1.3, -25.0, -1.7, -2.5]; final H1 alpha = 0, beta = 8: [0.57, 0.01, 0.29, 0.13]; final H3 full refresh: [1.0, 0.0, 0.0, 0.0] | confirmed |
| L2-117 | panel 117: "What tasks can you do?" answers ['Answer questions, provide information.', 'I can understand human language and provide information.', 'Asking questions, generating text, providing insights.', "I'm ready for inquiries and assistance."] | p0 [0.27, 0.55, 0.12, 0.06]; Skywork reward [-0.6, -8.7, -4.4, -7.5]; final H1 alpha = 0, beta = 8: [0.88, 0.05, 0.06, 0.01]; final H3 full refresh: [1.0, 0.0, 0.0, 0.0] | confirmed |
| K-resid-runs | main runs (batches 1-2) with per-round residual median <= 0.1 nats; the exceptions with their medians | 15 of 19; H4_dpo_hard 101.93, H5_b4_a09 0.18, H5_b4_a095 0.44, H6_B12_a02 0.12 | confirmed |
| K-resid-range | residual median range over the 15 passing runs | 0.065-0.100 | confirmed |
| I1b | 1.5B R1: KL from the policy averaged over the last 10 rounds to the B = 8 target, alpha = 0 / 0.8 (the measure in v4 Table 2); frozen R1 criterion uses the final KL <= 0.02 | 0.00005 / 0.0008 (final 0.0005 / 0.0015) | confirmed (recomputed 2026-10-17) |

## Further facts used in the text report (added 2026-10-17)
| # | Fact | Value | Status |
|---|---|---|---|
| A9 | Rounds per run: H1 30; H2 30 / 40 / 70 (alpha = 0.8 / 0.9 / 0.95); H3 40; H4 60; H5, H6 80; S1 30; S2 40; S3 80; R1, R2 30; R3 60. Panels: 130 in batch 1, 64 elsewhere | `results/runs*/<run>/run.npz` shapes | confirmed |
| C-ratio | smallest control distance over largest within-B=8 distance: 0.207 / 0.0076 | 27 | derived |
| D-KL | H2 KL final to target, alpha = 0.5 / 0.8 / 0.9 / 0.95 | 0.0013 / 0.0029 / 0.0043 / 0.0023 | confirmed (`eval_main.json`) |
| G-cells | H5/H6 oscillating shares, neural vs population, differ by at most one panel of 32 in every one of the 14 cells (0.9375 vs 0.90625; 0.125 vs 0.15625; 0.03125 vs 0) | max difference 1 panel | derived |
| G-amp | balanced amplitudes in the four oscillating configurations differ from the population by at most 0.004 (0.181 vs 0.185) | 0.004 | derived |
| G-fig5 | Figure 5 panel (id 0, balanced): alpha = 0.2 probability of the plotted answer over the last 30 rounds ranges 0.032-0.949; alpha = 0.9 final 0.411 (population 0.412) | as stated | confirmed |
| H-noise | n = 32 diagnosis: ratio of logit standard deviations alpha = 0 vs 0.8 is 3.4 (linear-noise prediction 3.0); in probability space 22, because at beta = 8 the policy jumps between nearly deterministic states | 3.4 / 3.0 | logged |
| I5 | 1.5B: residual medians 0.063-0.097; hard-label DPO median top-two log-odds gap 10, 81, 156, 179, 179 at t = 1, 5, 10, 20, 30 (plateau from numerical floors on token log-probabilities) | as stated | logged |
| L3 | mean initial panel entropy under builder scores / training-time scores (0.5B, 130 panels) | 1.159 / 1.140 | confirmed |


## Training loss per round (`analysis/round_losses.py` -> `results/eval_round_losses.json`; `shared/gen_dynamics.py` -> `shared/dynamics_facts.json`; computed 2026-10-17, no new training)
| # | Fact | Value | Status |
|---|---|---|---|
| N0 | logged training loss: only inner steps 1 and 40 of each round were logged (`vm/panel_po.py` lines 184-187); intermediate steps were not | - | confirmed (code read) |
| N1 | largest relative difference between the logged step-1 loss and the step-1 loss recomputed from the saved scores, over all expected-label runs | 1.5e-05 | confirmed |
| N2 | first round: loss after the round minus the round optimum, range over the 0.5B IPO runs | 0.09-0.14 | confirmed |
| N3 | oscillating runs (H5 alpha = 0.9, 0.95; H6 alpha = 0.2, 0.5): median excess loss at the start / at the end of a round | 0.47-3.16 / 0.0062-0.0114 | confirmed |
| N4 | converging IPO runs (12 runs): median excess at the start of a round; median excess at the end from round 5 on | 0.0012-0.0141; 0.0047-0.0131 | confirmed |
| N5 | converging IPO runs: share of rounds whose logged step-40 loss is below the step-1 loss (range); full refresh H3 and oscillating runs | 0.00-0.74; 1.00 | confirmed |
| N6 | soft-label DPO (H4): median excess at the end of a round (logistic loss scale) | 0.0006 | confirmed |
| N7 | full refresh H3: median excess at the start / end of a round | 0.250 / 0.0142 | confirmed |
| O-slow | slow oscillation (H5 alpha = 0.95, beta = 4) on panels 0, 1, 5: period of the answer swings (rounds between peaks above 0.5, neural and population, range); largest neural-population gap in period; largest gap in mean peak height | 14.2-18.7; 1.6; 0.04 | confirmed (derived from examples_series.json) |
| O-slow-phase | slow oscillation phase: neural peaks vs population peaks by round ~70 (panel 0: 65/66, 69/70, 75/76; panel 1: 69/66, 64/61, 74/72); panel 5: order of the first swings differs (answer 4 peaks at 8 vs 17) | 0-3 rounds on panels 0 and 1; larger on panel 5 | confirmed (derived) |

## Representative prompts (panels 0, 1, 5; all balanced under the planted oracle; `shared/examples_series.json`)
| # | Prompt / run | Final panel policy; min-max over the last 30 rounds (answers in panel order) | Status |
|---|---|---|---|
| O-p0-0 | panel 0 "Suggest a name for a pet golden retriever.": answers ['Golden Max', 'Golden Maxime', 'Golden Poochie', 'Golden Prodigy'] | initial panel policy [0.29, 0.44, 0.2, 0.06] | confirmed |
| O-p0-1 | panel 1 "Suggest a name for a robot vacuum cleaner.": answers ['AI Poochie', 'IntelligentSweep2000', 'QuantumGuru Vacuum', 'Mopman'] | initial panel policy [0.7, 0.13, 0.1, 0.07] | confirmed |
| O-p0-5 | panel 5 "Write a slogan for a public library.": answers ['Welcome to the heart of knowledge.', 'Welcome to our reading haven.', 'Read, Learn, Explore.', 'Reading is the Key to Life.'] | initial panel policy [0.29, 0.34, 0.27, 0.1] | confirmed |
| O-0-H1_a08_b16 | panel 0, H1_a08_b16 | final [0.7, 0.18, 0.08, 0.04]; last-30 min [0.3, 0.18, 0.06, 0.03], max [0.71, 0.42, 0.21, 0.08] | confirmed |
| O-0-H3_a1_b1 | panel 0, H3_a1_b1 | final [1.0, 0.0, 0.0, 0.0]; last-30 min [0.99, 0.0, 0.0, 0.0], max [1.0, 0.01, 0.0, 0.0] | confirmed |
| O-0-H4_dpo_hard | panel 0, H4_dpo_hard | final [1.0, 0.0, 0.0, 0.0]; last-30 min [1.0, 0.0, 0.0, 0.0], max [1.0, 0.0, 0.0, 0.0] | confirmed |
| O-0-H6_B12_a02 | panel 0, H6_B12_a02 | final [0.65, 0.06, 0.02, 0.28]; last-30 min [0.0, 0.03, 0.0, 0.0], max [0.82, 0.95, 0.02, 0.81] | confirmed |
| O-0-H5_b4_a095 | panel 0, H5_b4_a095 | final [0.56, 0.0, 0.0, 0.44]; last-30 min [0.0, 0.0, 0.0, 0.01], max [0.92, 0.92, 0.0, 0.97] | confirmed |
| O-0-H6_B12_a09 | panel 0, H6_B12_a09 | final [0.21, 0.41, 0.01, 0.37]; last-30 min [0.18, 0.38, 0.01, 0.3], max [0.24, 0.5, 0.01, 0.4] | confirmed |
| O-1-H1_a08_b16 | panel 1, H1_a08_b16 | final [0.27, 0.7, 0.02, 0.01]; last-30 min [0.22, 0.16, 0.01, 0.01], max [0.71, 0.76, 0.05, 0.08] | confirmed |
| O-1-H3_a1_b1 | panel 1, H3_a1_b1 | final [0.0, 1.0, 0.0, 0.0]; last-30 min [0.0, 0.95, 0.0, 0.0], max [0.05, 1.0, 0.0, 0.0] | confirmed |
| O-1-H4_dpo_hard | panel 1, H4_dpo_hard | final [0.0, 1.0, 0.0, 0.0]; last-30 min [0.0, 1.0, 0.0, 0.0], max [0.0, 1.0, 0.0, 0.0] | confirmed |
| O-1-H6_B12_a02 | panel 1, H6_B12_a02 | final [0.02, 0.74, 0.21, 0.02]; last-30 min [0.02, 0.01, 0.01, 0.02], max [0.08, 0.83, 0.75, 0.89] | confirmed |
| O-1-H5_b4_a095 | panel 1, H5_b4_a095 | final [0.0, 0.0, 0.71, 0.29]; last-30 min [0.0, 0.0, 0.0, 0.01], max [0.0, 0.91, 0.93, 0.97] | confirmed |
| O-1-H6_B12_a09 | panel 1, H6_B12_a09 | final [0.07, 0.24, 0.29, 0.4]; last-30 min [0.06, 0.22, 0.22, 0.35], max [0.07, 0.33, 0.32, 0.44] | confirmed |
| O-5-H1_a08_b16 | panel 5, H1_a08_b16 | final [0.03, 0.05, 0.91, 0.01]; last-30 min [0.03, 0.05, 0.49, 0.01], max [0.24, 0.28, 0.91, 0.07] | confirmed |
| O-5-H3_a1_b1 | panel 5, H3_a1_b1 | final [0.0, 0.0, 1.0, 0.0]; last-30 min [0.0, 0.0, 0.99, 0.0], max [0.0, 0.0, 1.0, 0.0] | confirmed |
| O-5-H4_dpo_hard | panel 5, H4_dpo_hard | final [0.0, 0.0, 1.0, 0.0]; last-30 min [0.0, 0.0, 1.0, 0.0], max [0.0, 0.0, 1.0, 0.0] | confirmed |
| O-5-H6_B12_a02 | panel 5, H6_B12_a02 | final [0.01, 0.47, 0.52, 0.01]; last-30 min [0.0, 0.01, 0.03, 0.0], max [0.01, 0.83, 0.91, 0.75] | confirmed |
| O-5-H5_b4_a095 | panel 5, H5_b4_a095 | final [0.0, 0.05, 0.05, 0.9]; last-30 min [0.0, 0.0, 0.01, 0.0], max [0.0, 0.94, 0.97, 0.91] | confirmed |
| O-5-H6_B12_a09 | panel 5, H6_B12_a09 | final [0.01, 0.49, 0.3, 0.2]; last-30 min [0.01, 0.38, 0.29, 0.15], max [0.01, 0.51, 0.42, 0.23] | confirmed |

## v4 Section 8 text against the record (read 2026-10-17, `paper/mirror/overleaf/JMLR-B-v4/jmlr_main.tex` lines 939-1020)
| # | Fact | Value | Status |
|---|---|---|---|
| M1 | v4 says every kept panel has initial probability >= 0.05 for every answer | true for the builder scores; false at training time on 13 of 130 panels (L1) | confirmed |
| M2 | v4 says the hypotheses and pass criteria were frozen before training | the oscillation measure was amended after 2 of 7 batch-2 runs and again after all batch-2 results were in (thresholds from batch 1), and the batch-3b criteria were set after batch 3 failed, before the 3b runs | confirmed |
| M3 | v4 robustness text reports the 512-pair results and the 32-pair link test, not the failed 32-pair same-target test or the inconclusive 32-pair cycle test | omission | confirmed |
| M4 | v4 Table 2 numbers match the recomputed values (time-averaged same-target KL; link; cycle amplitudes) | match | confirmed |


## Practice (natural feedback channels)
| # | Fact | Source | Status |
|---|---|---|---|
| J1 | 3-RM jury (vote share) on the 130 panels: 1 panel without a Condorcet winner; pairwise ordering agreement between the RMs 0.683-0.754 | session analysis 2026-10-16 of `data/panels_q05.json` | confirmed |
| J2 | SPPO released training panels (UCLA-AGI data-mistral-7b-instruct-sppo-iter1/3; 600 + 600 sampled): 0 majority 3-cycles; identity cyclic share 0.118 / 0.113 vs 0.200 for a hard order | `reviews/2026-10-15-sppo-pairrm-probe/results/hodge_out.jsonl` | confirmed |
| J3 | Yu He's Oracle1 runs (snapshot 2026-09-11): IPO entropy vs M_t coincides across alpha (1.11-1.14 at M_t = 8), plateaus about 1.26 / 0.94-0.96 / 0.61-0.64 for alpha = 0.8 / 0.9 / 0.95; hard-label DPO alpha = 0.8 entropy 0.44 at t = 10, 0.16 at t = 80 | `paper/reviews/2026-10-15-concentration-reframe/analysis/oracle1_entropy_vs_horizon.json` | confirmed (single seed) |
| J4 | v4 commits containing these results: 80533b0 (Sections 8.1-8.5, 8.7) and 4c4731e (8.6, Table 2), pushed 2026-10-17 before Fan asked to approve first | `paper/mirror/overleaf` git log | confirmed |


## 2026-10-18: controls chapter added to the visual deck (slides 15-24) and folded into slides 1, 2, 25-28

The numbers come from `reviews/2026-10-18-controls-neural/shared/sources.md` (the controls fact sheet; row ids below refer to it) and are read in the generator (`visual/build/slides_controls.py`) from `experiments/2026-10-17-controls/results/eval_controls.json` and `reviews/2026-10-18-controls-neural/shared/{fig_facts,dyn_facts,dyn_series}.json`. Nothing was recomputed. The deck went from 18 to 28 slides; the previous files are kept as `visual/*.prev-2026-10-17.*`.

| Slide | Change | Controls fact-sheet rows | Status |
|---|---|---|---|
| header | centre title "v4 neural validation + feedback controls"; new link "Controls report" to `reviews/2026-10-18-controls-neural/text/report.html` | - | - |
| 1 Title | meta line adds "16 control runs, 10-17/18", "updated 2026-10-18" | H2, H4 | confirmed |
| 2 The answer | new banner linking to slides 15-24 (four of six frozen tests pass; step budget and weight averaging fail); "Still open" now says 4 tests not run, and "v4 Section 8 revised and pushed (3b032bd); the controls await your decision (slide 28)" replaces "Section 8 needs 3 fixes" | E*-v, E3-v, E6-v; T3; P1 | confirmed |
| 15 Controls · the answer (new) | verdict grid E1-E6 and E3′ with observed values; how it was run; the two failures (my reading); take (revised after review): four controls worked as frozen predictions said, the two that failed act through shared weights, which the population model does not represent (second half our reading) | E1-C1_*, E2-*, E3-pr, E3-dr, E3s, E3s2, E4-*, E5, E5b, E6, S1, S2, S3, S5, H2, H4, H5, H7, I2 | confirmed; costs approximate; reading ours |
| 16 Controls · setting (new) | loop with the five control points and what each changes in the round | T1, T2, T4, T5, T6, T7, S1 | theory (ours) |
| 17 Controls · training loss (new) | per-round sawtooth (logged step 1 → step 40) against each round's exact optimum for per-round KL control, step budget and refresh-then-anchor; excess-loss summary of all 16 runs | L-*, H6, H8, E3s2; series `dyn_series.json` loss | confirmed; ln 2 start derived |
| 18 Controls · E1 judge (new, interactive) | buttons ε = 0 / 0.3 / 0.45 / 0.55 / 0.55 frozen; end entropy vs prediction; gauge of α + βε; gap at ε = 0.55 | E1-C1_*, E1-gap, E1-v, T1, I1 | confirmed; α + βε and the factor 2.1 derived from T1 |
| 19 Controls · E2 anchor (new) | schematic of the two references; entropy vs KL to the initial model, four runs with predictions | E2-*, E2-v, T2 | confirmed; left panel schematic |
| 20 Controls · E3 drift (new) | end entropy of per-round KL control, step budget and refresh-then-anchor; schematic of the missing restoring force; zero-step share and loss blow-up | E3-pr, E3-dr, E3-dr2, E3s2, H3, H6, H7 | confirmed; right panel schematic; per-round panel count from fig_facts.json |
| 21 Controls · E3′ rerun (new) | per-panel KL to the B = 8 target (median, 75th, 90th percentile, count above 0.1); rounds until the median KL < 0.001 | E3s, E3s2, E3s3, E2-C2_anchorB8, H3, H4, H9; quartiles from fig_facts.json | confirmed; speed comparison post hoc |
| 22 Controls · E4-E5 cycles (new) | oscillating share with and without replay at α = 0.5 and 0.8; adaptive α at rounds 1, 21, 41, 61, 81, 101, 150 against prediction and threshold (re-scored 2026-10-18: trained and predicted α now sampled at the same rounds, so the dots coincide through round 61; the earlier one-step lag was an evaluator indexing error, verdicts unchanged); the graduate-student panel under adaptive refresh | E4-*, E4-v4, E5, E5b, E5c, I3, S3, S4, X30-C5_adapt | confirmed; overshoot reading ours |
| 23 Controls · E6 deploy (new) | median KL to the target for last iterate, weight average, log-probability average, random checkpoint per request; one rotating panel with its last-30 average | E6, E6b, E6-v, I2, X30-C6_wavg | confirmed |
| 24 Try it · controls (new, interactive) | two prompts (best book: reward-model labels; graduate student: planted cycle), six settings each, trained vs population, regime tags, target or average ticks | X35-*, X30-* | confirmed |
| 25 What did not go as planned | new ledger row for the controls batch: the two failures, the logged E3 condition change and rerun, the four frozen passes; take adds the E3′ controller; "push 3b032bd" pill on the Post hoc · 10-17 row | E3-dr, E3-dr2, E6, E6b, H3, S5, P1 | confirmed, logged |
| 26 Limits | new "not run" card for pair weights (Prop. 3); new card: one seed per control setting, E4-E6 on planted cycles only, E3′ speed-up post hoc; headline "four tests"; take adds single runs | T3, S3, H9, fact-sheet header (single seeds) | confirmed |
| 27 Practice | six rules from the controls batch (frozen or debiased judge; anchor rather than bonus; budget drift with a restoring force, never per-round KL control; replay only with fast refresh; adapt α at fixed B; serve a random recent checkpoint) | E1-*, T1, E2-*, E3-*, E3s*, E4-*, E5, E6 | rules are our reading, one seed each |
| 28 Decision | rewritten after review: timeline ends with 3b032bd (Section 8 revised after Fan approved the v4 reports, pushed 2026-10-17); M1-M3 shown as resolved (panel-filter caveat, frozen-criteria wording, 32-pair failure disclosed; plus a loss sentence and the new 8.6 figure); the open decision is the controls, with the text report's Section 8 options (a) Controls subsection, (b) appendix, (c) not yet, two more seeds, about 2 VM hours; nothing edited for the controls; cost line for both batches | v4 J4, M1-M3, A7, A8; controls P1, C1, H2, H4, H5; controls text report Section 8 | confirmed; costs approximate; C1 an estimate |

Values on the slides that are derived or come from a cited file rather than a row of the controls fact sheet:
- α + βε = 0.50 / 0.80 / 0.95 / 1.05 and the frozen-judge factor 2.1 = 1 + βε/(1 − α): T1 with α = 0.5, β = 1 (derived).
- "each round starts at ln 2 = 0.693" under full refresh: the logged step-1 loss 0.69315 in `dyn_series.json` (derived reading).
- "In fifteen runs": the 16 runs minus the step budget (derived count); "half the rounds": 8 vs 16 (H9, derived).
- "0.125 of the initial model": 1 − α at α = 0.875; α = 0.875 / 0.75 for B = 8 / 4 and the bonus as judge bias ε = −η: `experiments/2026-10-17-controls/prereg/PREREG.md`, E2.
- β = 6 and 2.4 at α = 0.5 and 0.8 (B = 12): PREREG E4 (also (1 − α)B, derived). The adaptive rule (every 10 rounds, 20%, 30% of the way to 1): PREREG E5. β = 9.6 and weights averaged over rounds 51-80: PREREG E6. The drift budget as each panel's drift at its B = 8 target: PREREG E3.
- Per-round KL control: median KL 0.0132, 90th percentile 0.35 and 16 of 64 panels above 0.1; step budget 90th percentile 4.04; step budget below 0.01 at round 30: `shared/fig_facts.json` (E3, E3_speed) of the controls report.
- 32 balanced planted panels: v4 fact sheet, H5/H6 section header.
- Slide 28 option texts paraphrase the controls text report, Section 8 ("seven tests" = E1, E2, E3, E3′, E4, E5, E6).
- Rounded for display: 1.60 / 1.26 / 1.38 / 0.93 (E2: 1.602, 1.255, 1.376, 0.931); 1.26 (E3s3 1.259); 0.91 / 0.73 (E5: 0.906, 0.726); 0.052 / 0.016 (E6: 0.0519, 0.0157); 77% (E3-dr2 0.77); 0.0132 (fig_facts 0.01325).

## Update 2026-10-18 (late): the X1/X9 chapter

X1 (actual DPO on planted cycles) and X9 (inner Adam steps) were added as slides 25–31 after the controls chapter. The slides read their numbers from `reviews/2026-10-18-dpo-inner-neural/shared/facts.json` and `series.json` through `build/slides_x1x9.py`. Row ids refer to `reviews/2026-10-18-dpo-inner-neural/shared/sources.md`.

| Slide | Change | Source rows | Status |
|---|---|---|---|
| 2 The answer | Added a second banner for X1–X9. "Still open" now reads "controls and JMLR edits pushed (133d64c)" and "3 tests not run" (R46 is now run as X1). | X1c, X9c; the 133d64c push (2026-10-18, after Fan approved preview 7817d1de) | confirmed |
| 25 X1–X9 · the answer | New. Verdict grid (X1 pass; X9 ordinal fail; calibrated prediction reported only; target check not testable) and the conduct box. | X1a–c, X9a–f, S2, S3, S7, S8 | confirmed; cost approximate; right box my reading |
| 26 X1–X9 · training loss | New. Logged per-round loss against the exact round optimum for X1 (B = 4) and X9 (k = 10, 3, 1); medians over rounds 5+ of the excess before → after the inner steps. | X1k, X9j | confirmed |
| 27 X1 · actual DPO | New. Shares against T1 and the logit map with the pass band; per-panel agreement and one-step error. | X1a–b, X1e–f | confirmed; X1e–f post hoc |
| 28 Try it · X1 | New, interactive (3 prompts). Trained model with T1's map, and the logit map, for both settings. The maps are drawn at every second round to keep the single file under 512 KB. | X1i | confirmed |
| 29 X9 · inner steps | New. Share oscillating against inner steps (balanced, decisive) with the frozen gradient-descent reference; calibrated predictions. | X9a–b, X9d, X9g | confirmed; X9d reported only |
| 30 X9 · why | New. Median fraction of the exact move by quartile; loss above the optimum before and after the steps. | X9h–j | post hoc; Adam explanation my reading |
| 31 Try it · X9 | New, interactive (3 prompts). Trajectories at k = 40, 10, 3, 1 with target ticks. | X9k | confirmed |
| 32 What did not go as planned | Controls row now shows push 133d64c and is shortened. New row: X1–X9 batch. | X1c, X9c | confirmed |
| 33 Limits | "Actual DPO on cyclic labels" removed from "Not run". New card: X1–X9 batch (R46 now run as X1; X9 diagnosis post hoc; T6 under gradient descent untested). Headline now says three tests were not run. | X1c, X9h | confirmed |
| 35 Decision | Rewritten. Timeline adds a6edf05 and 133d64c. Three cards cover what X1–X9 change, plus options (a) X1 with T1 into Section 8.5, (b) X9 as a limitation, (c) test T6 with SGD. The spend line adds about 1.6 h and $11. | S7, S8, X1c, X9c, X9f; options from the X1/X9 text report, Section 7 | cost approximate; recommendation my reading |

**Encoding.** The prompt series on slides 24 (controls), 28 and 31 are now coded as one character per probability, with 64 levels (resolution 1/63 ≈ 0.016). Before, slide 24 used integer percent. The plotted curves are unchanged at slide scale. This was done to keep the single file under 512 KB (now 504 KB).

**Rounded for display.** 0.0007 (X1k 0.00066); 3.77 and 1.05 (X9j); shares as n/32 (X1a–b, X9a–b).
