# Preregistration: X11, mixed-criteria labels in the iterated loop (written 2026-10-19, before any measurement)

Approved by Fan on 2026-10-19 (translated: "approved; run the experiments and the theory in parallel, adding the new results to the current framework"). The theory is in `reviews/2026-10-19-mixed-criteria-cycles/theory.md`; its checks are in `analysis/theory_mix.py` there.

## Question
Each criterion alone is a scalar score, so it is transitive. When pairs are labelled under different criteria, and the context that fixes the criterion is hidden from the policy, does pooling put a cycle into the labels that the iterated loop follows? Do standard recipes then rotate between answers across rounds? And does slower refresh at the same target stop the rotation?

## Stage 0: real criteria on the existing panels (computed, no GPU)
- Script `analysis/predict_x11.py`, output `prereg/population_predictions.json`.
- Panels: the 130 v4 panels. Criteria:
  - quality: the Skywork-8B score;
  - concision: minus the number of characters;
  - originality: minus the base log-probability.
- Each criterion's labels are σ(gap/T_k), with T_k the standard deviation of that criterion's within-panel gaps.
- Result:
  - Pooling raises the mean logit-scale cyclic share from 0.017 (soft mixture) to 0.079 (salience routing) and 0.122 (margin filtering).
  - It gives a majority three-cycle on 1.5% and 3.1% of panels, and leaves no Condorcet winner on 1.5% and 1.5% (soft mixture: 1.5% and 0%). Corrected 2026-10-19 13:45 EDT: the first version called the three-cycle shares Condorcet-winner shares. The no-winner shares are in `prereg/stage0_no_condorcet_winner.json`.
  - Population DPO predicts oscillation on at most 1.5% of panels under every recipe.
- These criteria are nearly unrelated (mean Kendall τ between −0.09 and 0.19). They therefore look like the i.i.d. baseline, where systematic routing removes the Condorcet winner on 3–4% of four-answer panels.
- This motivates stage 1. One judge reads three contexts that each favour a different kind of answer.

## Stage 1: rubric-context labels (one VM pass, about 15 minutes, no training)
- **Script.** `vm/rubric_scores.py` scores every response of the 130 panels with each of the three reward models: Skywork-Reward-Llama-3.1-8B-v0.2 (the main judge, fixed now), Skywork-Reward-V2-Qwen3-1.7B, and GRM-Llama3.2-3B.
- **Contexts.** Each response is scored under the plain prompt and under three rubric contexts, each appended to the user prompt:
  - rigor: "I care most that the answer is clear, precise and accurately descriptive."
  - creativity: "I care most that the answer is creative and original, even unusual."
  - concision: "I care most that the answer is as short and punchy as possible."
- **Labels.** Each context's labels are σ(gap/T_k), with T_k the standard deviation of that context's within-panel gaps. This is the framework's `soft` rule.
- **Pooling rules.** The policy is trained on the plain prompt, so the context is hidden.
  - *soft:* the uniform mixture over contexts, so every pair gets each context with probability 1/3.
  - *salience:* each pair is labelled by the context whose standardized gap is largest.
  - *filtered:* each context keeps its most decisive third of pairs; a pair kept by no context falls back to the uniform mixture.
  - *visible:* the context is visible. Each context is its own transitive panel. This is reported from the population model only, as the control the theory says cannot cycle.
- **Measured per rule:**
  - the logit-scale Hodge cyclic share;
  - the share of panels without a Condorcet winner;
  - population DPO oscillation shares under the three recipes (below).
- **Run selection for stage 2.** The stage 2 runs are fixed from these population predictions, before any training, as follows.

## Stage 2: training (fixed after stage 1, before any training run)
- **Testbed.** As in `experiments/2026-10-18-dpo-inner/` (script `panel_po_ctrl.py`, unchanged): Qwen2.5-0.5B-Instruct, full fine-tuning, the DPO objective, λ = 1, 40 inner Adam steps at lr 2e-5, fp32, 120 rounds, with labels given as `--oracle npz:` with the pooled matrices.
- **Recipes.** All three have target sharpness B = 10:
  - A: fixed SFT reference, β_DPO = 0.1 (α = 0, β = 10);
  - B: α = 0.5, β = 5;
  - C (the remedy): α = 0.9, β = 1.
- **Oscillation.** A panel oscillates when its detrended probability amplitude over the last 30 rounds exceeds 0.0536 (the v4 rule).
- **Arms.**
  - The coupled rule with the most panels predicted to oscillate under recipe A, run under A and under C.
  - The soft rule under A, as the pooling control.
  - If no rule predicts oscillation on at least 8 panels, stage 2 is not run, and the stage 1 measurement is reported as the result.
- **Panels.** Every panel predicted to oscillate under the chosen rule and recipe A, plus panels predicted to settle, up to 64 panels in total, chosen with seed 0 and fixed before training.
- **Predictions.** Each arm's oscillating share is predicted by population DPO from the model's own initial scores. Pass: every arm's neural share is within 0.15 of its prediction. Directional pass: the coupled rule under A oscillates on a larger share than the same panels under C and than under soft A.
- **Decision rule.** As in the earlier batches. Failures are diagnosed and reported. Any change is logged here with a timestamp before a rerun.

## Log
- 2026-10-19: written before stage 1. Stage 0 numbers above are from `prereg/population_predictions.json`.
- **2026-10-19 13:10 EDT, stage 1 measured.** VM pass 12:55–13:07 EDT, then deallocated. Data: `data/rubric_scores.json`, sha256 prefix 8cb776df2ddce1f8. Script: `analysis/stage1_predict.py`, output `prereg/stage1_predictions.json`.
  - **Check.** Plain-prompt scores reproduce the panel file exactly for both Skywork models. GRM differs by up to 0.125, a bf16 difference, and is not used for decisions.
  - **The rubric contexts barely move the judges.** Mean pairwise sign agreement (Kendall τ), main judge:
    - with the plain prompt: rigor 0.84, concision 0.78, creativity 0.44;
    - between contexts: 0.46–0.79.
    - The three contexts pick three different top answers on only 7 of 130 panels (3 and 8 for the other judges).
  - **Pooling.** Main judge:
    - mean logit cyclic share 0.001 (soft), 0.035 (salience), 0.037 (filtered);
    - no pooled matrix without a Condorcet winner;
    - predicted oscillation 0% under every recipe.
    - Other judges: at most 0.8% without a Condorcet winner, 0% oscillation.
  - **Decision.** By the stage 2 rule (fewer than 8 panels predicted to oscillate), stage 2 is not run on these panels.
- **2026-10-19 13:20 EDT, amendment before any new measurement: stage 1b, style panels.** This is the approved X11 design ("writing panels with style answers (rigorous / creative / concise) from the base model"). The existing panels hold four answers of one kind, so no rubric has a specialist.
  - **Script.** `vm/style_panels.py`, run over the same prompt list as the v4 panels (up to 600 prompts, with the same short-answer suffix).
  - **Panel.** For each prompt, the policy samples M = 16 answers under the plain prompt and under each style instruction, appended after the suffix:
    - "Make it precise and accurately descriptive.";
    - "Make it creative and original, even whimsical.";
    - "Make it as short and punchy as possible."
    - The most probable valid answer under its own prompt is kept, giving four answers: plain, rigor, creativity, concision.
  - **Keep rule.** A panel is kept when the four answers are distinct and every initial probability under the plain prompt is at least 0.01. This is relaxed from the v4 rule of 0.05, because answers written to another instruction are less probable under the plain prompt. The number passing 0.05 is reported.
  - **Scoring.** The same three reward models, plain prompt plus the same three rubric contexts.
  - **Analysis.** Labels, pooling rules, recipes, the oscillation rule and the stage 2 selection rule are those of stage 1, with "130 panels" replaced by the kept style panels.
- **2026-10-19 13:40 EDT, stage 1b measured.**
  - **Run.** VM pass 13:12–13:36 EDT, then deallocated. Data: `data/style_panels.json`, sha256 prefix 1f088379ac73888a. Script: `analysis/stage1b_predict.py`, output `prereg/stage1b_predictions.json`.
  - **Panels.** 25 of 600 prompts give a style panel.
    - 470 fail the probability floor: an answer written to a style instruction is improbable under the plain prompt.
    - 71 lack a valid answer in some style; 34 repeat an answer.
    - 4 panels would pass the v4 floor of 0.05.
  - **Agreement.** The contexts disagree more than on the v4 panels. Main-judge Kendall τ: rigor–creativity 0.16, creativity–concision 0.17, rigor–concision 0.72.
  - **Specialists are absent.** Each context's top answer is the answer written in its own style on 16% (rigor), 48% (creativity) and 32% (concision) of panels; chance is 25%. No panel has all three; the other judges give 0 and 1 of 25.
  - **Pooling.** Main judge:
    - mean logit cyclic share 0.002 / 0.060 / 0.069 (soft / salience / filtered);
    - no panel without a Condorcet winner, for all three judges;
    - 0% predicted oscillation under every recipe.
  - **Decision.** By the stage 2 rule, stage 2 is not run. X11 ends with the measurement; no training run was made. VM use for X11 in total is about 40 minutes, about $5.
- **2026-10-19 14:40 EDT, post hoc solver recheck.** The shared DPO solver was fixed (see `experiments/2026-10-19-x10-x5-x12/prereg/PREREG.md`). Rerunning stages 0, 1 and 1b with it (`results/recheck_*.json`) leaves every reported number unchanged except stage 1's single-criterion "rigor" panel count under recipe A (1 → 0), a solver artefact. Decisions unchanged.
