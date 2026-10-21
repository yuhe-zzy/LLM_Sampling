# Oracle1 / Qwen3 generated-output dynamics

Authorized by the owner on 2026-10-15. This is a NEW six-arm campaign,
not a rewrite, recovery or continuation of the old Oracle2 experiments.
Training preferences AND generated-response win rates use Oracle1 only:
`nvidia/Llama-3.1-Nemotron-70B-Reward-HF`. Skywork is not used.

## Scientific settings

- Policy: pinned `Qwen/Qwen3-4B`, BF16, non-thinking chat template, LoRA.
- Data: exact existing disjoint100 calibration/500 train/200 evaluation panels,
  four real responses each. Reuse hash-verified Nemotron scores, rebuild
  `P_ij=sigmoid(reward_i-reward_j)` at temperature1. No old mixed P/group labels.
- Six arms: IPO/DPO x ordinary/reference/feedback, seed0, alpha=.9,
  lambda_current=.8, IPO beta=.2, DPO beta=.8, reference nu=.45,
  feedback kappa=.5. Sequence-sum including response EOS, never token-average.
- Reference `(1-alpha)s0+(alpha-nu)st+nu*s(t-1)`. IPO feedback uses the
  existing history operator. For a SINGLE BT reward, DPO's exact optimized
  reward increment is centered reward/beta, independent of sampling. Its
  feedback extrapolation is exactly zero. This arm is a requested null control,
  not an intervention expected to improve stability; do not manufacture solver
  noise. Reference changes can still have an effect.
-100 outer updates, recording states0..100. Same fixed-panel pair weighting,
  optimizer reset per outer step, training settings as the reviewed plan.
- Open generation at0,10,...100:200 prompts x4 answers, temperature.8,
  top_p.95, top_k0, cap256 new tokens. Shared freshly generated INITIAL Qwen3
  baseline uses separate frozen seeds. No reuse of the old1.5B model baseline.
- Oracle1 WR: average sigmoid reward differences over all4x4 answer pairs,
  then equal weight per prompt. Majority/tie-.5 WR is secondary. Offline
  scoring after complete generation, NOT an immediately available training log.
- Five simultaneous0..10 traits: elaboration, linguistic complexity, formality,
  politeness, hedging. Frozen Qwen3-32B BF16 non-thinking greedy rater, response
  only; no quality, arm, checkpoint or oracle information in rater input.
  Explicit null for unrateable text, never zero-fill. This is an operational
  measurement rubric, not a claimed validated five-item scale.
-200-response calibration pilot requires HUMAN review before bulk annotation.
  Do not fabricate review agreement. Full main annotation is2,000 original
  training responses +53,600 generated responses including baseline. Exact-text
  cache avoids duplicate inference but retains every observation's weight.

## Interpretation and analysis

Primary endpoint: generated-response trait means/distributions over outer step,
not just probabilities of fixed training responses. `analyze.py` emits raw
five-trait trajectories, Oracle1 WR and a descriptive2D PCA view. PCA is fitted
ONLY on original responses from seed0 selected80% of training prompts; axes,
centering, scaling and sign conventions stay fixed across arms/time. No selecting
axes or traits to maximize visible trajectory oscillations. Complete-prompt
cohorts remain fixed across all arms/states; missing coverage is explicit.

PCA is NOT an estimated cyclic preference operator. Oracle1 per-prompt exact
BT-logit Hodge cyclic residual is zero. A fitted-feature or aggregate residual
would not establish intrinsic cyclic ground truth. Ordinary drift, short-period
jitter and repeated oscillation must be distinguished; one seed supports
descriptive comparisons, not a robust seed-level conclusion. Prompt SE bands
do not quantify training-seed or annotation uncertainty. Preserve length-cap,
length/repetition and null-label diagnostics before interpreting trajectories.
The same-model-family policy/rater is not independent human validation.

## Durable continuation

Every outer boundary atomically commits adapter + s0/st/s(t-1) + token counts +
Python/NumPy/Torch/CUDA RNG + metrics/snapshot + file hashes. Temporary partial
directories are preserved but never treated as completed states. Resume checks
every committed hash/contiguous step and re-scores the loaded adapter against
the saved sequence sums BEFORE training. A round interrupted before commit is
replayed from its last complete boundary. There is no mid-round optimizer
resume: the protocol explicitly resets AdamW/scheduler EVERY outer iteration.

First run IPO/DPO ordinary for one real outer update; next allocation reloads
and runs one further update for both. Those states belong to the actual runs,
not disposable trials. Only after both complete successfully advance all six
arms in segments of at most10 updates, with18h soft stop/24h Slurm limit.
Signals request completion of the current boundary. Generation, WR and labels
have atomic per-response journals and deterministic identities; completed
responses are reused without overwrite. Hard kill/TIMEOUT/FAILED requires
diagnosis and accounting-backed lock recovery, never an automatic blind retry.

## Resources and safety

FOUR GPUs maximum for the FULL account. This campaign starts with two concurrent
one-H100 training/generation tasks; WR is one two-H100 job, annotation one H100.
The original launcher keeps phases nonoverlapping. October19 the owner explicitly
approved two extra H100s for interim0..80 WR alongside training, still FOUR total;
use only the [bound cooperative extension](campaigns/oracle1_qwen3_20261015/INTERIM_WR_80.md)
for this exception. The legacy pending-worker startup restriction must first clear.
Before submission and GPU loading, inspect full
`squeue -a -r` by owner/UID448057 and `scontrol -a show job`, count generic
AllocTRES GPU once including releasing allocations, then check sacct. Unrelated
running OR pending work blocks new submission. These are operational guards,
not an administrator-enforced account cap against independent submissions.

Every submission has exclusive immutable intent then receipt. Ambiguous intent,
non-successful predecessor, changed source/data/model/rubric or stale writer lock
blocks continuation. Investigate, do not delete the guard or blindly sbatch.
No credential/raw text/weights/adapters should be committed to GitHub.
Old Oracle2 jobs/data/source remain untouched; do not resume its recovery/WR.

## Entry points

Run with the frozen server Python under an immutable launch root:

```text
python queue_campaign.py --root ROOT --inspect
python queue_campaign.py --root ROOT --advance
python queue_campaign.py --root ROOT --advance --submit
python analyze.py --root ROOT --output NEW_RESULT_DIRECTORY
```

`advance` checks prior receipts and chooses the next missing authorized phase.
Repeated healthy segments resume, never start from0. Only the monitor/operator
calls it after verification; workers cannot self-submit or bypass the queue.
`worker.py` dispatches only the phase/index/update count in its scheduler receipt.
`freeze.py` seals source and verifies model/data identity and all six configs.

CPU tests cover byte-level corruption, duplicate guards, formulae, labels, seeds,
WR, phase transitions, RNG and real tiny Qwen3/LoRA train-save-load-train continuity.
GPU memory and actual full-model reload remain separate startup validations.

Campaign plan and dated evidence: [2026-10-15](campaigns/oracle1_qwen3_20261015/README.md).
