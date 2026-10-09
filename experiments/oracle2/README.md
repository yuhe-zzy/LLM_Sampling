# Oracle2: real fixed candidates and generated-response win rate

User approved the first campaign on **2026-10-09**, including the 6/4 mixture,
six configurations and 100/500/200 disjoint prompt split. Evaluation cadence is
**outer states 0,10,20,...,100**, eleven evaluations, not optimizer steps.
Implementation/authorization is not a submission receipt. See the dated
[campaign record](campaigns/oracle2_real_20261009/README.md) for actual status.

First GPU audit 4773811 failed before any score due to the Transformers 5
chat-template return-type default. The explicit-token-list repair is separate
from its frozen source. The user requested **code repair only, no resubmission**.
Corrected CPU deployment passed 101 tests and full data/tokenizer checks;
post-repair GPU inference remains untested. No retry or six-arm training was
submitted. See the campaign's latest status and validation records.

## Fixed judges and genuine pairwise mixing

Oracle1 remains `nvidia/Llama-3.1-Nemotron-70B-Reward-HF`. Oracle2 is the
comparator made from Nemotron and `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2`:

```text
P2(i > j | x) = .6 sigmoid((N_i-N_j)/T_N) + .4 sigmoid((S_i-S_j)/T_S)
T_N = T_S = 1, frozen for this first audit.
```

Skywork alone is not oracle2. We do not average scalar rewards before the
sigmoid, sample noisy hard training labels, or replace the mixed matrix with
a scalar ranking. Temperatures may not change silently. The .5/.5 and .7/.3
mixtures are offline audit diagnostics only, not extra training arms. Unequal
weights permit but do not guarantee cycles; saturation can suppress cycles.

`prepare_models.py` pins Skywork's Hub revision and inventories the existing
Nemotron/Qwen files without changing them. The lock records SHA256 for small
model/tokenizer files and download-revision/ETag/size provenance for weights;
it does not claim to rehash every old weight shard. Inference uses local
files and model-specific templates with no doubled BOS or text truncation.
Nemotron uses its causal-LM one-step reward interface; Skywork uses scalar
sequence-classification logits. Both are frozen BF16 models, loaded
**sequentially** in one three-H100 scoring allocation. This interface still
requires a fresh GPU check; CPU tests cannot establish inference correctness.

Primary model documentation:
[Skywork model card](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2),
[Nemotron model card](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward-HF).
Respect their model/base-model licenses. No redistribution of weights here.

## Data and pretraining grouping

Use the original four distinct real responses in each panel from the existing
`eval_prompt_responses_1000.jsonl`, verifying exact prompt/response membership
in raw HelpSteer. Dataset attribute scores and old synthetic preference
matrices are ignored. No response is invented, calibrated or edited.

Exclude an entire panel if any policy prompt+answer+EOS exceeds 1537 tokens,
if prompt+256 generation tokens exceeds 1537, or if either judge's formatted
conversation exceeds 4096 tokens. Filter before looking at reward scores.
Deduplicate normalized prompt text, shuffle with seed0, then split into
**100 calibration, 500 train, 200 held-out evaluation prompts**, four real
candidates each. Reject undersized data rather than silently shrinking it.
These prompts are held out from this campaign's adaptation, not guaranteed
unseen by reward-model pretraining (both judge/data provenance matter).

Score all 3,200 real responses once with each judge. Freeze all six pairwise
probabilities per panel, their reciprocals and diagonal .5. Before training:

- Cyclic: at least one directed triangle with all three probabilities >.52.
- Transitive: all six edges have |P-.5| >=.02 and no strict majority triangle.
- Ambiguous: everything else, retained and reported separately.

Audit group counts, disagreement, saturation, margins and DPO BT solver
residuals. A recorded review of the exact audit is required before training;
the software never infers that some mixture must be suitably cyclic. Missing
cyclic/transitive train or eval groups blocks training. Small nonzero groups
also need scientific review, not an automatic claim of adequate power.
Group labels concern this fixed candidate support, not every possible
generated answer. Calibration and evaluation prompts do not enter training.

## Six matched arms and unchanged sequence-sum engine

| Objective | Scheme | beta | alpha | lambda_current | nu | kappa |
|---|---|---:|---:|---:|---:|---:|
| IPO | ordinary | .2 | .9 | .8 | 0 | 0 |
| IPO | reference | .2 | .9 | .8 | .45 | 0 |
| IPO | feedback | .2 | .9 | .8 | 0 | .5 |
| DPO | ordinary | .8 | .9 | .8 | 0 | 0 |
| DPO | reference | .8 | .9 | .8 | .45 | 0 |
| DPO | feedback | .8 | .9 | .8 | 0 | .5 |

All start independently from the same Qwen2.5-1.5B policy, seed0, and train
100 outer updates with states 0..100. `s_t(i)` sums response token log
probabilities, including EOS and excluding prompt/padding; never token mean.

```text
q_t = softmax(s_t) on the four fixed candidates
mu_t = .2 Uniform(4) + .8 q_t
pair weights proportional to mu_t(i) mu_t(j), i<j
ordinary reference = .1 s_0 + .9 s_t
lagged reference   = .1 s_0 + .45 s_t + .45 s_(t-1)
feedback reference = ordinary reference + .5 (d_t - d_(t-1))
```

IPO uses `d_t = center((P2-.5) mu_t / beta)`. DPO uses the actual weighted
BT optimum divided by beta, not elementwise logit(P2). This BT projection
only determines the feedback displacement; training still uses the complete
mixed P2 matrix. At the first update previous=current, so history correction
is zero. State/reference/sampling weights stay fixed during each inner epoch.

With `z=(s_theta(i)-s_theta(j))-(reference(i)-reference(j))`, DPO uses
`BCEWithLogits(beta*z,P2_ij)`; IPO uses
`P2_ij*(z-1/(2 beta))^2+(1-P2_ij)*(z+1/(2 beta))^2`.
Every epoch enumerates all six unordered pairs with positive importance
weights, giving equal prompt mass. LoRA r16/alpha32/dropout0, BF16,
LR1e-5, batch1, accumulation4, **one epoch per outer update**, linear warmup
ratio .03, AdamW reset each outer iteration, gradient clip1. The tested
attention context excludes cuDNN SDPA backward throughout checkpoint
recomputation. Nonfinite loss/gradient/parameters fail rather than being masked.

The wrapper reuses the existing history engine without changing its old
protocols. Real-candidate initial scores are freshly computed and checked
finite, not forced to match a synthetic calibrated fixed point. Snapshot
scores/token counts, source commit, data/judge hashes and versions are retained.
The old synthetic-panel beta calibration does not prove these new settings
are stable; this is an empirical comparison, not an exact-theory gate.

## Open generation WR, every ten outer iterations

For each of the 200 evaluation prompts, generate **four initial-model baseline
responses once**, shared across every arm and checkpoint. Independently
generate four checkpoint responses at each state 0,10,...,100, with matched
random seeds across arms, temperature .8, top_p .95 and max_new_tokens256.
Use the same raw-prompt policy format as training. Store true step0 evaluation;
do not assign it .5 by definition. At each prompt compare all 4x4 new/base
pairs using the same fixed oracle2, then average prompts equally.

Primary metric: `oracle2_expected_win_rate`, the average mixed probability.
Secondary metric: `oracle2_majority_win_rate`, P2>.5 with .5 credit for exact
ties. This is not oracle1's historical strict-wins/ties-zero number. Summarize
all/cyclic/transitive/ambiguous prompts; standard errors treat prompts, not
the sixteen within-prompt comparisons, as independent units. Keep numeric
per-prompt results for paired comparisons between arms.

Adapters are saved every ten outer updates. Evaluation is a **post-training
batch over those saved states**, so six training processes need not each load
the 70B judge. It yields the requested cadence in outer time, but WR is not
available live during each update. The first version refuses incomplete
generation banks rather than silently dropping missing states. Baseline and
checkpoint banks are immutable; no overwrite/retry occurs automatically.

Track response length, EOS/max-length rate, token repetition and unique-token
ratio alongside WR. Every outer state also records panel q, panel entropy,
adjacent-step TV, and **relative-sequence entropy**:
`H(softmax(s_t-s_0))`, distinct from raw panel `H(softmax(s_t))`.
A WR plateau against one fixed opponent does not prove policy convergence
or globally meaningful human quality. A transitive fixed panel does not
guarantee its newly generated responses remain transitive.

## Execution, audit trail and privacy

CPU preparation -> three-GPU candidate scoring/audit -> review -> one-GPU
baseline generation -> six one-GPU training tasks -> six one-GPU generation
tasks -> one three-GPU scoring/summarization job. These are **nonoverlapping
phases**, with fresh full owner/UID queue checks before each submission.
`queue.py` requires an empty full account and preserves exclusive intent and
receipt files. No automated resubmission, phase chaining, cancellation or
account-wide hard scheduler cap is claimed. Pending jobs count in preflight.
Source bytes are frozen and verified at startup. All old campaigns remain intact.

Keep raw candidate text, generated text, score records, adapters and snapshots
outside Git under the private data/run roots. Publish reviewed code, frozen
configs, tests, numeric aggregate audits, receipts and later figures only.
Never put SSH credentials or access tokens in configuration or source.

CPU tests (no model downloads or allocations):

```bash
python -m unittest discover -s experiments/oracle2 -p 'test_oracle2.py' -v
python -m unittest discover -s experiments/cyclic_history -p 'test_*.py' -v
```
