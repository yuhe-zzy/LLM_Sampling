# Oracle1 / oracle2 design review, 2026-10-06

Historical design review. The later **2026-10-09** user approval and
[implementation/campaign record](../experiments/oracle2/campaigns/oracle2_real_20261009/README.md)
supersede the proposed-only status below. The approved main mixture is .6/.4,
with WR at outer states 0,10,...,100; no silent temperature changes.

Status: terminology confirmed by the user; implementation choices below are
recommendations. No new oracle model has been downloaded, deployed or tested
on the server, and no new oracle experiment is submitted by this document.

## Names and intended evaluation

- **oracle1**: the original frozen Nemotron-70B scalar reward oracle used for
  transitive IPO/DPO and generated-response win-rate evaluation.
- **oracle2**: a proposed mixture of oracle1's BT comparison probability and
  a second frozen reward model's BT comparison probability. The second reward
  model alone is not named oracle2. Neither component is a new trainable policy.

The user wants checkpoints to generate fresh responses to the same prompts,
then compare them with initial-policy responses using a fixed, extensible
judge. Dataset attribute scores exist only for stored responses and cannot
score arbitrary new generations. The synthetic fixed-panel preference matrix
likewise defines a panel game, not a judge on all possible strings. Its panel
expected WR must not substitute for the requested generated-response WR.

## Source and practical second-model choice

COMAL (arXiv:2410.23223v2), section 5.1, equation (4), uses a 50/50 mixture
of Skywork-Reward-Llama-3.1-8B-v0.2 and ArmoRM-Llama3-8B-v0.1 probabilities.
Our proposal retains Nemotron, so it is an adaptation, not an exact replication.

Official model cards checked on 2026-10-06:

- [Skywork](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2):
  public 8B reward-model weights; standard `AutoModelForSequenceClassification`
  with `num_labels=1`, model-specific chat template, and scalar `.logits`.
  Recommended first candidate because this interface is simpler to integrate.
  Respect the model card's Skywork license and applicable base-model terms.
- [ArmoRM](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1): also a
  candidate; custom code and multi-objective output. Its aggregated `.score`,
  not the entire `.rewards` vector, is the scalar for this BT component.
  Requires reviewing/pinning custom code and respecting Llama 3 terms.
- [Nemotron](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward-HF):
  original judge. The repository's `HelpfulnessRewardOracle` uses the supplied
  causal-LM reward interface; Skywork cannot be dropped into it by changing
  only the model-path argument. Preserve oracle1's existing behavior.

Both 8B candidates can score a prompt plus a new generated response without
fine-tuning. Public documentation is not a server inference/access test.
Do not silently upgrade the live training environment or use a newer model
revision under the same experimental name.

## Mixed probability, calibration and cyclic audit

For the same prompt and two responses a,b, propose

```text
P2(a > b | x) = w * sigmoid((r_N(x,a)-r_N(x,b))/T_N)
             + (1-w) * sigmoid((r_S(x,a)-r_S(x,b))/T_S).
```

Here w=.6 or .7 denotes the weight on Nemotron, not a chosen parameter yet.
T_N and T_S are positive frozen calibration temperatures. Start by inspecting
native-scale T=1 scores; any rescaling must be recorded and fixed before
training/evaluation, using a separate calibration subset. Do not standardize
anew at each checkpoint: that would change the evaluator during training.
Do not mix scalar scores and then take a single sigmoid; that remains BT.

Exactly equal weights give a transitive majority order: for two scaled score
differences a,b, sigmoid(a)+sigmoid(b)>1 iff a+b>0. Unequal weights permit
strict cycles but do not guarantee them. If both components saturate to hard
votes, any weight above .5 effectively follows the majority component's
ranking, also limiting cycles. Neither .7 nor a stronger mixture is inherently
more cyclic than .6. Audit disagreement, saturation, strict triangle frequency
and margins on real candidates before committing to a training configuration.

Compute full pairwise P on fixed candidate panels with at least three
responses. Define cyclic/transitive/ambiguous groups from this pre-training
audit, not from noisy sampled labels or favorable post-training trajectories.
Groups describe the inspected support: a transitive panel does not prove
the oracle is transitive over every response that can later be generated.
Freeze the grouping for primary comparisons; separately report how newly
generated responses alter any expanded-support cycle audit.

## Generated-response win rate

Freeze prompts, decoding settings, initial-policy baseline response bank,
model revisions, chat templates, length limits and mixture calibration.
At each checkpoint generate new responses and score each with both models.
For each prompt compare all checkpoint/base response pairs and average P2,
then average prompts equally. Call this `oracle2_expected_win_rate`, not an
unqualified copy of historical `oracle_win_rate`.

Optionally also report `oracle2_majority_win_rate` based on P2>.5, with a
declared tie rule (recommend .5 for exact ties). Keep it distinct from expected
WR and from oracle1's historical strict-wins/ties-zero metric. Store both
component scores/probabilities to audit disagreements. Oracle2 is a pairwise
evaluation system, not necessarily a globally transitive scalar quality rank.

Report generated WR for all prompts and the frozen cyclic/transitive groups.
Use paired prompts across arms and prompt-level uncertainty; the many response
pairs within one prompt are not independent samples. Keep actual step-0
generation evaluation rather than forcing it to .5. WR fluctuation includes
sampling noise and judge bias; a plateau alone does not establish convergence
or general human quality. Retain panel motion and generation-length/repetition
checks as complementary diagnostics, not substitutes for generated WR.

## Efficient deployment and next decisions

Fixed training candidates can be scored once, with both component rewards
cached. Every genuinely new generated response still needs both reward-model
passes; cached initial responses do not. Compare 6/4 and 7/3 offline from the
same reward cache before choosing a frozen evaluator. Loading two judges in
each simultaneous training job is not required: a separate batched scoring
stage/service or sequential model loading can avoid duplicating Nemotron.

An 8B BF16 model has roughly 16 GB of weights alone; this is not a total-memory
or throughput guarantee. The existing 70B judge remains the main cost. Do not
claim six one-GPU end-to-end oracle jobs fit the account budget. Measure memory
and throughput for the complete workflow before planning allocations; the
six-GPU full-account ceiling still applies.

Next proposed steps: review/download a pinned second-model revision when
authorized; run a bounded scoring/interface audit; inspect 6/4 and 7/3 cycle
statistics on fixed data; then choose the smallest matched ordinary/reference/
feedback training campaign. Do not start GPU work merely because this design
or the results atlas is published.
