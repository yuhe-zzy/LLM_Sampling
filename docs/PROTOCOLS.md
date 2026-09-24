# Experimental protocols

## Likelihood and reference conventions

For a prompt x and response y, corrected training uses the response-token sum
`s_theta(x,y) = sum_k log pi_theta(y_k | x,y_<k)`.
EOS is included, prompt and padding tokens excluded, and the causal shift is
applied. Standard tokenization left-truncates the combined sequence to
max_length=1537. This can remove response prefixes for extremely long answers;
audit lengths before interpreting a score as an untruncated response likelihood.
The new history pipeline instead shares one prompt context across candidates
and rejects response truncation.

The ordinary frozen outer-round reference is

```text
r_t = (1-alpha)*s_initial + alpha*s_t
delta(i,j) = (s_theta(i)-r_t(i)) - (s_theta(j)-r_t(j))
```

These are geometric-mixture **log scores**, not a parameter average of LLMs.
Prompt-dependent normalization constants cancel in pair differences. References
are cached before training and remain fixed during all inner optimizer steps.

For an oriented hard chosen/rejected pair:

```text
IPO(delta) = (delta - 1/(2*beta_train))**2
DPO(delta) = -log sigmoid(beta_train*delta)
```

The history code generalizes these to P_ij by averaging both orientations.
`beta_train` is a loss coefficient; the IPO feedback gain scales as its inverse.
It is not automatically the same beta used as a gain in a theoretical recursion.

## Three different sampling controls

### Dynamic scalar-oracle labels

Each response independently uses the initial generator with probability
`lambda_base` and the current generator otherwise. Both use temperature 0.8,
top-p 0.95, maximum 256 new tokens. Thus the actual decoding law is a mixture
of two **decoders**, not necessarily unmodified ancestral model sampling.
Four responses produce two disjoint pairs per selected prompt. The one frozen
Nemotron-70B scalar reward model ranks them, skips exact ties and empty outputs,
and assigns unit pair weights. No BT sampling of labels is used.

The `lambda_on` command-line name is retained for compatibility, but it means
`lambda_base` here, not the current-generator weight.

### Fixed static labels, transitive or cyclic

At a fixed prompt, compute the sequence-sum chosen/rejected margin
`m(e) = s_theta(x,chosen) - s_theta(x,rejected)`.
The within-prompt target is

```text
u(e) = softmax_e(tau*m(e))
q_pair(e) = lambda_pair*u(e) + (1-lambda_pair)*Uniform(e)
q_pair(e) = (1-mix_eps)*q_pair(e) + mix_eps*Uniform(e)
```

Uniform proposals are drawn with replacement. Target weights are applied once,
normalized by their sampled mean and clipped to the configured range. This is
the inherited self-normalized estimator, not exact unbiased importance sampling.
The sequence-sum-only revision uses the same response-score reduction for
sampling and training. This changes static pair probabilities relative to
earlier average-margin sampling. The cyclic sweep uses mix_eps=0 and clip
range [0,1e6]; those estimator choices remain unchanged.

### New history pilot

This uses an explicitly different, matched protocol:

```text
q_panel = softmax(sequence_sum_scores)
mu = (1-lambda_current)*Uniform(4) + lambda_current*q_panel
Q(i,j) = mu(i)*mu(j) / sum_{a<b} mu(a)*mu(b), i<j
```

Draw two uniform proposals among six unordered pairs per prompt and apply
weight `6*Q(i,j)` exactly once. There is no sample-mean weight normalization.
Proposal random numbers are matched across the three arms of each objective.
Only nu or kappa changes in a controlled comparison; other protocol changes
are shared with the **new** ordinary control.

The fixed cyclic preference matrix is

```text
P = [[.5, 1, 1, 0],
     [0, .5, 1, 1],
     [0, 0, .5, 1],
     [1, 0, 0, .5]]
```

Unused historical diagonal zeros are treated as .5. Missing off-diagonal
comparisons are errors. This tournament has both cyclic and directional
components; a uniform equilibrium is not assumed.

Lagged reference uses
`r_nu = (1-alpha)*s_initial + (alpha-nu)*s_t + nu*s_previous`.
The pilot has nu=alpha/2=.45, a partial lag, not nu=alpha.

Feedback extrapolation shifts the cached ordinary reference by
`kappa*(d_t-d_previous)`, then trains the same **positive** pair objective.
Here d is the population-optimal log-ratio feedback:

- IPO: `center((P-.5) @ mu / beta_train)`.
- DPO: minimize `sum_{i<j} Q_ij [softplus(v_i-v_j)-P_ij*(v_i-v_j)]`
  with centered v, then use `d=v/beta_train`.

The unconstrained population target is therefore
`r_t + (1+kappa)*d_t - kappa*d_previous`, up to an additive constant.
For IPO, d is linear in mu. For DPO, the BT projection is nonlinear:
this is **actual-DPO feedback extrapolation**, not entrywise-logit PsiPO.
No logit of a hard 0/1 preference and no negative-weight DPO loss is used.
The CPU Newton solve must meet its residual tolerance. Shared population
fixed-point equations do not guarantee shared neural endpoints.

## Evaluation metrics

### Relative-sequence entropy (primary oracle quantity)

On each selected prompt, generate ten initial-policy candidates, deduplicate,
and retain up to five by initial sequence-sum likelihood. Freeze the panel.
This ranking change means newly generated panels need not equal historical
panels. The support selection score is not the later relative-entropy score:

```text
z_t(i) = s_t(i) - s_initial(i)
q_relative(i) = softmax_i(z_t)
H_relative = mean_prompt[-sum_i q_relative(i)*log q_relative(i)]
```

Entropy is in nats. Supports may contain fewer than five candidates.
This normalizes likelihood **ratios** on a finite panel, not the complete
LLM output distribution. New runs emit raw sequence-panel entropy
`prompt_sequence_entropy_mean` from `softmax(tau*s_t)` as a separate quantity;
they no longer compute or emit token-average scores or entropy. The formulas
above assume tau=1, as in the supplied recipes; otherwise multiply logits by tau.

Older dumps may reconstruct z as
`token_count*(average_score_t-average_score_at_0)` only with identical support,
tokenization, EOS, truncation and causal shift. Preserve whether the baseline
was recorded step zero or directly scored frozen initial model. Never force
the initial value to log(5), smooth away reversals, or reorder curves.

### Oracle winning rate

At 0/20/40/60/80, draw four current-model responses on each of 500 prompts,
and compare with four cached initial-model responses generated with seed 777.
Average all 16 within-prompt indicators `R(current)>R(initial)`.
Exact ties contribute zero. The separately logged sigmoid reward difference
is a **soft** score, not the strict WR. The same frozen oracle supplies
training and evaluation rewards: this is an in-oracle diagnostic, not an
independent human or task benchmark. A WR plateau is not a convergence proof.

### Cyclic trajectories and validity

Keep prompt IDs, response ordering and colors fixed across an intervention.
One-round TV is `0.5*sum_i abs(q_t(i)-q_previous(i))`.
Windowed temporal variance includes slow drift; it does not itself establish
an attracting cycle. Report prompt variability as descriptive statistics,
not across-training-seed uncertainty.

Historical plots use average-token panel probabilities and mask the known
invalid DPO lambda_pair=1 interval from snapshot 82. The new history pilot
uses sequence-sum panel probabilities. It also logs panel mass, relative entropy,
and discrepancy between post-update scores and the current population target.
That discrepancy is **inner-update error**, not distance to a solved fixed point.

## Optimization and indexing

All recipes use Qwen2.5-1.5B + LoRA r=16 / scale=32, targeting q/k/v/o and
gate/up/down projections, AdamW LR=1e-5, warmup=.03, one inner epoch, pair
batch=1, accumulation=4, and gradient clipping=1. AdamW defaults are
betas=(.9,.999), eps=1e-8, weight_decay=.01. Optimizer state and the linear
schedule reset each outer round; policy/adapter parameters persist.

The oracle/static core preserves the archived FP16 and
LoRA dropout=.05 conventions. They **sum** four microbatch mean losses before
an optimizer step (not divide by four). The new history code uses BF16,
FP32 scoring/loss reductions, zero dropout, and **averages** accumulation
groups, including short final groups. These differences prohibit using old
results as matched history controls. No automatic claim of identical effective
step size across IPO and DPO is made.

The standard core's `iters` counts evaluated states: 81 yields states 0..80
and 80 training rounds. The cyclic sweep's 151 states mean 150 updates.
History `iters=100` means 100 updates plus initial state 0.

Release safeguards stop the corrected core on nonfinite loss, gradients or
scored likelihoods rather than accepting a uniform probability fallback.
Historical trainers are not included as runnable alternatives.
These checks do not guarantee semantic quality or absence of token collapse.
