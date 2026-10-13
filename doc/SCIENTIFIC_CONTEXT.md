# Scientific context and interpretation

## Confirmed definitions

Current runnable IPO/DPO experiments use sequence-sum response log likelihood,
including the protocol's EOS, not token-average training. Historical archived
token-average figures retain their old label; publishing them does not restore
an old training mode or make them sequence-sum results.

- Raw panel policy: softmax(s_t) over the fixed candidate responses. It is not
  normalized over every possible generated response.
- Relative-sequence distribution: softmax(s_t-s_0). Its entropy is distinct
  from raw panel entropy and historical length-normalized entropy.
- History reference: (1-alpha)*s_0+(alpha-nu)*s_t+nu*s_(t-1).
  Alpha belongs to the reference refresh, not an independent policy mixture.
  Ordinary uses nu=0; initial-anchor control alpha=.1 yields(.9,.1,0).
- History sampling lambda_current and older static/oracle lambda_on do not
  share one meaning. Read [protocols](../docs/PROTOCOLS.md) before comparisons.
- Generated-response WR requires an external comparison rule. The synthetic
  cyclic fixed-panel tournament is not an unrestricted-generation judge.

## Approved Oracle2 question and settings

Empirical stability trends, not a demand to eliminate oscillations or prove
convergence. Oracle1 is Nemotron's scalar reward judge. Oracle2 mixes .6 of
Nemotron BT probabilities and .4 of Skywork BT probabilities, each temperature1;
it does not mix raw rewards. Fixed real candidates train the model; open
generations are compared with an independent shared pretraining baseline.

Disjoint calibration/train/evaluation panels100/500/200, four candidates each.
Audited robust cyclic counts0/1/0, transitive81/436/160, ambiguous19/63/40.
The owner explicitly accepted the sparse cyclic case on2026-10-09. Do not
retune data, mixture, temperature or cycle thresholds to manufacture cycles.
Six arms: IPO/DPO ordinary/reference/feedback, seed0, alpha=.9,lambda_current=.8;
IPO beta=.2, DPO beta=.8, reference nu=.45, feedback kappa=.5; states0..100.

## Checked limitations, not favorable-result filters

The sole cyclic training prompt416 starts with nearly unit raw panel mass on
a two-token candidate. Its flat raw trajectory is not history stabilization.
Ambiguous does not mean non-ST; no cyclic evaluation subgroup exists, so its
WR is missing, not0. Single-seed/all-prompt findings are descriptive.
Lower temporal SD can coexist with higher adjacent-step TV. Neither alone
proves convergence or absence of periodicity. Preserve counterexamples.

## Proposal versus execution

The collaborator's Hodge notes and two-week roadmap are hypotheses/plans unless
linked to an actual launch receipt and validated result. Theory trajectories
must not be labeled LLM measurements. Training provenance is the frozen commit
recorded per campaign, never the current documentation/plotting commit.
