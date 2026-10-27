# Aligned latent-preference signal: plan bundle for Yu He

Shared at Fan's explicit request on2026-10-13. This is the500-panel static
feasibility analysis discussed at the coauthor meeting, plus the conditional
checkpoint projection. It is not a new training campaign.

Read in order:

1. [Execution clarifications](EXECUTION_CLARIFICATIONS.md): these supersede
   ambiguous/overstrong statistical clauses in the Claude drafts.
2. [Main handoff](HANDOFF_TO_YU_HE_2026-10-13.md): S0–S5 and deliverables.
3. [Detailed plan v2](draft-v2.md): the framing and Fan's choices.
4. [Latent-preference background](references/latent_preference_framework.md),
   Sections0–3 first; [partial-sharing background](references/partial_sharing_theorem.md)
   is context only, not a request to run rho scans.

## Immediate task and order

- Work from the exact frozen500 training panels and their per-oracle scores,
  not an inferred candidate order or an elementwise mean P.
- First compute frozen sentence embeddings, PCA8/16, and S2/S3 signal diagnostics
  under identity and logit for each oracle and mixture. No annotation fee.
- Also prepare the style-judge route:2000 responses, one annotation per response,
  shared continuous style attributes plus token length. Record model, rubric,
  token/compute estimates and owner-confirmed budget before running it.
- Return embedding results first. The later interpreted-feature comparison
  uses the same prompt splits, links and diagnostics.
- Preserve the existing six training arms, checkpoint banks and WR task.
  S4 uses existing logged panels only; do not launch new inference/training
  merely because a field is missing. List missing fields and ask for direction.

## Minimal input contract

For each prompt, preserve prompt key/ID, response index and exact response
text/hash, individual oracle scores or probabilities, mixed probability matrix,
mixture rule/temperatures, and initial sequence-sum panel policy. Record run and
frozen-source identity. Keep raw texts/embeddings and judge payloads private;
share sanitized numeric summaries, code and provenance only.

The completed slot-average analysis remains in
`experiments/oracle2/campaigns/oracle2_real_20261009/average_hodge_20261013/`.
It is a baseline, not this aligned analysis. The500 panels are shared by six
arms, so they are500 preference observations, not3000 independent samples.

Deliver an English REPORT.md, numerical tables, exact commands/configs, fitted
matrices/planes, validation/null diagnostics and a brief updated handoff. Mark
what was executed versus proposed. A negative/positive diagnostic is conditional
on the feature model and does not prove neural convergence/instability.
