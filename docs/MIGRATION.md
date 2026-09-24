# September 2026 experiment release

## Source selection

The old default branch ended at commit
83b3157 (four-H100 oracle launcher update). That history remains available.
The release is an ordinary descendant commit; it does not rewrite history.

Production Python sources were read from the experiment server on 2026-09-24.
The corrected shared oracle core includes sequence-sum IPO/DPO, outer-round
reference caching, direct relative-sequence metrics, and a final
evaluation-only snapshot. The original non-oracle runner hashes match the
source snapshot used to prepare the history pilot. SOURCE_MANIFEST.json
records imported source hashes and release hashes for auditing.

## File changes

| Before | Current role |
|---|---|
| scripts/run_ipo.py, scripts/run_dpo.py | Standard sequence-sum, cached-reference non-oracle entry points |
| Original non-oracle implementations | Preserved in scripts/legacy/ with original source bytes |
| scripts/run_preference_oracle_core.py | Current production core plus small publication safeguards |
| scripts/run_ipo_oracle.py, scripts/run_dpo_oracle.py | Current scalar-oracle entry points |
| run_iterative_ipo_fast.py | Retired duplicate; retrieve from old Git commits if needed |
| run_oracle_experiment.sh | Replaced by portable slurm/oracle.sh and explicit JSON recipes |
| README_REPRODUCE_EXPERIMENTS.md | Redirects to the single authoritative root README |
| Data builders/download helpers | Retained and synchronized, not discarded |
| experiments/cyclic_history/ | New matched history pilot, portable paths, tests and documentation |

Do not assume the root non-oracle entry points reproduce the old scripts
unchanged. Use scripts/legacy/ for historical non-oracle experiments.
Standardized non-oracle recipes expose the corrected core on static labels;
their availability is not a completed-result claim.

## Deliberate publication changes

1. Shared helper imports refer to scripts/legacy/run_ipo.py, preserving their
   behavior without making the public non-oracle entry points ambiguous.
2. Standard non-oracle wrappers disable both oracle training and evaluation.
3. Scalar-oracle training is rejected with cyclic preference metadata: a scalar
   labeler would replace the cyclic environment.
4. Nonfinite scored likelihoods, loss, or gradient norm stop the corrected
   core. Invalid scores are not silently represented as a uniform policy.
5. An unsupported auto_stop flag and nonpositive/nonfinite beta are rejected.
6. Public recipes preview before execution and refuse existing run directories;
   history entry points additionally accept portable path overrides.
7. Cache locations are isolated per oracle run to avoid support/cache collisions.
8. Metrics include the training seed and explicit sampling/reduction semantics;
   static runs no longer claim the oracle generator-mixture convention.

No learning-rate, accumulation-normalization, reference formula or loss reduction
was silently changed in the imported corrected oracle core. The history pilot
already had different precision/accumulation choices; those remain explicit.
Legacy runner files themselves are retained unchanged.

Publication did not stop, restart, resubmit or allocate GPUs for any experiment.
The private working directory, paper drafts, credentials, dataset text, generated
responses, adapters and weight files are not repository payloads.

## Reproducibility limits

Current standard-core state 80 is obtained with iters=81 because the final
state is evaluation-only. Some historical launchers had different horizons.
Use recorded source/config versions when reproducing an old result.

Adapter-only loading is not full RNG/support/optimizer restoration. Prepared
history runs have not supplied empirical stabilization evidence. Neither CPU
tests nor successful compilation is a substitute for a complete GPU run.

The requirements-server snapshot records relevant package versions observed
during release validation; it does not assert that all older jobs used exactly
those versions. Record per-run environments for a final published result.
