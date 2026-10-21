# Project context

Canonical root: this repository. Enrolled by the owner on 2026-10-13.
Start with [current handoff](../handoff/current.md), then the newest
[daily note](../daily-notes/). Do not infer live job status from dated files.

- [Current experiment status and decisions](EXPERIMENT_STATUS.md)
- [IPO ordinary0..100 WR results](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/ipo_only_wr_results_20261021/README.md): October21 generation/9600 scores complete; inspected figures and summary delivered, monitor PAUSED. Expected WR47.569% to47.536%, no clear improvement; one seed, prompt SE only.
- [Oracle1/Qwen3 campaign and provenance](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/README.md): retained index0 complete; other five arms withdrawn, outputs preserved, no automatic new work.
- [Interim0..80 WR execution](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/INTERIM_WR_80.md): prior six-arm automatic advancement withdrawn; first-arm-only continuation must not bypass failed receipts or mutate frozen sources.
- [Definitions and interpretation limits](SCIENTIFIC_CONTEXT.md)
- [Results atlas](../docs/RESULTS_INDEX.md): batch reports, figures and tables
- [Publication coverage and exclusions](../docs/RESULT_PUBLICATION_20261013.md)
- [Protocols](../docs/PROTOCOLS.md) and [reproduction guide](../README_REPRODUCE_EXPERIMENTS.md)
- [Oracle2 execution record](../experiments/oracle2/campaigns/oracle2_real_20261009/README.md)
- [Cyclic campaign registry](../experiments/cyclic_history/campaigns/README.md)
- [Hodge collaborator material](../experiments/hodge_diagnostics/README.md)
- [Fan's aligned latent-preference analysis plan](../experiments/hodge_diagnostics/plans/latent-estimation-2026-10-13/README.md): embedding first, judged style second; execution clarifications take precedence over draft shorthand.
- [Generated-output dynamics discussion plan](../experiments/hodge_diagnostics/plans/generated-output-dynamics-2026-10-15/PLAN.md): background rationale; the new Oracle1/Qwen3 campaign above supersedes its earlier execution hold and mixed-oracle options.

`doc/` is the current context layer. Existing `docs/` and campaign directories
remain authoritative for their detailed protocols and dated evidence. The daily
log is append-only; this index and the handoff are current snapshots. GitHub
publication does not authorize new GPU experiments.
