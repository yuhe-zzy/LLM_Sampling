# Oracle1 Qwen3 campaign, 2026-10-15

See [protocol and continuation](../../README.md). Owner approved implementation,
submission and early5-minute checks, changing the same monitor to5-hour checks
only after real IPO/DPO training AND checkpoint reload/continued training work.
No new Oracle2 work. Full-account FOUR-GPU ceiling.

## Preparation evidence

At approximately03:42 CDT, SSH reconnected and full owner/UID account inspection
found no queued/running/releasing jobs and zero allocated GPUs. This is a dated
snapshot, not a reservation. Old final Oracle2 generators report COMPLETED in
sacct; their output completeness was NOT audited or used for this new campaign.

Models downloaded on the login node without GPU allocation and pinned:

| Role | Model | Revision |
|---|---|---|
| Policy | Qwen/Qwen3-4B | 1cfa9a7208912126459214e8b04321603b3df60c |
| Five-trait rater | Qwen/Qwen3-32B | 9216db5781bf21249d130ec9da846c4624c16137 |
| Oracle1 | nvidia/Llama-3.1-Nemotron-70B-Reward-HF | existing audited lock, unchanged |

Model lock fingerprints small configuration/tokenizer files and weight metadata;
it is not a newly computed full SHA256 of every weight shard. Models are BF16,
no quantization. Memory sufficiency is not established just by downloading.

New roots:
- Launch: `/work/users/y/u/yuhe32/ipo/diagnostics/oracle1_qwen3_20261015_v1`
- Output: `/work/users/y/u/yuhe32/ipo_runs/oracle1_qwen3_20261015_v1`
- Local private receipts: `oracle1_qwen3_launch_20261015` in the outer workspace.

The real-data audit rebuilt pure Oracle1 preferences from all800 original
panels, preserving100/500/200 and4 candidates. Full Qwen3 chat+answer+EOS length
checks pass at inherited1537 cap; no truncation or dropped data. Evaluation
prompts plus256 generated tokens also pass. No repeated candidate judge job.

The first approved scheduler action will be just two actual first-round tasks:
IPO ordinary(index0), DPO ordinary(index3), one H100 each. Receipt/publication
records below determine whether this has been submitted; preparation alone
does not mean inference or training passed. First resume allocation likewise
contains0/3 for one more outer update, then all six arms use at most2 concurrency.

Do not edit plan/lock/data/source after deployment. Technical fixes must be tested,
synced to GitHub and deployed as a new immutable version with explicit provenance.
Never treat a failed scheduler task's exit0 or stale file manifest as completion.
