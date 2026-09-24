# Archived-result caveats (no legacy trainer)

Current code is sequence-sum-only. The token-average trainers, scoring mode and
old launcher protocol are removed. This page documents existing results; it
does not provide an alternate training implementation. Git history remains
available for source provenance.

Historical non-oracle IPO used token-average loss and a dynamic inner reference.
Historical DPO used sequence-sum training but average-score sampling and panel
probabilities. Those results are not results of the current sequence-sum-only
sampler and frozen-reference protocol.

The historical cyclic sweep used alpha=.99, tau=1, lambda_pair in
{1, .75, .5, .25, 0}, seed=0; IPO beta=10 and 1000 pairs/round; DPO beta=1 and
500 pairs/round. It saved 150 pre-update states, with another update after state
149. The new `configs/cyclic_sampling_sweep.json` retains the parameter grid
but has 150 updates and an evaluation-only state 150. It is not a reproduction.

The archived DPO lambda_pair=1 run first has nonfinite loss at round 81 and
invalid raw panel scores from states 82..149. The old uniform fallback must
not be plotted as convergence. Scheduler completion is not numerical validity.
Keep valid-state masks and disclose unequal diagnostic windows in old figures.

Previously selected descriptive prompt IDs were IPO 94/333/882 and DPO
89/367/947, near baseline TV percentiles 50/90/99. Keep prompt and response
ordering fixed across an intervention; examples are not independent training
replicates. Old runs are not matched controls for the new history pilot.

The only compatibility code is the offline CSV importer, which can convert
saved averages to sequence sums with exact token counts. It cannot train a
model, compute new average scores, or modify archived checkpoints.
