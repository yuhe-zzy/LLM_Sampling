# Stage B: panel pi versus outer iteration

Six figures: IPO/DPO each ordinary, reference and feedback; every figure has
all six prompts and four fixed-response curves. Axes are outer iteration
0..100 and probability 0..1. Response colors and line styles stay fixed.

`pi_panel(i,t) = exp(s_i(t)) / sum_j exp(s_j(t))`, with response sequence-sum
log probability including EOS. This is conditional on the four-response panel,
not full response-space probability, relative-to-initial pi, or sampling mu.
No interpolation, smoothing, mode projection or theoretical rollout is used.

The combined PDF has six pages. `pi_trajectories.csv` has 14,544 numeric points;
response indices use the original fixed support order, not cyclic-role order.
Private prompt/response text and response_mapping.json are deliberately omitted.
See the [campaign reproduction instructions](../../README.md) for replotting
directly from the public CSV without raw snapshots or a model.
