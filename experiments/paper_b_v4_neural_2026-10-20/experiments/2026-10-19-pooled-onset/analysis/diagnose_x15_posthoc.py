"""Post hoc diagnosis of X15 (2026-10-20, after scoring; not preregistered).
(1) Rotation vs jitter: for K = 3 panels, the winding of the centred probability vector in the plane orthogonal to (1,1,1)
    over the last 30 rounds. A sustained rotation winds consistently (|net turns| >= 1); label noise gives near-zero net winding.
(2) Per-round fit for the IPO arm: loss at the start and end of each round against the round's exact optimum."""
import json, os, sys
import numpy as np
sm = lambda x: np.exp(x - x.max(-1, keepdims=True)) / np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True)
E = json.load(open('evaluation_x15.json'))['arms']; THR = 0.0536
out = {}
for arm in sorted(os.listdir('runs')):
    Z = np.load(f'runs/{arm}/run.npz'); s = Z['s'].astype(float)
    if s.shape[2] != 3: continue
    p = sm(s[-31:]); u = np.array([1, -1, 0]) / np.sqrt(2); v = np.array([1, 1, -2]) / np.sqrt(6)
    ang = np.arctan2(p @ v, p @ u)                                   # rounds x panels
    d = np.diff(ang, axis=0); d = (d + np.pi) % (2 * np.pi) - np.pi
    turns = d.sum(0) / (2 * np.pi); same_sign = np.abs(np.sign(d).sum(0)) / d.shape[0]
    amp = np.array(list(E[arm]['per_panel_amp'].values())); osc = amp > THR
    out[arm] = dict(osc_share=float(osc.mean()), median_abs_turns_osc=float(np.median(np.abs(turns[osc]))) if osc.any() else None,
                    share_osc_with_turns_ge1=float((np.abs(turns[osc]) >= 1).mean()) if osc.any() else None,
                    median_sign_consistency_osc=float(np.median(same_sign[osc])) if osc.any() else None)
    print(arm, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in out[arm].items()})
json.dump(out, open('diagnose_x15.json', 'w'), indent=1)
