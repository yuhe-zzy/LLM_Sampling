# X15: onset of rotation under pair-dependent pooling (2026-10-19/20)

Neural test of v4 cor:pooled-onset: on specialist panels (three transitive criteria ranking three answers in rotation), assigning comparisons to the criterion that owns them (share c) makes the fixed-reference DPO loop rotate, with onset step scaling as (cm)⁻¹; the uniform mixture does not rotate; slower refresh at the same target removes the rotation.

- Design: `DESIGN.md`. Preregistration with timestamped log, amendments, the voided first batch and the scorecard: `prereg/PREREG.md`.
- Panels: `data/panels3.json` (76 K = 3), `data/panels4.json` (4 K = 4), floor 0.02, M = 96 (`vm/build_specialist_panels.py`); the 0.05-floor pass (15 panels) in `data/floor05/`.
- Predictions (frozen before training): `prereg/predictions_x15.json`, oracles `prereg/oracle_*.npz`, script `analysis/predict_x15.py`.
- Runs: `results/runs/<arm>/run.npz` (+ config, log); batch files `vm/batch_x15*.json`; VM logs `results/vm_logs/`. Voided mis-pointed runs: `results/runs_bad_wrongpanels/` (configs and logs only).
- Scoring: `analysis/evaluate_x15.py` → `results/evaluation_x15.json`.

Headline (64 K = 3 panels, Qwen2.5-0.5B, DPO, B = 10): oscillating share 0.000 / 0.078 / 0.484 / 0.812 at c = 0 / 0.25 / 0.5 / 1 (population 0.000 / 0.078 / 0.484 / 0.938); remedy α = 0.9, β = 1 at c = 1: 0.188 (population 0.156); population onset slopes −1.28 in c, −0.98 in m̃. Random-assignment and IPO arms failed for identified reasons (see the scorecard).
