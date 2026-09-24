# Release validation, 2026-09-24

Validation ran in an isolated source directory using the existing experiment
environment, with CUDA_VISIBLE_DEVICES empty, OMP_NUM_THREADS=1,
OPENBLAS_NUM_THREADS=1, and Hugging Face offline mode.

- Release/data/plot/core suite: **22 tests passed**, no skips in the server environment.
- Cyclic history suite: **29 tests passed**, including both tiny CPU LoRA microsteps.
- Python compilation succeeded for scripts, experiments, and tests.
- All three Slurm templates passed bash syntax validation.
- Standard grid previews and the six history configurations were validated.
- Plot tests produced IPO/DPO entropy and WR PNG/PDF outputs from synthetic
  metrics, and rejected legacy-metric substitution and duplicate attempts.
- Imported historical IPO/DPO source hashes are retained in SOURCE_MANIFEST.json.

The relevant Python/package versions are in requirements-server.txt.
The local lightweight environment additionally ran data/plot/recipe tests;
Torch-dependent tests were skipped there and completed in the server validation.

This is CPU/source validation, **not a new pretrained-model experiment**.
No scheduler job was submitted, no GPU was allocated, and no running training
process was canceled or modified. Numerical behavior of a full FP16/BF16 GPU
run, reward-model memory requirements on another device, and long-run
convergence remain outside these tests. GitHub CI runs the CPU suites again
after publication.
