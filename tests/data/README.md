Synthetic elastic-model fixtures for regression tests. `B10000_*.npz` contains
three challenging batch cases and `witness_*.npz` contains eight independently
checked cases. References use FP64 CPS/disba scans and a separately built pinned
QEDispInv reference. Tests require matching phase velocities or explicit
unresolved status and NaNs.

`accuracy_high_cpu/` and `accuracy_high_cuda/` are bitwise 0.2.1 baselines for
the `Solver(accuracy="high")` contract. They were generated from installed
modisp 0.2.1 wheels with `tools/make_accuracy_fixtures.py` on 2026-10-28
(CPU: macOS arm64 CPython 3.12; CUDA: A800/sm_80, CUDA 12.6, CPython 3.10). The
cases cover solid P-SV, surface-water/Scholte, Love and a 70-layer model.
`manifest.json` records the source package/extension hashes and per-case hashes.
No field observation or third-party model data is included.
