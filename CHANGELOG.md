# Changes

## 0.2.0 — 2026-10-28

- Remove `PhaseDispersion`, `DispersionCurve` and `DispersionError`. Use
  `dispersion`, `dispersion_batch` or `Solver` for both single and batch models.
- Return complete `DispersionBatch` arrays at every entry point. Preserve NaN,
  status and trace diagnostics instead of filtering curves or raising a
  curve-specific numerical exception.
- Use `Yuan Yusong` as the formal author and copyright holder, with AI-assisted
  development acknowledged separately in the README.
- Add a reproducible same-host solid-Rayleigh accuracy and performance driver
  for modisp, disba and the pinned external QEDispInv reference.
- Publish full 6083-trace accuracy results and B=1/8/64 steady-state timings,
  including external wrong/missing/extra counts and first-call costs.

## 0.1.0 — 2026-10-27

Initial modisp release.

- Provide FP64 multimodal Rayleigh/P–SV and Love phase-dispersion forward
  modeling for layered elastic media, including a surface acoustic water layer
  and Scholte-type roots.
- Provide a persistent-thread CPU backend and an optional version-matched CUDA
  12 plugin behind one Python API.
- Add disba-style single curves, complete batch arrays, reusable solvers,
  automatic backend selection and explicit numerical diagnostics.
- Include the `modisp doctor` and `modisp forward` command-line tools, examples,
  numerical regression fixtures and source/distribution release auditing.
- Retain the legacy uniform CUDA solid-Rayleigh search as an explicit option.
