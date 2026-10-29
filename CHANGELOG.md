# Changes

## 0.3.1 — 2026-10-29

- Default to local shape-adaptive sampling: density/critical/interface seeds,
  five-point integral consistency, sample reuse and bounded local refinement.
  Independently adapt Zhu et al. (2026) to the existing Dunkin/SH functions.
- Keep the original signed FP64 function for bracketing, refinement and
  acceptance. Extremum searches propose samples; amplitudes never accept roots.
- Expose `sampling_rtol` separately from refinement and close-root tolerances;
  retain the 0.5% balanced/fast ambiguity scale and quality markers.
- Preserve `qe_adaptive` as the 0.3.0 comparison path and legacy `uniform`.
- Use persistent CUDA blocks with compact local activity queues and shared
  model constants; account for the CPU stack and device queues in workspace.
- Add subdivision/reuse/limit/fallback counters and `TRACE_ADAPTIVE_FALLBACK`.
  Persist strategy and sampling configuration in CLI outputs.
- Upgrade the matching CPU/CUDA distributions to private ABI 5. See the
  [0.3.1 validation report](docs/SHAPE_ADAPTIVE_0_3_1.md) for independent-reference
  results, performance comparisons and retained numerical limitations.

## 0.3.0 — 2026-10-28

- Use a configurable `close_root_rtol=0.005`: balanced/fast logarithmic velocity
  steps are at most 0.25%/0.5%; high retains the full 0.2.1 schedule.
- Keep FP64 evaluation/refinement, with default refinement tolerances
  1e-6/1e-5/1e-4 for high/balanced/fast. Explicit c_tol_rel overrides the preset.
- Flag every approximate search and expose recovered-rank uncertainty, observed
  close-root masks and sampled cluster-risk hints; persist them in CLI NPZs.
- Separate ordinary errors, close-cluster omissions, ordinal shifts and unjudged
  reference-prefix/baseline-unresolved results in validation reports.
- Bump the private ABI to 4 for the configurable observation-scale interface.
- Measured balanced/fast gains versus same-build high: CPU 3.15x/5.41x
  (32 threads, 128 models); CUDA 3.63x/4.92x (A800, 1024 models).
  Both workloads use 70 layers, 100 periods and five modes. Keep isolated-root
  failures, close clusters, ordinal shifts and unjudged results separate.
- Validate 140 CPU tests and 143 CUDA tests; all four CUDA sanitizer checks
  pass. Standalone source distributions reproduce the tested native binaries.

## Historical strict-fix experiment (superseded for balanced/fast)

- Remove spacing-based early acceptance and skipped independent verification:
  existing witness cases could silently lose a close pair and shift mode labels.
- Restore all high density levels and the 32,768/65,536 dense pair in every
  preset; coarse-grid agreement must not bypass unresolved failure signaling.
- Retain exact FP64 arithmetic and accelerate CPU exact-bracket, sampling-union
  and nested-dense reuse, with smaller CPU prefix tiles and workspace accounting.
  Reject larger CUDA caches and smaller GPU tiles after matched experiments
  showed negligible speed-up and substantially higher memory use.
- Run witness regressions in all presets and both automatic/explicit ranges;
  add complete-diagnostic equivalence checks and a strict independent-reference
  plus interleaved-performance driver.
- Keep `adaptive_period_threshold` accepted for compatibility, with early
  acceptance disabled. Original 0.3.0 speed claims describe the unsafe heuristic
  and do not apply to this corrected implementation.

## Original 0.3.0 candidate (superseded, not the release implementation)

- Add `Solver(accuracy=...)` presets `"high"`, `"balanced"` (new default) and
  `"fast"`. `high` reproduces the 0.2.1 schedule and outputs bit for bit.
- Accept sparse root sets after a density level once adjacent roots are safely
  separated from the local bracket width and no low-|F| hidden-pair trough is
  present. Skip the independent guarded family only for those safe traces.
- Use an 8,192/32,768 point dense floor in `balanced` (4,096/16,384 in `fast`),
  accept a coarse dense result only when its spacing is safe, and fall back to
  the exact high dense schedule for traces that cannot prove it. This prevents
  newly unresolved extreme close-root traces.
- Add `adaptive_period_threshold`, `numerical_flags`, `accuracy` on
  `DispersionBatch`, new `search_level` values 1/2/4, and accuracy/early/dense
  counters in `stats()`. Document that `workspace_bytes` is not the per-trace
  sampling capacity (`sampling_capacity=2048`).
- Bump the private extension ABI to 3; install matching 0.3.0 CPU/CUDA wheels.
- Recorded 1,024-model steady gains of 2.39x CUDA and 2.73x CPU against 0.2.1,
  with zero wrong/missing/extra/newly-unresolved results, maximum common error
  2.15e-7 km/s. The complete Base workload (29,379 models) improved 2.44x in
  solver time and 2.38x end to end; the Aug 10,000 stress set improved 2.48x.
  Both retained zero wrong/missing/extra/newly-unresolved cells. `fast`
  measured 2.45x on 1,024 models CUDA, 2.74x on 1,024 models CPU, 2.53x
  solver/2.47x end-to-end on complete Base and 2.55x on Aug 10,000, also with
  zero wrong/missing/extra/newly-unresolved cells. The 11 independent FP64 suites remain
  0 wrong/missing/extra/unresolved for high, balanced and fast; `high` remains
  byte-identical to 0.2.1.

## 0.2.1 — 2026-10-28

- Precompute exact model/period constants in the FP64 CPU and CUDA recurrences;
  retain every search density, guarded verification and dense fallback.
- Reuse CPU model/sort buffers and native capacity across short tail batches.
- Isolate CUDA work in a solver-owned stream with persistent pinned staging;
  aggregate sampling counters per warp and opt into shared memory for 256 layers.
- Add optional `Solver(profile=True)` stage timings and cache/activity counters.
- Add streaming FP64 ModePhys preparation, full-diagnostic output, checked resume
  and interleaved installation benchmarks that reject observed GPU contention.
- Bump the private extension ABI to 2; install matching 0.2.1 CPU/CUDA wheels.
- Record matched ModePhys throughput gains of 4.56% CPU and 5.34% CUDA; retain
  byte-identical full Base outputs and document four legacy unresolved fixtures.

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
