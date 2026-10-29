# Same-host modisp 0.3.1 comparison — 2026-10-29

This reruns the original external comparison with the **released** 0.3.1 CPU
and CUDA extensions. No solver source or release tag is changed. The CPU native
SHA256 is `09f5088869abdcab9895da18278334e1f92c4238917dd4a015b909b09695780b`; CUDA is
`486d709fbcc915312ff7c3ac9dae1de615ab8d674f5eef51350e18b2a790189d`. Python is 3.10.12, NumPy 2.2.6, Numba
0.68.0, disba 0.7.0. The machine is a dual-socket Intel Xeon Gold 6430 host with
NVIDIA A800 80GB PCIe. The complete environment and input hashes are in the
[statistical receipt](benchmarks/samehost-0.3.1-20261029.json).

## Accuracy and reference boundary

All 11 archived solid-Rayleigh suites are retained: 171 models, 6,083 traces,
19,161 finite independent reference cells. Every archived trace is stable
under the original 32,769/65,537-grid check; no unstable traces are filtered out
of this run. Four modes are requested. Water, Love and Scholte are not covered
by this particular external-comparison workload; their separate evidence is
in the [0.3.1 upgrade report](SHAPE_ADAPTIVE_0_3_1.md).

The following is the original **strict, rank-aligned** criterion for every
implementation: `abs(actual-reference) > max(3e-6, 2e-6*abs(reference))`.
Missing/extra mean finite-slot changes. They must not be interpreted as proof
that a flagged approximate trace contains every reference root.

| Solver / preset | Wrong | Missing | Extra | Exceptions / unresolved | Max ordinal common abs error (km/s) |
|---|---:|---:|---:|---:|---:|
| modisp CPU / high | 0 | 0 | 0 | 0 | 2.621352e-07 |
| modisp CUDA / high | 0 | 0 | 0 | 0 | 2.621352e-07 |
| modisp CPU / balanced | 6 | 0 | 0 | 0 | 3.287436e-01 |
| modisp CUDA / balanced | 6 | 0 | 0 | 0 | 3.287436e-01 |
| modisp CPU / fast | 14220 | 0 | 0 | 0 | 3.287391e-01 |
| modisp CUDA / fast | 14211 | 0 | 0 | 0 | 3.287391e-01 |
| disba 0.7.0 | 1338 | 1034 | 0 | 12 exceptions | 2.868654e+00 |
| QEDispInv ed1b5dd | 0 | 1 | 0 | 0 | 1.629918e-12 |

Balanced/fast deliberately use looser refinement and permit flagged close-root
ambiguity. The large 0.329 km/s maximum ordinal deviation reflects a near-root
omission shifting later recovered ranks; it is not the error of a correctly
matched isolated root. Fast's many strict wrong cells also reflect its 1e-4
refinement setting exceeding the old strict threshold. The separate budgets
remain high 2e-6, balanced 2e-5, fast 2e-4 (relative); `close_root_rtol=.005`.
All arithmetic is FP64. The defaults for `sampling_rtol` are .001/.01/.05.

| Backend / preset | Strict wrong | Isolated-root failures in budget | Strict unmatched cluster roots | Ordinal shifts | Outside independent prefix | Max matched relative error (with guards) |
|---|---:|---:|---:|---:|---:|---:|
| modisp CPU / high | 0 | 0 | 0 | 0 | 0 | 6.279798e-08 |
| modisp CUDA / high | 0 | 0 | 0 | 0 | 0 | 6.279798e-08 |
| modisp CPU / balanced | 6 | 0 | 5 | 6 | 5 | 6.289137e-07 |
| modisp CUDA / balanced | 6 | 0 | 5 | 6 | 5 | 6.289137e-07 |
| modisp CPU / fast | 14220 | 0 | 5 | 6 | 5 | 6.277271e-06 |
| modisp CUDA / fast | 14211 | 0 | 5 | 6 | 5 | 6.277271e-06 |

For boundary classification only, retained `qe_adaptive` high supplies up to
eight guard roots, conditional on agreeing with the independent four-mode
prefix. Independent cells are never replaced by guard values. Guard matches
beyond that prefix are not independent physics validation: five returned
balanced/fast cells per backend remain outside the independent prefix, even
though guards classify their recovered-rank shifts. Zero isolated-root
failures is therefore a scoped budgeted result, not a claim of arbitrary root
completeness or certified approximate mode indices. Every approximate trace
retains the approximate-search / mode-index uncertainty flags. Original wrong,
missing, extra, exception and reference-prefix counts are retained.

## Performance

Use the same `seed20260916` prefix and original 10 layers, 64 periods, four
modes, `cmin=.2`, default 256 MiB workspace. B<=64 uses distinct prefix models.
Larger batches map task `i` to `i % 64`; B=1000 is CUDA only. Each call runs the
native forward solver. Modisp first and every steady output must reproduce
scientific values, unresolved states and flags bitwise, and first/last outputs
are independently checked. Warmup is separate from five timed steady calls.
The receipt preserves initialization, first call, warmup, five samples,
median/min/max, native counters and all first/last strict/quality checks.

| Solver / preset | B=1 (s) | B=8 (s) | B=64 (s) | B=64 models/s |
|---|---:|---:|---:|---:|
| modisp CPU, 1 thread / high | 0.059395 | 0.485008 | 3.706599 | 17.267 |
| modisp CPU, 8 threads / high | 0.007600 | 0.061062 | 0.466389 | 137.225 |
| modisp CUDA / high | 0.007215 | 0.012756 | 0.062621 | 1022.020 |
| modisp CPU, 1 thread / balanced | 0.024730 | 0.180012 | 1.495611 | 42.792 |
| modisp CPU, 8 threads / balanced | 0.003929 | 0.023435 | 0.189008 | 338.611 |
| modisp CUDA / balanced | 0.002159 | 0.003135 | 0.012605 | 5077.232 |
| modisp CPU, 1 thread / fast | 0.023362 | 0.169055 | 1.407595 | 45.468 |
| modisp CPU, 8 threads / fast | 0.003528 | 0.022099 | 0.178267 | 359.012 |
| modisp CUDA / fast | 0.001931 | 0.002745 | 0.011008 | 5813.716 |
| disba, serial | 0.011069 | 0.096820 | 0.778879 | 82.169 |
| disba, 8 processes | 0.012213 | 0.014693 | 0.107443 | 595.662 |
| QEDispInv, serial | 0.012751 | 0.118384 | 0.842285 | 75.984 |

| CUDA preset | Tasks | Independent models | Unique reference traces | First call (s) | Warm median (s) | Tasks/s | First/last strict issues | Outside independent prefix (each output) |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| high | 1,000 | 64 | 4,096 | 0.685045 | 0.674103 | 1483.452 | 0 / 0 | 0 |
| balanced | 1,000 | 64 | 4,096 | 0.195296 | 0.187797 | 5324.900 | 64 / 64 | 48 |
| fast | 1,000 | 64 | 4,096 | 0.170717 | 0.164922 | 6063.464 | 156672 / 156672 | 78 |

These figures rerun third parties at B=1/8/64. There is no same-size B=1000
third-party result. Disba uses the default Dunkin / dc=.005. Its parallel
configuration uses eight persistent fork workers; every worker is warmed and
verified by PID, and pool setup / worker warmup are excluded from steady calls
but reported separately. Its first serial call uses a fresh Numba cache
location; process cold calls and warmed workers are not conflated.

QEDispInv remains pinned to `ed1b5dd7b449a2b3a27bb8e6581278790f5df8aa`.
The adapter is the same one used by the original performance table, with
`qed_benchmark_roots` for performance and `qed_reference_roots` for accuracy;
its binary SHA256 is `aae549de5d52885178c28fba313a72ed1373b25b1c71db3c83ce4c12fb2a023b`.
An initial attempt selected an older reference-only adapter without the
benchmark symbol; that failed attempt is preserved privately and is not used
for the final third-party rows. Third-party source/binaries are not bundled.
GPU process monitoring reports no competing compute in any timed CUDA run;
resident idle contexts are not treated as compute or altered. Read speed and
accuracy together; these settings do not establish the best possible tuned
performance of every solver or general physical root completeness.

## Reproduction and evidence

Install the exact matching 0.3.1 CPU/plugin wheels, or select their extracted
runtime with PYTHONPATH. Use the existing frozen baseline archives and the
pinned benchmark adapter; do not substitute new high as reference. The driver
prints per-suite/per-batch progress and writes checkpoint summaries.

```sh
python tools/benchmark_external.py performance --baseline "$baseline_dir" \
  --qed-library "$qed_library" --out cpu-high --batches 1 8 64 \
  --solvers modisp-cpu1 modisp-cpu8 --accuracy high \
  --search-strategy shape_adaptive --repeats 5
python tools/benchmark_external.py performance --baseline "$baseline_dir" \
  --qed-library "$qed_library" --out cuda-high --batches 1 8 64 1000 \
  --solvers modisp-cuda --accuracy high --search-strategy shape_adaptive \
  --repeats 5 --guard-gpu
python tools/benchmark_external.py accuracy --baseline "$baseline_dir" \
  --qed-library "$qed_library" --out accuracy-high --threads 8 \
  --solvers modisp-cpu modisp-cuda --accuracy high \
  --search-strategy shape_adaptive --guard-gpu
```

Repeat modisp commands with balanced and fast in fresh output directories.
Run third parties with `performance --solvers disba-serial disba-process8
qed-serial --batches 1 8 64`, and `accuracy --solvers disba qed`, using the same
baseline and benchmark adapter. The original inputs, arrays, logs, cache/setup
records and the failed adapter attempt remain in private validation folders;
only source/docs and statistical summaries are public. Historical 0.2.0
measurements remain in [the validation archive](VALIDATION.md).
