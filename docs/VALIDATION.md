# Validation

## 0.3.0 — 2026-10-28

Installed-wheel checks run outside the source tree:

| Platform | Python | Backend | Public suite | Additional checks |
|---|---|---|---|---|
| macOS arm64 | 3.12.12 | CPU | 64 passed, 3 skipped | pip check, installed CLI/smoke |
| Ubuntu 22.04 x86_64 | 3.10.12 | CPU | 64 passed, 3 skipped | pip check, CLI, examples |
| Same host, A800/sm80, CUDA 12.6 | 3.10.12 | CUDA | 67 passed, 0 skipped | memcheck/racecheck/synccheck/initcheck: 0 errors, 0 leaked bytes |

The 11 independent solid-Rayleigh reference suites contain 171 models, 6,083
period traces and 24,332 stable mode cells. All cells were judged; none was
excluded as reference-unstable. `high`, `balanced` and `fast` each have zero
wrong, missing, extra or unresolved results. Maximum common absolute errors are
2.586418e-7 km/s for `high` and 2.624469477296998e-7 km/s for `balanced`/`fast`;
the threshold remains max(3e-6, 2e-6*abs(reference)) km/s.

`accuracy="high"` reproduces 0.2.1 bit for bit on seven compact CPU/CUDA
fixtures and on the complete Base output: 10,764,863 finite cells, maximum
difference 0.0, and all status/flags/level/root-count/bounds diagnostics equal.
The 1,024-model before/after A/B also retained exact identity for `high` and
measured a 1.0014x steady speed ratio (no regression).

`balanced` against 0.2.1 on the same 1,024-mode task retained zero wrong,
missing, extra or newly-unresolved cells; maximum common error was 2.1494e-7
km/s, with no cell above 1e-6. The CPU comparison had five extra finite cells
(roots found by balanced where 0.2.1 had no value), zero missing, zero new
unresolved and the same maximum. The only changed trace flag was
`TRACE_WORKSPACE_LIMIT` (1,520 traces, all reduced). Search levels differ by
design. On the complete Base workload, balanced retained zero
wrong/missing/extra/newly-unresolved cells and maximum common error 2.8042e-7
km/s; mode mask, status, root count and bounds were unchanged. The Aug 10,000
stress set also retained zero wrong/missing/extra/newly-unresolved cells and
maximum 2.9777e-7 km/s.

The separate public extreme-fixture set retains four unresolved traces per
backend from 0.2.0 under `accuracy="high"`: all three B10000 fixtures and
witness_model9384_period11. Their NaNs, status, flags and root counts match the
baseline. These cases are not part of the 6,083-trace archive above. Passing
their tests validates honest failure signaling; it does not mean those physical
roots were recovered under the high schedule. Other tests explicitly reject an
all-UNRESOLVED regression.

`fast` measured 2.45x on the 1,024-model CUDA task, 2.74x on the 1,024-model
64-thread CPU task, 2.53x solver/2.47x end-to-end on complete Base, and 2.55x on
Aug 10,000. It retained zero
wrong/missing/extra/newly-unresolved cells on all four, with maximum common
errors 2.2580e-7, 2.8042e-7 and 2.9777e-7 km/s. The earlier 16,384-point fine
dense floor produced six shifted close-root Aug cells (maximum 0.088 km/s); the
shipped fast preset uses 32,768 points and the four-cell dense safety margin.

The A/B runs used interleaved order, a separate warmup and five steady samples.
CUDA steadies were 7.4857 s (0.2.1) and 3.1352 s (balanced), a 2.39x gain. CPU
64-thread steadies were 31.6240 s and 11.5768 s, a 2.73x gain. Complete Base
solver time was 207.92 s and 85.27 s (2.44x); end-to-end time was 211.46 s and
88.69 s (2.38x). The Aug 10,000 stress set measured 74.11 s and 29.84 s (2.48x).
All accepted first/last outputs had zero wrong, missing, extra and newly
unresolved results.

## 0.2.1 — 2026-10-28

Installed-wheel checks run outside the source tree:

| Platform | Python | Backend | Public suite | Additional checks |
|---|---|---|---|---|
| macOS arm64 | 3.12.12 | CPU | 49 passed, 3 CUDA-only skipped | pip check, installed CLI/smoke |
| Ubuntu 22.04 x86_64 | 3.10.12 | CPU | 49 passed, 3 CUDA-only skipped | pip check, CLI, examples |
| Same host, A800/sm80, CUDA 12.6 | 3.10.12 | CUDA | 52 passed | 7 memcheck tests; 0 errors, 0 leaked bytes |

The 11 independent solid-Rayleigh reference suites contain 171 models, 6,083
period traces and 24,332 stable mode cells. All cells were judged; none was
excluded as reference-unstable. CPU and CUDA each have zero wrong, missing, extra
or unresolved results, with maximum common absolute error 2.586418e-7 km/s.
The threshold remains max(3e-6, 2e-6*abs(reference)) km/s.

The separate public extreme-fixture set retains four unresolved traces per
backend from 0.2.0: all three B10000 fixtures and witness_model9384_period11.
Their NaNs, status, flags and root counts match the baseline. These cases are
not part of the 6,083-trace archive above. Passing their tests validates honest
failure signaling; it does not mean those physical roots were recovered. Seven
previously successful witness fixtures now explicitly reject an all-UNRESOLVED
regression in CI.

The 1,024-model first/last before/after checks and all eight full Base NPY files
retain exact numerical/diagnostic identity. Changed input values, tail capacity,
concurrent solvers, 256 layers, damaged-output rejection and interrupted resume
are tested. See [performance measurements](PERFORMANCE.md) and the compact
[numeric receipt](benchmarks/modephys-0.2.1.json). Raw models/results remain in
the separate validation archive.

## 0.2.0 release archive

This document records checks executed for this release. Build configuration and
unexecuted platform combinations are not treated as validation evidence.

### Acceptance scope

The public suite covers CPU/CUDA numerical consistency, multimode Rayleigh and
Love solutions, surface-water and Scholte cases, analytic half-space/two-layer
references, disba comparisons, batching, padding, solver reuse, input checks,
workload diagnostics and the NPZ command-line interface.

Release checks use installed wheels from outside the source tree. CPU and CUDA
run separately with `MODISP_TEST_BACKEND`; the legacy uniform compatibility test
runs only on CUDA. Additional smoke checks cover examples, `doctor`, `forward`,
package metadata and installed extension paths.

### Release results

| Platform | Python | Backend | Public suite | Installation and smoke |
|---|---|---|---|---|
| macOS 27.0.1 arm64 | 3.12.12 | CPU | 44 passed, 1 CUDA-only skipped | Wheel and sdist installs; examples, CLI and smoke passed |
| Ubuntu 22.04 / Linux x86_64 | 3.10.12 | CPU | 44 passed, 1 CUDA-only skipped | Wheel install; examples, CLI and smoke passed |
| Same Linux host, NVIDIA A800/sm_80, CUDA 12.6 | 3.10.12 | CUDA | 45 passed | Matching plugin wheel; examples, CLI and smoke passed |

Both CPU runs included the disba comparisons; no test was skipped for a missing
test dependency. `pip check` passed in the release environments. On the A800,
`backend="auto"` selected CUDA, and explicit CPU/CUDA CLI runs completed with
installed extensions loaded from the isolated `.venv-0.2.0` environment.

### External solid-Rayleigh accuracy

Measured on 2026-10-28 on the Linux host above. The comparison used all 11
archived suites: 171 models, 6083 period traces and four requested Rayleigh
modes per trace. The independent judge is the stable two-grid FP64 reference
(32769/65537 nodes), not either external solver.

| Solver | Wrong | Missing | Extra | Exceptions / unresolved traces | Max common abs error (km/s) |
|---|---:|---:|---:|---:|---:|
| modisp CPU | 0 | 0 | 0 | 0 | 2.586418e-07 |
| modisp CUDA | 0 | 0 | 0 | 0 | 2.586418e-07 |
| disba 0.7.0 | 1338 | 1034 | 0 | 12 exceptions | 2.868654 |
| QEDispInv `ed1b5dd` | 0 | 1 | 0 | 0 | 1.629918e-12 |

Wrong values use `abs(actual-reference) > max(3e-6, 2e-6*abs(reference))`.
Maximum common absolute error includes only points finite in both outputs and
is not a substitute for the wrong/missing/extra counts. disba used its default
Dunkin/`dc=0.005` settings. All cases, including external failures, were kept.

The raw outputs, per-suite reports, input hashes and environment are retained in
the external release archive under `accuracy-full/`; they are not bundled in the
public source or runtime distributions.

### External solid-Rayleigh performance

The fixed-shape performance comparison used the first 64 `seed20260916` models,
`L=10`, `T=64`, four modes and `cmin=0.2`. Each configuration recorded
initialization and a first complete call, then a separate warmup and five
steady-state calls. The table reports first-call and median steady seconds;
models/s uses the median. Every first and last output was checked against the
independent reference.

| Solver / configuration | B | First call (s) | Steady median (s) | Models/s |
|---|---:|---:|---:|---:|
| modisp CPU, 1 thread | 1 | 0.253329 | 0.252565 | 3.959 |
| modisp CPU, 1 thread | 8 | 1.888680 | 1.908055 | 4.193 |
| modisp CPU, 1 thread | 64 | 13.475390 | 13.487279 | 4.745 |
| modisp CPU, 8 threads | 1 | 0.052106 | 0.040718 | 24.559 |
| modisp CPU, 8 threads | 8 | 0.324607 | 0.259115 | 30.874 |
| modisp CPU, 8 threads | 64 | 1.914629 | 1.759717 | 36.369 |
| modisp CUDA, A800 | 1 | 0.192401 | 0.022400 | 44.642 |
| modisp CUDA, A800 | 8 | 0.032447 | 0.030346 | 263.625 |
| modisp CUDA, A800 | 64 | 0.095017 | 0.094644 | 676.218 |
| disba, serial | 1 | 0.756975 | 0.011027 | 90.684 |
| disba, serial | 8 | 0.098097 | 0.097004 | 82.471 |
| disba, serial | 64 | 0.803140 | 0.809310 | 79.080 |
| disba, 8 processes | 1 | 0.012416 | 0.012252 | 81.620 |
| disba, 8 processes | 8 | 0.030993 | 0.024412 | 327.704 |
| disba, 8 processes | 64 | 0.146907 | 0.144897 | 441.694 |
| QEDispInv, serial | 1 | 0.026260 | 0.013284 | 75.279 |
| QEDispInv, serial | 8 | 0.116775 | 0.117171 | 68.276 |
| QEDispInv, serial | 64 | 0.870068 | 0.864273 | 74.051 |

#### GPU-only 1,000-task run

At the user's direction, the larger-batch measurement was run on the CUDA
backend only. It submits **1,000 solve tasks using 64 independent baseline
models**: task `i` maps to model `i % 64`. Each task has 64 periods, so the
batch has 64,000 model-period tasks and 4,096 distinct reference traces. This
is a repeated-model throughput measurement, not a run with 1,000 independent
models. The command was `tools/benchmark_external.py performance --batches
1000 --solvers modisp-cuda --repeats 5`.

| Solver | Task count | Independent models | Unique reference traces | First call (s) | Warm median (s) | Tasks/s | First/last issues |
|---|---:|---:|---:|---:|---:|---:|---|
| modisp CUDA, A800 | 1,000 | 64 | 4,096 | 1.450736 | 1.298926 | 769.867 | 0 / 0 |

The first and last calls each had zero wrong, missing, extra or unresolved
results; maximum common absolute error was `2.140507e-07 km/s`. CPU and
third-party solvers were not measured at B=1,000 in this pass. Their full
same-host comparison remains the B=1/8/64 results above; no same-size external
speedup is implied by this CUDA-only row. Raw timing samples and outputs are
retained alongside the external release archive under
`performance-B1000-cuda/`.

The host has dual-socket Xeon Gold 6430 CPUs and NVIDIA A800/sm_80 GPUs. The
measurement used Python 3.10.12, NumPy 2.2.6, Numba 0.68.0 and disba 0.7.0.
The disba serial first call includes JIT. Its persistent eight-process pool was
created and all eight workers were warmed before timed calls. QEDispInv timing
uses the exact four-mode entry, not the extra-root reference path. modisp had
zero wrong/missing/extra/unresolved in every performance first/last check;
external failures are retained in the accuracy table above.

Raw outputs, five timing samples per configuration, initialization details and
environment are retained in the external release archive under
`performance-full/`. The public reproducibility driver is
`tools/benchmark_external.py`.

The release source archive, wheel inventories and SHA-256 manifest are generated
from this checked source tree. Numerical diagnostics and their interpretation
are documented in [Python API](API.md). Test outcomes apply to the executed
cases and do not by themselves establish completeness for every layered model.
