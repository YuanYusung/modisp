# modisp 0.2.0 validation

This document records checks executed for this release. Build configuration and
unexecuted platform combinations are not treated as validation evidence.

## Acceptance scope

The public suite covers CPU/CUDA numerical consistency, multimode Rayleigh and
Love solutions, surface-water and Scholte cases, analytic half-space/two-layer
references, disba comparisons, batching, padding, solver reuse, input checks,
workload diagnostics and the NPZ command-line interface.

Release checks use installed wheels from outside the source tree. CPU and CUDA
run separately with `MODISP_TEST_BACKEND`; the legacy uniform compatibility test
runs only on CUDA. Additional smoke checks cover examples, `doctor`, `forward`,
package metadata and installed extension paths.

## Release results

| Platform | Python | Backend | Public suite | Installation and smoke |
|---|---|---|---|---|
| macOS 27.0.1 arm64 | 3.12.12 | CPU | 44 passed, 1 CUDA-only skipped | Wheel and sdist installs; examples, CLI and smoke passed |
| Ubuntu 22.04 / Linux x86_64 | 3.10.12 | CPU | 44 passed, 1 CUDA-only skipped | Wheel install; examples, CLI and smoke passed |
| Same Linux host, NVIDIA A800/sm_80, CUDA 12.6 | 3.10.12 | CUDA | 45 passed | Matching plugin wheel; examples, CLI and smoke passed |

Both CPU runs included the disba comparisons; no test was skipped for a missing
test dependency. `pip check` passed in the release environments. On the A800,
`backend="auto"` selected CUDA, and explicit CPU/CUDA CLI runs completed with
installed extensions loaded from the isolated `.venv-0.2.0` environment.

## External solid-Rayleigh accuracy

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

## External solid-Rayleigh performance

The performance run used the first 64 `seed20260916` models, `L=10`, `T=64`,
four modes and `cmin=0.2`. Each configuration recorded initialization and a
first complete call, then a separate warmup and five steady-state calls. The
table reports first-call and median steady seconds; models/s uses the median.
Every first and last output was checked against the independent reference.

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
