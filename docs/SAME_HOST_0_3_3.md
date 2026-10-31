# Same-host modisp 0.3.3 measurements — 2026-10-31

Only modisp was selected for the final timing refresh. Per user instruction,
the published disba 0.7.0 and QEDispInv `ed1b5dd` rows are copied unchanged from
the [2026-10-29 receipt](benchmarks/samehost-0.3.2-20261029.json), not from any new external
output. Early external accuracy calls had already completed before scope was
narrowed; those outputs are excluded. No new external performance run was made.

The solver is installed modisp 0.3.3 (`1cb70cd`), ABI 7, default per-instance
Brent refinement, shape-adaptive discovery, CPU Dunkin and CUDA fast. Standalone
mode reliability checking is excluded from solve timings. The comparison is
solid-Rayleigh only; Love phase counting is not exercised here.

## Load, dates and interpretation

The host is dual-socket Xeon Gold 6430 (128 logical CPUs), with A800 80GB PCIe
physical GPU 1. Python 3.10.12, NumPy 2.2.6, Numba 0.68.0, disba 0.7.0.
The pinned QED adapter hash matches the retained record. New modisp CPU stages
started at one-minute load averages 184.2–194.4; a separate snapshot reached
190.7. In particular, raw minimum/maximum samples show scheduling variation.
Historical external timings have different load/date provenance. These tables
do not establish equal-load superiority or a causal 0.3.2-to-0.3.3 speedup.

CUDA pmon records for each preset have no competing-compute samples. Resident
contexts and host load remain; short GPU calls include CPU scheduling overhead.
No process belonging to another user was altered.

## Independent-reference accuracy

All 11 NPZ hashes agree with the old comparison: 171 models, 6,083 traces,
19,161 finite reference cells. Original reference-stability masks are kept.
New modisp output is compared to the independent reference, not to disba/QED.

| Solver / accuracy | Test date | Wrong | Missing | Extra | Exceptions / unresolved | Max ordinal common abs error (km/s) |
|---|---|---:|---:|---:|---:|---:|
| modisp 0.3.3 CPU / high | 2026-10-31 | 0 | 0 | 0 | 0 | 1.976388e-07 |
| modisp 0.3.3 CUDA / high | 2026-10-31 | 0 | 0 | 0 | 0 | 1.976388e-07 |
| disba 0.7.0 (retained) | 2026-10-29 | 1338 | 1034 | 0 | 12 exceptions | 2.868654e+00 |
| QEDispInv ed1b5dd (retained) | 2026-10-29 | 0 | 1 | 0 | 0 | 1.629918e-12 |

Strict threshold: `max(3e-6,2e-6*abs(reference))` km/s, rank-aligned. Slot missing /
extra is not equivalent to setwise root recall. Approximate preset matching:

| Backend / accuracy | Strict wrong | Isolated-root failures in budget | Strict unmatched cluster roots | Ordinal shifts | Outside independent prefix | Max matched relative error (with guards) |
|---|---:|---:|---:|---:|---:|---:|
| CPU / high | 0 | 0 | 0 | 0 | 0 | 6.248794e-08 |
| CUDA / high | 0 | 0 | 0 | 0 | 0 | 6.248794e-08 |
| CPU / balanced | 6 | 0 | 5 | 6 | 5 | 6.263855e-07 |
| CUDA / balanced | 6 | 0 | 5 | 6 | 5 | 6.263855e-07 |
| CPU / fast | 8312 | 0 | 5 | 6 | 5 | 6.249694e-06 |
| CUDA / fast | 8312 | 0 | 5 | 6 | 5 | 6.249694e-06 |

Conditional guard roots classify prefix boundaries; they do not replace the
independent four-mode reference. Outside-prefix returned values remain
independently unjudged. Broad root completeness is not established.

## Fixed-shape timings

Five steady calls after first call/warmup; native solvers/process pools persist
through each configuration. B<=64 uses distinct ten-layer seed models, 64 periods,
four modes, cmin=.2. Thread/process counts are explicit.

| Solver / accuracy | Test date | B=1 (s) | B=8 (s) | B=64 (s) | B=64 models/s |
|---|---|---:|---:|---:|---:|
| modisp CPU, 1 thread / high | 2026-10-31 | 0.097861 | 0.825126 | 6.430636 | 9.952 |
| modisp CPU, 8 threads / high | 2026-10-31 | 0.021021 | 0.211781 | 0.847331 | 75.531 |
| modisp CUDA / high | 2026-10-31 | 0.006762 | 0.009803 | 0.045213 | 1415.533 |
| modisp CPU, 1 thread / balanced | 2026-10-31 | 0.043905 | 0.301913 | 2.566146 | 24.940 |
| modisp CPU, 8 threads / balanced | 2026-10-31 | 0.008928 | 0.041338 | 0.580373 | 110.274 |
| modisp CUDA / balanced | 2026-10-31 | 0.001920 | 0.002384 | 0.008646 | 7402.372 |
| modisp CPU, 1 thread / fast | 2026-10-31 | 0.041180 | 0.287676 | 2.456856 | 26.050 |
| modisp CPU, 8 threads / fast | 2026-10-31 | 0.014233 | 0.048632 | 0.302077 | 211.867 |
| modisp CUDA / fast | 2026-10-31 | 0.001787 | 0.002163 | 0.007844 | 8158.722 |
| disba, serial (retained) | 2026-10-29 | 0.011032 | 0.096813 | 0.788136 | 81.204 |
| disba, 8 processes (retained) | 2026-10-29 | 0.012103 | 0.014981 | 0.109398 | 585.022 |
| QEDispInv, serial (retained) | 2026-10-29 | 0.012634 | 0.114149 | 0.840837 | 76.115 |

## Repeated CUDA workload

Task i uses source model i % 64, so B=1,000 is 1,000 tasks and 64 independent
models / 4,096 distinct model-period reference traces.

| CUDA accuracy | Tasks | Independent models | Unique reference traces | First call (s) | Warm median (s) | Tasks/s | First/last strict issues | Outside independent prefix (each output) |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| high | 1,000 | 64 | 4,096 | 0.476671 | 0.469490 | 2129.969 | 0 / 0 | 0 |
| balanced | 1,000 | 64 | 4,096 | 0.127757 | 0.120532 | 8296.564 | 64 / 64 | 48 |
| fast | 1,000 | 64 | 4,096 | 0.116973 | 0.108554 | 9211.995 | 94839 / 94839 | 78 |

## Provenance and reproduction

The [machine-readable receipt](benchmarks/samehost-0.3.3-20261031.json) retains
all samples, binary hashes, manifests, load snapshots, quality metrics and the
source receipt hash for external rows. The exact external rows are unchanged.
The private raw directory is `modisp-samehost-0.3.3-20261031` on yaolab; copies
and logs are retained locally under the matching results directory.

To repeat modisp only with the retained independent inputs and adapter:

```sh
python tools/benchmark_external.py accuracy --baseline /path/to/reference-npz \
  --qed-library /path/to/libqed_reference.so --out /new/accuracy-high \
  --solvers modisp-cpu modisp-cuda --threads 8 --accuracy high \
  --search-strategy shape_adaptive --require-clean-modisp

python tools/benchmark_external.py performance --baseline /path/to/reference-npz \
  --qed-library /path/to/libqed_reference.so --out /new/performance-cpu-high \
  --solvers modisp-cpu1 modisp-cpu8 --batches 1 8 64 \
  --accuracy high --search-strategy shape_adaptive --repeats 5

CUDA_VISIBLE_DEVICES=1 python tools/benchmark_external.py performance \
  --baseline /path/to/reference-npz --qed-library /path/to/libqed_reference.so \
  --out /new/performance-cuda-high --solvers modisp-cuda --batches 1 8 64 1000 \
  --accuracy high --search-strategy shape_adaptive --repeats 5 --guard-gpu
```

Repeat only the modisp selections for balanced/fast, omitting the strict-clean
gate for approximate accuracy. The adapter argument is required by the generic
tool, but modisp-only solver selections do not run QED or disba. Third-party
retained source: [0.3.2 methods/report](SAME_HOST_0_3_2.md).
