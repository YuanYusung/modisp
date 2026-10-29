# Same-host modisp 0.3.2 comparison — 2026-10-29

These measurements use installed modisp 0.3.2 CPU/CUDA binaries and the backend
defaults from solver commit `ce1ab41d6894d1f885517b4ad8c02e2c06b40da0`.
CPU uses Dunkin; CUDA uses fast. The CPU native SHA256 is
`2e80ff5ed2d11010e0e6b332f15b5535be96cd7d2d4346c7294a92adf134c734`; CUDA is
`ebefe6627065aa87fe31bef84d54adaf33b3f65244f566fafb7d11f14c31f307`. The host is a dual-socket Xeon Gold 6430 with
A800 80GB PCIe; the timed CUDA device is physical GPU 1 (sm_80).
Python 3.10.12, NumPy 2.2.6, Numba 0.68.0 and disba 0.7.0 are shared by the
comparison. BLAS/OpenMP worker counts are one; modisp CPU thread counts are
explicit. Environment snapshots and all samples are in the
[statistical receipt](benchmarks/samehost-0.3.2-20261029.json).

## Independent-reference accuracy

All 11 archived solid-Rayleigh suites retain their previous hashes: 171
independent models, 6,083 period traces and 19,161 finite reference cells.
All archived reference traces are stable under their recorded 32769/65537-node
FP64 check. Four modes are requested. This comparison does not externally
validate Love, water or Scholte. The strict criterion is
`abs(actual-reference) > max(3e-6, 2e-6*abs(reference))` with ordinal alignment.

| Solver / accuracy preset | Wrong | Missing | Extra | Exceptions / unresolved | Max ordinal common abs error (km/s) |
|---|---:|---:|---:|---:|---:|
| modisp CPU / high | 0 | 0 | 0 | 0 | 2.621352e-07 |
| modisp CUDA / high | 0 | 0 | 0 | 0 | 2.621352e-07 |
| disba 0.7.0 | 1338 | 1034 | 0 | 12 exceptions | 2.868654e+00 |
| QEDispInv ed1b5dd | 0 | 1 | 0 | 0 | 1.629918e-12 |

Approximate presets use their own relative budgets (balanced 2e-5, fast 2e-4)
and retain the configurable 0.5% close-root scale. Their actual strict counts
must be read alongside the separate root matching:

| Backend / accuracy | Secular | Strict wrong | Isolated-root failures in budget | Strict unmatched cluster roots | Ordinal shifts | Outside independent prefix | Max matched relative error (with guards) |
|---|---|---:|---:|---:|---:|---:|---:|
| CPU / high | dunkin | 0 | 0 | 0 | 0 | 0 | 6.279798e-08 |
| CUDA / high | fast | 0 | 0 | 0 | 0 | 0 | 6.279798e-08 |
| CPU / balanced | dunkin | 6 | 0 | 5 | 6 | 5 | 6.289137e-07 |
| CUDA / balanced | fast | 6 | 0 | 5 | 6 | 5 | 6.289137e-07 |
| CPU / fast | dunkin | 14220 | 0 | 5 | 6 | 5 | 6.277271e-06 |
| CUDA / fast | fast | 14211 | 0 | 5 | 6 | 5 | 6.277271e-06 |

All six modisp configurations have zero unresolved traces. Approximate traces
retain uncertainty flags. Their 0.329 km/s maximum ordinal deviation results
from a near-root omission shifting later recovered ranks, while many fast
strict wrong cells also reflect its looser refinement tolerance. The maximum
matched relative errors in the table include conditional retained guard roots.
Those guards classify prefix boundaries only: they never replace independent
reference cells, and five returned balanced/fast cells per backend remain
outside the independent four-mode prefix. Budget acceptance does not certify
arbitrary root completeness or approximate mode indices.

## Fresh same-host performance

Use the `seed20260916` prefix, L=10, T=64, four modes, cmin=.2, 256 MiB native
workspace and shape-adaptive search. B<=64 uses distinct models. Every row was
rerun, including third parties. First call and warmup are separate from the
median of five steady samples; complete first/last outputs are checked against
the independent reference. Modisp's steady outputs also preserve the first
output and diagnostics bitwise.

| Solver / accuracy preset | B=1 (s) | B=8 (s) | B=64 (s) | B=64 models/s |
|---|---:|---:|---:|---:|
| modisp CPU, 1 thread / high | 0.060497 | 0.491111 | 3.786716 | 16.901 |
| modisp CPU, 8 threads / high | 0.007695 | 0.062077 | 0.472353 | 135.492 |
| modisp CUDA / high | 0.006737 | 0.010052 | 0.051538 | 1241.798 |
| modisp CPU, 1 thread / balanced | 0.025082 | 0.182592 | 1.506705 | 42.477 |
| modisp CPU, 8 threads / balanced | 0.003789 | 0.023845 | 0.191617 | 333.999 |
| modisp CUDA / balanced | 0.001890 | 0.002435 | 0.009840 | 6504.294 |
| modisp CPU, 1 thread / fast | 0.023595 | 0.171697 | 1.435598 | 44.581 |
| modisp CPU, 8 threads / fast | 0.003520 | 0.022506 | 0.180825 | 353.933 |
| modisp CUDA / fast | 0.001749 | 0.002083 | 0.008630 | 7415.677 |
| disba, serial | 0.011032 | 0.096813 | 0.788136 | 81.204 |
| disba, 8 processes | 0.012103 | 0.014981 | 0.109398 | 585.022 |
| QEDispInv, serial | 0.012634 | 0.114149 | 0.840837 | 76.115 |

The larger CUDA batch repeats these same 64 models by modulo indexing, giving
1,000 tasks, 64 independent models and 4,096 unique reference traces:

| CUDA accuracy preset | Tasks | Independent models | Unique reference traces | First call (s) | Warm median (s) | Tasks/s | First/last strict issues | Outside independent prefix (each output) |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| high | 1,000 | 64 | 4,096 | 0.558628 | 0.553372 | 1807.102 | 0 / 0 | 0 |
| balanced | 1,000 | 64 | 4,096 | 0.150879 | 0.145254 | 6884.478 | 64 / 64 | 48 |
| fast | 1,000 | 64 | 4,096 | 0.133247 | 0.126782 | 7887.525 | 156672 / 156672 | 78 |

First/last high outputs have zero strict issues. Balanced/fast retain their
strict errors, rank shifts and outside-prefix counts, with zero isolated-root
failures within preset budgets. Third-party B=1000 timing was not measured.

Disba uses default Dunkin/dc=.005. Its eight-process pool persists during each
measured configuration; all eight workers are warmed and verified by PID.
Pool setup and worker warmup are excluded from medians and recorded separately.
The third-party performance process used a separate fresh Numba cache directory
from the accuracy run. QEDispInv is pinned to
`ed1b5dd7b449a2b3a27bb8e6581278790f5df8aa`; its benchmark-capable adapter SHA256 is
`aae549de5d52885178c28fba313a72ed1373b25b1c71db3c83ce4c12fb2a023b`.
Third-party implementations, input arrays and caches are not bundled.

## Device contention and retained evidence

The first CUDA high timing attempt was rejected before timed results because
the monitor observed another process computing on GPU 1. Its logs remain in
the private run directory and none of its samples is used. All three reported
CUDA retries completed with empty competing-compute records. Other allocations
occupied GPU memory, and GPU 0 was used by another task; the host was shared.
These monitored results do not promise a completely isolated platform.

The CPU/CUDA accuracy jobs are correctness checks, not GPU speed measurements;
they did not require the timing monitor. Runtime noise does not change their
reported independent-reference counts.

## Reproduce with retained tools

Install matching 0.3.2 base/plugin packages and use the original 11 NPZ suites
and pinned benchmark-capable QED adapter:

```sh
python tools/benchmark_external.py accuracy \
  --baseline /path/to/reference-npz --qed-library /path/to/libqed_reference.so \
  --out /new/accuracy-high --solvers modisp-cpu modisp-cuda disba qed \
  --threads 8 --accuracy high --search-strategy shape_adaptive --require-clean-modisp

python tools/benchmark_external.py performance \
  --baseline /path/to/reference-npz --qed-library /path/to/libqed_reference.so \
  --out /new/performance-cpu-high --solvers modisp-cpu1 modisp-cpu8 \
  --batches 1 8 64 --accuracy high --search-strategy shape_adaptive --repeats 5

CUDA_VISIBLE_DEVICES=1 python tools/benchmark_external.py performance \
  --baseline /path/to/reference-npz --qed-library /path/to/libqed_reference.so \
  --out /new/performance-cuda-high --solvers modisp-cuda \
  --batches 1 8 64 1000 --accuracy high --search-strategy shape_adaptive \
  --repeats 5 --guard-gpu
```

Repeat accuracy and performance with `--accuracy balanced` and `fast`; omit
`--require-clean-modisp` for approximate accuracy runs and keep their budgeted
quality metrics. For third-party timing use `performance --solvers disba-serial
disba-process8 qed-serial --batches 1 8 64 --repeats 5` with the same reference
and adapter arguments. The selected GPU index is host-specific.

The previous [0.3.1 report](SAME_HOST_0_3_1.md) remains historical evidence;
its measurements were not relabeled as 0.3.2. The 70-layer random-model secular
benchmark is a separate workload in [the method report](SECULAR_FAST_0_3_2.md).
