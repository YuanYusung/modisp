# Performance measurement and ModePhys workflow

## 0.3.0 accuracy presets

Version 0.3.0 adds `accuracy="high"|"balanced"|"fast"`. The default is
`balanced`; `high` reproduces the 0.2.1 search schedule and outputs bit for bit.
Every preset keeps the FP64 secular function, FP64 bracket refinement, root
ordering and diagnostics. The presets only change adaptive sampling, the
independent guarded-verification schedule and the first dense fallback floor.
Traces that cannot prove a safe root spacing fall back to the exact `high` dense
schedule, so the coarse floor cannot make an extreme close-root trace newly
unresolved.

Measured on 2026-10-28 on the same host as 0.2.1, with 1,024 ModePhys models,
70 layers, 100 periods, five Rayleigh modes and `workspace_bytes=1 GiB`. Calls
are interleaved, warmup and first call are separate, and each value is the median
of five steady samples. All comparisons had zero wrong, missing, extra or
newly-unresolved results.

| Workload | 0.2.1 high | 0.3.0 balanced | Speed-up |
|---|---:|---:|---:|
| A800 CUDA, 1,024 models | 7.4857 s | 3.1352 s | 2.39x |
| 64-thread CPU, 1,024 models | 31.6240 s | 11.5768 s | 2.73x |
| A800 CUDA, `accuracy="high"` | 7.4857 s | 7.49 s | 1.00x (bitwise) |
| A800 CUDA, full Base 29,379 models, solver | 207.92 s | 85.27 s | 2.44x |
| A800 CUDA, full Base 29,379 models, end to end | 211.46 s | 88.69 s | 2.38x |
| A800 CUDA, Aug 10,000-model stress set | 74.11 s | 29.84 s | 2.48x |

The balanced 1,024-model run used 144,862,574 evaluations instead of
435,649,857; 96,026 traces accepted early, 86,396 skipped the guarded family,
and 3,578 accepted the coarse dense level. Maximum common absolute difference
from 0.2.1 was 2.1494e-7 km/s; there were no differences above 1e-6 km/s. Full
Base and the Aug 10,000 stress set had zero wrong, missing, extra or
newly-unresolved cells and maximum 2.8042e-7 and 2.9777e-7 km/s respectively;
the five Aug dense-root differences found in the first candidate were fixed by
requiring a four-cell dense safety margin and escalating to the high dense
schedule.

`fast` was measured with the same interleaved protocol after its fine dense
floor was raised to 32,768 points and its four-cell dense safety margin was
aligned with `balanced`:

| Workload | 0.2.1 high | 0.3.0 fast | Speed-up |
|---|---:|---:|---:|
| A800 CUDA, 1,024 models | 7.4912 s | 3.0545 s | 2.45x |
| A800 CUDA, full Base, solver | 207.92 s | 82.31 s | 2.53x |
| A800 CUDA, full Base, end to end | 211.46 s | 85.66 s | 2.47x |
| A800 CUDA, Aug 10,000 stress set | 74.11 s | 29.03 s | 2.55x |

Fast retained zero wrong, missing, extra and newly-unresolved cells on the
1,024-model set, complete Base and Aug 10,000 stress set; maximum common errors
were 2.2580e-7, 2.8042e-7 and 2.9777e-7 km/s respectively. The earlier fast
candidate used a 16,384-point fine floor and produced six shifted close-root
cells in Aug 10,000 (maximum 0.088 km/s); that configuration is not shipped.

The 11 independent FP64 suites (171 models, 6,083 period traces) remain clean:

| Preset | Wrong | Missing | Extra | Unresolved | Max common abs error (km/s) |
|---|---:|---:|---:|---:|---:|
| high | 0 | 0 | 0 | 0 | 2.586418e-07 |
| balanced | 0 | 0 | 0 | 0 | 2.624469477296998e-07 |
| fast | 0 | 0 | 0 | 0 | 2.624469477296998e-07 |

The design note's 12–15x estimate is not claimed: the measured per-trace FP64
evaluation count and dense fallback floor do not support it, and this release
keeps the physical reference acceptance rather than reproducing the old coarse
scan's dense-root errors. `fast` is an explicit opt-in with a larger error bound;
`balanced` is the production default.


## 0.2.1 history

Version 0.2.1 preserves the FP64 recurrence, search densities, shifted/guarded
verification, dense fallback, root ordering and diagnostics from 0.2.0. It does
not add a Numba backend or enable fast math. CPU work remains in compiled C++.

The changes precompute invariant coefficients using the original FP64
expressions, reuse CPU scratch/model capacity and preserve CUDA allocation
capacity for short tail batches. CUDA uses a solver-owned nonblocking stream,
persistent pinned staging and warp reductions of integer sampling counters.
Larger shared-memory requests opt in explicitly for the 256-layer boundary.

## Measured 0.2.1 results

Measured on 2026-10-28 against 0.2.0 `ae5aed5`, with unchanged FP64/root-search
settings. The 1,024 distinct ModePhys models have 70 layers, 100 periods and
five Rayleigh modes. Both installations retain their solver; call order reverses
each round. First call and warmup are separate from five steady samples.

| Workload | 0.2.0 seconds | 0.2.1 seconds | Throughput gain |
|---|---:|---:|---:|
| CPU, 64 threads, 1,024 models, steady median | 33.4922 | 32.0329 | 4.56% |
| A800 CUDA, 1,024 models, steady median | 7.8863 | 7.4865 | 5.34% |
| Complete Base, 29,379 models, solver time | 218.7598 | 207.9634 | 5.19% |
| Complete Base, end-to-end time | 221.5735 | 211.3284 | 4.85% |

The full Base rows are one complete pass per installation, including file I/O,
checksums, the contention guard and cleanup; they are not five-run medians. All
eight Base NPY result/diagnostic files are byte-identical. Both first/last 1,024
model outputs also retain identical finite values, NaN patterns and diagnostics.
CPU uses dual Xeon Gold 6430 with affinity 0-63; CUDA uses the same A800 per paired
comparison. Public [numeric receipt](benchmarks/modephys-0.2.1.json) includes all
five raw samples and the independent-reference scope. Model data, large raw
outputs, private paths and process logs stay outside the public source.

Accepted coverage is complete Base plus the 1,024-model comparison. An observed
competing GPU job invalidated a separate large Aug baseline measurement; no Aug
paired speedup is reported. [Validation](VALIDATION.md) separately records stable-reference success
and four retained extreme-fixture limitations.

## Profiling

```python
with Solver(backend="cuda", workspace_bytes=1024**3, profile=True) as solver:
    result = solver(h, vp, vs, rho, periods, nmodes=5)
    print(solver.stats()["timings_s"])
    print(solver.kernel_report())
```

`profile=False` is the default. Profiled calls are for diagnosis and must be
separate from reported steady-state performance: detailed CPU clocks and CUDA
events add overhead. Since 0.3.0, unprofiled `stats()` also exposes
`accuracy`, `period_threshold`, `sampling_capacity`, `early_accepts`,
`verification_skipped`, `dense_coarse_accepted`, `dense_fine_escalated` and
`trough_risks`. The `high` preset reports zero early/skip/coarse counters. Timings are seconds; CUDA stages are event durations on
the solver's stream. CPU fields ending in `_worker` sum worker durations and
overlap each other; they must not be added to wall time. `solver_total` includes
validation, bounds, native execution and result packaging. The uniform legacy
backend does not provide this detailed adaptive-search profile.

Profiled CPU statistics include sampling-cache hits and misses. CUDA also
returns `active_traces_by_stage`: six full-search levels, six guarded-search
levels and two reconciliation dense levels. These are queue lengths; the first
level can include analytically empty traces that the kernel immediately skips.
`device_workspace_bytes` is the
actual native allocation, `pinned_host_bytes` is separate host staging, and
`workspace_bytes` remains a conservative native-workspace budget estimate.
None includes the driver context or the user's complete input/result arrays.

`MODISP_CUDA_BLOCK=128|256` and `MODISP_CUDA_BLOCKS_PER_SM=1..8` are experimental
launch controls read when the native solver is created. Defaults remain 256
and 2. Their product is a persistent-grid size, not a promise that that many
blocks can reside on every SM. Use `kernel_report()` and measurements. A CUDA
Graph experiment showed no material benefit on the measured ModePhys cases and
was not retained. Pinned double buffering was not added because measured
transfer time was negligible compared with search/recurrence work.

## Reproducible real-task inputs

The source models are OpenSWI FlatVelA profiles used by the ModePhys task. They
are real workload inputs, not field observations or independent higher-mode
truth. Data remain outside the package and keep their original provenance.

```sh
python tools/benchmark_modephys.py prepare \
  --root /path/to/ModePhys --output /new/run/inputs
```

Preparation reads the model NPZ members only, extracts independent NPY files,
and records source/output hashes and the one-time unpack cost. The fixed seed
is 20261028. Aug selection contains 2,000 distinct stress profiles (500 each
ranked by internal speed excess, downward Vs jump, speed contrast and optical
thickness), then 8,000 uniformly sampled distinct profiles. Parameter hashes
exclude duplicate physical profiles. The 1,024-model tuning set interleaves
512 Base and 512 selected Aug profiles; its first 256 rows form the balanced
correctness subset. The manifest saves all original row indices.

Convention A is fixed: 69 layers of 0.04 km plus the final half-space; no input
unit conversion. All calculations use five Rayleigh roots on 100 FP64 logarithmic
periods from 0.2 to 10 s. The source dispersion member has other period grids
and is not used as five-mode truth.

## Streaming, full diagnostics and checked resume

```sh
python tools/benchmark_modephys.py run \
  --input /new/run/inputs/base.npy --periods /new/run/inputs/periods.npy \
  --backend cuda --workspace-bytes 4294967296 --chunk 10000 \
  --output /new/run/results/base
```

One persistent solver serves all blocks. Each output field has its own NPY
file: FP64 velocity and periods, mask, status, flags, search level, root count
and bounds. Every completed block is flushed and checksummed before an atomic
checkpoint update. `--resume` requires matching input/configuration, installed
extension hashes and completed-output checksums. It reports resumed/session
timing rather than inventing an uninterrupted end-to-end throughput.

`report.json` separates setup/validation, block input reads, conversion, solver,
output flush/checksum and the complete elapsed time. OS page-cache state is not
forcibly changed; report repeated reads as cache-warm, not cold storage latency.
FP32 export, cutoff/continuity filtering, compression and training products
must be written separately; none modifies the raw benchmark result.

```sh
python tools/benchmark_modephys.py compare \
  --before /new/run/baseline --after /new/run/candidate \
  --output /new/run/comparison.json
```

This checks the complete finite/NaN pattern and every diagnostic. It is a
before/after regression check, not an independent physical validation. Use the
11 archived stable FP64 suites separately with `benchmark_external.py accuracy`.
Retain reference-unstable points and numerical failures in their own counts.

## Interleaved comparisons

```sh
CUDA_VISIBLE_DEVICES=0 python tools/benchmark_ab.py \
  --before-python /new/run/venv-baseline/bin/python \
  --after-python /new/run/venv-final/bin/python \
  --input /new/run/inputs/tuning1024.npy \
  --periods /new/run/inputs/periods.npy --models 1024 \
  --backend cuda --workspace-bytes 1073741824 --output /new/run/ab
```

Both child processes retain their solver but only one computes at a time.
Round order reverses to reduce thermal/load drift. Initialization/first call,
warmup and five steady samples are separate. First/last outputs and installed
module hashes accompany the medians. The GPU process monitor records per-process
memory and rejects observed competing compute; it never stops another task.
An interrupted or contaminated run is not an accepted speed measurement.

For CPU scaling, bind physical cores explicitly and keep BLAS/OpenMP worker
counts at one when adding an external process pool. The modisp pool itself uses
`threads`. For an accuracy-matched external comparison, read disba's wrong,
missing and exception counts together with timings, including its first-call
JIT cost; speed alone does not establish equivalent root completeness.
