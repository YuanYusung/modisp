# Performance measurement and ModePhys workflow

> For 0.3.2 see [the fresh same-host comparison](SAME_HOST_0_3_2.md) and
> [the fast-secular method report](SECULAR_FAST_0_3_2.md).
> For 0.3.1 use [the shape-adaptive validation report](SHAPE_ADAPTIVE_0_3_1.md).
> The 0.3.0 results below describe the retained `qe_adaptive` path.

## CUDA FP32 on RTX 5050

The benchmark uses `random_model_batch(B, L, seed=20261110)`,
`np.geomspace(.3, 10., T)`, four requested modes and the adaptive shape
search. Wet Rayleigh replaces the first layer with
`(h, Vp, Vs, rho) = (.04, 1.5, 0., 1.02)`. Times are end-to-end medians of
12 warm calls per build, interleaved in alternating order on the same RTX
5050 (`sm_120`, CUDA 12.9). The previous FP32 build used the FP64 secular
recurrence at low Rayleigh velocities.

| Wave | Models × layers × periods | Preset | Previous FP32 | Updated FP32 | Speedup |
|---|---:|---|---:|---:|---:|
| Dry Rayleigh | 24 × 40 × 24 | balanced | 21.79 ms | 10.68 ms | 2.04× |
| Wet Rayleigh | 24 × 40 × 24 | balanced | 31.98 ms | 15.56 ms | 2.06× |
| Love | 24 × 40 × 24 | balanced | 3.34 ms | 3.29 ms | 1.02× |
| Dry Rayleigh | 192 × 70 × 32 | balanced | 316.73 ms | 110.33 ms | 2.87× |
| Wet Rayleigh | 192 × 70 × 32 | balanced | 402.30 ms | 159.44 ms | 2.52× |
| Dry Rayleigh | 8 × 40 × 12 | high | 156.14 ms | 99.31 ms | 1.57× |

Low-velocity Rayleigh samples now use a factored FP32 compound matrix.
With automatic bounds, the FP32 shape search also reuses the CPU lower
bound. On the 192-model wet workload, separate profiled calls put GPU
shape preparation at 27.27 ms before and 0.28 ms after, and GPU shape
search at 345.08 ms before and 137.57 ms after. Search geometry and density
sampling still use FP64.

The mode masks and root counts matched FP64 across the six workloads,
with 56,448 finite entries and no unresolved traces. One balanced dry
trace differed in the fourth mode in both FP32 builds. The high preset
in both precisions recovered the close pair missed by balanced FP64 on
that trace. Excluding that rank difference, the maximum relative error
was `1.26e-6`. On the eight independent close-root witnesses in
`tests/data/witness_*.npz`, high FP32 recovered the requested prefix in
six cases and marked two unresolved. High FP64 recovered all eight.

### Shape-search update for FP32 high and balanced

The RTX 5050 comparison reuses the same inputs, CUDA 12.9 and the
12-call interleaved protocol. The baseline is the FP32 build with the
factored low-velocity recurrence. The thin-layer batches use seed
`20261111`, `h_min=.02`, `h_max=.25`, `lvz_prob=.5` and periods from
.15 to 10 s. Each batch requests four modes.

| Wave | Models × layers × periods | Preset | Previous FP32 | Updated FP32 | Trace fallbacks |
|---|---:|---|---:|---:|---:|
| Dry Rayleigh | 8 × 40 × 12 | high | 101.15 ms | 83.57 ms | 60 → 0 |
| Dry Rayleigh | 24 × 40 × 24 | balanced | 11.04 ms | 11.66 ms | 11 → 0 |
| Wet Rayleigh | 24 × 40 × 24 | balanced | 16.06 ms | 16.06 ms | 0 → 0 |
| Love | 24 × 40 × 24 | balanced | 3.25 ms | 3.25 ms | 0 → 0 |
| Dry Rayleigh | 192 × 70 × 32 | balanced | 111.72 ms | 109.85 ms | 108 → 0 |
| Wet Rayleigh | 192 × 70 × 32 | balanced | 163.07 ms | 165.87 ms | 2 → 0 |
| Thin dry Rayleigh | 48 × 16 × 48 | balanced | 40.49 ms | 41.36 ms | 193 → 0 |
| Thin Love | 48 × 16 × 48 | balanced | 6.48 ms | 6.55 ms | 0 → 0 |

The table counts entries into the dense ladder for `high` and into the
logarithmic observation grid for `balanced`. In the high batch,
evaluations fell from 1,480,774 to 204,748. One of 96 traces triggered
the FP64 shape recheck, which took 33.22 ms in a separate profiled call.
The prior upstream FP64 timing on these high inputs was 53.70 ms.

In the eight timing workloads, all 68,508 finite slots and root counts
matched upstream FP64. One known balanced ordinal difference arose because
the upstream balanced run skipped a close pair; both high solvers resolved
it. For the corresponding roots, the velocity difference was 1.25 mm/s
at P99 and 7.32 mm/s at the maximum, with a maximum relative difference
of `5.79e-6`.

The 11 independent Rayleigh fixtures provide 49 requested roots. High
FP32 recovered 30 roots in six complete traces before this update and all
49 afterward. Six of these deliberately difficult traces triggered a
per-trace FP64 check. Of the five previously unresolved traces, three
converged on FP32 secular evaluations and two used the FP64 recheck.
Maximum relative velocity error on the recovered prefix was `1.82e-7`.
Balanced improved from 34 to 36 one-to-one matches and from three to
four complete prefixes, using FP32 secular evaluations throughout.

## FP64 model preparation on YAOLAB

The A800 80 GB PCIe ran CUDA 12.6 with five requested modes and
`shape_adaptive`. Times are wall-clock medians of 12 warm calls per build
and workload, with alternating order. The upstream baseline is `3db2451`.

| Wave | Models × layers × periods | Preset | Upstream | Updated | Speedup | Reported roots |
|---|---:|---|---:|---:|---:|---:|
| Dry Rayleigh | 2,048 × 24 × 1 | balanced | 48.23 ms | 15.88 ms | 3.04× | 5,168 |
| Dry Rayleigh | 8 × 8 × 28 | balanced | 2.31 ms | 2.27 ms | 1.02× | 784 |
| Dry Rayleigh | 64 × 24 × 64 | balanced | 18.78 ms | 17.82 ms | 1.05× | 14,678 |
| Wet Rayleigh | 24 × 12 × 40 | high | 15.22 ms | 15.06 ms | 1.01× | 2,686 |
| Love | 64 × 24 × 64 | balanced | 17.23 ms | 17.29 ms | 0.996× | 13,373 |

In table order, the generated batches use seeds `20261110+i`, minimum
layer thicknesses `[.04, .1, .04, .04, .04]` km and maximum periods
`[3., 2.5, 10., 8., 10.]` s. All use `h_max=.25` km and
`lvz_prob=.5`. Periods are logarithmically spaced from .15 s to the
maximum, or set to the maximum for the single-period batch. Wet Rayleigh
replaces the first layer with `(.08, 1.5, 0., 1.02)`.

The number of secular evaluations and all seven result arrays matched
bitwise across the paired runs. Separate profiled calls show where the
time went. These stage readings are separate from the medians above.

| Workload | Stage | Upstream | Updated |
|---|---|---:|---:|
| 2,048-model dry Rayleigh | Host bounds | 37.72 ms | 5.43 ms |
| 2,048-model dry Rayleigh | GPU shape search | 8.46 ms | 8.11 ms |
| 64-model dry Rayleigh | Host bounds | 1.19 ms | 0.18 ms |
| 64-model dry Rayleigh | GPU shape search | 16.50 ms | 16.43 ms |
| Wet Rayleigh | GPU shape search | 13.12 ms | 12.77 ms |
| Love | GPU phase search | 10.52 ms | 10.36 ms |
| Love | GPU shape search | 4.15 ms | 4.04 ms |

Bounds are computed once per model. Skipping layers that cannot lower the
automatic Rayleigh bound saves about 32 ms in the single-period batch. The
64-period batch amortizes that work across periods and spends most of its
time in shape search. The wet Rayleigh bound also accounts for the acoustic
layer, and Love has its own phase search and scan fallback.

## 0.3.2 experimental fast secular evaluator

`secular="fast"` keeps the released FP64 dispersion algebra but removes scalar
overhead. On the documented fixed-node CPU workload it improved function
evaluation 1.03–1.19×; the full 32-model/100-period/5-mode high-preset CPU run
improved 1.07–1.10×. On an A800, 1,024 models at the same preset improved from
6.128 s to 4.602 s (1.33×). Root sets were equal in both comparisons. The
CPU default is `secular="dunkin"`; CUDA defaults to `secular="fast"`. Protocol and caveats are in
[the 0.3.2 report](SECULAR_FAST_0_3_2.md).


## 0.3.0 observation-scale presets

Balanced/fast now trade close-cluster resolution for throughput using a 0.5%
configurable observation scale. See [current measured results](TOLERANT_PRESETS.md)
for speed, ordinary-root accuracy, flagged cluster/ordinal differences and
explicitly unjudged reference-prefix values. High retains its full schedule.

The earlier strict-fix experiment is archived in
[CORRECTNESS_PERFORMANCE_FIX.md](CORRECTNESS_PERFORMANCE_FIX.md); its 1.31x CPU
and unchanged CUDA results do not describe the current observation grids.

## Historical 0.3.0 measurements (superseded)

The following timings used the original spacing-based acceptance heuristic.
Two existing witness cases reveal silently missed root pairs outside those
benchmarked inputs. These speed ratios do **not** apply to the corrected code.

### Original 0.3.0 accuracy presets

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
| 64-thread CPU, 1,024 models | 31.7626 s | 11.6101 s | 2.74x |
| A800 CUDA, full Base, solver | 207.92 s | 82.31 s | 2.53x |
| A800 CUDA, full Base, end to end | 211.46 s | 85.66 s | 2.47x |
| A800 CUDA, Aug 10,000 stress set | 74.11 s | 29.03 s | 2.55x |

Fast retained zero wrong, missing, extra and newly-unresolved cells on the
1,024-model CUDA and CPU comparisons, complete Base and Aug 10,000 stress set;
maximum common errors were 2.2580e-7 (1,024), 2.8042e-7 (Base) and 2.9777e-7
(Aug 10,000) km/s. The CPU comparison had five extra finite cells, no missing
cells. The earlier fast
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
