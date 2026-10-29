# modisp 0.3.1

modisp is an FP64, GPU-accelerated package for batched multimode Rayleigh/P–SV
and Love phase-velocity calculations in 1-D layered elastic media. A portable
CPU backend and an optional Linux CUDA 12 plugin share one Python interface.
The package is designed for large-scale forward modeling in model-space
exploration and synthetic-data workflows. Units are km, km/s, g/cm³ and seconds.

## Install

Install the wheel matching your Python and platform, or build the CPU source:

```sh
python -m pip install /path/to/modisp-0.3.1-<matching-tags>.whl
modisp doctor
```

CPU installation and import require no CUDA. CUDA support is provided by the
separate `modisp-cuda12` wheel with exactly the same version. NumPy is the only
Python runtime dependency. See [installation and deployment](docs/INSTALL.md).

The default `shape_adaptive` search uses density seeds and local five-point
integral consistency to refine difficult intervals and reuse samples. The
implementation independently adapts Zhu et al. (2026) to the existing Dunkin/SH
functions; their GRT validation does not establish this implementation's
accuracy. See [0.3.1 methods and validation](docs/SHAPE_ADAPTIVE_0_3_1.md).
`search_strategy="qe_adaptive"` retains the 0.3.0 path for comparisons.

## Unified dispersion interface

`dispersion`, `dispersion_batch` and `Solver` all return a complete
`DispersionBatch`. A single model is simply a batch with `B=1`.

```python
import numpy as np
from modisp import dispersion

model = np.array([[0.3, 2.0, 0.8, 2.0], [0.0, 5.0, 2.8, 2.6]])
periods = np.logspace(-1, 1, 30)
result = dispersion(model, periods, nmodes=2, wave="rayleigh",
                    backend="cpu", threads=4)

rayleigh = result.phase_velocity[0, 0]
print(result.period, rayleigh)
print(result.status, result.trace_flags)
```

The output arrays have shape `[B,nmodes,T]`; periods may be unsorted or
repeated. Undetected modes remain NaN with `SolveStatus.NOT_FOUND`. Numerical
failures remain NaN with `UNRESOLVED` or `REJECTED` and retain all diagnostics
instead of raising a curve-specific exception.

For repeated calls, reuse a solver:

```python
from modisp import Solver

with Solver(backend="cpu", threads=4) as solver:
    result = solver(*model.T, periods, nmodes=3, wave="love")
    print(solver.stats(), solver.last_result is result)
```

`accuracy="balanced"` is the default. Balanced/fast permit flagged ambiguity
within a configurable close-root scale (`close_root_rtol=0.005`). Their local
sampling is controlled separately by `sampling_rtol` (high `1e-3`, balanced
`1e-2`, fast `5e-2`). All three use FP64 evaluation and refinement. Balanced/fast
flag approximate searches and expose `mode_index_uncertain` and
`close_root_mask`. Select `high` to search close roots individually:

```python
with Solver(backend="cuda", accuracy="high") as exact:
    reference = exact(...)
```

Models with different layer counts can use zero-thickness solid padding.
Reusable solvers retain workspaces and release the Python GIL during native
computation. `backend="auto"` selects a compatible CUDA device when available,
otherwise CPU and records the reason. Numerical failure never changes backend.

### Large model batches

The solver accepts parameter arrays shaped `[B,L]` and returns velocity arrays
shaped `[B,nmodes,T]`. For long runs, process and save outer batches so the
Python input and result arrays do not need to hold the entire model set at once.
The native workspace is budgeted and the solver can chunk internally as well.

```python
import numpy as np
from modisp import Solver

h = np.load("thickness.npy", mmap_mode="r")
vp = np.load("vp.npy", mmap_mode="r")
vs = np.load("vs.npy", mmap_mode="r")
rho = np.load("rho.npy", mmap_mode="r")
n_models = h.shape[0]
periods = np.logspace(-1, 1, 64)
chunk_size = 512
velocity = np.lib.format.open_memmap(
    "phase_velocity.npy", mode="w+", dtype=np.float64,
    shape=(n_models, 4, periods.size),
)
trace_flags = np.lib.format.open_memmap(
    "trace_flags.npy", mode="w+", dtype=np.uint32,
    shape=(n_models, periods.size),
)

with Solver(backend="auto", threads=8, cmin=0.2) as solver:
    for start in range(0, n_models, chunk_size):
        stop = min(start + chunk_size, n_models)
        result = solver(
            h[start:stop], vp[start:stop], vs[start:stop], rho[start:stop],
            periods, nmodes=4, wave="rayleigh",
        )
        velocity[start:stop] = result.phase_velocity
        trace_flags[start:stop] = result.trace_flags

velocity.flush()
trace_flags.flush()
```

Inspect `status`, `trace_flags`, `numerical_flags`, `search_level` and
`root_count` alongside the velocities. A requested mode can be absent without being a numerical failure;
see [API and diagnostics](docs/API.md) for their meanings.

### Migrating from 0.1.0

`PhaseDispersion`, `DispersionCurve` and `DispersionError` have been removed.
Replace `pd(periods, mode=m)` with
`dispersion(model, periods, nmodes=m+1, ...)`, and `curve.velocity` with
`result.phase_velocity[0, m]`. Complete status and NaN arrays replace filtered
curves and curve-specific numerical exceptions.

## Water and mixed batches

```python
water_model = np.vstack([[0.025, 1.5, 0.0, 1.02], model])
result = dispersion(water_model, periods, nmodes=3, backend="auto")
print(result.phase_velocity.shape, result.backend_used, result.trace_flags)
```

A first layer with Vs exactly zero is an inviscid acoustic fluid. In that case
`wave="rayleigh"` denotes the coupled P–SV family, including Scholte-type roots.
`wave="love"` solves SH waves with zero shear traction at the solid surface and
is unaffected by inviscid water depth.

## Scope and numerical diagnostics

The final layer is a solid half-space. Solid Vp/Vs must exceed sqrt(4/3). The
current scope is isotropic, non-attenuating layered media with at most one
surface fluid layer. Internal fluids, anisotropy, attenuation, leaky/complex
modes, group velocity, eigenfunctions, sensitivity kernels and inversion are not
included.

Roots are sorted by phase velocity independently at each period; they are not
tracked physical branches across periods. Explicit `cmin` or `cmax` crops the
search spectrum, so mode 0 is the first detected root inside that interval, not
necessarily the model's fundamental mode.

`status`, `trace_flags`, `numerical_flags`, `search_level` and `root_count`
describe the numerical search. `numerical_flags` excludes the user-cropping
`TRACE_RANGE_TRUNCATED` bit. A zero flag or agreement between sampling levels describes the completed
checks, not a mathematical guarantee that every physical root was found. See
[API and diagnostics](docs/API.md).

## Same-host comparison (0.2.0 baseline)

Measured on 2026-10-28 with disba 0.7.0 and QEDispInv `ed1b5dd`, using the same
solid-Rayleigh inputs on dual-socket Xeon Gold 6430 CPUs and one NVIDIA A800.
Accuracy uses all 171 models / 6083 period traces with four requested modes.
The independent judge is a stable two-grid FP64 reference (32769/65537 nodes).
The current presets are checked against all 11 reference suites,
plus the separate extreme witness fixtures. Ordinary errors, close-cluster omissions, ordinal shifts and unjudged values
are reported separately in [current validation](docs/SHAPE_ADAPTIVE_0_3_1.md).
The superseded local 0.3.0 candidate speed claims below/elsewhere are historical and do not apply
to the corrected execution path.

| Solver | Wrong | Missing | Extra | Exceptions / unresolved | Max common abs error (km/s) |
|---|---:|---:|---:|---:|---:|
| modisp CPU | 0 | 0 | 0 | 0 | 2.586418e-07 |
| modisp CUDA | 0 | 0 | 0 | 0 | 2.586418e-07 |
| disba | 1338 | 1034 | 0 | 12 exceptions | 2.868654 |
| QEDispInv | 0 | 1 | 0 | 0 | 1.629918e-12 |

Wrong values use `abs(actual-reference) > max(3e-6, 2e-6*abs(reference))`.
Maximum common absolute error covers only points finite in both outputs and does
not replace wrong/missing/extra counts. disba used its default Dunkin/`dc=0.005`
settings.

The fixed-shape comparison uses the first 64 `seed20260916` models (`L=10`,
`T=64`, `nmodes=4`, `cmin=0.2`). A separate CUDA run contains **1,000 solve
tasks and 64 independent baseline models**: task `i` cycles to source model
`i % 64`. This produces 64,000 model-period tasks but only 4,096 distinct
model-period reference traces. The B=1/8/64 table below compares all solver
configurations; the B=1,000 measurement is CUDA only. All timings are medians of
five steady-state calls after warmup; the B=1,000 output's first and last calls
were also checked against the independent reference.

| Solver / configuration | B=1 (s) | B=8 (s) | B=64 (s) | B=64 models/s |
|---|---:|---:|---:|---:|
| modisp CPU, 1 thread | 0.252565 | 1.908055 | 13.487279 | 4.745 |
| modisp CPU, 8 threads | 0.040718 | 0.259115 | 1.759717 | 36.369 |
| modisp CUDA, A800 | 0.022400 | 0.030346 | 0.094644 | 676.218 |
| disba, serial | 0.011027 | 0.097004 | 0.809310 | 79.080 |
| disba, 8 processes | 0.012252 | 0.024412 | 0.144897 | 441.694 |
| QEDispInv, serial | 0.013284 | 0.117171 | 0.864273 | 74.051 |

The B=1,000 CUDA-only run completed in a median 1.298926 s, or 769.867 solve
tasks/s. The task count is 1,000 and the unique model count is 64.

| Solver | Task count | Independent models | Unique reference traces | First call (s) | Warm median (s) | Tasks/s | First/last issues |
|---|---:|---:|---:|---:|---:|---:|---|
| modisp CUDA, A800 | 1,000 | 64 | 4,096 | 1.450736 | 1.298926 | 769.867 | 0 / 0 |

Both the first and last output had zero wrong, missing, extra or unresolved
results; their maximum common absolute error was `2.140507e-07 km/s`. Third-party
solvers were not rerun at B=1,000, so the same-size external comparison remains
the B=64 table above.

The disba parallel configuration uses a persistent eight-worker process pool;
QEDispInv is serial. First-call/JIT costs and raw samples are reported in
[validation](docs/VALIDATION.md). These measurements are specific to this input
and configuration; timing and correctness should be read together rather than
as a claim of superiority on every model.

## Command line

```sh
python examples/make_input.py input.npz
modisp forward input.npz --output output.npz --backend auto --nmodes 3 --accuracy balanced
```

The CLI reads NPZ arrays and writes phase velocities together with numerical
diagnostics. Run [the examples](examples/quickstart.py) after installation from
any directory.

## Documentation and methods

- [Python API](docs/API.md)
- [Installation](docs/INSTALL.md)
- [中文快速开始](docs/QUICKSTART_ZH.md)
- [Validation](docs/VALIDATION.md)
- [Changes](CHANGELOG.md)
- [Sources and licenses](THIRD_PARTY_NOTICES.md)

Methods: [Dunkin (1965)](https://doi.org/10.1785/BSSA0550020335),
[CPS](https://rbherrmann.github.io/ComputerProgramsSeismology/),
[disba](https://github.com/keurfonluu/disba), and density/extremum-guided search
motivated by [Pan et al. (2026)](https://doi.org/10.1785/0120250207).

BSD-3-Clause. Author: Yuan Yusong.

AI-assisted development: GPT 6.1 Sol and Kimi k3.1 pro.
