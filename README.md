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

## Same-host comparison (0.3.1)

Rerun on **2026-10-29** using the released modisp 0.3.1 CPU/CUDA binaries,
disba 0.7.0 and QEDispInv `ed1b5dd`, on the same dual-socket Xeon Gold 6430
host and NVIDIA A800 80GB PCIe. All modisp runs explicitly use
`search_strategy="shape_adaptive"`; the preset is named in each row.

Accuracy covers all **171 models / 6,083 period traces / 19,161 finite reference
cells** in 11 solid-Rayleigh archives. All reference traces pass their archived
32769/65537-node FP64 stability checks. The independent reference is the judge;
third-party solvers are compared implementations. For continuity with the old
strict comparison, the main accuracy table uses `high`:

| Solver / preset | Wrong | Missing | Extra | Exceptions / unresolved | Max ordinal common abs error (km/s) |
|---|---:|---:|---:|---:|---:|
| modisp CPU / high | 0 | 0 | 0 | 0 | 2.621352e-07 |
| modisp CUDA / high | 0 | 0 | 0 | 0 | 2.621352e-07 |
| disba 0.7.0 | 1338 | 1034 | 0 | 12 exceptions | 2.868654e+00 |
| QEDispInv ed1b5dd | 0 | 1 | 0 | 0 | 1.629918e-12 |

`Wrong` means `abs(actual-reference) > max(3e-6, 2e-6*abs(reference))` with
outputs aligned by recovered mode rank. `Missing`/`Extra` count finite-slot
changes, not all setwise root omissions. Maximum common error includes only
finite aligned cells; it does not replace the other counts. disba uses its
default Dunkin/`dc=0.005` configuration.

The approximate presets are **not zero-error under this strict threshold**:
balanced has 6 wrong cells on each backend; fast has 14,220 CPU / 14,211 CUDA.
With their documented 2e-5 / 2e-4 relative budgets, isolated-root failures are
zero. Each has five strictly unmatched close-cluster roots and six mode-index
shifts; five returned cells lie beyond the independent four-mode prefix and
remain independently unjudged. Those values are marked uncertain, rather than
silently removed. The largest rank-aligned deviation is about 0.329 km/s after
a near-root omission shifts later ranks. See [the full accuracy accounting](docs/SAME_HOST_0_3_1.md).

The fixed-shape performance comparison uses the first 64 `seed20260916` models
(`L=10`, `T=64`, `nmodes=4`, `cmin=0.2`). All configurations below were rerun;
values are medians of five steady calls after a separate first call and warmup.
CPU thread counts are explicit; disba's eight-process pool persists across the
measured calls. Accuracy and speed must be read together:

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

The CUDA-only larger run contains **1,000 solve tasks and 64 independent
models**: task `i` uses source model `i % 64`. It has 64,000 model-period tasks
and **4,096 distinct reference traces**, not 1,000 independent models.

| CUDA preset | Tasks | Independent models | Unique reference traces | First call (s) | Warm median (s) | Tasks/s | First/last strict issues |
|---|---:|---:|---:|---:|---:|---:|---|
| high | 1,000 | 64 | 4,096 | 0.685045 | 0.674103 | 1483.452 | 0 / 0 |
| balanced | 1,000 | 64 | 4,096 | 0.195296 | 0.187797 | 5324.900 | 64 / 64 |
| fast | 1,000 | 64 | 4,096 | 0.170717 | 0.164922 | 6063.464 | 156672 / 156672 |

The first and last high outputs have zero wrong/missing/extra/unresolved
results under the strict threshold. Balanced/fast have zero isolated-root
failures within their preset budgets, but their strict issue counts above
retain the lower-refinement-precision differences. Third parties were not run
at B=1,000; their same-size comparison remains B=1/8/64. GPU timing runs reject
competing compute. Cold/JIT/setup costs, all five samples and per-configuration
first/last accuracy checks are in the [same-host report](docs/SAME_HOST_0_3_1.md)
and its [statistical receipt](docs/benchmarks/samehost-0.3.1-20261029.json).
These measurements describe these inputs/settings and do not establish
root completeness or superiority on arbitrary models.

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
