# modisp 0.3.3

modisp is an FP64, GPU-accelerated package for batched multimode Rayleigh/P–SV
and Love phase-velocity calculations in 1-D layered elastic media. A portable
CPU backend and an optional Linux CUDA 12 plugin share one Python interface.
The package is designed for large-scale forward modeling in model-space
exploration and synthetic-data workflows. Units are km, km/s, g/cm³ and seconds.

## Install

Install the wheel matching your Python and platform, or build the CPU source:

```sh
python -m pip install /path/to/modisp-0.3.3-<matching-tags>.whl
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

CUDA defaults to the optimized `secular="fast"` evaluator, which keeps the same FP64 Dunkin algebra
but removes per-layer density divisions, uses one reciprocal normalization and
contracts the 5-vector without materializing `5x5` entries. CPU defaults to
`secular="dunkin"`; explicit Dunkin arithmetic passes the retained bitwise
frozen-fixture checks. The gains depend on workload and hardware. See
[0.3.2 methods and measurements](docs/SECULAR_FAST_0_3_2.md); the same-host
accuracy and performance tables below were rerun with the 0.3.2 backend defaults.

## Changes in 0.3.3

Adaptive CPU/CUDA solvers default to `refinement="brent"`; choose
`"chandrupatla"` or `"legacy"` explicitly. Love defaults to
`love_strategy="phase"`: unwrapped SH phase proposes root intervals, the
original SH function validates/refines them, and ambiguous phase states fall
back to the configured scan. Phase roots use a tolerance no larger than
`1e-8`; the P–SV search presets retain their existing tolerances.

Cross-period diagnostics run separately, without changing recovered roots:

```python
from modisp import check_mode_reliability
check = check_mode_reliability(result)  # or result.check_modes()
print(check.summary())
# check.flags, check.assessed, check.passes_checks follow original input order.
```

The checker flags count changes, close roots, nearest-rank ambiguity, jumps,
prediction discrepancies, duplicate inconsistency and large period gaps.
Passing these heuristic checks does not certify physical branch identity or
rule out a smooth missing/wrong branch. See [0.3.3 methods and validation](docs/ROOTS_0_3_3.md).
The same-host tables below refresh modisp to **0.3.3**; external
baselines retain their explicitly labeled 2026-10-29 measurements.

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

`secular=None` selects `fast` on CUDA and `dunkin` on CPU after backend
selection; legacy `uniform` defaults to `dunkin`. Either evaluator can be
selected explicitly. The fast evaluator changes FP64 rounding but not the model, sampling policy or root
acceptance rules; rounding can change the actual adaptive sampling path. `accuracy="balanced"` is the default. Balanced/fast permit flagged ambiguity
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

## Same-host comparison (0.3.3; retained external baselines)

**Modisp rows were rerun on 2026-10-31; disba/QED rows are retained from
2026-10-29, as requested.** All use the same dual-socket Xeon Gold 6430 host
and the same frozen inputs. CUDA uses A800 80GB PCIe, physical GPU 1.
Modisp 0.3.3 is commit `1cb70cd`, with `search_strategy="shape_adaptive"`,
**`refinement="brent"`**, CPU `secular="dunkin"` and CUDA `secular="fast"`.
The high/balanced/fast labels select accuracy, independently of the evaluator.

**CPU load caveat:** the new CPU runs started at one-minute load averages
**184.2–194.4 on 128 logical CPUs**, with visible timing variation. The retained
external rows were measured at a different time and load. These are same-host
records with explicit dates, **not a simultaneous equal-load speed ranking**.
Do not infer a version speedup from the old and new tables.

Accuracy uses **171 independent models / 6,083 period traces / 19,161 finite
reference cells** in 11 solid-Rayleigh archives. Input hashes match the retained
comparison, and original 32769/65537-node FP64 stability masks remain intact.
Water, Love and Scholte are outside this particular external comparison.

| Solver / accuracy | Test date | Wrong | Missing | Extra | Exceptions / unresolved | Max ordinal common abs error (km/s) |
|---|---|---:|---:|---:|---:|---:|
| modisp 0.3.3 CPU / high | 2026-10-31 | 0 | 0 | 0 | 0 | 1.976388e-07 |
| modisp 0.3.3 CUDA / high | 2026-10-31 | 0 | 0 | 0 | 0 | 1.976388e-07 |
| disba 0.7.0 (retained) | 2026-10-29 | 1338 | 1034 | 0 | 12 exceptions | 2.868654e+00 |
| QEDispInv ed1b5dd (retained) | 2026-10-29 | 0 | 1 | 0 | 0 | 1.629918e-12 |

`Wrong` means `abs(actual-reference) > max(3e-6,2e-6*abs(reference))` km/s,
with ordinal root alignment. Missing/extra count finite slots, not all setwise
omissions. Maximum common error excludes nonfinite cells. disba uses default
Dunkin / `dc=0.005`; its exceptions and missing results are retained.

Balanced has **6 CPU / 6 CUDA strict wrong cells**;
fast has **8,312 CPU / 8,312 CUDA**. Their isolated-root
failures within the documented 2e-5 / 2e-4 relative budgets remain separately
accounted for. Close-cluster omissions, rank shifts and independently unjudged
prefix tails are retained in [the full 0.3.3 accounting](docs/SAME_HOST_0_3_3.md).
Neither a budget pass nor matching finite masks proves root completeness.

Performance uses the first 64 `seed20260916` models (`L=10`, `T=64`, four
modes, `cmin=0.2`). Modisp entries are medians of **five fresh steady calls**
after a separate first call and warmup. External entries preserve their original
five-call medians, including the persistent eight-process disba pool.

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

The CUDA-only larger run has **1,000 tasks repeating 64 independent models**
via `task i -> source i % 64`: 64,000 model-period tasks but only **4,096
distinct reference traces**. All rows below are new modisp 0.3.3 measurements.

| CUDA accuracy | Tasks | Independent models | Unique reference traces | First call (s) | Warm median (s) | Tasks/s | First/last strict issues | Outside independent prefix (each output) |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| high | 1,000 | 64 | 4,096 | 0.476671 | 0.469490 | 2129.969 | 0 / 0 | 0 |
| balanced | 1,000 | 64 | 4,096 | 0.127757 | 0.120532 | 8296.564 | 64 / 64 | 48 |
| fast | 1,000 | 64 | 4,096 | 0.116973 | 0.108554 | 9211.995 | 94839 / 94839 | 78 |

High first/last outputs have zero wrong/missing/extra/unresolved results.
Approximate presets retain strict issues and unjudged prefix tails; read these
with their budgeted matching metrics. Third parties were not run at B=1,000.
The monitored CUDA runs recorded no competing compute on the selected device;
other resident allocations and heavy host load remained. This does not certify
an isolated machine. All samples, cold/setup costs, provenance and input hashes
are in [the report](docs/SAME_HOST_0_3_3.md) and
[the receipt](docs/benchmarks/samehost-0.3.3-20261031.json).

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

## Development checks

Run `python -m pytest -q` for the CPU suite, or
`MODISP_TEST_BACKEND=cuda python -m pytest -q` for CUDA. Tests are grouped into
API/lifecycle, physics/conditioning, solver/regression and tool checks.
Use `python tools/validate.py {paper,external,stress} --help` for extended
validation and `tools/benchmark_secular.py` for evaluator timing.
See [development workflows](docs/DEVELOPMENT.md) for the retained tools and the
consolidation of version-specific scripts.
