# Python API

Speeds use km/s, thicknesses km, densities g/cm³ and periods seconds. The final
layer is an elastic half-space; its thickness is ignored by the array API.
`LayeredModel` uses zero final thickness. See
[Scope and numerical diagnostics](../README.md#scope-and-numerical-diagnostics)
for supported media and mode-numbering conventions.

## Unified solving interfaces

`dispersion`, `dispersion_batch` and `Solver` return `DispersionBatch` for both
single and batch models. There is no separate single-curve return type.

```python
import numpy as np
from modisp import dispersion

model = np.array([[0.3, 2.0, 0.8, 2.0], [0.0, 5.0, 2.8, 2.6]])
periods = np.logspace(-1, 1, 30)
result = dispersion(model, periods, nmodes=3, wave="rayleigh",
                    backend="cpu", threads=4)
velocity = result.phase_velocity[0, 0]
```

`dispersion(model, periods, nmodes=5, wave="rayleigh", **solver_options)` accepts
`LayeredModel`, an `[L,4]` array ordered h/Vp/Vs/rho, or a model dictionary.
`dispersion_batch(thickness, vp, vs, rho, periods, nmodes=5, wave="rayleigh",
solver=None, **solver_options)` accepts the four arrays directly.

Periods must be finite, positive and one-dimensional; batch interfaces preserve
unsorted and repeated values. Modes are zero-based. Undetected modes and
numerical failures stay in the output as NaNs with explicit status rather than
being filtered or raising a curve-specific exception. `release()` closes the
cached default solver.

## Reusable Solver

```python
from modisp import Solver

with Solver(backend="auto", device=0, threads=4) as solver:
    result = solver(thickness, vp, vs, rho, periods, nmodes=3, wave="love")
    print(result.phase_velocity.shape)  # [B, 3, T]
    print(solver.stats(), solver.kernel_report())
```

Each model array has shape `[B,L]`; `[L]` is promoted to one model. All four
shapes must agree. Models may mix dry and wet first layers. Layer and mode
counts are limited to 256. `names` is accepted for batch compatibility and does
not affect physics.

Use one solver per concurrent caller. Instances reject simultaneous calls and
reuse their native thread pool or device allocation across compatible calls.
`close()` releases resources; a later call may allocate them again.
`solver.last_result` refers to the most recent complete `DispersionBatch`.

| Option | Default | Meaning |
|---|---:|---|
| `backend` | `"auto"` | CUDA when a compatible plugin/device is available, otherwise CPU |
| `device` | `0` | CUDA ordinal after `CUDA_VISIBLE_DEVICES` mapping |
| `threads` | up to 8 available CPUs | Persistent CPU worker count, 1–256 |
| `cmin`, `cmax` | `None` | Automatic range; explicit values define a cropped search window |
| `c_tol_rel` | `1e-6` | Relative phase-speed tolerance |
| `f_accept` | `1.0` | Optional normalized-function acceptance bound |
| `search_strategy` | `"qe_adaptive"` | FP64 adaptive search; `"uniform"` is legacy CUDA solid Rayleigh only |
| `workspace_bytes` | `256*1024**2` | Native work-buffer budget, excluding inputs/results and CUDA context |
| `chunk_size` | `None` | Optional model-count cap per chunk |
| `nc`, `block` | `512`, `256` | Legacy uniform settings |

An explicit unavailable CUDA request raises `BackendUnavailableError`.
Automatic selection records its choice and reason. Version/ABI mismatches raise
an installation error. Numerical failures never trigger a backend change.

## Results and diagnostics

`DispersionBatch` contains:

- `phase_velocity`, `mode_mask`, `status`: `[B,nmodes,T]` arrays.
- `period`: a copy of the input periods; `mode`: ascending integers.
- `trace_flags`, `search_level`, `root_count`: `[B,T]` search diagnostics.
- `search_bounds`: `[B,2]`, determined before chunking.
- `backend_used`, `backend_reason`, `wave`, `mode_convention`.

`SolveStatus` values are `VALID=0`, `NOT_FOUND=1`, `UNRESOLVED=2` and
`REJECTED=3`. `NOT_FOUND` means the completed search did not detect that root in
its window. An unresolved trace has all modes NaN. `root_count` is capped at
`nmodes+2`, including two guard roots; an unresolved count is -1.

`search_level=3` denotes adaptive acceptance and 5 dense-fallback acceptance.
-1 denotes an unresolved search, -2 legacy uniform, -3 an analytically empty SH
family, and -4 an empty intersection between the requested and trapped domains.
Flags are cumulative: a later successful fallback can coexist with an earlier
workspace or root-conflict flag.

| Bit | Exported constant | Meaning |
|---:|---|---|
| 1 | `TRACE_UNRESOLVED_BRACKET` | Legacy unresolved bracket |
| 2 | `TRACE_RELOCATED` | Legacy recovered bracket |
| 4 | `TRACE_SEARCH_UNCONVERGED` | Final search failed; all modes NaN |
| 8 | `TRACE_WORKSPACE_LIMIT` | Sampling/workload capacity was reached |
| 16 | `TRACE_NEAR_CUTOFF` | A requested root lies within 1% of half-space Vs |
| 32 | `TRACE_NONFINITE` | Nonfinite or collapsed secular samples |
| 64 | `TRACE_REFINEMENT_FAILED` | Bracket refinement failed |
| 128 | `TRACE_F_REJECTED` | A root failed the optional acceptance threshold |
| 256 | `TRACE_RANGE_TRUNCATED` | User supplied at least one search bound |
| 512 | `TRACE_SAMPLING_UNCERTAIN` | Density re-evaluation failed its spacing constraint |
| 1024 | `TRACE_ROOT_CONFLICT` | Sampling families disagreed; inspect final status |

A budget too small for one model raises `MemoryError` before allocation.
Hardware allocation and driver failures raise explicitly. `stats()` reports
algorithm counters and estimated native work-buffer bytes.

## Model helpers

`to_batch` and `pad_batch` provide zero-thickness solid padding for models with
different layer counts. CUDA uses standard FP64 fused multiply-add; CPU
contraction is disabled. Root tolerances and acceptance checks are shared, but
bitwise CPU/GPU identity is not part of the API contract.
