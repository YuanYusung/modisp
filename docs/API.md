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
| `c_tol_rel` | preset | FP64 refinement tolerance: high `1e-6`, balanced `1e-5`, fast `1e-4`; an explicit value overrides it |
| `f_accept` | `1.0` | Optional normalized-function acceptance bound |
| `search_strategy` | `"shape_adaptive"` | Local shape consistency; `"qe_adaptive"` retains 0.3.0; `"uniform"` is legacy CUDA solid Rayleigh only |
| `sampling_rtol` | preset | Shape-integral consistency tolerance: high `1e-3`, balanced `1e-2`, fast `5e-2`; explicit values apply only to shape search |
| `accuracy` | `"balanced"` | Search/accuracy preset: `"high"`, `"balanced"` or `"fast"` (see below) |
| `secular` | `None` | Actual backend default: CUDA `"fast"`, CPU `"dunkin"`; legacy uniform `"dunkin"`. Explicit values override this policy. |
| `refinement` | `"brent"` (adaptive) | `"brent"`, `"chandrupatla"`, `"legacy"`; uniform defaults to and requires legacy |
| `love_strategy` | `"phase"` | SH phase counting with validated original-function brackets and scan fallback; `"scan"` retains the original search |
| `close_root_rtol` | `0.005` | Relative close-root scale; permits flagged balanced/fast close-cluster ambiguity |
| `adaptive_period_threshold` | preset | Compatibility metadata only; shape decisions do not depend on this threshold |
| `workspace_bytes` | `256*1024**2` | Native work-buffer budget, excluding inputs/results and CUDA context |
| `chunk_size` | `None` | Optional model-count cap per chunk |
| `profile` | `False` | Collect stage timings and cache/queue analysis in `stats()` |
| `nc`, `block` | `512`, `256` | Legacy uniform settings |

An explicit unavailable CUDA request raises `BackendUnavailableError`.
Automatic selection records its choice and reason. Version/ABI mismatches raise
an installation error. Numerical failures never trigger a backend change.

### Accuracy presets

| Preset | Shape consistency `sampling_rtol` | Default refinement `c_tol_rel` |
|---|---:|---:|
| `high` | `1e-3` | `1e-6` |
| `balanced` | `1e-2` | `1e-5` |
| `fast` | `5e-2` | `1e-4` |

All arithmetic remains FP64. Density and model-anchored critical/interface
seeds initialize intervals; five-point integral comparisons drive local
subdivision. High verifies density coordinates directly and does not stop or
merge based on the observation threshold. Balanced/fast use a coarser density
schedule and may stop an inconsistent interval at `0.5*close_root_rtol` /
`close_root_rtol`, retaining uncertainty markers. This is a local stopping
scale, not a promised maximum grid step in already smooth intervals. Shape
sampling does not change the original signed secular function used for root
brackets, refinement and `f_accept`. Searches collect two extra guard roots
and return the requested ascending recovered prefix.

Balanced/fast deliberately allow unresolved close pairs or clusters within the
configured scale. Their returned integer ranks may shift after an omitted pair;
they are **not certified physical mode indices**. Every balanced/fast trace
has `TRACE_APPROXIMATE_SEARCH`, `approximate_search=True`, and
`mode_index_uncertain=True`, even when no close pair was detected. This is a
conservative warning about reduced verification, not a claim that every value
is erroneous. `close_root_mask` and `close_root_risk` give additional local
hints; a false hint cannot exclude an unseen close pair.

Use `high` when exact individual-root ordering matters. `close_root_rtol=.005`
is an algorithmic observation-scale choice, not a measured resolution estimate
for a particular survey. `adaptive_period_threshold` remains accepted for
compatibility but does not affect shape search. `uniform` is unchanged.

See [0.3.1 validation](SHAPE_ADAPTIVE_0_3_1.md) for ordinary-root,
close-cluster, ordinal-shift and explicitly unjudged counts.

## Results and diagnostics

`DispersionBatch` contains:

- `phase_velocity`, `mode_mask`, `status`: `[B,nmodes,T]` arrays.
- `period`: a copy of the input periods; `mode`: ascending integers.
- `trace_flags`, `numerical_flags`, `search_level`, `root_count`: `[B,T]`
  search diagnostics. `numerical_flags` excludes only
  `TRACE_RANGE_TRUNCATED`, separating user cropping from numerical failures.
- `search_bounds`: `[B,2]`, determined before chunking.
- `accuracy`: the preset used (`"high"`, `"balanced"`, `"fast"` or `"uniform"`).
- `secular`: the resolved `"dunkin"` or `"fast"` evaluator; `Solver.secular` retains the requested setting (`None` means backend default).
- `approximate_search`, `mode_index_uncertain`, `close_root_risk`,
  `potentially_ambiguous`: `[B,T]` quality arrays.
- `close_root_mask`: `[B,nmodes,T]`, observed adjacent recovered roots within
  `close_root_rtol`. It cannot locate pairs missed by both samples.
- `close_root_rtol`: the configured relative observation scale.
- `search_strategy`, `sampling_rtol`: search configuration (sampling tolerance is `None` for legacy strategies).
- `secular` is also reported in `stats()` and `kernel_report()`; the legacy
  `uniform` path rejects `secular="fast"`. Love waves use the SH recurrence and
  are unaffected by this selector; their actual `secular_kernel` is `"sh"`.
- `backend_used`, `backend_reason`, `wave`, `mode_convention`.

`SolveStatus` values are `VALID=0`, `NOT_FOUND=1`, `UNRESOLVED=2` and
`REJECTED=3`. `NOT_FOUND` means the completed search did not detect that root in
its window. On an approximate trace this is not evidence of physical absence.
`VALID` means the active search accepted a finite value; it does not certify a
mode index on an approximate trace. An unresolved trace has all modes NaN. `root_count` is capped at
`nmodes+2`, including two guard roots; an unresolved count is -1.

`search_level=8/9/10` denotes high/balanced/fast shape acceptance.
`3/5` denotes retained high adaptive/dense acceptance; `6` is the balanced
observation grid and `7` the fast grid. Values 1/2/4 belong to historical 0.3.0 results. -1 denotes an unresolved
search, -2 legacy uniform, -3 an analytically empty SH family, and -4 an empty
intersection between the requested and trapped domains.
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
| 2048 | `TRACE_APPROXIMATE_SEARCH` | Observation-grid search; close clusters and ordinal ranks may be unresolved |
| 4096 | `TRACE_CLOSE_ROOT_RISK` | Observed close roots or a sampled hidden-pair trough hint |
| 8192 | `TRACE_ADAPTIVE_FALLBACK` | Local shape search required the retained 0.3.0 search |

A budget too small for one model raises `MemoryError` before allocation.
Hardware allocation and driver failures raise explicitly. `stats()` reports
algorithm counters and estimated native work-buffer bytes.

`workspace_bytes` controls allocation/chunking, not the observation threshold.
The bounded CPU interval stack and CUDA compact activity queues are included
in that budget, as are shared model seeds and the retained fallback buffers.
CUDA conservatively reserves 5,632 queue bytes for every trace in the allocated
model capacity (`shape_queue_reserved_bytes`), although only resident blocks
use shared queues simultaneously. `device_workspace_bytes` separately reports
actual global device allocations; `kernel_report()` reports shared resources.
Depth, queue capacity, representability and workload failures trigger a local
review or the corresponding retained 0.3.0 path. A successful fallback keeps
`TRACE_ADAPTIVE_FALLBACK` and any earlier diagnostic bits. An unresolved final
search keeps NaNs and its failure status. No fallback certifies completeness.

`stats()` reports actual/requested strategy, `sampling_rtol`, refinement and
close-root tolerances, `shape_accepted`, `shape_subdivisions`, `shape_reused`,
`shape_depth_limit`, `shape_work_limit` (including CUDA queue capacity),
`legacy_fallbacks`, evaluations,
workspace and timings. `verification_skipped` includes completed approximate
shape traces. `velocity_step_rtol` is `None` for shape search; its local close
leaf stopping scale is `close_cluster_leaf_rtol`. Legacy counters remain.

Version 0.3.0 retains an existing compatible allocation for smaller tail calls;
`capacity_models` reports this high-water capacity and `engine_allocations`
reports new allocations during the latest call. A reduced budget/chunk cap or
changed geometry still replaces the workspace. Input values are refreshed on
every call, including arrays modified in place. With profiling enabled,
`timings_s` contains stage durations; summed CPU worker times overlap wall time.
See [performance measurement](PERFORMANCE.md) for interpretation and the
streaming ModePhys benchmark tools.

## Model helpers

`to_batch` and `pad_batch` provide zero-thickness solid padding for models with
different layer counts. CUDA uses standard FP64 fused multiply-add; CPU
contraction is disabled. Root tolerances and acceptance checks are shared, but
bitwise CPU/GPU identity is not part of the API contract.


## Independent cross-period reliability checks (0.3.3)

`check_mode_reliability(result)` or `result.check_modes()` returns a
`ModeReliability`. Alternatively pass `velocity[B,K,T], periods`, optionally
`trace_flags[B,T]` and `root_count[B,T]`. It never changes velocities, solve
status or modal ranks and does not run a new solver.

Options: `close_root_rtol=.005`, `max_relative_step=.1`,
`prediction_rtol=.05`, `max_log_period_gap=.25`, all finite positive.
Periods are sorted stably internally; arrays are returned in original order.
`previous_index[T]` maps each incoming edge to its predecessor's original
index, with -1 for the first period. Duplicates are compared with their first
representative; large log-period gaps and the first sample are unassessed.

`flags[B,K,T]` uses `ModeReliabilityFlag`: UNASSESSED=1,
NUMERICAL_FAILURE=2, ROOT_COUNT_CHANGE=4, CLOSE_ROOT=8,
RANK_AMBIGUOUS=16, STEP_JUMP=32, PREDICTION_MISMATCH=64,
DUPLICATE_MISMATCH=128, NONFINITE=256, PERIOD_GAP=512,
SEARCH_UNCERTAIN=1024. `assessed` means a finite, sufficiently sampled incoming
edge was available; `passes_checks` additionally requires zero flags. First
samples, missing modes and large gaps cannot pass by absence of evidence.

Nearest-rank consistency is a heuristic, not eigenfunction matching. Physical
cutoffs/births and rapid valid dispersion may legitimately be flagged. A smooth
wrong branch or persistent omission can pass. No field certifies physical mode
identity, and the separate check is excluded from default solver timings.

CLI: `modisp forward ... --refinement chandrupatla --love-strategy phase
--check-continuity` stores continuity arrays in the solve NPZ. Or run
`modisp check-modes solved.npz --output checked.npz` without solving again.
The check command returns zero on successful execution; inspect its flags and
summary to interpret the numerical observations.

Successful Love phase traces have `search_level=11`, without approximate-search
flags even under balanced/fast. A lost/ambiguous phase state falls back to the
configured scan and retains `TRACE_LOVE_PHASE_FALLBACK=16384`. Inspect
`love_phase_accepted` and `love_phase_fallbacks` in stats. The default phase root
tolerance is `min(c_tol_rel,1e-8)` before original-function refinement. General
Rayleigh root discovery, preset budgets and close-cluster limitations remain
as documented above. CPU and CUDA package versions and private ABI 7 must match.
