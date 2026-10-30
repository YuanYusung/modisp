# modisp 0.3.3: root refinement, Love phase counting and mode checks

2026-10-31. Base commit: `4efefb3524fa96f72d995ff423362729c0a1a5ee`
(0.3.2). CPU and CUDA retain FP64 evaluation. Private ABI is **7**.

## Algorithms and API

```python
from modisp import Solver, check_mode_reliability

with Solver(backend="auto", refinement="brent", love_strategy="phase",
            accuracy="high") as solver:
    result = solver(thickness, vp, vs, rho, periods, nmodes=4, wave="love")
check = check_mode_reliability(result)  # also result.check_modes()
print(check.summary())
```

Adaptive solvers default to **Brent-Dekker**. `refinement="chandrupatla"`
selects safeguarded three-point inverse interpolation; `"legacy"` retains the
previous secant/periodic-bisection method. All have the same FP64 bracket-width
termination budget for a given bracket and requested tolerance. The selection
travels with each Solver/Medium, including separate CUDA streams; no global
selector can change another Solver. Legacy uniform CUDA requires legacy
refinement. Full SciPy attribution for the Chandrupatla interpolation test is
in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

Love defaults to **unwrapped SH phase counting**. The decaying half-space state
is propagated upward through the finite layers. Propagating layers accumulate
phase; interfaces rescale traction/displacement coordinates; evanescent
layers use bounded hyperbolic factors. Integer-pi phase levels propose roots.
Each root must also have a valid original SH sign bracket and pass original-
function refinement. Opaque-layer cancellation, nonfinite phase states,
invalid brackets or unordered results trigger the existing scan instead.
This guard is important: a floating-point phase winding is not an unconditional
mathematical completeness certificate.

CPU uses the persistent trace pool; CUDA assigns one warp to a trace and
refines separate roots concurrently across lanes, reusing shared model and
frequency constants. The initial serial-per-thread GPU prototype was slower
and is not the released implementation.

`love_strategy="scan"` retains the previous root-discovery path. Phase-accepted
traces use `search_level=11` and a root tolerance `min(c_tol_rel,1e-8)`, before
original-function refinement. Scan fallbacks use the requested preset budget
and retain `TRACE_LOVE_PHASE_FALLBACK=16384`. Stats expose
`love_phase_accepted` and `love_phase_fallbacks`; kernel reports distinguish the
actual phase kernel and its scan fallback. Successful phase searches carry no
approximate-search flag, including under balanced/fast. Physical roots remain
sorted by velocity independently at each period.

Rayleigh root discovery and the high/balanced/fast search policies are retained;
replacing refinement does not repair roots omitted during discovery.

## Independent accuracy and bounded Love validation

For both refiners on CPU and CUDA, the same 11 frozen solid-Rayleigh archives
contain **171 independent models, 6,083 traces and 19,161 finite reference
cells**. All traces retain their original stability masks. Strict wrong-value
tolerance is `max(3e-6,2e-6*abs(reference))` km/s, aligned by recovered rank.

| Preset, each refiner/backend | Wrong | Missing finite slots | Extra finite slots | Unresolved |
|---|---:|---:|---:|---:|
| high | 0 | 0 | 0 | 0 |
| balanced | 6 | 0 | 0 | 0 |

Balanced retains the existing near-root rank shifts, with about 0.329 km/s
maximum rank-aligned deviation. Equal finite masks do not prove equal root
sets. The independent four-mode prefix does not judge roots beyond that
prefix. These measurements do not establish arbitrary-model completeness.

The analytic two-layer Love spectrum is checked at five periods, with multiple
refiners, stricter requested tolerances and cropped velocity windows. Existing
water, Scholte, near-kissing pairs, scaling, conditioning and lifecycle tests
remain in the suite. Frozen bitwise tests explicitly use legacy refinement and
scan discovery; ordinary tests exercise the new defaults.

A separate **2,304-trace** Love regression uses 32 models each at 8, 16 and
32 solid layers, 24 periods, velocity reversals and a top-water batch. The phase
path and retained high scan have zero wrong/missing/extra results at the same
strict velocity threshold and zero unresolved traces. CPU phase accepted
1,926 traces and fell back on 378; CUDA accepted 1,922 and fell back on 382.
Those different fallback decisions reflect floating-point conditioning, not
different returned physical spectra in this set. This is a regression against
retained SH arithmetic/search, not an independent general multilayer Love
reference. The independent analytic Love tests cover a narrower class.

## Same-host timing against installed 0.3.2

Separate persistent processes load released 0.3.2 and candidate 0.3.3. Cold
calls and warmup are separate; the table uses medians of **seven interleaved
steady calls**. First/last outputs are saved and compared. Backend secular
defaults are unchanged: CPU Dunkin, CUDA fast. Separate mode checks are excluded
from solver timings.

CUDA uses **A800 80GB PCIe, physical GPU 1 / sm80 / CUDA 12.6**. A GPU-0 timing
attempt was rejected when another user's compute appeared. The final GPU-1
pmon receipt observed no competing computation; other resident allocations
remain, and this is a shared host rather than an exclusivity guarantee.

| Workload / preset | 0.3.2 seconds | Brent seconds | Chandrupatla seconds | Brent speedup |
|---|---:|---:|---:|---:|
| seed64 / high | 0.052521 | 0.045753 | 0.045821 | 1.148x |
| seed64 / balanced | 0.010620 | 0.009136 | 0.009071 | 1.162x |
| random70 / high | 0.152401 | 0.141389 | 0.139718 | 1.078x |
| Love32 / high | 0.012771 | 0.011715 | 0.011560 | 1.090x |
| seed1000 / high | 0.512688 | 0.429919 | 0.429813 | 1.192x |
| seed1000 / balanced | 0.145510 | 0.117877 | 0.117415 | 1.234x |

seed64: 64 independent ten-layer models, 64 periods, four modes. seed1000:
**1,000 tasks repeating those 64 models via `i % 64`**, 4,096 independent
model-period traces. random70: 32 independent 70-layer models, 100 periods,
five modes. Love32: 32 independent eight-layer reversal models, 24 periods,
eight modes; this includes guarded fallback overhead and a stricter accepted-
phase root budget than the old high scan.

macOS arm64, eight CPU workers, same workload definitions:

| Workload / preset | 0.3.2 seconds | Brent seconds | Chandrupatla seconds |
|---|---:|---:|---:|
| seed64 / high | 0.239835 | 0.232711 | 0.233429 |
| seed64 / balanced | 0.099304 | 0.092961 | 0.092978 |
| random70 / high | 2.146120 | 2.111221 | 2.109279 |
| Love32 / high | 0.029101 | 0.022292 | 0.022505 |

CPU Rayleigh gains are small and workload/load dependent; Love improved about
1.31x in this run. Guarded, tighter-precision production results should not be
replaced by the larger gains from the earlier unguarded research prototype.
Full precision claims must be read with the finite-slot and rank limitations
above. Original search tolerances are still configurable with `c_tol_rel`.

## Independent continuity / mode-reliability diagnostics

`check_mode_reliability` does **not** accelerate by continuation, alter the
solution, track eigenfunctions or change the reported rank convention. It
stably sorts periods internally and returns flags in original order for count
changes, close roots, nearest-rank ambiguity, relative jumps, extrapolation
discrepancies, inconsistent duplicates and numerical/search uncertainty.
The first period and edges with large log-period gaps are explicitly
unassessed. It inherits `close_root_rtol` from a DispersionBatch unless the user
overrides it. Full flag definitions are in [API.md](API.md).

The checker was additionally applied to the prior continuation prototype's
known failing synthetic output: **five of eight wrong finite cells were
flagged; three smooth wrong-branch cells were not**. Thus `passes_checks`
means only that available heuristic observations passed. It cannot certify
physical modal identity, prove complete root discovery or authorize automatic
curve repair. Births/cutoffs and valid rapid dispersion can also be flagged.

CLI: `modisp forward ... --check-continuity`, or independently
`modisp check-modes output.npz --output checks.npz`. Both expose assessed masks,
flags, original predecessor indices and stable sort order without editing
the velocities. Standalone checking has no GPU/runtime dependency.

The numerical receipt with all steady timing samples and validation accounting
is [roots-0.3.3-20261031.json](benchmarks/roots-0.3.3-20261031.json).

## Installed-package release checks

- macOS arm64 CPU: 208 passed, 5 CUDA-only skips.
- Linux x86-64 CPU: 208 passed, 5 CUDA-only skips.
- A800 CUDA / installed sm80 wheel: 213 passed.
- Bounded installed-package compute-sanitizer smoke, covering both refiners,
  P–SV and the Love warp path: 0 errors, 0 leaked bytes. This is not an all-input memory audit.
- Both installed base wheels contain the same final Python sources, and the
  self-contained plugin sdist includes both shared numerical headers.
