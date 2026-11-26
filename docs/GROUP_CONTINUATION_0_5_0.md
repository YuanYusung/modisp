# Group velocity and frequency continuation (0.5.0)

Both capabilities are opt-in. Defaults remain independent per-period phase
searches without group-velocity work. CPU and CUDA packages must both be
version 0.5.0, with private ABI 9.

```python
import numpy as np
from modisp import Solver

model = np.array([[.3, 2., .8, 2.], [0., 5., 2.8, 2.6]])
periods = np.geomspace(.1, 10., 64)
with Solver(backend="cuda", group_velocity=True,
            frequency_strategy="anchor_continuation", anchor_stride=4) as solver:
    result = solver(*model.T, periods, nmodes=3)
    print(result.phase_velocity.shape, result.group_velocity.shape)
    print(result.group_velocity_valid)
    print(result.frequency_strategy_used, result.frequency_strategy_reason)
```

## Implicit group velocity

For the signed secular equation `F(c, omega)=0`, put `x=ln(omega)` and
`k=omega/c`. Direct implicit differentiation gives

```text
dc/dx = -F_x/F_c
U = d(omega)/dk = c*c*F_c / (c*F_c + F_x)
dc/dx = c*(1 - c/U)
```

This also supplies the local predictor used by continuation. Derivatives are
evaluated at a locally polished root of the original secular function; they
do not require adjacent requested periods. A single-period input can return
a group velocity. `group_rtol=1e-3` controls derivative consistency separately
from phase-root tolerance.

The private FP64 Brent polish targets relative root tolerance `1e-11`.
Its expanding bracket is clipped to neighboring-root midpoints. Central
derivatives in logarithmic coordinates start at step `1e-3` and use at most
six halvings. Success requires agreement between successive Richardson
estimates with propagated error below one quarter of
`group_rtol * max(c, abs(U))`. Smooth frequency-dependent recurrence
normalization is used for the derivative calculation. These checks reject
poorly conditioned roots instead of inferring groups from output-curve
smoothness.

Group calculation uses FP64 even when phase output uses FP32. Reported phase
roots are preserved. Failed group calculations leave NaNs with explicit
flags without changing phase status. `group_velocity`,
`group_velocity_flags` and `group_velocity_valid` have shape `[B,K,T]`;
`group_velocity_mask` aliases `group_velocity_valid`. All four fields are
`None` when group output is disabled.

`GroupVelocityFlag` bits are `MISSING_ROOT=1`, `POLISH_FAILED=2`,
`DERIVATIVE_UNSTABLE=4`, `ILL_CONDITIONED=8`, `CLOSE_ROOT=16`, and
`NONFINITE=32`. A group value is numerically valid only when finite with
zero flags. This does not establish physical mode identity. Phase-search
uncertainty arrays remain relevant to its recovered modal rank.

The [CPS documentation](https://www.eas.slu.edu/eqc/ComputerProgramsSeismology/cps_programs.html)
describes the underlying phase/group dispersion context and eigenfunction
methods. [disba](https://github.com/keurfonluu/disba) supplies ordinary-model
group-dispersion comparisons as a test dependency. The formula above is a
direct derivation; the derivative estimator does not reproduce the CPS
energy-integral method.

## Anchor continuation

`frequency_strategy="anchor_continuation"` requests CPU or CUDA auto, balanced or fast
adaptive solving on at least eight distinct periods. The wrapper sorts and
deduplicates periods for this path, then restores every output array to the
original order, including repeated periods. Adaptive independent solves also
reuse exactly repeated input periods. CUDA FP32 high accuracy preserves the
full input grid for its batch-dependent repair budget.

Anchors use the configured global search. Intermediate periods use nearby
accepted roots to propose local brackets, validate them with the original
signed function, and fall back to global search when the attempt is unsafe.
`anchor_stride=4` controls anchor spacing and must be an integer of at least
two. Internal prediction does not require `group_velocity=True`.

The two bounding anchors must report the same positive root count
`NR <= nmodes + 2`, with isolated roots and usable slopes for all `NR` roots.
The search can retain two additional guard roots beyond the requested modes.
Local solving corrects the available `NR` roots; any remaining requested
modes stay NaN. If `NR < nmodes + 2`, the coarse audit also covers the entire
tail up to the original upper search bound to look for omitted or newly born
roots. For a full `nmodes + 2` prefix, the audit covers the guarded prefix.
Zero or unequal anchor counts, cutoff diagnostics, observed close roots and
unsafe windows cause global fallback. Inspect acceptance and fallback
counters when assessing a workload.

CPU and CUDA share the same Hermite interpolation in log period. The slope
at each endpoint is `dc/dln(T) = c*(c/U - 1)`. Anchor counts, spacing and bounds
are checked before computing slopes. Root recovery brackets and refines the
original signed equation within each window. The Love audit counts integer-π
levels of the unwrapped phase. The Rayleigh audit probes the signed function on
both sides of each recovered root, including inside a sign-changing window.
See the [targeted FP64 review rules](API.md#phase-continuity-review).

Candidate geometry is checked before secular evaluation and eligible targets
are packed into a compact queue. Their frequency-dependent recurrence
coefficients are prepared once for local correction and audit; derivative
stencils compute their own frequency-dependent coefficients. Valid anchor
group values are reused for requested group output.

Local continuation acceptance is approximate. A coarse domain audit can
detect some omissions but cannot certify completeness. Accepted local
traces carry `TRACE_CONTINUATION_APPROXIMATE=65536`, participating in
`approximate_search` and `mode_index_uncertain`. Outputs retain ascending
recovered-root ranks; continuation does not assign eigenfunction-based
branch identities. Use independent `accuracy="high"` for close-root ordering
comparisons.

`prediction_flags[B,T]` uses `PredictionFlag`: `ANCHOR=1`, `ACCEPTED=2`,
`FALLBACK=4`, `BYPASSED=8`, `AUDIT=16`, `AUDIT_FAILED=32`,
`SOURCE_SEARCH_UNCERTAIN=64`, `ROOT_COUNT_UNSAFE=128`, `ANCHOR_UNSAFE=256`,
`WINDOW_UNSAFE=512`, `BRACKET_FAILED=1024`, `CORRECTION_FAILED=2048` and `REVIEW=4096`. The review bit records spectral evidence sent for independent FP64 verification.
`AUDIT` means a
coarse domain check, not a full independent search. The array is `None` for
the independent strategy. High accuracy and grids with fewer than eight
distinct periods use independent solves and return `BYPASSED` when
continuation was requested. Results, `stats()` and `kernel_report()` record
the requested strategy, strategy used and reason. Legacy `uniform` rejects
both features.

A fallback records one mutually exclusive reason: `ROOT_COUNT_UNSAFE` for
zero, unequal or invalid source counts; `ANCHOR_UNSAFE` for unavailable
source slopes; `WINDOW_UNSAFE` for unsafe prediction order, bounds or local
window size; `BRACKET_FAILED` when no validated sign bracket is found;
`CORRECTION_FAILED` for refinement, ordering, acceptance or close-root failure;
or `AUDIT_FAILED` when the coarse domain check rejects the corrected prefix.
The `FALLBACK`, `AUDIT` and source-uncertainty bits can accompany that reason.

`stats()` reports `continuation_fallback_root_count_unsafe`,
`continuation_fallback_anchor_unsafe`, `continuation_fallback_window_unsafe`,
`continuation_fallback_bracket_failed`,
`continuation_fallback_correction_failed`, and
`continuation_fallback_audit_failed`. Counts are collected per native chunk
before restoring duplicate periods, so each computed model/unique-period
trace contributes once. `continuation_local_success_rate` is
`accepted / (accepted + fallbacks)`, including geometry rejections but
excluding anchors and bypasses; it is `None` when there are no such targets.

The existing trace bit `TRACE_CLOSE_ROOT_RISK=4096` can also result from a
conservative sampling-floor hint without an observed close pair. Continuation
checks the actual spacing of all source roots, including guards, rather than
rejecting every hinted anchor. A dependent target retains
`SOURCE_SEARCH_UNCERTAIN` if either anchor had this hint, including when the
target falls back. Source trace flags are preserved. All local acceptances
remain approximate; this does not certify the absence of an unobserved pair.

A grid with no three consecutive periods within a log-period span of `0.25`
also uses independent solving: otherwise every period would be an anchor,
leaving no interior work to accelerate.

## CLI and validation

```sh
modisp forward input.npz --output output.npz --backend cuda \
  --group-velocity --group-rtol 0.001 \
  --frequency-strategy anchor_continuation --anchor-stride 4
```

Optional NPZ arrays are omitted when absent and remain readable with
`allow_pickle=False`. Phase arrays and diagnostics retain their names.

Validate group output against nondispersive Rayleigh `U=c`, independently
differentiated two-layer Love and ordinary-model disba comparisons. Cover
missing/cutoff/close roots, water, scaled thicknesses and periods, FP32 phase
output, unsorted/repeated periods, chunking and workspace reuse. Compare
continuation with independent search on smooth grids and adverse cases,
retaining missing/extra/ordinal differences separately. Benchmark phase-only,
phase+group, continuation-only and combined workloads in unprofiled,
interleaved runs. Evaluation-count savings alone do not establish wall-time
speedup.

Comparing FP32 search strategies requires checking the full ascending root
prefix, including finite masks, counts and values. Repeated roots can shift
later ranks while preserving the number of returned modes.

With continuation, `active_traces_by_stage` describes the final fallback
batch. `continuation_anchor_search` and `continuation_fallback_search` are
outer timing intervals that overlap their detailed search/refinement timings;
do not sum outer and inner intervals. `continuation_prepare` measures
prediction and candidate compaction, `continuation_constants` prepares the
fixed-frequency coefficients, and `continuation_local_search` includes
correction and the coarse audit. `continuation_candidates` counts targets
that pass geometry checks; `group_reused` counts cached anchor group values.
