# modisp 0.3.2: experimental `secular="fast"`

This release adds an opt-in evaluator for the same FP64 Dunkin compound-matrix
dispersion function. `secular="dunkin"` remains the default and is unchanged;
`secular="fast"` is an explicit experimental path. The two are compared on the
same models, the same fixed velocity nodes and the same FP64 arithmetic.

## Why the released function was revisited

For a 70-layer model the released recurrence performs four divisions by
`rho`/`rho^2` per finite layer and five more divisions in its max-normalization
step. The 5x5 compound matrix is also written to `ca[5][5]` and then contracted.
Those are scalar costs around an otherwise compact 5-vector recurrence.

The 0.3.2 fast path keeps the identical boundary conditions, half-space vector,
layer ordering, water branch and signed return value, but changes the scalar
mechanics:

| Step | Released `dunkin` | Experimental `fast` |
|---|---|---|
| Density divisions | 4 per layer, from `rho` | `irho=1/rho`, `irho2` prepared per model |
| Normalization | five divisions by `t1` | one reciprocal + five multiplications |
| Matrix apply | write 25 entries, then 25 products | 15 distinct entries, contract directly |
| Products | separate multiply/add | explicit `fma` in the five contractions, with a multiply/add fallback on x86 builds without `-mfma` |
| Evanescent decay | `exp(-2p)` | `exp2(-2p*log2(e))` |
| Propagating phase | separate `sin`/`cos` | platform `sincos` where available |

No FP32 conversion, no interpolation and no approximate root acceptance is
introduced. The fast path returns the same signed function up to ordinary FP64
rounding; it is not promised to be bitwise identical.

## API

```python
with modisp.Solver(backend="cpu", secular="fast", accuracy="high") as s:
    out = s(thickness, vp, vs, rho, periods, nmodes=5)

out.secular                          # "fast"
s.stats()["secular_kernel"]          # "dunkin_fast"
s.kernel_report()["secular"]         # "fast"
```

`secular="fast"` is accepted by the CPU and CUDA adaptive paths. The legacy
`search_strategy="uniform"` path rejects it. `wave="love"` is unaffected because
Love uses the independent SH 2×2 recurrence; `stats()` and `kernel_report()`
report `secular_kernel="sh"` while preserving the requested `secular` value.
The command line exposes
`modisp forward --secular {dunkin,fast}` and stores the choice in the output NPZ.
`modisp doctor` reports the available values. The diagnostic
`native.secular_grid(method="fast")` requires normalized FP64; unsupported raw
or FP32 requests raise an error.

The private extension ABI is now 6, so the CPU wheel and optional CUDA plugin
must come from the same release.

## Measurement protocol

Stage 1 used one fixed 1,000,000-node velocity grid per model/period, shared by
both evaluators, at FP64. Stage 2 used identical model/period/mode arrays and
interleaved `Solver` calls in one process. Each stage reports the median of three samples. The historical stage-1
receipts ran each evaluator in a block without an explicit warmup; the reviewed
benchmark now warms up both variants and alternates their order. Stage 2 already
used separate warmups and alternating calls. The CPU host was an Apple
M4 with 8 worker threads.

The stage-2 workload was 32 distinct random 70-layer models, 100 periods,
five requested modes, `accuracy="high"`, `threads=8`.

## Results

Fixed-node function evaluation (1,000,000 evaluations per row):

| Model | Period (s) | dunkin (ns/eval) | fast (ns/eval) | speed-up | max scaled difference |
|---|---:|---:|---:|---:|---:|
| canonical_0 | 1 | 100.4 | 78.3 | 1.282 | 8.2e-15 |
| canonical_0 | 10 | 97.2 | 85.5 | 1.136 | 7.9e-15 |
| canonical_3 | 1 | 90.7 | 78.8 | 1.152 | 3.6e-13 |
| canonical_3 | 10 | 97.2 | 85.3 | 1.141 | 1.7e-14 |
| canonical_7 | 1 | 70.9 | 60.5 | 1.171 | 9.1e-15 |
| canonical_7 | 10 | 72.2 | 62.8 | 1.149 | 9.6e-15 |
| water_two_layer | 1 | 33.7 | 28.8 | 1.173 | 3.6e-15 |
| water_two_layer | 10 | 31.1 | 27.2 | 1.143 | 3.4e-15 |
| random_70_solid | 1 | 1624.9 | 1370.8 | 1.185 | 3.6e-13 |
| random_70_solid | 10 | 1678.4 | 1448.1 | 1.159 | 7.8e-14 |
| random_70_water | 1 | 1588.6 | 1407.5 | 1.129 | 3.8e-13 |
| random_70_water | 10 | 1632.8 | 1582.7 | 1.032 | 4.8e-14 |

Across all twelve rows there were zero sign mismatches away from zero, identical
zero-crossing counts, and identical grid intervals containing sign changes.
This does not measure displacement of refined roots. The reported 3.8e-13 is
`max(abs(f_fast-f_dunkin)) / max(1, max(abs(f_dunkin)))`, a globally scaled
absolute difference, not a pointwise relative error near roots. The historical
JSON fields `max_relative_difference` and `max_root_difference` have those
meanings; reviewed receipts use `max_scaled_difference` and
`max_crossing_node_difference`. The 70-layer speed-up
ranged from 1.03× to 1.19× across the rows (median about 1.15×); the lowest row
may reflect host timing noise.

Cross-platform note: a generic x86-64 Linux build does not enable the FMA ISA by
default. On the 32-thread Xeon test host the same fixed-node comparison measured
1.02–1.09× (70-layer models 1.037–1.042×) and the full 32-model solver measured
1.063×. The fast path uses a multiply/add fallback there instead of a libm
`fma` call, avoiding libm call overhead. The measured gain was smaller than on
AArch64 or CUDA; this does not guarantee that every workload is faster. The
result is platform-dependent and is not presented as a universal speed-up.

Full solver (same 32 models, 100 periods, five modes, high preset):

| Implementation | median seconds | models/s | evaluations |
|---|---:|---:|---:|
| `secular="dunkin"` | 2.5430 | 12.58 | 7,812,821 |
| `secular="fast"` | 2.3670 | 13.52 | 7,812,820 |

The solver speed-up was **1.07×** in this run (1.10× in an earlier run of the
same build), smaller than the function-level speed-up
because sampling, sorting, bracketing and refinement are not accelerated. The
comparison had zero missing, zero extra and zero wrong finite cells; the finite
mask and per-trace root counts were equal; the maximum common absolute velocity
difference was 3.6e-15 km/s.

On the same Linux host, an NVIDIA A800/sm_80 run with 1,024 models, 100
periods, five modes and `accuracy="high"` measured 6.1281 s for `dunkin` and
4.6020 s for `fast`, a **1.33×** solver speed-up (167.1 vs 222.5 models/s).
Both runs performed about 255.76 million evaluations, had equal finite masks and
root counts, zero wrong/missing/extra cells, and a maximum common velocity
difference of 1.30e-7 km/s. A 32-model run gave the same 1.34× ratio. A competing
user process held unused GPU memory during these measurements; it was idle at
the start, but the absolute times should be treated as contention-affected.
Interleaving reduces order bias but cannot rule out changing contention; the
measured timing ratio should be rechecked on an idle device. The recorded
root-set comparison is independent of that timing caveat.

The CPU and CUDA solver measurements use the same random 70-layer model
family. They do not establish a universal speed-up or root completeness.

## Measurements after review fixes

The reviewed benchmark uses density constants even on the diagnostic path,
warms up both evaluators and alternates sample order. All rows below passed
finite-mask/root-count equality and zero wrong/missing/extra cells. The CPU
workload has 32 models; CUDA has 1,024. Both use 70 layers, 100 periods, five
modes and the high preset.

| Platform | Dunkin seconds | Fast seconds | Solver speed-up | Max common velocity difference (km/s) |
|---|---:|---:|---:|---:|
| Apple M4, 8 CPU threads | 2.1095 | 1.9155 | 1.101× | 3.55e-15 |
| Shared Xeon Gold 6430, 32 CPU threads | 1.1985 | 1.0712 | 1.119× | 5.75e-11 |
| Shared A800/sm_80 | 6.1539 | 4.7415 | 1.298× | 1.30e-7 |

The Mac fixed-node comparison measured 1.118–1.218×, with a maximum globally
scaled difference of 3.80e-13. Every row was finite, with identical signs away
from zero and identical crossing intervals. On the shared Linux host, the
fixed-node ratios ranged from 0.677× to 1.004×, including slowdowns. Its load
average reached 41.85 during that stage. These timings do not isolate kernel
cost from changing host contention and do not establish a universal gain.
The A800 had about 21 GiB occupied by other allocations before the run;
interleaving does not guarantee uncontended timings.

Raw samples: [Mac](benchmarks/secular-fast-review-cpu-0.3.2-20261029.json),
[Linux CPU](benchmarks/secular-fast-review-linux-x86-0.3.2-20261029.json),
[A800](benchmarks/secular-fast-review-cuda-0.3.2-20261029.json).
See [the implementation review](REVIEW_0_3_2.md) for fixes and checks.

## Relationship to Buchen & Ben-Hador (1996) and Xi et al. (2026)

Buchen & Ben-Hador review the delta-matrix, reduced-delta, Schwab–Knopoff and
Kennett formulations and derive a five-component fast delta recursion. Xi et al.
extend a five-scalar Plücker recursion to Scholte waves by coupling the solid
stack to a scalar normal impedance for the water layer. Both are more compact
than assembling and multiplying a full 5×5 compound matrix.

They are not drop-in replacements for the current vector. The released Dunkin
vector and Xi's vector use different minor normalizations (`rho*xi`, `g*xi`,
`g^2*xi/rho`); a component-by-component transcription does not preserve the
layer transfer. The 0.3.2 fast path therefore uses the algebraically identical
Dunkin recurrence with the scalar overhead removed. The Buchen and Xi kernels
remain research candidates: they must first pass an explicit basis/normalization
audit and the same root-set gates before they can be exposed as a second
evaluator. They are not claimed as validated here.

## Reproduce

```bash
python tools/benchmark_secular.py \
  --mode both --evaluations 1000000 --periods-eval 1 10 \
  --models 32 --periods 100 --threads 8 --accuracy high \
  --output secular-cpu.json
```

The scalar contraction can also be checked independently of layer propagation:

```bash
c++ -std=c++17 -O3 -ffp-contract=off tools/check_secular_algebra.cpp -o /tmp/modisp-algebra
/tmp/modisp-algebra
```

It compares 100,000 arbitrary compound matrices (500,000 output components),
using the sum of absolute matrix-product terms as the rounding-error scale.
This is an algebra check, not an independent physical reference.

The root-set regression is in `tests/test_secular_fast.py`. The released
`secular="dunkin"` path is checked by the existing frozen-fixture suite.

## Default-policy recommendation after review

Keep `secular="dunkin"` as the 0.3.2 default and retain it as a reproducible
reference. The fast contraction matches all 25 matrix coefficients algebraically,
and the observed agreement supports explicit use of `secular="fast"`, especially
for the measured A800 workload. It is an optimized Dunkin evaluator, not an
independent fast-delta formulation or a different root-search algorithm.

Reciprocal multiplication, FMA, summation order, `exp2` and platform `sincos`
change FP64 rounding. Adaptive samples and refinement paths can consequently
differ even when policy and acceptance rules are identical (the recorded CPU
runs differ by one function evaluation). Matching Dunkin on these inputs does
not establish external root completeness or identical behavior near every
multiple/near-cutoff root. A future default change should be backed by broader
paired stress tests and compiler/GPU coverage; the current CUDA binary evidence
is for sm_80. Retain the explicit Dunkin option even after any default change.
