# modisp 0.3.1: local shape-adaptive search

> Historical report. Version-specific test/benchmark scripts were consolidated
> during 0.3.2 development. Use [current workflows](DEVELOPMENT.md); exact historical
> scripts remain in Git history at `e6bd811`.


This release independently adapts the initial density sampling and five-point
quadratic integral-consistency tests of Zhu, Xu, Hao and Yao (2026), BSSA 116,
1383–1394, [doi:10.1785/0120260071](https://doi.org/10.1785/0120260071).
Their numerical validation concerns GRT. The Dunkin/SH transfer, safeguards and
measurements below are modisp results, not conclusions established by that paper.

## Implementation and API

`search_strategy="shape_adaptive"` is the default for all three presets.
`"qe_adaptive"` retains the released 0.3.0 execution paths; `"uniform"` remains
the legacy CUDA solid-Rayleigh option. Matching CPU/CUDA distributions use
version 0.3.1 and private ABI 5.

Model-anchored velocity probes, body-wave critical velocities and top-interface
Rayleigh/Scholte seeds augment the complete requested bounds. Density targets
initialize intervals without allocating or sorting a full per-period grid.
High verifies its density-coordinate inversions and uses a finer density
schedule than the paper. Its seed halos and critical probes address observed
weak/closely spaced features; they are not a completeness proof.

Three integral comparisons from equations 13–14 decide local subdivision.
Parent values are reused at child endpoints/midpoints. The shape feature is the
signed secular component divided by a positive smooth L2 scale. Unlike the
paper's magnitude-GRT function, it preserves sign. Balanced/fast supplement the
test with centered variation near saturation or a sampled approach to zero.
This catches weak pair features that the uncentered relative integral can hide.

The original signed FP64 function is unchanged and still supplies brackets,
root refinement and `f_accept`. Quadratic/golden-section extrema only propose
additional locations; no fixed amplitude threshold accepts a root. High does
not merge roots using the close-root scale. Balanced/fast permit flagged local
stopping at half/one times that scale. Sampling, refinement and ambiguity
tolerances have separate meanings:

| Preset | `sampling_rtol` | `c_tol_rel` | `close_root_rtol` |
|---|---:|---:|---:|
| high | 1e-3 | 1e-6 | 0.005, diagnostic only |
| balanced | 1e-2 | 1e-5 | 0.005 |
| fast | 5e-2 | 1e-4 | 0.005 |

CPU workers use a bounded interval stack. CUDA uses persistent blocks, compact
32-node activity queues, shared model constants and reused samples. The queue,
stack, model seeds and retained fallback buffers count toward the native
workspace budget. Queue/work/depth limits and representability/nonfinite
failures trigger the corresponding retained path. Earlier diagnostics remain
cumulative; `TRACE_ADAPTIVE_FALLBACK=8192` identifies the fallback. Final failure
remains NaN with explicit status. `stats()` reports subdivisions, inherited
sample reuse, limits and fallback counts. CLI outputs save the strategy and
sampling tolerance. See [API](API.md) for details.

## Independent accuracy evidence

The paper's Table 1 solid model is reproduced in km, km/s and g/cm³:

| Thickness | Vp | Vs | Density |
|---:|---:|---:|---:|
| .01 | 1.55 | .18 | 1.78 |
| .01 | 1.80 | .35 | 1.85 |
| .02 | 1.00 | .12 | 1.50 |
| half-space | 2.00 | .60 | 1.93 |

Both Rayleigh and Love cover 0.1–100 Hz in 0.1 Hz steps, including kissing
regions. References combine external `pygrt-kit==0.18.0` at two search controls,
65,537-node and shifted 98,305-node raw grids, and local raw-function brackets.
Love uses an independent unnormalized displacement/traction 2×2 recurrence;
Rayleigh uses the separate unnormalized CPS probe. A one-to-one source union
preserves multiple roots from any one source rather than merging by proximity.
All 1,000 reference frequencies per wave pass the recorded cross-checks.
The resulting reference contains 24,952 Rayleigh and 24,471 Love roots.

GRT omits two raw-SH-confirmed roots close to the half-space cutoff (31.8 and
41.6 Hz); one Rayleigh trace differs between the two dense grids. Those cases
are resolved using the other independent witnesses and local brackets. Neither
GRT, a single dense grid, nor the new high path is treated as sole truth.

Strict unmatched reference roots, with zero spurious roots in every table cell:

| Wave | 0.3.0 high / balanced / fast | 0.3.1 CPU high / balanced / fast | 0.3.1 CUDA high / balanced / fast |
|---|---:|---:|---:|
| Rayleigh | 138 / 440 / 1164 | 0 / 34 / 64 | 0 / 34 / 64 |
| Love | 63 / 386 / 1334 | 0 / 46 / 72 | 0 / 314 / 72 |

All balanced/fast misses in this table belong to flagged close clusters;
isolated-root errors are zero within the 2e-5 / 2e-4 budgets. CUDA balanced
falls back on 384 densely clustered Love traces because of queue capacity.
This explains its larger close-cluster count and the nonmonotonic tolerance
sensitivity; a stricter fit tolerance need not improve a capacity-limited
result. CPU/GPU close-cluster sets are therefore not promised identical.
The one high Rayleigh fallback still matches the complete independent reference.

The 11 retained external archives contain 6,083 stable solid-Rayleigh traces.
High has no missing/spurious independent-prefix roots on either backend.
Balanced/fast strict close-cluster unmatched counts decrease from 33/71 to 5/5,
with zero isolated-root failures. Optional guard roots come only from retained
high and never replace an independent reference cell; unknown prefix boundaries
and values beyond the independent prefix remain explicitly unjudged.

The random stress set has 640 distinct six-layer models, 24 periods and both
waves (30,720 traces). The 117 legacy-unresolved traces remain unjudged. Thirteen stable
high prefixes change; two external GRT controls independently confirm all new
first-five-root prefixes. Two far-tail GRT control disagreements are retained
in the raw reports and do not certify those tails. The 0.54% Rayleigh and 0.575%
Love isolated-pair counterexamples are regression tests, not waived as close
clusters. Full ordinary, strict-cluster, ordinal-shift and unjudged counts are
kept in the accompanying statistical receipt.

Tolerance sensitivity sweeps use half/default/twice each preset's
`sampling_rtol` against the same frozen full paper reference. Chunking, model
mutation, sample reuse, work exhaustion, CLI persistence, blocks of 32/64/128/256
threads and layer counts through 256 are covered separately. Retained high
fixtures remain bitwise identical; original-function values remain bitwise
identical with shape sampling enabled. CPU/GPU FP64 FMA differences do not
imply a bitwise cross-backend contract.

## Performance and release checks

The machine-readable [release receipt](benchmarks/shape-presets-20261029.json)
records exact inputs/binary hashes, cold calls, five interleaved steady samples,
medians, error accounting, gates and executed checks. The baseline is the real
released 0.3.0 wheel, not a reimplemented approximation. Each forward call runs
the native solver; returned curves are not replayed from a result cache.

Both workloads use 70 layers, 100 periods and five requested modes, with 128
distinct models / 32 CPU threads or 1,024 distinct models / one A800. Initial
calls and warmup are separate. Each steady scientific output must reproduce
its own worker's first output bitwise; retained high must also reproduce the
real 0.3.0 high output bitwise. Timings exclude saving/statistical analysis.
GPU monitoring rejects competing compute processes during the timed section.

| Workload | Preset | 0.3.0 median seconds | 0.3.1 median seconds | Speedup |
|---|---|---:|---:|---:|
| CPU, 32 threads / 128 models | high | 8.323351 | 2.925676 | 2.84× |
| same | balanced | 2.695121 | 1.168737 | 2.31× |
| same | fast | 1.547245 | 1.084714 | 1.43× |
| A800 / 1,024 models | high | 7.513213 | 4.496071 | 1.67× |
| same | balanced | 2.069366 | 1.269399 | 1.63× |
| same | fast | 1.526930 | 1.114626 | 1.37× |

All four balanced/fast comparisons pass 1.2× and high improves on both
workloads. On these standard workloads, each new preset has zero ordinary
errors, strict close-cluster unmatched roots and ordinal differences against
the frozen 0.3.0 high regression witness. That witness is not an independent
physics reference; its role is distinct from the accuracy suites above.

Each platform passes its installed-wheel tests (176 CPU tests, with 5 CUDA-only
skips; 181 CUDA tests). All four CUDA sanitizer tools pass 25 targeted tests
with zero errors and warnings. Independently built CPU/plugin sdists reproduce
the tested native binaries byte for byte. GitHub CI checks the Ubuntu/macOS
CPython 3.10–3.14 matrix before publication.

Final timing and packaging receipts are linked above. The release
gates require balanced and fast to exceed 1.2× on both workloads and high to
remain within 5% of baseline time. Accuracy gates use the same frozen references
for both implementations. New high alone cannot establish a correct reference.

Supported physics remains solids plus one surface water layer. The paper's
multiple-fluid model, multiple-reference-layer GRT, eigenfunctions and waveform
synthesis are outside this release. The 0.5% scale is a configurable algorithmic
choice, not a measured resolution limit for every survey. Flags, empirical
agreement and these finite model sets do not prove root completeness for
arbitrary models. Raw arrays, inputs, failure experiments and sanitizer logs
remain private; the public source contains code, tests and statistical receipts.
