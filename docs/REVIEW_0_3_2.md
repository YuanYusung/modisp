# modisp 0.3.2 implementation review — 2026-10-29

The review found no coefficient, layer-order or fluid-boundary error in the
new contraction. Keep the evaluator opt-in and retain `dunkin` as the default
and reproducible reference. `fast` optimizes Dunkin evaluation; it does not
replace the root-search algorithm or implement the Buchen/Xi research kernels.

## Fixed findings

- Streaming ModePhys checkpoints omitted `secular` from their configuration
  identity. Resuming with another evaluator could mix output blocks. The
  identity now includes the selector and a regression rejects the mismatch.
- `native.secular_grid(method="fast")` silently used Dunkin for raw or FP32
  probes. Those unsupported combinations now raise an explicit error.
- Diagnostic fast calls allocated density constants but discarded them when
  frequency constants were absent. The fallback now consumes the supplied
  model constants. The production prepared path keeps the same arithmetic.
- Love-wave reports named a Dunkin kernel although they execute SH. Stats and
  kernel reports now identify `sh` while retaining the requested selector.
- The pointwise benchmark could report zero displacement when crossing counts
  differed and did not check finite masks. It now rejects nonfinite/crossing
  mismatches, reports unavailable displacement as null and exits nonzero on
  failed comparison. Both variants are warmed up and their order alternates.
  M-evaluations/s now has the correct million-evaluation unit.
- Removed unsupported claims of universally faster execution and contention-free
  paired ratios. Historical `max_relative_difference` is explicitly defined as
  a globally scaled absolute difference; crossing intervals are distinguished
  from refined roots. Installation docs link to published releases without
  implying that a source push has already published 0.3.2 binary assets.

## Verification

| Check | Reviewed result |
|---|---|
| Installed macOS arm64 CPU wheel, full suite | 204 passed, 5 CUDA-only skips |
| CPU sdist independently built/installed; full suite outside the checkout | 204 passed, 5 CUDA-only skips |
| Installed Linux x86-64 CPU wheel, full suite | 204 passed, 5 CUDA-only skips |
| Installed A800/sm_80 CUDA wheel, full suite | 209 passed |
| CUDA compute-sanitizer memcheck, all 27 secular regression cases | 27 passed, 0 errors |
| Retained Dunkin frozen fixtures | Passed in the full suites |
| Fast vs Dunkin, shape and QE paths, all three accuracy presets | Passed |
| CLI NPZ evaluator metadata and unsupported precision probes | Passed |
| Algebra-only contraction check | 100,000 arbitrary matrices / 500,000 components passed; maximum error / sum(abs(terms)) = 6.52e-16 |
| Shared CPU/plugin C++ sources | Exact copies |

The algebra-only check uses arbitrary eigenfunction-product coefficients and
compares every output component with `dunkin_dnka` plus the original matrix
contraction. It checks the scalar rewrite independently of layer propagation;
it is not an independent physical dispersion reference.

The extended tests cover both adaptive search strategies, CLI selection,
resume identities, Love reports and benchmark failure detection. The full
suites include the retained frozen-input and frozen-output regressions.

## Performance and default choice

Fresh paired solver measurements gave 1.101× on M4, 1.119× on the shared Xeon
host and 1.298× on A800. All three had matching finite masks/root counts and
zero wrong/missing/extra cells within the benchmark's stated tolerance. Linux
fixed-node measurements included slowdowns on the loaded shared host. Raw
samples and limitations are in [the method report](SECULAR_FAST_0_3_2.md).

Use `Solver(secular="fast")` explicitly for workloads whose paired tests and
measurements support it, particularly the measured A800 batch. Keep the 0.3.2
default unchanged. FMA, reciprocal multiplication, summation order and platform
math functions can change adaptive paths near numerical thresholds; paired
agreement on these cases is not a proof of root completeness. Broader compiler,
architecture and near-root stress coverage should precede a future default
change. Preserve the explicit Dunkin evaluator for regression and reproducibility.

## Packaging

The macOS CPU wheel, CPU sdist, rebuilt Linux CPython 3.10 CUDA wheel and
self-contained CUDA plugin sdist are version 0.3.2. CPU/CUDA runtime imports
validated private ABI 6. The six central runtime source hashes on the remote
build matched the local reviewed files; shared CPU/plugin C++ trees are equal.
The distribution audit covered all four archives and found no excluded runtime
caches, build directories or pinned external reference implementation files.
The independent sdist installation ran the full suite outside the checkout.

The refreshed local distribution checksums are in `release/SHA256SUMS.0.3.2`;
those generated release assets are excluded from Git.
