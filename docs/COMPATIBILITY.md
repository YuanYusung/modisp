# Compatibility

modisp 0.4.0 requires Python 3.10 or newer. The CPU extension and the optional
CUDA plugin share private ABI version 8. Install the matching distributions
together.
CUDA adaptive solvers default to `secular="fast"`; CPU and legacy uniform
default to `dunkin`. Auto fallback follows the actual backend.
All presets default to local shape-adaptive sampling. Balanced/fast retain
quality markers; `search_strategy="qe_adaptive"` retains the 0.3.0 paths.
`adaptive_period_threshold` is accepted for compatibility but no longer controls
shape search. `sampling_rtol` controls fit consistency separately from
`close_root_rtol` and velocity refinement.
The CI configuration includes CPython 3.10–3.14 on Ubuntu and macOS.

FP64 is the default. Opt-in FP32 supports the adaptive CUDA search paths.
The `high` shape search may recheck a bounded set of ambiguous traces in
FP64. The CPU backend and legacy uniform search use the FP64 precision
setting. See [precision settings](API.md#precision) and [measured results](PERFORMANCE.md).

Binary wheels are platform- and CPython-specific. Linux wheels use native
`linux_x86_64` tags. CUDA source builds default to sm_70, 75, 80, 86, 89 and 90
and add sm_120 when supported by the selected nvcc. Use an explicit target
for a device-specific wheel. The toolkit, host compiler and driver must be
compatible with the target. See [installation](INSTALL.md) for build commands.

See [validation](VALIDATION.md) for reference scope and historical
platform-specific checks.
