# Compatibility

modisp 0.3.2 requires Python 3.10 or newer. The CPU extension and the optional
CUDA plugin share private ABI version 6; mixing distribution/ABI versions raises
an installation error. All paired test wheels must come from this build.
CUDA adaptive solvers default to `secular="fast"`; CPU and legacy uniform
default to `dunkin`. Auto fallback follows the actual backend.
All presets default to local shape-adaptive sampling. Balanced/fast retain
quality markers; `search_strategy="qe_adaptive"` retains the 0.3.0 paths.
`adaptive_period_threshold` is accepted for compatibility but no longer controls
shape search. `sampling_rtol` controls fit consistency separately from
`close_root_rtol` and velocity refinement.
The CI configuration includes CPython 3.10–3.14 on Ubuntu and macOS; configured
jobs are distinct from the executed release checks listed below.

| Platform | Python | Backend | Executed release checks |
|---|---|---|---|
| macOS arm64 | 3.12.12 | CPU | 197 passed, 5 CUDA-only tests skipped |
| Ubuntu / Linux x86_64 | 3.10.12 | CPU | 197 passed, 5 CUDA-only tests skipped |
| Same Linux host, NVIDIA A800/sm_80, CUDA 12.6 | 3.10.12 | CUDA | 202 passed |

Replace both CPU/plugin packages together when upgrading from 0.3.1 / ABI 5.
See [0.3.2 fast-secular measurements](SECULAR_FAST_0_3_2.md) and [0.3.1 validation](SHAPE_ADAPTIVE_0_3_1.md) for accuracy,
performance, packaging and reference limitations. Earlier release counts and
benchmarks describe historical builds.

Binary wheels are platform- and CPython-specific. Linux wheels use native
`linux_x86_64` tags rather than claiming manylinux portability. CUDA source
builds can target sm_70, 75, 80, 86, 89 and 90; runtime release verification is
performed on A800/sm_80.

The attached Linux CUDA binary targets sm_80. Other supported architectures
require a source build. The macOS wheel is tagged macOS 27 arm64 / CPython 3.12;
the Linux wheels are CPython 3.10. Use source distributions for other platforms
or Python versions; the CI matrix is not a promise of an attached wheel.

Windows, free-threaded CPython and PyPy are not part of this release's verified
platforms. See [validation](VALIDATION.md) for the executed test counts and
[installation](INSTALL.md) for deployment commands.
