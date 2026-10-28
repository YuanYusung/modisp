# Compatibility

modisp 0.3.0 requires Python 3.10 or newer. The CPU extension and the optional
CUDA plugin share private ABI version 4; mixing distribution/ABI versions raises
an installation error. All paired test wheels must come from this build.
Balanced/fast now use observation-scale grids and quality markers;
`accuracy="high"` retains the original 0.2.1 execution path.
`adaptive_period_threshold` is accepted for compatibility but no longer controls
the grid; `close_root_rtol` does.
The CI configuration includes CPython 3.10–3.14 on Ubuntu and macOS; configured
jobs are distinct from the executed release checks listed below.

| Platform | Python | Backend | Executed release checks |
|---|---|---|---|
| macOS 27.0.1 arm64 | 3.12.12 | CPU | 140 passed, 3 CUDA-only tests skipped |
| Ubuntu 22.04 / Linux x86_64 | 3.10.12 | CPU | 140 passed, 3 CUDA-only tests skipped |
| Same Linux host, NVIDIA A800/sm_80, CUDA 12.6 | 3.10.12 | CUDA | 143 passed; memcheck/racecheck/synccheck/initcheck each passed on 6 targeted tests |

Release 0.3.0 uses paired CPU/plugin builds with private ABI 4. Earlier local
0.3.0 candidate wheels with ABI 3 must be replaced together.
See [current validation](TOLERANT_PRESETS.md) for accuracy,
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
