# Compatibility

modisp 0.2.0 requires Python 3.10 or newer. The CI configuration includes
CPython 3.10–3.14 on Ubuntu and macOS; configured jobs are distinct from the
local release checks listed below.

| Platform | Python | Backend | Release checks |
|---|---|---|---|
| macOS 27.0.1 arm64 | 3.12.12 | CPU | 44 passed, 1 CUDA-only skipped; wheel/sdist, examples, CLI and smoke |
| Ubuntu 22.04 / Linux x86_64 | 3.10.12 | CPU | 44 passed, 1 CUDA-only skipped; wheel, examples, CLI and smoke |
| Same Linux host, NVIDIA A800/sm_80, CUDA 12.6 | 3.10.12 | CUDA | 45 passed; matching plugin wheel, examples, CLI and smoke |

Binary wheels are platform- and CPython-specific. Linux wheels use native
`linux_x86_64` tags rather than claiming manylinux portability. CUDA source
builds can target sm_70, 75, 80, 86, 89 and 90; runtime release verification is
performed on A800/sm_80.

Windows, free-threaded CPython and PyPy are not part of this release's verified
platforms. See [validation](VALIDATION.md) for the executed test counts and
[installation](INSTALL.md) for deployment commands.
