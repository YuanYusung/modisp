# Installation and deployment

The Python import package and command-line program are both named `modisp`.

## CPU wheel

```sh
python -m venv .venv
# Activate the environment using your shell's normal command.
python -m pip install /path/to/modisp-0.4.0-<matching-tags>.whl
modisp doctor
python /path/to/examples/quickstart.py
```

NumPy is the only Python runtime dependency. CPU import and
`Solver(backend="cpu")` do not require CUDA. Test dependencies (pytest, SciPy
and disba/Numba) are optional.

## Optional CUDA 12 plugin

Install the exact matching plugin after the CPU package:

Version 0.4.0 uses private ABI 8. Replace both packages together when
upgrading. Published assets are listed under
[GitHub Releases](https://github.com/YuanYusung/modisp/releases).

```sh
python -m pip install /path/to/modisp_cuda12-0.4.0-<matching-tags>.whl
modisp doctor
```

Use a wheel that matches the OS, Python ABI and GPU target, or build from
source. The CUDA plugin links its runtime statically and requires a compatible
NVIDIA driver. A toolkit is needed for source builds. Use `backend="cuda"` to
require CUDA or `backend="auto"` for automatic selection. `device=0` is the
first device visible to the process.

## Source builds

CPU builds require CMake >=3.22, a C++17 compiler and Python development
headers:

```sh
python -m pip install /path/to/modisp-0.4.0.tar.gz
```

To build a wheel explicitly:

```sh
python -m pip install build
python -m build /path/to/unpacked-cpu-source
```

The plugin source distribution is self-contained and requires a CUDA 12
toolkit plus a supported host compiler:

```sh
CUDACXX=/usr/local/cuda/bin/nvcc python -m pip install /path/to/modisp_cuda12-0.4.0.tar.gz
```

For a combined source tree, first run `python tools/prepare_release.py` to copy
the shared C++ sources into the plugin, then build the base and plugin
separately. By default the plugin builds for `sm_70`, `sm_75`, `sm_80`,
`sm_86`, `sm_89` and `sm_90`, and includes `sm_120` when nvcc lists it.
Use `-Ccmake.define.CMAKE_CUDA_ARCHITECTURES=80` for an A800-only build.
Installed extensions reside inside their Python packages.

Build both packages from this checkout to use the FP32 solver. For example,
an RTX 5050 needs a CUDA 12 toolkit with `sm_120` support:

```sh
python -m pip install .
CUDACXX=/path/to/cuda-12.9/bin/nvcc python -m pip install ./cuda-plugin \
  -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=120
modisp doctor
```

`doctor` reports `cuda_float32_available` for the loaded plugin and device.

## Offline deployment

Provide a wheelhouse with runtime dependencies and matching modisp wheels:

```sh
python -m pip install --no-index --find-links /path/to/wheelhouse modisp==0.4.0
python -m pip install --no-index --find-links /path/to/wheelhouse modisp-cuda12==0.4.0
```

For an offline source build, also collect scikit-build-core, pybind11 and their
build dependencies. After installing them, use
`pip install --no-build-isolation --no-deps` with the supplied source archive.

## Command line

```sh
python examples/make_input.py input.npz
modisp forward input.npz --output output.npz --backend cpu --threads 4 --nmodes 3
modisp forward input.npz --output love.npz --wave love --backend auto
modisp forward input.npz --output fp32.npz --backend cuda --precision float32
```

Input keys are `thickness`, `vp`, `vs`, `rho` (1-D or `[B,L]`) and `periods`.
Output includes phase velocity, mask, status, flags, search bounds/level/count,
periods, backend, wave and precision. `--cmin`, `--cmax` and `--device` are optional.
Exit 0 indicates no unresolved search; exit 2 preserves unresolved diagnostics
in the output. `doctor` reports versions, extension paths, backend selection,
device name and memory.

## Optional external reference

`tools/external_reference.py` builds a separate pinned QEDispInv reference for
solid-Rayleigh validation. It is not part of runtime execution:

```sh
python tools/external_reference.py --cache /path/to/reference-cache --fetch --eigen /path/to/eigen3
```

This requires Git, a C++17 compiler and Eigen3 headers. On a restricted network,
supply verified snapshots with `--qed-source` and `--fmt-source`. Keep the
reference cache outside runtime distributions.

For the prepared source directory, `scripts/python.sh` selects its local
`.venv-release`. Set `MODISP_PYTHON` to use another installed interpreter;
ordinary installations use the generated `modisp` command directly.
