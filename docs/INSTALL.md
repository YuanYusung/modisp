# Installation and deployment

The Python import package and command-line program are both named `modisp`.

## CPU wheel

```sh
python -m venv .venv
# Activate the environment using your shell's normal command.
python -m pip install /path/to/modisp-0.3.0-<matching-tags>.whl
modisp doctor
python /path/to/examples/quickstart.py
```

NumPy is the only Python runtime dependency. CPU import and
`Solver(backend="cpu")` do not require CUDA. Test dependencies (pytest, SciPy
and disba/Numba) are optional.

## Optional CUDA 12 plugin

Install the exact matching plugin after the CPU package:

Release 0.3.0 uses private ABI 4. Replace both packages if an earlier local
0.3.0 candidate with ABI 3 is installed; the same public version number alone
does not make those candidate binaries compatible. Release assets are at
[GitHub v0.3.0](https://github.com/YuanYusung/modisp/releases/tag/v0.3.0).

```sh
python -m pip install /path/to/modisp_cuda12-0.3.0-<matching-tags>.whl
modisp doctor
```

The attached CUDA wheel targets Linux x86_64, CPython 3.10 and sm_80, and
requires a compatible NVIDIA driver. The macOS CPU wheel targets macOS 27
arm64 / CPython 3.12. Use a source build for other supported configurations.
Its static CUDA runtime is linked into the extension; a toolkit is needed only
for source builds. Use `backend="cuda"` to require CUDA or `backend="auto"` for
automatic selection. `device=0` is the first device visible to the process.

## Source builds

CPU builds require CMake >=3.22, a C++17 compiler and Python development
headers:

```sh
python -m pip install /path/to/modisp-0.3.0.tar.gz
```

To build a wheel explicitly:

```sh
python -m pip install build
python -m build /path/to/unpacked-cpu-source
```

The plugin source distribution is self-contained and requires a CUDA 12
toolkit plus a supported host compiler:

```sh
CUDACXX=/usr/local/cuda/bin/nvcc python -m pip install /path/to/modisp_cuda12-0.3.0.tar.gz
```

For a combined source tree, first run `python tools/prepare_release.py` to copy
the shared C++ sources into the plugin, then build the base and plugin
separately. `-Ccmake.define.CMAKE_CUDA_ARCHITECTURES=80` can target one device
during development. Installed extensions reside inside their Python packages;
no source-tree build directory or `PYTHONPATH` is needed at runtime.

## Offline deployment

Provide a wheelhouse with runtime dependencies and matching modisp wheels:

```sh
python -m pip install --no-index --find-links /path/to/wheelhouse modisp==0.3.0
python -m pip install --no-index --find-links /path/to/wheelhouse modisp-cuda12==0.3.0
```

For an offline source build, also collect scikit-build-core, pybind11 and their
build dependencies. After installing them, use
`pip install --no-build-isolation --no-deps` with the supplied source archive.

## Command line

```sh
python examples/make_input.py input.npz
modisp forward input.npz --output output.npz --backend cpu --threads 4 --nmodes 3
modisp forward input.npz --output love.npz --wave love --backend auto
```

Input keys are `thickness`, `vp`, `vs`, `rho` (1-D or `[B,L]`) and `periods`.
Output includes phase velocity, mask, status, flags, search bounds/level/count,
periods, backend and wave. `--cmin`, `--cmax` and `--device` are optional.
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
