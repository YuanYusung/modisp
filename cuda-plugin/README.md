# modisp CUDA 12 plugin

This optional distribution adds the CUDA backend to `modisp==0.3.0`. It requires
an NVIDIA driver compatible with CUDA 12 and a supported GPU. Binary wheels
include the static CUDA runtime; a CUDA toolkit is required only to compile the
source distribution. The CPU base package requires neither a GPU nor CUDA.

Install matching base and plugin wheels, or use
`python -m pip install --no-index --find-links /path/to/wheelhouse modisp[cuda12]`.
Then select `Solver(backend="cuda")` or use the default automatic selection.
The public API and diagnostics are documented in the base package.

Release 0.3.0 uses private ABI 4. Install the base and plugin from the same
release; earlier local 0.3.0 candidates used ABI 3. Balanced/fast use a
configurable 0.5% observation scale and expose approximate-search, close-root
and mode-index uncertainty markers. Use high for the full search schedule.

The attached Linux CPython 3.10 CUDA wheel targets sm_80 and was tested on A800
with CUDA 12.6. The source build supports the architectures listed in CMake.
