# modisp CUDA 12 plugin

This optional distribution adds the CUDA backend to `modisp==0.3.2`. It requires
an NVIDIA driver compatible with CUDA 12 and a supported GPU. Binary wheels
include the static CUDA runtime; a CUDA toolkit is required only to compile the
source distribution. The CPU base package requires neither a GPU nor CUDA.

Install matching base and plugin wheels, or use
`python -m pip install --no-index --find-links /path/to/wheelhouse modisp[cuda12]`.
Then select `Solver(backend="cuda")` or use the default automatic selection.
The public API and diagnostics are documented in the base package.

Release 0.3.2 uses private ABI 6. Install the base and plugin from the same
release; 0.3.1 uses ABI 5 and 0.3.0 uses ABI 4. The default shape-adaptive search uses a compact
activity queue and sample reuse. Balanced/fast retain the configurable 0.5%
close-root scale and approximate-search / mode-index uncertainty markers.
Use `search_strategy="qe_adaptive"` for the retained 0.3.0 path. See the base
package's 0.3.2 fast-secular report for measured scope and limitations.

The attached Linux CPython 3.10 CUDA wheel targets sm_80 and was tested on A800
with CUDA 12.6. The source build supports the architectures listed in CMake.
