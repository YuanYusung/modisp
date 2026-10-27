# modisp CUDA 12 plugin

This optional distribution adds the CUDA backend to `modisp==0.2.0`. It requires
an NVIDIA driver compatible with CUDA 12 and a supported GPU. Binary wheels
include the static CUDA runtime; a CUDA toolkit is required only to compile the
source distribution. The CPU base package requires neither a GPU nor CUDA.

Install matching base and plugin wheels, or use
`python -m pip install --no-index --find-links /path/to/wheelhouse modisp[cuda12]`.
Then select `Solver(backend="cuda")` or use the default automatic selection.
The public API and diagnostics are documented in the base package.
