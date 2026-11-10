# modisp CUDA 12 plugin

This optional distribution adds the CUDA backend to `modisp==0.4.0`. It requires
an NVIDIA driver compatible with CUDA 12 and a supported GPU. Binary wheels
include the static CUDA runtime; a CUDA toolkit is required only to compile the
source distribution. The CPU base package requires neither a GPU nor CUDA.

Install matching base and plugin wheels, or use
`python -m pip install --no-index --find-links /path/to/wheelhouse modisp[cuda12]`.
Then select `Solver(backend="cuda")` or use the default automatic selection.
CUDA adaptive solvers default to the optimized FP64 `secular="fast"` evaluator;
use `secular="dunkin"` explicitly for the retained reference evaluator.
CPU fallback and legacy uniform default to Dunkin.
The public API and diagnostics are documented in the base package.

Version 0.4.0 uses private ABI 8. Install the matching CPU and CUDA packages.
Adaptive Rayleigh and Love can use `Solver(backend="cuda", precision="float32")`.
The default remains FP64. FP32 high can recheck a bounded set of ambiguous
traces in FP64, recorded in the trace flags and stats. Balanced and fast
retain the approximate-search and mode-index uncertainty markers.

The source build includes native `sm_120` when the selected CUDA toolkit
supports it. Use a wheel matching the platform, Python ABI and GPU
architecture, or set `CMAKE_CUDA_ARCHITECTURES` explicitly when building.

Adaptive solvers default to per-instance Brent refinement, with optional
Chandrupatla/legacy choices. Love defaults to warp-parallel SH phase counting
with original-function validation and scan fallback. Cross-period mode checks
run separately in the Python API or `modisp check-modes`.
