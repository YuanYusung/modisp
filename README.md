# modisp

modisp computes multimode Rayleigh and Love dispersion in 1-D layered media.
It supports single models and large batches through the same Python API, with
an FP64 CPU backend and an optional CUDA 12 backend.

## Install

Build the CPU package from the 0.5.0 source with Python 3.10 or newer,
CMake 3.22 or newer, and a C++17 compiler.

```sh
git clone --branch v0.5.0 https://github.com/YuanYusung/modisp.git
cd modisp
python -m pip install .
modisp doctor
```

For NVIDIA GPUs on Linux, build the matching `modisp-cuda12` plugin.
See the [installation guide](docs/INSTALL.md).

## Quick start

```python
import numpy as np
from modisp import dispersion

model = np.array([[0.3, 2.0, 0.8, 2.0],
                  [0.0, 5.0, 2.8, 2.6]])
periods = np.geomspace(0.1, 10.0, 30)
result = dispersion(model, periods, nmodes=2, wave="rayleigh",
                    backend="auto", group_velocity=True)
print(result.phase_velocity[0, 0], result.group_velocity[0, 0])
```

Model columns are thickness, P velocity, S velocity and density in km, km/s
and g/cm³. A zero-thickness last row represents the half-space. Use
`dispersion_batch` for many models or reuse a `Solver` across calls. Results
include validity masks and numerical diagnostics.

## What’s new in 0.5.0

- Adaptive `auto` accuracy and a default continuity check that rechecks
  selected phase results in FP64.
- Optional FP64 group velocities on CPU and CUDA, including single-period
  calculations.
- Optional anchor continuation across periods on CPU and CUDA, with
  global-search fallback and prediction diagnostics.

Phase velocity uses FP64 and independent per-period searches by default.
Adaptive CUDA supports opt-in FP32.

[API](docs/API.md) · [Methods](docs/GROUP_CONTINUATION_0_5_0.md) ·
[中文快速开始](docs/QUICKSTART_ZH.md) · [Changelog](CHANGELOG.md)
