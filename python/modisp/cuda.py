"""Lazy low-level access to the optional CUDA extension.

Most callers should use :func:`modisp.dispersion_batch` or
:class:`modisp.Solver` to reuse backend state automatically.
"""

from __future__ import annotations

from ._loader import CudaUnavailable, load

__all__ = ["CudaUnavailable", "CoarseScan", "Refine", "QeSolver", "device_name"]


def _module():
    return load()


def device_name() -> str:
    """Name of the CUDA device the extension will use."""
    return _module().device_name()


class _Proxy:
    """Defers extension import until the first attribute is touched."""

    def __init__(self, name):
        self._name = name

    def _cls(self):
        return getattr(_module(), self._name)

    def __call__(self, *a, **kw):
        return self._cls()(*a, **kw)

    def __getattr__(self, item):
        return getattr(self._cls(), item)


CoarseScan = _Proxy("CoarseScan")
Refine = _Proxy("Refine")
QeSolver = _Proxy("QeSolver")
