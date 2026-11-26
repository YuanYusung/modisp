"""Versioned, optional CUDA 12 extension. Imported only on backend selection."""
__version__ = "0.5.0"


def load():
    from . import _cuda
    return _cuda
