"""Versioned, optional CUDA 12 extension. Imported only on backend selection."""
__version__ = "0.3.3"


def load():
    from . import _cuda
    return _cuda
