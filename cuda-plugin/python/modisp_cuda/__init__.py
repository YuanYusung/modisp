"""Versioned, optional CUDA 12 extension. Imported only on backend selection."""
__version__ = "0.3.2"


def load():
    from . import _cuda
    return _cuda
