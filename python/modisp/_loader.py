"""Import installed extensions; CPU imports never require a CUDA runtime."""
import importlib

ABI_VERSION = 6
VERSION = "0.3.2"


class BackendUnavailableError(RuntimeError):
    pass


CudaUnavailable = BackendUnavailableError


def load():
    try:
        package = importlib.import_module("modisp_cuda")
    except ModuleNotFoundError as exc:
        if exc.name != "modisp_cuda":
            raise BackendUnavailableError(f"CUDA plugin dependency is unavailable: {exc}") from exc
        raise BackendUnavailableError("CUDA plugin is not installed; install modisp[cuda12] or select backend='cpu'") from exc
    try:
        module = package.load()
    except ImportError as exc:
        raise BackendUnavailableError(f"CUDA extension cannot be loaded: {exc}") from exc
    if getattr(package,"__version__",None) != VERSION or module.ABI_VERSION != ABI_VERSION or module.VERSION != VERSION:
        raise RuntimeError("CPU/CUDA package versions differ; install matching modisp and CUDA plugin wheels")
    return module


def extension_path():
    return load().__file__


def candidate_dirs():
    return []
