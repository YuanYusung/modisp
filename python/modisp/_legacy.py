"""Legacy CUDA Rayleigh solver used by the uniform search option.

Units are km, km/s, g/cm³ and seconds. Results contain NaN for undetected modes
and explicit trace flags for numerical diagnostics.
"""

from __future__ import annotations

import numpy as np

from ._loader import CudaUnavailable
from .models import (
    CANONICAL_MODELS,
    LayeredModel,
    pad_batch,
    random_model_batch,
    to_batch,
)

__all__ = [
    "CudaUnavailable",
    "DispersionBatch",
    "Solver",
    "TRACE_RELOCATED",
    "TRACE_UNRESOLVED_BRACKET",
    "TRACE_SEARCH_UNCONVERGED",
    "TRACE_WORKSPACE_LIMIT",
    "TRACE_NEAR_CUTOFF",
    "TRACE_NONFINITE",
    "TRACE_REFINEMENT_FAILED",
    "TRACE_F_REJECTED",
    "dispersion_batch",
    "release",
    "device_name",
    "LayeredModel",
    "CANONICAL_MODELS",
    "to_batch",
    "pad_batch",
    "random_model_batch",
    "models",
    "cuda",
]

__version__ = "0.3.3"

WAVES = ("rayleigh",)


# Trace-flag bits mirror md::TraceFlag in cpp/cuda/refine.cuh.
TRACE_UNRESOLVED_BRACKET = 1 << 0
TRACE_RELOCATED = 1 << 1
TRACE_SEARCH_UNCONVERGED = 1 << 2
TRACE_WORKSPACE_LIMIT = 1 << 3
TRACE_NEAR_CUTOFF = 1 << 4
TRACE_NONFINITE = 1 << 5
TRACE_REFINEMENT_FAILED = 1 << 6
TRACE_F_REJECTED = 1 << 7


class DispersionBatch:
    """Phase velocities for a batch of models.

    ``phase_velocity`` is ``[B, nmodes, T]`` in km/s, with **NaN** wherever a
    root was not found or a numerical failure was flagged. ``mode_mask`` is the same shape and is
    True exactly where ``phase_velocity`` is finite -- it is offered because a
    bare NaN is easy to miss and easy to propagate, and callers routinely want
    the boolean for indexing.

    Modes are ordered by ascending phase velocity at each period, which is the
    same rule DISBA uses (mode m is the m-th root above mode m-1).  A mode that
    exists at some periods and not others is represented by NaN at the missing
    ones, not by a shorter array -- the period axis is never ragged.
    """

    __slots__ = ("phase_velocity", "mode_mask", "period", "mode", "wave",
                  "trace_flags", "search_level", "root_count")

    def __init__(self, phase_velocity, period, mode, wave, mode_mask=None,
                  trace_flags=None, search_level=None, root_count=None):
        self.phase_velocity = phase_velocity
        self.mode_mask = (
            np.isfinite(phase_velocity) if mode_mask is None else mode_mask
        )
        self.period = period
        self.mode = mode
        self.wave = wave
        self.trace_flags = trace_flags
        # QE levels 0..3 are adaptive; 4..5 are dense fallback.
        self.search_level = search_level
        self.root_count = root_count

    @property
    def n_models(self) -> int:
        return self.phase_velocity.shape[0]

    @property
    def n_modes(self) -> int:
        return self.phase_velocity.shape[1]

    def __len__(self) -> int:
        return self.n_models

    def __repr__(self) -> str:
        b, m, t = self.phase_velocity.shape
        frac = 100.0 * self.mode_mask.mean() if self.mode_mask.size else 0.0
        return (
            f"<DispersionBatch {self.wave} B={b} nmodes={m} T={t} "
            f"modes_present={frac:.1f}%>"
        )


def _as_2d(x, name, B=None):
    a = np.ascontiguousarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError(f"{name} must be [B, L] or [L], got shape {a.shape}")
    if B is not None and a.shape[0] != B:
        raise ValueError(
            f"{name} has {a.shape[0]} models but the batch has {B}; all of "
            f"thickness/vp/vs/rho must agree"
        )
    return a


class Solver:
    """Reusable legacy CUDA solver; not thread-safe.

    Device buffers are rebuilt when input geometry changes and retained until
    the solver is closed.
    """

    def __init__(
        self,
        nc: int = 512,
        cmin: float = 0.2,
        cmax: float | None = None,
        c_tol_rel: float = 1e-6,
        f_accept: float = 1.0,
        block: int = 256,
        search_strategy: str = "qe_adaptive",
        workspace_bytes: int = 256 * 1024**2,
        chunk_size: int | None = None,
    ):
        if nc < 3:
            raise ValueError(f"nc must be at least 3, got {nc}")
        if cmin <= 0:
            raise ValueError(f"cmin must be positive, got {cmin}")
        if search_strategy not in ("qe_adaptive", "uniform"):
            raise ValueError("search_strategy must be 'qe_adaptive' or 'uniform'")
        if not np.isfinite(cmin) or (cmax is not None and not np.isfinite(cmax)):
            raise ValueError("cmin/cmax must be finite")
        if not np.isfinite(c_tol_rel) or c_tol_rel <= 0:
            raise ValueError("c_tol_rel must be finite and positive")
        if not np.isfinite(f_accept) or f_accept < 0:
            raise ValueError("f_accept must be finite and non-negative")
        if not isinstance(workspace_bytes, (int, np.integer)) or workspace_bytes <= 0:
            raise ValueError("workspace_bytes must be a positive integer")
        if chunk_size is not None and (not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1):
            raise ValueError("chunk_size must be a positive integer or None")
        self.nc = int(nc)
        self.cmin = float(cmin)
        self.cmax = None if cmax is None else float(cmax)
        self.c_tol_rel = float(c_tol_rel)
        self.f_accept = float(f_accept)
        self.block = int(block)
        self.search_strategy = search_strategy
        self.workspace_bytes = int(workspace_bytes)
        self.chunk_size = chunk_size
        self._qe = None
        self._qe_key = None
        self._last_stats = None
        self._scan = None
        self._ref = None
        self._key = None
        self._periods = None
        self._calls = 0

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        for attr in ("_ref", "_scan", "_qe"):
            obj = getattr(self, attr, None)
            if obj is not None:
                if attr == "_qe":
                    obj.close()
                del obj
                setattr(self, attr, None)
        self._key = None
        self._qe_key = None
        self._periods = None
        self._last_stats = None

    def __enter__(self) -> "Solver":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass

    @property
    def calls(self) -> int:
        """How many times this Solver has been called."""
        return self._calls

    # -- the call ----------------------------------------------------------
    def __call__(
        self,
        thickness,
        vp,
        vs,
        rho,
        periods,
        nmodes: int = 5,
        wave: str = "rayleigh",
        names=None,
    ) -> DispersionBatch:
        # `names` is carried by models.to_batch() for identification and is
        # ignored here; accepting it is what lets `dispersion_batch(**batch,
        # periods=...)` work without the caller pruning the dict first.
        del names
        if wave not in WAVES:
            raise NotImplementedError(
                f"wave={wave!r} is unavailable in the legacy solver; use rayleigh"
            )
        if not isinstance(nmodes, (int, np.integer)) or nmodes < 1:
            raise ValueError(f"nmodes must be at least 1, got {nmodes}")

        d = _as_2d(thickness, "thickness")
        vp_a = _as_2d(vp, "vp", d.shape[0])
        vs_a = _as_2d(vs, "vs", d.shape[0])
        rho_a = _as_2d(rho, "rho", d.shape[0])
        if not (d.shape == vp_a.shape == vs_a.shape == rho_a.shape):
            raise ValueError(
                "thickness/vp/vs/rho must all have the same shape, got "
                f"{d.shape}, {vp_a.shape}, {vs_a.shape}, {rho_a.shape}"
            )
        B, L = d.shape
        if B < 1:
            raise ValueError("batch must contain at least one model")
        if L < 2:
            raise ValueError(
                f"need at least 2 layers (one plus the half-space), got L={L}"
            )

        # Validate physical parameters before allocating device state.
        for name, arr in (("thickness", d), ("vp", vp_a), ("vs", vs_a),
                          ("rho", rho_a)):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be finite everywhere")
        # Zero-thickness solid layers are valid batch padding.
        if np.any(d < 0):
            raise ValueError(
                "thickness must be non-negative; the last entry is the "
                "half-space and its thickness is ignored"
            )
        for name, arr in (("vp", vp_a), ("vs", vs_a), ("rho", rho_a)):
            if np.any(arr <= 0):
                raise ValueError(
                    f"{name} must be positive; a zero or negative value is not "
                    f"a solid layer (fluid layers are deliberately out of scope)"
                )
        # Match the solid-model constraint in LayeredModel.validate().
        if np.any(vp_a <= vs_a):
            raise ValueError(
                "vp must exceed vs in every layer; a P velocity at or below "
                "the S velocity is not an elastic solid"
            )

        per = np.ascontiguousarray(periods, dtype=np.float64)
        if per.ndim != 1 or per.size == 0:
            raise ValueError("periods must be a non-empty 1-D array")
        if not np.all(np.isfinite(per)):
            raise ValueError("periods must be finite (inf passes a > 0 test)")
        if not np.all(per > 0):
            raise ValueError("periods must be positive")

        # Trapped Rayleigh modes lie below the half-space S velocity.
        ceiling = float(vs_a[:, -1].max())
        cmax = self.cmax
        if cmax is None:
            cmax = ceiling
        elif cmax > ceiling:
            raise ValueError(
                f"cmax={cmax:g} is above the largest half-space Vs in the "
                f"batch ({ceiling:g}); trapped Rayleigh modes do not exist "
                f"there. Leaky modes are out of scope and would need a "
                f"different solver definition, not a wider scan."
            )
        if cmax <= self.cmin:
            raise ValueError(
                f"cmax ({cmax}) must exceed cmin ({self.cmin}); the scan range "
                f"is empty. Check the Vs column."
            )

        if self.search_strategy == "qe_adaptive":
            return self._adaptive(d, vp_a, vs_a, rho_a, per, nmodes, wave, cmax)

        # Cache period values, not only their count, and compare against a copy
        # so later caller-side mutations cannot invalidate device state.
        key = (B, L, per.size, self.nc, nmodes)
        periods_changed = (
            self._periods is None
            or self._periods.shape != per.shape
            or not np.array_equal(self._periods, per)
        )
        if key != self._key:
            self.close()
            self._scan = cuda.CoarseScan(B, L)
            self._ref = cuda.Refine(B, L)
            self._scan.upload_periods(per)
            self._scan.allocate(self.nc)
            self._ref.upload_periods(per)
            self._ref.allocate(self.nc, nmodes)
            self._key = key
            self._periods = per.copy()
        elif periods_changed:
            # Reuse allocations and refresh only the changed period axis.
            self._scan.upload_periods(per)
            self._ref.upload_periods(per)
            self._periods = per.copy()

        # Models change every call in an inversion loop, so this is the part
        # that cannot be hoisted; it is one H2D per array per stage.
        self._scan.upload_models(d, vp_a, vs_a, rho_a)
        self._ref.upload_models(d, vp_a, vs_a, rho_a)

        self._scan.run(self.cmin, cmax, self.block)
        self._ref.run(
            self._scan, self.cmin, cmax, self.c_tol_rel, self.f_accept,
            nmodes, self.block,
        )
        self._calls += 1
        return DispersionBatch(
            trace_flags=self._ref.flags(),
            phase_velocity=self._ref.download(),
            period=per.copy(),
            mode=np.arange(nmodes),
            wave=wave,
        )

    def _adaptive(self, d, vp, vs, rho, per, nmodes, wave, cmax):
        B, L = d.shape
        nf = per.size
        # Exact persistent device-array sizes, plus a small allocation margin.
        # Sampling scratch is fixed block shared memory, not B*T*Nc storage.
        per_model = 8*(6*L+2*1024+4)+4+nf*(64*(nmodes+2)+8*nmodes+48+2*256*8)
        fixed = nf * 8 + 13 * 8 + 4096
        capacity = (self.workspace_bytes - fixed) // per_model
        if capacity < 1:
            raise ValueError("workspace_bytes is too small for one model at this geometry")
        capacity = min(B, capacity, self.chunk_size or B)
        key = (capacity, L, nf, nmodes)
        if key != self._qe_key:
            self.close()
            self._qe = cuda.QeSolver(*key)
            self._qe_key = key
        self._key = (B, L, nf, self.nc, nmodes)
        velocity = np.empty((B, nmodes, nf), dtype=np.float64)
        flags = np.empty((B, nf), dtype=np.uint32)
        levels = np.empty((B, nf), dtype=np.int32)
        counts = np.empty((B, nf), dtype=np.int32)
        totals = {}
        chunks = 0
        from . import _cpu
        bounds = _cpu.bounds(d,vp,vs,rho,0,self.cmin,cmax)
        # cmax was determined over the entire input before slicing. Thus this
        # budget and explicit chunk_size cannot change the search domain.
        for start in range(0, B, capacity):
            stop = min(B, start + capacity)
            v, f, stats, lev, cnt = self._qe.run(
                d[start:stop], vp[start:stop], vs[start:stop], rho[start:stop],
                per, bounds[start:stop], self.c_tol_rel, self.f_accept,0,False,
            )
            velocity[start:stop] = v
            flags[start:stop] = f
            levels[start:stop] = lev
            counts[start:stop] = cnt
            for name, value in stats.items():
                if name not in ("strategy", "capacity"):
                    totals[name] = totals.get(name, 0) + int(value)
            chunks += 1
        totals.update(strategy="qe_adaptive", capacity=2048, chunks=chunks,
                      workspace_bytes=capacity * per_model + fixed,
                      workspace_budget_bytes=self.workspace_bytes)
        # Counters for the retained uniform paths remain available. Those
        # paths do not execute in QE, so their event counts are zero here.
        for name in ("rejected_sign", "troughs", "hidden_roots", "relocated"):
            totals.setdefault(name, 0)
        self._last_stats = totals
        self._periods = per.copy()
        self._calls += 1
        return DispersionBatch(velocity, per.copy(), np.arange(nmodes), wave,
                               trace_flags=flags, search_level=levels,
                               root_count=counts)

    def stats(self) -> dict:
        """Refinement counters from the last call."""
        if self.search_strategy == "qe_adaptive":
            if self._last_stats is None:
                raise RuntimeError("call the solver first")
            return dict(self._last_stats)
        if self._ref is None:
            raise RuntimeError("call the solver first")
        return self._ref.stats()

    def kernel_report(self) -> dict:
        """Return strategy metadata and available kernel resource figures."""
        if self.search_strategy == "qe_adaptive":
            if self._qe is None:
                raise RuntimeError("call the solver first")
            return {"strategy": "qe_adaptive", "sampling_capacity": 2048,
                    "sampling_block": 256, "precision": "float64",
                    "device": cuda.device_name()}
        if self._scan is None:
            raise RuntimeError("call the solver first")
        return {
            "coarse_scan": self._scan.kernel_report(self.block),
            "refine": self._ref.kernel_report(self.block),
        }


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------
_DEFAULT: Solver | None = None


def _default_solver(search_strategy="qe_adaptive") -> Solver:
    global _DEFAULT
    if _DEFAULT is None or _DEFAULT.search_strategy != search_strategy:
        if _DEFAULT is not None:
            _DEFAULT.close()
        _DEFAULT = Solver(search_strategy=search_strategy)
    return _DEFAULT


def release() -> None:
    """Free the device state held by the module-level default solver.

    Call this before measuring memory, or when done with the functional API.
    It is recreated on the next :func:`dispersion_batch` call.
    """
    global _DEFAULT
    if _DEFAULT is not None:
        _DEFAULT.close()
        _DEFAULT = None


def dispersion_batch(
    thickness,
    vp,
    vs,
    rho,
    periods,
    nmodes: int = 5,
    wave: str = "rayleigh",
    solver: Solver | None = None,
    names=None,
    search_strategy: str | None = None,
) -> DispersionBatch:
    """Rayleigh phase-velocity dispersion for a batch of 1-D layered models.

    Parameters
    ----------
    thickness, vp, vs, rho : array_like [B, L] or [L]
        km, km/s, km/s, g/cm^3.  ``L`` includes the half-space, which must come
        last and whose thickness entry is ignored.
    periods : array_like [T]
        Seconds.
    nmodes : int
        How many modes to return per model.  Modes that do not exist are NaN.
    wave : str
        Only ``"rayleigh"``.  Anything else raises -- rather than quietly
        solving a different problem.
    names : ignored
        ``models.to_batch()`` returns model names alongside the four physics
        arrays; this absorbs that key so ``dispersion_batch(**batch, ...)``
        works unchanged.
    solver : Solver, optional
        Reuse device state.  Pass one in a loop; omit for a one-off.  The
        module keeps a default solver so repeated calls do not rebuild state,
        but an explicit Solver avoids surprises if the shapes vary.
    search_strategy : {"qe_adaptive", "uniform"}, optional
        Defaults to qe_adaptive without an explicit solver, otherwise uses
        that solver's setting. An explicitly conflicting choice raises.

    Returns
    -------
    DispersionBatch
        ``phase_velocity`` is ``[B, nmodes, T]`` in km/s, NaN where a mode does
        not exist, plus ``mode_mask`` of the same shape.

    Examples
    --------
    >>> import numpy as np
    >>> from modisp import dispersion_batch, random_model_batch
    >>> batch = random_model_batch(1000, 8, seed=0)
    >>> out = dispersion_batch(periods=np.logspace(0, 2, 64), **batch)
    >>> out.phase_velocity.shape
    (1000, 5, 64)
    """
    if solver is not None and search_strategy is not None and solver.search_strategy != search_strategy:
        raise ValueError("search_strategy disagrees with the explicit solver")
    s = solver if solver is not None else _default_solver(
        "qe_adaptive" if search_strategy is None else search_strategy
    )
    return s(thickness, vp, vs, rho, periods, nmodes=nmodes, wave=wave,
             names=names)


def device_name() -> str:
    """Name of the CUDA device in use."""
    return cuda.device_name()


from . import cuda, models  # noqa: E402  (circular: cuda imports _loader only)
