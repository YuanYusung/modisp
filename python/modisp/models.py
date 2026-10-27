"""Layered models and reproducible regression inputs.

Use km, km/s and g/cm^3. Last thickness is zero for LayeredModel. A first
acoustic water layer has Vs=0; all other layers and the half-space are solid.
Zero-thickness solid padding immediately before the half-space is supported.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = [
    "LayeredModel",
    "CANONICAL_MODELS",
    "random_model_batch",
    "to_batch",
    "pad_batch",
]


class LayeredModel(NamedTuple):
    """A single 1-D layered model.  All arrays have length ``n_layers + 1``."""

    name: str
    thickness: np.ndarray  # km, last entry 0 (half-space)
    vp: np.ndarray  # km/s
    vs: np.ndarray  # km/s
    rho: np.ndarray  # g/cm3

    @property
    def n_layers(self) -> int:
        return len(self.thickness)

    def validate(self) -> None:
        values=[np.asarray(v,dtype=float) for v in (self.thickness,self.vp,self.vs,self.rho)]
        if any(v.ndim!=1 or not v.size or not np.isfinite(v).all() for v in values):
            raise ValueError(f"{self.name}: expected finite nonempty 1-D arrays")
        h,a,b,r=values
        if any(v.size!=h.size for v in values):raise ValueError(f"{self.name}: ragged layer arrays")
        if h[-1]!=0:raise ValueError(f"{self.name}: last layer must be a half-space (h=0)")
        if np.any(h<0):raise ValueError(f"{self.name}: negative layer thickness")
        if np.any(b<0) or b[-1]<=0 or np.any(b[1:]==0):raise ValueError(f"{self.name}: only one first water layer may have Vs=0")
        solid=b>0
        if np.any(a<=0) or np.any(a[solid]/b[solid]<=np.sqrt(4./3.)):raise ValueError(f"{self.name}: solid bulk modulus must be positive")
        if np.any(r<=0):raise ValueError(f"{self.name}: non-positive density")


def _m(name, thickness, vp, vs, rho) -> LayeredModel:
    m = LayeredModel(
        name,
        np.asarray(thickness, dtype=np.float64),
        np.asarray(vp, dtype=np.float64),
        np.asarray(vs, dtype=np.float64),
        np.asarray(rho, dtype=np.float64),
    )
    m.validate()
    return m


# --------------------------------------------------------------------------
# 1-3: well conditioned.  These validate the kernel arithmetic.
# --------------------------------------------------------------------------

# 1. Velocity increasing smoothly with depth -- the easy, monotone case.
_MODEL_1 = _m(
    "m01_gradient",
    thickness=[2.0, 3.0, 5.0, 8.0, 0.0],
    vp=[2.0, 2.6, 3.4, 4.4, 5.6],
    vs=[0.9, 1.3, 1.9, 2.6, 3.3],
    rho=[1.9, 2.1, 2.3, 2.5, 2.7],
)

# 2. Two layers: a single contrast over a half-space.
_MODEL_2 = _m(
    "m02_two_layer",
    thickness=[5.0, 0.0],
    vp=[3.0, 6.0],
    vs=[1.5, 3.5],
    rho=[2.2, 2.8],
)

# 3. Soft sediment over a fast basement -- strong impedance contrast and the
#    classic situation where higher modes appear.
_MODEL_3 = _m(
    "m03_sediment_basement",
    thickness=[1.0, 2.0, 6.0, 0.0],
    vp=[1.8, 2.4, 4.8, 6.2],
    vs=[0.5, 0.9, 2.7, 3.6],
    rho=[1.8, 2.0, 2.5, 2.7],
)

# --------------------------------------------------------------------------
# 4-8: the hard cases.  Each targets a specific failure mode.
# --------------------------------------------------------------------------

# 4. Single low-velocity zone: a mid-layer slower than the layer above it.
#    Target failure: the root count in the LVZ interval is misjudged and a
#    branch is skipped.
_MODEL_4 = _m(
    "m04_lvz",
    thickness=[2.0, 3.0, 4.0, 6.0, 0.0],
    vp=[2.6, 3.0, 2.8, 4.0, 5.0],
    vs=[1.2, 1.5, 1.1, 2.2, 2.9],
    rho=[2.0, 2.2, 2.1, 2.4, 2.6],
)

# 5. Strong LVZ: a large velocity drop, producing a pronounced discontinuity
#    in the dispersion curve and closely spaced modes.
_MODEL_5 = _m(
    "m05_strong_lvz",
    thickness=[1.5, 2.0, 3.0, 5.0, 0.0],
    vp=[3.2, 4.0, 2.6, 4.6, 6.0],
    vs=[1.8, 2.4, 0.9, 2.8, 3.5],
    rho=[2.2, 2.4, 2.0, 2.5, 2.7],
)

# 6. Multiple velocity reversals: two LVZs stacked, so roots interleave and
#    velocity-sorted labeling is very likely to swap branches.
_MODEL_6 = _m(
    "m06_multi_reversal",
    thickness=[1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 0.0],
    vp=[2.4, 3.0, 2.6, 3.6, 3.0, 4.8, 6.0],
    vs=[1.1, 1.6, 1.2, 2.0, 1.5, 2.9, 3.6],
    rho=[1.9, 2.1, 2.0, 2.3, 2.1, 2.5, 2.7],
)

# 7. Thin low-velocity layer: a thin (100 m) LVZ.  Its influence is small, so
#    the perturbation of the dispersion curve is subtle and easily missed by a
#    fixed coarse grid.
_MODEL_7 = _m(
    "m07_thin_lvz",
    thickness=[2.0, 0.1, 2.0, 4.0, 0.0],
    vp=[3.0, 1.6, 3.2, 4.5, 5.8],
    vs=[1.7, 0.7, 1.8, 2.7, 3.4],
    rho=[2.2, 1.9, 2.2, 2.5, 2.7],
)

# 8. Near mode-kissing: the fundamental and first higher mode approach each
#    other in (c, T) space and exercise velocity-sorted mode numbering.
_MODEL_8 = _m(
    "m08_near_kissing",
    thickness=[3.0, 3.0, 6.0, 0.0],
    vp=[3.4, 3.6, 5.0, 6.4],
    vs=[1.90, 1.98, 2.9, 3.7],
    rho=[2.2, 2.25, 2.5, 2.7],
)

# Water-covered examples used by the P-SV and Scholte regression tests.

# 9. Liquid layer over a solid half-space (Vs = 0 in the water layer).
MODEL_9_LIQUID_SOLID = dict(
    name="m09_liquid_solid",
    thickness=[2.0, 4.0, 0.0],
    vp=[1.5, 3.0, 5.5],
    vs=[0.0, 1.6, 3.2],  # water: Vs = 0
    rho=[1.0, 2.2, 2.6],
)

# 10. Scholte wave: water over a slow seabed, where the interface wave is the
#     physically relevant solution and Rayleigh modes are not the right target.
MODEL_10_SCHOLTE = dict(
    name="m10_scholte",
    thickness=[3.0, 0.0],
    vp=[1.5, 2.0],
    vs=[0.0, 0.6],
    rho=[1.0, 1.9],
)


CANONICAL_MODELS: tuple[LayeredModel, ...] = (
    _MODEL_1, _MODEL_2, _MODEL_3, _MODEL_4,
    _MODEL_5, _MODEL_6, _MODEL_7, _MODEL_8,
)
"""The eight supported validation models, easiest first."""


def to_batch(models, pad: bool = True) -> dict[str, np.ndarray]:
    """Pack a sequence of models into ``[B, L]`` batch arrays.

    This is the exact layout the CUDA kernels expect, so the test harness and
    the solver agree on memory order by construction.

    Models of differing layer counts are padded to the maximum by inserting
    zero-thickness layers before the half-space (see :func:`pad_batch`).  Pass
    ``pad=False`` to require a uniform layer count instead.
    """
    models = list(models)
    if not models:
        raise ValueError("empty model list")

    for m in models:m.validate()
    counts = {m.n_layers for m in models}
    if len(counts) > 1 and not pad:
        raise ValueError(
            f"ragged layer counts {sorted(counts)}; pass pad=True to pad them"
        )
    n = max(counts)

    def field(attr: str) -> np.ndarray:
        out = np.zeros((len(models), n), dtype=np.float64)
        for i, m in enumerate(models):
            src = getattr(m, attr)
            k = len(src)
            if k == n:
                out[i] = src
            else:
                # Real layers, then zero-thickness padding, then the half-space.
                out[i, :k - 1] = src[:-1]
                out[i, k - 1:n - 1] = src[-1]  # dummy layers inherit half-space
                out[i, -1] = src[-1]
        return out

    return {
        "thickness": field("thickness"),
        "vp": field("vp"),
        "vs": field("vs"),
        "rho": field("rho"),
        "names": np.array([m.name for m in models]),
    }


def pad_batch(batch: dict[str, np.ndarray], n_layers: int | None = None) -> dict:
    """Pad a ragged batch to a common layer count by inserting zero-thickness layers.

    The CUDA kernels require every model in a chunk to have the same ``L``.  Real
    model sets (including our own canonical suite, which spans L = 2..7) are
    ragged, so padding is a hard requirement of the batch interface rather than
    a convenience.

    Padding layers are inserted **immediately before the half-space** and given
    ``h = 0``.  A zero-thickness layer contributes an identity propagation step,
    so a correctly scaled formulation is unaffected by them.  The dummy layer is
    given the half-space's own Vp/Vs/rho so that it can never introduce a
    velocity inversion or a ``c > vs`` branch that the real model did not have.

    Note: this exactness is a property of the *formulation*.  Any implementation
    that divides by thickness, or that normalizes per layer in a way that
    degenerates at ``w*h = 0``, will break here.  The unit tests assert that
    padding changes results by exactly zero, which is what makes this safe to
    rely on.
    """
    for key in ("thickness", "vp", "vs", "rho"):
        if batch[key].ndim != 2:
            raise ValueError(f"expected a [B, L] array for {key!r}")

    lmax = n_layers if n_layers is not None else batch["thickness"].shape[1]
    if lmax < batch["thickness"].shape[1]:
        raise ValueError("n_layers is smaller than an existing model")

    b = batch["thickness"].shape[0]
    if lmax == batch["thickness"].shape[1] and n_layers is None:
        return dict(batch)

    out = {}
    for key in ("thickness", "vp", "vs", "rho"):
        src = batch[key]
        dst = np.zeros((b, lmax), dtype=np.float64)
        # Real layers (all but the half-space), then padding, then half-space.
        n_real = src.shape[1] - 1
        dst[:, :n_real] = src[:, :n_real]
        dst[:, -1] = src[:, -1]                     # half-space stays last
        if lmax - 1 > n_real:                       # dummy layers inherit it
            dst[:, n_real:lmax - 1] = src[:, -1][:, None]
        out[key] = dst

    if "names" in batch:
        out["names"] = batch["names"]
    return out


def random_model_batch(
    n_models: int,
    n_layers: int,
    seed: int = 0,
    *,
    vs_min: float = 0.6,
    vs_max: float = 3.6,
    vp_vs: float = 1.73,
    rho_min: float = 1.8,
    rho_max: float = 2.9,
    h_min: float = 0.5,
    h_max: float = 6.0,
    allow_lvz: bool = True,
    lvz_prob: float = 0.25,
) -> dict[str, np.ndarray]:
    """Generate a reproducible batch of random layered models.

    The defaults include both monotone profiles and velocity inversions.

    Parameters
    ----------
    n_layers
        Total layers **including** the half-space, so the batch shape is
        ``[n_models, n_layers]``.
    allow_lvz
        If False, Vs is forced to be non-decreasing with depth.
    lvz_prob
        Probability that a given model is allowed to contain an inversion.
    """
    if n_layers < 2:
        raise ValueError("need at least one layer plus a half-space")
    rng = np.random.default_rng(seed)

    # Start from a monotone (normally dispersive) profile, then inject
    # inversions explicitly.  Drawing Vs uniformly and hoping for inversions
    # instead gives an uncontrolled rate -- with 5 layer pairs essentially
    # every model ends up inverted, which misrepresents the intended mix.
    vs = np.sort(rng.uniform(vs_min, vs_max, size=(n_models, n_layers)), axis=1)

    if allow_lvz and lvz_prob > 0:
        n_inverted = int(round(lvz_prob * n_models))
        if n_inverted:
            chosen = rng.choice(n_models, size=n_inverted, replace=False)
            for i in chosen:
                # One or two layers dropped below their neighbour.
                for _ in range(int(rng.integers(1, 3))):
                    j = int(rng.integers(1, n_layers - 1))
                    vs[i, j] = vs[i, j - 1] * rng.uniform(0.45, 0.8)

    vp = vs * vp_vs * rng.uniform(0.97, 1.03, size=(n_models, n_layers))
    rho = rng.uniform(rho_min, rho_max, size=(n_models, n_layers))
    h = rng.uniform(h_min, h_max, size=(n_models, n_layers))
    h[:, -1] = 0.0  # half-space

    return {
        "thickness": h.astype(np.float64),
        "vp": vp.astype(np.float64),
        "vs": vs.astype(np.float64),
        "rho": rho.astype(np.float64),
    }
