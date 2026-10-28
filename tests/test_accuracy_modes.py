"""Accuracy presets: default balanced, high bitwise baseline, fast opt-in."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import modisp as md

BACKEND = os.environ.get("MODISP_TEST_BACKEND", "cpu")
FIELDS = ("phase_velocity", "mode_mask", "status", "trace_flags",
          "search_level", "root_count", "search_bounds")
MODEL_FIELDS = ("thickness", "vp", "vs", "rho")


def assert_same(a, b):
    for name in FIELDS:
        x, y = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        assert x.shape == y.shape, name
        if x.dtype.kind == "f":
            np.testing.assert_array_equal(x.view(np.uint64), y.view(np.uint64))
        else:
            np.testing.assert_array_equal(x, y)


def fixture_dir():
    root = Path(__file__).parent / "data"
    if BACKEND == "cuda":
        directory = root / "accuracy_high_cuda"
    elif sys.platform.startswith("linux") and (root / "accuracy_high_cpu_linux").is_dir():
        directory = root / "accuracy_high_cpu_linux"
    else:
        directory = root / "accuracy_high_cpu"
    if not directory.is_dir():
        pytest.skip(f"no 0.2.1 high fixture for backend {BACKEND} on {sys.platform}")
    return directory


def test_default_is_balanced_and_high_explicit():
    with md.Solver(backend=BACKEND, threads=1) as solver:
        assert solver.accuracy == "balanced"
        out = solver([0.], [6.], [3.5], [2.7], [1.], nmodes=1)
    assert out.accuracy == "balanced"
    with md.Solver(backend=BACKEND, threads=1, accuracy="high") as solver:
        high = solver([0.], [6.], [3.5], [2.7], [1.], nmodes=1)
    assert high.accuracy == "high"


def _fixture_paths():
    directory = fixture_dir()
    return sorted(p for p in directory.glob("*.npz") if not p.name.startswith("._"))


@pytest.mark.parametrize("path", _fixture_paths(), ids=lambda p: Path(p).stem)
def test_high_matches_0_2_1_bitwise(path):
    f = np.load(path, allow_pickle=True)
    options = dict(backend=BACKEND, threads=4, accuracy="high")
    if BACKEND == "cuda":
        options["workspace_bytes"] = 1024 ** 3
    if np.isfinite(f["cmin"]):
        options["cmin"] = float(f["cmin"])
    if np.isfinite(f["cmax"]):
        options["cmax"] = float(f["cmax"])
    with md.Solver(**options) as solver:
        out = solver(f["thickness"], f["vp"], f["vs"], f["rho"],
                     f["periods"], nmodes=int(f["nmodes"]), wave=str(f["wave"]))
    for name in FIELDS:
        reference = np.asarray(f[name])
        actual = np.asarray(getattr(out, name))
        assert actual.shape == reference.shape, name
        if actual.dtype.kind == "f":
            np.testing.assert_array_equal(actual.view(np.uint64), reference.view(np.uint64))
        else:
            np.testing.assert_array_equal(actual, reference)


def _cases():
    water = dict(thickness=[.025, 0.], vp=[1.5, 6.], vs=[0., 3.5], rho=[1.02, 2.7])
    random = md.random_model_batch(3, 8, seed=20261028)
    return [
        ("random8", random, "rayleigh", 3, [.2, .63, 1., 2., 10.]),
        ("m04_lvz", {k: np.asarray(getattr(next(m for m in md.CANONICAL_MODELS if m.name == "m04_lvz"), k))
                     for k in MODEL_FIELDS}, "rayleigh", 3, [.2, .63, 1., 2., 10.]),
        ("water", water, "rayleigh", 3, [.2, .63, 1., 2., 10.]),
        ("love", {k: np.asarray(getattr(next(m for m in md.CANONICAL_MODELS if m.name == "m02_two_layer"), k))
                  for k in MODEL_FIELDS}, "love", 3, [.2, .63, 1., 2., 10.]),
    ]


@pytest.mark.parametrize("accuracy,max_error", [("balanced", 1e-5), ("fast", 1e-2)])
def test_presets_bound_error_and_do_not_lose_unresolved(accuracy, max_error):
    for name, model, wave, nmodes, periods in _cases():
        with md.Solver(backend=BACKEND, accuracy="high", threads=2) as s:
            expected = s(**model, periods=periods, nmodes=nmodes, wave=wave)
        with md.Solver(backend=BACKEND, accuracy=accuracy, threads=2) as s:
            actual = s(**model, periods=periods, nmodes=nmodes, wave=wave)
        new_unresolved = (((actual.trace_flags & md.TRACE_SEARCH_UNCONVERGED) != 0)
                          & ((expected.trace_flags & md.TRACE_SEARCH_UNCONVERGED) == 0))
        assert not np.any(new_unresolved), (name, accuracy)
        finite = np.isfinite(expected.phase_velocity) & np.isfinite(actual.phase_velocity)
        assert not np.any(np.isfinite(expected.phase_velocity) & ~np.isfinite(actual.phase_velocity)), name
        if finite.any():
            error = np.abs(expected.phase_velocity[finite] - actual.phase_velocity[finite])
            assert error.max() <= max_error, (name, accuracy, float(error.max()))


def test_period_threshold_and_stats():
    model = np.array([[.2, 2., .8, 2.], [0., 5., 2.8, 2.6]])
    periods = np.array([.5, 1., 2., 10.])
    with md.Solver(backend=BACKEND, accuracy="balanced", threads=2,
                   adaptive_period_threshold=2.) as solver:
        out = solver(*model.T, periods, nmodes=2)
        stats = solver.stats()
    assert stats["accuracy"] == "balanced"
    assert stats["period_threshold"] == 2.
    assert stats["sampling_capacity"] == 2048
    assert stats["early_accepts"] >= 0
    assert stats["verification_skipped"] >= 0
    assert set(np.unique(out.search_level)).issubset({-1, -4, 1, 2, 3, 4, 5})
    assert np.array_equal(out.numerical_flags,
                          out.trace_flags & np.uint32(0xFFFFFFFF ^ md.TRACE_RANGE_TRUNCATED))
    report = solver.kernel_report()
    assert report["accuracy"] == "balanced"
    with pytest.raises(ValueError):
        md.Solver(backend=BACKEND, accuracy="high", adaptive_period_threshold=1.)


def test_default_period_thresholds():
    model = np.array([[.2, 2., .8, 2.], [0., 5., 2.8, 2.6]])
    for accuracy, expected in (("high", 0.), ("balanced", 1.), ("fast", .63)):
        with md.Solver(backend=BACKEND, accuracy=accuracy, threads=1) as solver:
            solver(*model.T, [1.], nmodes=1)
            assert solver.stats()["period_threshold"] == expected
            assert solver.kernel_report()["period_threshold"] == expected


def test_invalid_and_uniform_accuracy():
    with pytest.raises(ValueError):
        md.Solver(backend=BACKEND, accuracy="medium")
    with pytest.raises(ValueError):
        md.Solver(backend=BACKEND, search_strategy="uniform", accuracy="balanced")
    with md.Solver(backend="cpu", search_strategy="uniform") as solver:
        assert solver.accuracy == "uniform"
        with pytest.raises(NotImplementedError):
            solver([0.], [6.], [3.5], [2.7], [1.])


def test_balanced_chunking_and_reuse_deterministic():
    batch = md.random_model_batch(5, 7, seed=654)
    periods = np.array([.4, 1., 2.])
    with md.Solver(backend=BACKEND, accuracy="balanced", threads=2) as one:
        first = one(**batch, periods=periods, nmodes=3)
    with md.Solver(backend=BACKEND, accuracy="balanced", threads=3, chunk_size=2) as many:
        chunked = many(**batch, periods=periods, nmodes=3)
    assert_same(first, chunked)

def test_functional_accuracy_conflict():
    model = np.array([[.2, 2., .8, 2.], [0., 5., 2.8, 2.6]])
    with md.Solver(backend=BACKEND, accuracy="high") as solver:
        with pytest.raises(ValueError):
            md.dispersion_batch(*model.T, [1.], nmodes=1, solver=solver, accuracy="balanced")
    a = md.dispersion_batch(*model.T, [1., 2.], nmodes=1, backend=BACKEND, accuracy="high")
    b = md.dispersion_batch(*model.T, [1., 2.], nmodes=1, backend=BACKEND, accuracy="balanced")
    assert a.accuracy == "high" and b.accuracy == "balanced"
    assert a.phase_velocity.shape == b.phase_velocity.shape
    md.release()
