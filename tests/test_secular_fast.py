#!/Users/yusong/miniforge3/bin/python
"""Pointwise and root-set contracts for the experimental secular='fast' path.

The fast recurrence is algebraically the same FP64 Dunkin compound matrix. It
is not required to be bitwise identical to the released path; roots must agree
one-to-one and the signed function must satisfy a strict globally scaled bound.
"""
import os
from pathlib import Path
import numpy as np
import pytest
import modisp as md

BACKEND = os.environ.get("MODISP_TEST_BACKEND", "cpu")


def _models():
    out = []
    for m in md.CANONICAL_MODELS:
        out.append((np.asarray(m.thickness, dtype=np.float64).ravel(),
                    np.asarray(m.vp, dtype=np.float64).ravel(),
                    np.asarray(m.vs, dtype=np.float64).ravel(),
                    np.asarray(m.rho, dtype=np.float64).ravel()))
    # surface acoustic water layer over a solid stack
    out.append((np.array([.2, 1., 0.]), np.array([1.5, 3., 6.]),
                np.array([0., 1.5, 3.5]), np.array([1.02, 2., 2.7])))
    return out


def _crossings(c, f):
    i = np.flatnonzero(np.signbit(f[:-1]) != np.signbit(f[1:]))
    return c[i]


@pytest.mark.parametrize("model_index", range(len(_models())))
def test_pointwise_and_fixed_node_roots(model_index):
    h, vp, vs, rho = _models()[model_index]
    period = 1.0
    c = np.linspace(max(.05, .1 * vs[-1]), np.nextafter(vs[-1], 0.), 4097)
    f0 = np.asarray(md._cpu.secular(h, vp, vs, rho, period, c, 0, 0))
    f1 = np.asarray(md._cpu.secular(h, vp, vs, rho, period, c, 0, 1))
    assert np.array_equal(np.isfinite(f0), np.isfinite(f1))
    scale = max(1.0, float(np.max(np.abs(f0))))
    assert np.max(np.abs(f0 - f1)) <= 2e-10 * scale
    # Away from a zero, the two functions must have the same sign.
    away = np.abs(f0) > 1e-9 * scale
    assert np.array_equal(np.signbit(f0[away]), np.signbit(f1[away]))
    r0, r1 = _crossings(c, f0), _crossings(c, f1)
    assert r0.size == r1.size
    np.testing.assert_allclose(r1, r0, atol=2e-10 * max(1.0, float(np.max(c))), rtol=0)


def test_shape_feature_and_love_are_stable():
    h, vp, vs, rho = _models()[-1]
    c = np.linspace(.2, np.nextafter(vs[-1], 0.), 2049)
    f0, g0 = md._cpu.secular_shape(h, vp, vs, rho, 1.0, c, 0, 0)
    f1, g1 = md._cpu.secular_shape(h, vp, vs, rho, 1.0, c, 0, 1)
    np.testing.assert_allclose(f1, f0, atol=2e-10 * max(1.0, np.max(np.abs(f0))), rtol=0)
    np.testing.assert_allclose(g1, g0, atol=2e-9, rtol=0)
    with md.Solver(backend=BACKEND, secular="fast", accuracy="high") as s:
        fast = s(h, vp, vs, rho, [.5, 1., 2.], nmodes=2, wave="love")
    with md.Solver(backend=BACKEND, secular="dunkin", accuracy="high") as s:
        slow = s(h, vp, vs, rho, [.5, 1., 2.], nmodes=2, wave="love")
    np.testing.assert_allclose(fast.phase_velocity, slow.phase_velocity, atol=0, rtol=0)
    assert fast.secular == "fast"
    assert s.stats()["secular_kernel"] == "sh"
    assert s.kernel_report()["secular_kernel"] == "sh"


@pytest.mark.parametrize("accuracy", ["high", "balanced", "fast"])
@pytest.mark.parametrize("strategy", ["shape_adaptive", "qe_adaptive"])
def test_solver_root_sets_match(accuracy, strategy):
    periods = np.logspace(-1, 1, 12)
    for h, vp, vs, rho in _models():
        with md.Solver(backend=BACKEND, secular="dunkin", accuracy=accuracy, threads=2, search_strategy=strategy) as s:
            slow = s(h, vp, vs, rho, periods, nmodes=3)
        with md.Solver(backend=BACKEND, secular="fast", accuracy=accuracy, threads=2, search_strategy=strategy) as s:
            fast = s(h, vp, vs, rho, periods, nmodes=3)
        assert np.array_equal(np.isfinite(slow.phase_velocity), np.isfinite(fast.phase_velocity))
        assert np.array_equal(slow.root_count, fast.root_count)
        both = np.isfinite(slow.phase_velocity) & np.isfinite(fast.phase_velocity)
        if both.any():
            np.testing.assert_allclose(fast.phase_velocity[both], slow.phase_velocity[both],
                                       atol=2e-7, rtol=1e-9)
        assert fast.secular == "fast"
        assert s.stats()["secular"] == "fast"
        assert s.kernel_report()["secular_kernel"] == "dunkin_fast"


def test_invalid_secular_and_legacy_uniform():
    with pytest.raises(ValueError, match="secular"):
        md.Solver(secular="other")
    with pytest.raises(NotImplementedError, match="uniform"):
        with md.Solver(secular="fast", search_strategy="uniform", backend="cpu") as s:
            s([1., 0.], [3., 6.], [1.5, 3.5], [2., 2.7], [1.], nmodes=1)

def _fixture_paths():
    directory = Path(__file__).parent / "data" / "accuracy_high_cpu"
    return sorted(directory.glob("*.npz")) if directory.is_dir() else []


@pytest.mark.parametrize("path", _fixture_paths(), ids=lambda p: Path(p).stem)
def test_fast_matches_dunkin_on_frozen_high_inputs(path):
    f = np.load(path, allow_pickle=True)
    options = dict(backend=BACKEND, threads=4, accuracy="high",
                   search_strategy="qe_adaptive")
    if BACKEND == "cuda":
        options["workspace_bytes"] = 1024 ** 3
    if np.isfinite(f["cmin"]):
        options["cmin"] = float(f["cmin"])
    if np.isfinite(f["cmax"]):
        options["cmax"] = float(f["cmax"])
    kwargs = dict(periods=f["periods"], nmodes=int(f["nmodes"]), wave=str(f["wave"]))
    with md.Solver(secular="dunkin", **options) as slow:
        expected = slow(f["thickness"], f["vp"], f["vs"], f["rho"], **kwargs)
    with md.Solver(secular="fast", **options) as fast:
        actual = fast(f["thickness"], f["vp"], f["vs"], f["rho"], **kwargs)
    assert np.array_equal(np.isfinite(expected.phase_velocity),
                          np.isfinite(actual.phase_velocity))
    assert np.array_equal(expected.root_count, actual.root_count)
    both = np.isfinite(expected.phase_velocity) & np.isfinite(actual.phase_velocity)
    if both.any():
        np.testing.assert_allclose(actual.phase_velocity[both],
                                   expected.phase_velocity[both],
                                   atol=2e-7, rtol=1e-9)


@pytest.mark.parametrize("options", [{"normalize": False}, {"precision": "float32"}])
def test_fast_precision_probe_rejects_unsupported_options(options):
    from modisp.native import secular_grid
    with pytest.raises(ValueError, match="normalized float64"):
        secular_grid([1., 0.], [3., 6.], [1.5, 3.5], [2., 2.7],
                     [1.], .2, 3., 100, method="fast", **options)


def test_benchmark_rejects_crossing_and_finite_mismatches():
    import importlib.util
    tool = Path(__file__).resolve().parents[1] / "tools" / "benchmark_secular.py"
    if not tool.exists():
        pytest.skip("source-only benchmark driver is not installed in runtime wheel")
    spec = importlib.util.spec_from_file_location("secular_benchmark", tool)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    c = np.array([1., 2., 3.])
    f = np.array([-1., 1., 1.])
    good = driver.compare_pointwise(c, f, f.copy())
    assert good["accepted"] and good["max_crossing_node_difference"] == 0.
    missing = driver.compare_pointwise(c, f, np.ones(3))
    assert not missing["accepted"] and not missing["crossing_count_equal"]
    assert missing["max_crossing_node_difference"] is None
    invalid = driver.compare_pointwise(c, f, np.array([-1., np.nan, 1.]))
    assert not invalid["accepted"] and not invalid["finite_mask_equal"]
    invalid = driver.compare_pointwise(c, np.full(3, np.nan), np.full(3, np.nan))
    assert not invalid["accepted"] and invalid["max_abs_difference"] is None
