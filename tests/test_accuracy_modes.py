"""Accuracy presets: default balanced, high bitwise baseline, fast opt-in."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import modisp as md
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))

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
@pytest.mark.parametrize("accuracy", ["high"])
def test_presets_match_0_2_1_bitwise(path, accuracy):
    f = np.load(path, allow_pickle=True)
    options = dict(backend=BACKEND, threads=4, accuracy=accuracy,search_strategy="qe_adaptive")
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


@pytest.mark.parametrize("accuracy,error_budget", [("balanced",2e-5),("fast",2e-4)])
def test_presets_ordinary_roots_and_flagged_clusters(accuracy,error_budget):
    for name,model,wave,nmodes,periods in _cases():
        with md.Solver(backend=BACKEND,accuracy="high",threads=2) as solver:
            expected=solver(**model,periods=periods,nmodes=nmodes)
        with md.Solver(backend=BACKEND,accuracy=accuracy,threads=2) as solver:
            actual=solver(**model,periods=periods,nmodes=nmodes)
        from tolerant_metrics import compare
        result=compare(actual.phase_velocity,expected.phase_velocity,actual.trace_flags,.005,error_budget,expected.trace_flags)
        assert result['accepted'],(name,result)
        assert np.array_equal(actual.mode_index_uncertain,actual.approximate_search)


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
    assert stats["early_acceptance_enabled"] is False
    assert stats["approximate_search_enabled"] is True
    assert stats["early_accepts"] >= 0
    assert stats["verification_skipped"] >= 0
    assert set(np.unique(out.search_level)).issubset({-1, -4, 3, 5, 6, 7,8,9,10})
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


@pytest.mark.parametrize("accuracy",["balanced","fast"])
def test_tolerant_chunking_and_quality_flags(accuracy):
    batch=md.random_model_batch(12,8,seed=3490,lvz_prob=1.)
    periods=np.array([.1,.3,1.,2.,10.,.3])
    with md.Solver(backend=BACKEND,threads=2,accuracy=accuracy) as solver:
        expected=solver(**batch,periods=periods,nmodes=5)
    with md.Solver(backend=BACKEND,threads=2,accuracy=accuracy,chunk_size=5) as solver:
        actual=solver(**batch,periods=periods,nmodes=5)
        stats=solver.stats()
        assert stats['early_accepts']==0
        assert stats['strategy']=='shape_adaptive'
        assert stats['verification_skipped']>0
    assert_same(expected,actual)
    assert np.any(actual.approximate_search)
    assert np.all(actual.potentially_ambiguous[actual.approximate_search])


@pytest.mark.parametrize("accuracy", ["balanced", "fast"])
def test_exact_love_root_on_dense_tile_boundary(accuracy):
    # Construct F(1.5)=0 on the first 64-point tile boundary of the coarse
    # dense grid. Moving tile seams must not turn a valid root into NONFINITE.
    import math
    c, h = 1.5, 1000.
    k = 1./c
    q = math.sqrt((k+1.)*abs(k-1.))
    r = math.sqrt((k+.5)*abs(k-.5))
    z = -q*math.sin(q*h)
    rho = -(2.7*r)*math.cos(q*h)/(.25*z)
    upper = np.nextafter(2., 0.)
    lower = (c-upper*63/32768)/(1.-63/32768)
    model = ([h, 0.], [2., 4.], [1., 2.], [rho, 2.7])
    with md.Solver(backend=BACKEND, accuracy="high", threads=1, cmin=lower) as solver:
        expected = solver(*model, [2*math.pi], nmodes=3, wave="love")
    with md.Solver(backend=BACKEND, accuracy=accuracy, threads=1, cmin=lower) as solver:
        actual = solver(*model, [2*math.pi], nmodes=3, wave="love")
    assert not np.any(actual.trace_flags & md.TRACE_SEARCH_UNCONVERGED)
    np.testing.assert_allclose(actual.phase_velocity, expected.phase_velocity, atol=3e-5, rtol=2e-4)
    assert abs(actual.phase_velocity[0, 0, 0]-c) < (2e-5 if accuracy=="balanced" else 2e-4)*c


@pytest.mark.parametrize("rtol",[0.,-1.,1.,np.nan,np.inf,1j,[.005]])
def test_invalid_close_root_threshold(rtol):
    with pytest.raises(ValueError):md.Solver(close_root_rtol=rtol)


def test_tolerance_defaults_and_explicit_override():
    for preset,value in (("high",1e-6),("balanced",1e-5),("fast",1e-4)):
        with md.Solver(backend=BACKEND,accuracy=preset) as solver:
            assert solver.c_tol_rel==value
            assert solver.close_root_rtol==.005
        with md.Solver(backend=BACKEND,accuracy=preset,c_tol_rel=1e-8,close_root_rtol=.01) as solver:
            assert solver.c_tol_rel==1e-8
            assert solver.close_root_rtol==.01


def test_cluster_judge_does_not_waive_flagged_ordinary_errors():
    from tolerant_metrics import compare
    reference=np.array([[[1.],[1.003],[2.],[3.]]])
    flagged=np.array([[md.TRACE_APPROXIMATE_SEARCH]],dtype=np.uint32)
    allowed=compare(np.array([[[1.],[2.],[3.]]]),reference,flagged)
    assert allowed['accepted'] and allowed['mode_index_shifts']>0
    bad=compare(np.array([[[1.],[2.2],[3.]]]),reference,flagged)
    assert not bad['accepted'] and bad['ordinary_failures']>0
    unflagged=compare(np.array([[[1.],[2.],[3.]]]),reference,np.zeros_like(flagged))
    assert not unflagged['accepted'] and unflagged['unflagged_uncertainty']>0


@pytest.mark.parametrize("accuracy,factor",[("balanced",.5),("fast",1.)])
def test_observation_threshold_controls_grid_and_flags(accuracy,factor):
    with md.Solver(backend=BACKEND,accuracy=accuracy,close_root_rtol=.01,search_strategy="qe_adaptive") as solver:
        result=solver([.2,0.],[2.,5.],[.8,2.8],[2.,2.6],[.5,1.,2.],nmodes=3)
        assert solver.stats()['velocity_step_rtol']==.01*factor
    assert result.close_root_rtol==.01
    assert np.all(result.approximate_search)
    assert np.all(result.mode_index_uncertain)
    assert np.all(result.trace_flags&md.TRACE_APPROXIMATE_SEARCH)


@pytest.mark.parametrize("accuracy,budget",[("high",2e-6),("balanced",2e-5),("fast",2e-4)])
def test_isolated_love_roots_are_not_waived_as_close_clusters(accuracy,budget):
    # The old spacing heuristic missed the first two roots, separated by
    # 1.65%. Verify them against an independent unnormalised 2x2 SH recurrence.
    import math
    from scipy.optimize import brentq
    batch=md.random_model_batch(160,6,seed=3492,lvz_prob=1.,h_min=.05,h_max=8.)
    h,vp,vs,rho=[batch[k][98] for k in MODEL_FIELDS]
    period=.6682628855998671
    omega=2*math.pi/period
    def secular(c):
        k=omega/c
        traction=rho[-1]*vs[-1]**2*math.sqrt(k*k-(omega/vs[-1])**2)
        displacement=1.
        for j in range(len(h)-2,-1,-1):
            square=(omega/vs[j])**2-k*k
            q=math.sqrt(abs(square))
            if square>0:
                C=math.cos(q*h[j]);S=math.sin(q*h[j])/q;D=-q*math.sin(q*h[j])
            else:
                C=math.cosh(q*h[j]);S=math.sinh(q*h[j])/q;D=q*math.sinh(q*h[j])
            mu=rho[j]*vs[j]**2
            traction,displacement=traction*C+displacement*mu*D,traction*S/mu+displacement*C
        return traction
    guesses=[1.9943967252,2.0276803379,2.0535759900,2.0969734965,2.1495630704]
    reference=np.array([brentq(secular,c-.001,c+.001,xtol=1e-13) for c in guesses])
    assert np.all(np.diff(reference)>.005*reference[1:])
    with md.Solver(backend=BACKEND,accuracy=accuracy,threads=1,cmin=.1) as solver:
        out=solver(h,vp,vs,rho,[period],nmodes=5,wave='love')
    np.testing.assert_allclose(out.phase_velocity[0,:,0],reference,rtol=budget,atol=1e-7)
