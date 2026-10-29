#!/Users/yusong/miniforge3/bin/python
"""Search presets, shape/fast evaluators, frozen fixtures and near-root regressions."""
import os
import sys
from pathlib import Path
import numpy as np
import pytest
import modisp as md
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
BACKEND=os.environ.get("MODISP_TEST_BACKEND","cpu")


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
    options = dict(backend=BACKEND, threads=4, accuracy=accuracy,search_strategy="qe_adaptive",secular="dunkin")
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


    with md.Solver(**dict(options,secular="fast")) as solver:
        fast=solver(f["thickness"],f["vp"],f["vs"],f["rho"],f["periods"],
                    nmodes=int(f["nmodes"]),wave=str(f["wave"]))
    np.testing.assert_array_equal(fast.mode_mask,out.mode_mask)
    np.testing.assert_array_equal(fast.root_count,out.root_count)
    np.testing.assert_allclose(fast.phase_velocity,out.phase_velocity,atol=2e-7,rtol=1e-9)


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
            expected=solver(**model,periods=periods,nmodes=nmodes,wave=wave)
        with md.Solver(backend=BACKEND,accuracy=accuracy,threads=2) as solver:
            actual=solver(**model,periods=periods,nmodes=nmodes,wave=wave)
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


@pytest.mark.parametrize('accuracy,tolerance',[('high',1e-3),('balanced',1e-2),('fast',5e-2)])
def test_shape_default_and_sampling_metadata(accuracy,tolerance):
    if BACKEND=='cuda':
        import modisp_cuda
        assert modisp_cuda.__version__==md.__version__
    with md.Solver(backend=BACKEND,accuracy=accuracy,threads=1) as solver:
        result=solver([.2,0.],[2.,5.],[.8,2.8],[2.,2.6],[.5,1.,2.],nmodes=3)
        assert solver.search_strategy=='shape_adaptive'
        assert result.search_strategy=='shape_adaptive'
        assert result.sampling_rtol==tolerance
        assert solver.stats()['sampling_rtol']==tolerance
        assert solver.stats()['shape_accepted']>0
        assert solver.stats()['workspace_bytes']<=solver.workspace_bytes
        assert (solver.stats()['shape_queue_reserved_bytes']>0)==(BACKEND=='cuda')
        assert solver.stats()['velocity_step_rtol'] is None
        if accuracy!='high':assert np.all(result.mode_index_uncertain)


@pytest.mark.parametrize('value',[0.,-1.,1.01,np.nan,np.inf,1j,[.01]])
def test_sampling_tolerance_validation(value):
    with pytest.raises(ValueError,match='sampling_rtol'):md.Solver(sampling_rtol=value)


def test_sampling_override_and_legacy_separation():
    with md.Solver(backend=BACKEND,accuracy='fast',sampling_rtol=.003) as solver:
        result=solver([0.],[6.],[3.5],[2.7],[1.],nmodes=1)
        assert result.sampling_rtol==.003
        assert solver.c_tol_rel==1e-4
        assert solver.close_root_rtol==.005
    with pytest.raises(ValueError,match='sampling_rtol'):
        md.Solver(search_strategy='qe_adaptive',sampling_rtol=.01)


@pytest.mark.parametrize('wave',[0,1])
@pytest.mark.parametrize('water',[False,True])
def test_search_feature_preserves_signed_original_function(wave,water):
    model=np.array([[1.,3.,1.5,2.],[0.,6.,3.5,2.7]])
    if water:model=np.vstack([[.1,1.5,0.,1.02],model])
    c=np.linspace(.3,3.49,173)
    original=md._cpu.secular(*model.T,1.3,c,wave)
    signed,feature=md._cpu.secular_shape(*model.T,1.3,c,wave)
    np.testing.assert_array_equal(original.view(np.uint64),signed.view(np.uint64))
    assert np.isfinite(feature).all()
    nonzero=signed!=0
    np.testing.assert_array_equal(np.signbit(signed[nonzero]),np.signbit(feature[nonzero]))


PAPER=np.array([[.01,1.55,.18,1.78],[.01,1.8,.35,1.85],
                [.02,1.,.12,1.5],[0.,2.,.6,1.93]])


@pytest.mark.parametrize('frequency,count,index,pair',[
    (56.4,28,12,[.17180457878813038,.17184656429811684]),
    (60.6,30,13,[.17180436959946785,.17181893982553648]),
    (90.,44,22,[.1812415672074288,.18124177425326074]),
    (92.5,45,29,[.21422351364294456,.21422374155533766]),
    (98.3,48,22,[.17180423024080432,.1718666665002732]),
])
def test_high_resolves_paper_kissing_pairs(frequency,count,index,pair):
    with md.Solver(backend=BACKEND,accuracy='high',threads=1,workspace_bytes=1024**3) as solver:
        result=solver(*PAPER.T,[1/frequency],nmodes=64)
    roots=result.phase_velocity[0,:,0]
    assert np.count_nonzero(np.isfinite(roots))==count
    np.testing.assert_allclose(roots[index:index+2],pair,rtol=1e-7,atol=3e-8)
    assert roots[index+1]>roots[index]


def test_fast_retains_observably_separate_love_pair():
    # The roots are 0.575% apart, outside the approved 0.5% ambiguity scale.
    pair=np.array([.296497864967473,.2982134013475273])
    assert np.diff(pair)[0]>.005*pair[1]
    with md.Solver(backend=BACKEND,accuracy='fast',threads=1) as solver:
        result=solver(*PAPER.T,[1/71.6],nmodes=64,wave='love')
    roots=result.phase_velocity[0,:,0]
    for c in pair:assert np.any(np.isfinite(roots)&(abs(roots-c)<=2e-4*c))


def test_fast_retains_observably_separate_rayleigh_pair():
    batch=md.random_model_batch(160,6,seed=3491,lvz_prob=1.,h_min=.05,h_max=8.)
    model={k:v[138] for k,v in batch.items()}
    pair=np.array([.92300532,.92796816])
    assert np.diff(pair)[0]>.005*pair[1]
    with md.Solver(backend=BACKEND,accuracy='fast',cmin=.1,threads=1) as solver:
        result=solver(**model,periods=[6.046013057825198],nmodes=5)
    roots=result.phase_velocity[0,:,0]
    for c in pair:assert np.any(np.isfinite(roots)&(abs(roots-c)<=2e-4*c))


@pytest.mark.parametrize('accuracy',['balanced','fast'])
def test_sampling_finer_and_coarser_keep_analytic_root(accuracy):
    reference=md.dispersion(np.array([[0.,6.,3.5,2.7]]),[.1,1.,10.],
                            accuracy='high',backend=BACKEND,nmodes=1).phase_velocity
    for tolerance in [1e-4,1e-3,1e-2,5e-2]:
        out=md.dispersion(np.array([[0.,6.,3.5,2.7]]),[.1,1.,10.],accuracy=accuracy,
                          backend=BACKEND,nmodes=1,sampling_rtol=tolerance)
        np.testing.assert_allclose(out.phase_velocity,reference,rtol=2e-4,atol=1e-6)


@pytest.mark.parametrize('layers',[70,128,210,227,256])
@pytest.mark.parametrize('wave',['rayleigh','love'])
def test_shape_padding_across_shared_memory_boundaries(layers,wave):
    padded=np.tile(PAPER[-1],(layers,1));padded[:,0]=0.;padded[:3]=PAPER[:3]
    for accuracy in ['high','balanced','fast']:
        with md.Solver(backend=BACKEND,accuracy=accuracy,threads=1) as solver:
            small=solver(*PAPER.T,[1/42.,1/71.6],nmodes=5,wave=wave)
            large=solver(*padded.T,[1/42.,1/71.6],nmodes=5,wave=wave)
        np.testing.assert_allclose(large.phase_velocity,small.phase_velocity,rtol=2e-6,atol=1e-8,equal_nan=True)
        np.testing.assert_array_equal(large.status,small.status)


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


@pytest.mark.parametrize("options", [{"normalize": False}, {"precision": "float32"}])
def test_fast_precision_probe_rejects_unsupported_options(options):
    from modisp.native import secular_grid
    with pytest.raises(ValueError, match="normalized float64"):
        secular_grid([1., 0.], [3., 6.], [1.5, 3.5], [2., 2.7],
                     [1.], .2, 3., 100, method="fast", **options)
