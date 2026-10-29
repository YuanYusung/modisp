#!/Users/yusong/miniforge3/bin/python
"""Public API, backend selection, workspace ownership and concurrency."""
import concurrent.futures
import importlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest
import modisp as md
from modisp.cli import main
BACKEND=os.environ.get("MODISP_TEST_BACKEND","cpu")


FIELDS=('thickness','vp','vs','rho')


def test_zero_padding_periods_threads_chunks_and_reuse():
    batch=md.random_model_batch(7,8,seed=666);per=np.array([10.,1.,3.,1.])
    with md.Solver(backend=BACKEND,threads=1,chunk_size=1) as s:
        one=s(**batch,periods=per,nmodes=4)
        assert s.stats()['chunks']==7
        for threads,chunk in [(4,3),(8,7)]:
            with md.Solver(backend=BACKEND,threads=threads,chunk_size=chunk) as q:
                many=q(**batch,periods=per,nmodes=4)
            np.testing.assert_array_equal(one.phase_velocity,many.phase_velocity)
            np.testing.assert_array_equal(one.trace_flags,many.trace_flags)
            np.testing.assert_array_equal(one.search_bounds,many.search_bounds)
        different=s(**{k:v[:2] for k,v in batch.items()},periods=[.5,2.],nmodes=2,wave='love')
        again=s(**batch,periods=per,nmodes=4)
        np.testing.assert_array_equal(one.phase_velocity,again.phase_velocity)
        assert different.backend_used==BACKEND
        assert np.all(one.status[one.mode_mask]==md.SolveStatus.VALID)
    # zero-thickness duplicate material padding leaves the physical model intact
    model=[np.array([.2,0.]),np.array([2.,5.]),np.array([.8,2.8]),np.array([2.,2.6])]
    padded=[np.r_[x[0],x[0],x[1]] if i else np.array([.2,0.,0.]) for i,x in enumerate(model)]
    with md.Solver(backend=BACKEND) as s:
        a=s(*model,periods=[.2,1.,4.],nmodes=3)
        b=s(*padded,periods=[.2,1.,4.],nmodes=3)
    np.testing.assert_allclose(a.phase_velocity,b.phase_velocity,atol=3e-6,rtol=2e-6)


def test_mixed_dry_wet_batch_and_explicit_ranges():
    h=np.array([[0.,.2,0.],[.1,.2,0.]])
    a=np.array([[1.5,2.,5.],[1.5,2.,5.]])
    b=np.array([[0.,.8,2.8],[0.,.8,2.8]])
    r=np.array([[1.02,2.,2.6],[1.02,2.,2.6]])
    with md.Solver(backend=BACKEND) as s:
        both=s(h,a,b,r,[.1,1.],nmodes=3)
        for i in range(2):
            one=s(h[i],a[i],b[i],r[i],[.1,1.],nmodes=3)
            np.testing.assert_array_equal(one.phase_velocity[0],both.phase_velocity[i])
    with md.Solver(backend=BACKEND,cmin=.5,cmax=1.) as s:
        cropped=s(h,a,b,r,[.1,1.],nmodes=3)
    assert np.all(cropped.trace_flags&md.TRACE_RANGE_TRUNCATED)
    assert np.nanmin(cropped.phase_velocity)>=.5
    assert np.nanmax(cropped.phase_velocity)<=1.


def assert_tolerant(out, reference, accuracy):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
    from tolerant_metrics import compare
    metrics=compare(out.phase_velocity,np.asarray(reference)[None,:,None],out.trace_flags,
                    close_rtol=.005,error_rtol=2e-5 if accuracy=="balanced" else 2e-4)
    assert metrics["accepted"],metrics


@pytest.mark.parametrize('path',sorted((Path(__file__).parent/'data').glob('B10000_*.npz')),ids=lambda x:x.stem)
@pytest.mark.parametrize('accuracy',['high','balanced','fast'])
def test_previously_unresolved_benchmark_cases(path,accuracy):
    f=np.load(path)
    with md.Solver(backend=BACKEND,cmin=.2,accuracy=accuracy) as s:
        out=s(**{k:f[k] for k in FIELDS},periods=f['periods'],nmodes=3)
    ref=f['reference'][0,:3];x=out.phase_velocity[0,:,0]
    if np.any(out.trace_flags&4):
        assert np.isnan(x).all()
        assert np.all(out.status==md.SolveStatus.UNRESOLVED)
    else:
        if accuracy=="high":np.testing.assert_allclose(x,ref,atol=3e-6,rtol=2e-6)
        else:assert_tolerant(out,f["reference"].ravel(),accuracy)


def test_unavailable_and_version_checks(monkeypatch):
    from modisp import _loader
    def unavailable():raise _loader.BackendUnavailableError('test unavailable')
    monkeypatch.setattr(md,'_load_cuda',unavailable)
    with md.Solver(backend='auto') as s:
        out=s([0.],[6.],[3.5],[2.7],[1.],nmodes=1)
        assert out.backend_used=='cpu' and 'unavailable' in out.backend_reason
    with md.Solver(backend='cuda') as s:
        with pytest.raises(_loader.BackendUnavailableError):s([0.],[6.],[3.5],[2.7],[1.])
    class Wrong:
        VERSION='99';ABI_VERSION=1
    class Package:
        @staticmethod
        def load():return Wrong()
    monkeypatch.setitem(sys.modules,'modisp_cuda',Package())
    with pytest.raises(RuntimeError,match='versions differ'):_loader.load()


@pytest.mark.parametrize('accuracy',['high','balanced','fast'])
def test_budget_and_failed_work_diagnostics(accuracy):
    with md.Solver(backend=BACKEND,workspace_bytes=1,accuracy=accuracy) as s:
        with pytest.raises(MemoryError,match='workspace'):s([0.],[6.],[3.5],[2.7],[1.])
    # Extreme optical thickness exceeds the explicit search workload ceiling.
    with md.Solver(backend=BACKEND,accuracy=accuracy) as s:
        out=s([1e6,0.],[2.,6.],[.8,3.5],[2.,2.7],[.01],nmodes=2)
    if accuracy=='high':
        assert np.all(out.trace_flags&md.TRACE_WORKSPACE_LIMIT)
        assert np.all(out.trace_flags&md.TRACE_SEARCH_UNCONVERGED)
        assert np.all(out.status==md.SolveStatus.UNRESOLVED)
        assert np.isnan(out.phase_velocity).all()
    else:
        # Observation-scale scans have bounded work even for extremely dense
        # clusters; their recovered ranks are explicitly uncertain.
        assert np.all(out.trace_flags&md.TRACE_APPROXIMATE_SEARCH)
        assert np.all(out.mode_index_uncertain)
    assert s.last_result is out


@pytest.mark.parametrize("secular", [None, "dunkin", "fast"])
def test_cli_npz(tmp_path,capsys,secular):
    source=tmp_path/'input.npz';target=tmp_path/'output.npz'
    np.savez(source,thickness=[0.],vp=[6.],vs=[3.5],rho=[2.7],periods=[1.,2.])
    assert main(['doctor'])==0
    doctor=json.loads(capsys.readouterr().out)
    assert doctor['default_secular_by_backend']=={'cpu':'dunkin','cuda':'fast','uniform':'dunkin'}
    assert doctor['default_secular']==doctor['default_secular_by_backend'].get(doctor['auto_backend'])
    command=['forward',str(source),'--output',str(target),'--backend',BACKEND,
             '--nmodes','1','--sampling-rtol','.003']
    if secular is not None:command.extend(['--secular',secular])
    assert main(command)==0
    out=np.load(target)
    assert out['phase_velocity'].shape==(1,1,2)
    assert out['status'].dtype==np.uint8
    assert str(out['secular'])==(secular or ("fast" if BACKEND=="cuda" else "dunkin"))
    assert str(out['search_strategy'])=="shape_adaptive"
    assert float(out['sampling_rtol'])==.003


def test_functional_conveniences_and_input_validation():
    for name in ('PhaseDispersion','DispersionCurve','DispersionError'):
        assert not hasattr(md,name)
    model=np.array([[.2,2.,.8,2.],[0.,5.,2.8,2.6]])
    one=md.dispersion(model,[1.,2.],nmodes=2,backend=BACKEND)
    other=md.dispersion_batch(*model.T,[1.,2.],nmodes=2,backend=BACKEND)
    with md.Solver(backend=BACKEND) as solver:
        third=solver(*model.T,[1.,2.],nmodes=2)
    assert all(isinstance(out,md.DispersionBatch) for out in (one,other,third))
    np.testing.assert_array_equal(one.phase_velocity,other.phase_velocity)
    np.testing.assert_array_equal(one.phase_velocity,third.phase_velocity)
    with pytest.raises(ValueError):md.Solver(cmin=2,cmax=1)
    with pytest.raises(ValueError):md.Solver(threads=0)
    with pytest.raises(ValueError):md.dispersion_batch(*model.T,[1.],search_strategy='',backend=BACKEND)
    with md.Solver(backend='cpu',search_strategy='uniform') as s:
        with pytest.raises(NotImplementedError):s(*model.T,[1.])
    md.release()


@pytest.mark.skipif(BACKEND!='cuda',reason='legacy uniform is CUDA-only')
def test_uniform_legacy_facade_compatibility():
    from modisp._legacy import Solver as Legacy
    batch=md.random_model_batch(3,6,seed=20261027);per=[1.,3.,10.]
    with Legacy(search_strategy='uniform',cmin=.2) as old:
        expected=old(**batch,periods=per,nmodes=3)
    with md.Solver(backend='cuda',search_strategy='uniform',cmin=.2) as new:
        result=new(**batch,periods=per,nmodes=3)
    np.testing.assert_array_equal(result.phase_velocity,expected.phase_velocity)
    np.testing.assert_array_equal(result.trace_flags&255,expected.trace_flags)
    assert result.backend_used=='cuda' and result.secular=='dunkin'


@pytest.mark.parametrize('accuracy',['high','balanced','fast'])
def test_finite_unrepresentable_model_is_flagged(accuracy):
    with md.Solver(backend=BACKEND,accuracy=accuracy) as s:
        out=s([1e200,0.],[1e200,2e200],[5e199,1e200],[1e200,1e200],[1.],nmodes=3)
    assert np.isnan(out.phase_velocity).all()
    assert out.trace_flags[0,0]&md.TRACE_SEARCH_UNCONVERGED
    assert out.trace_flags[0,0]&md.TRACE_NONFINITE


@pytest.mark.parametrize('wave',['rayleigh','love'])
def test_empty_cropped_window_has_explicit_state(wave):
    with md.Solver(backend=BACKEND,cmin=4.,cmax=5.) as s:
        out=s([0.],[6.],[3.5],[2.7],[1.,2.],nmodes=2,wave=wave)
    assert np.isnan(out.phase_velocity).all()
    assert np.all(out.status==md.SolveStatus.NOT_FOUND)
    assert np.all(out.search_level==-4)
    assert np.all(out.root_count==0)


@pytest.mark.parametrize('path',sorted((Path(__file__).parent/'data').glob('witness_*.npz')),ids=lambda x:x.stem)
@pytest.mark.parametrize('accuracy',['high','balanced','fast'])
@pytest.mark.parametrize('cmin',[None,.2])
def test_independent_witness_regressions(path,accuracy,cmin):
    f=np.load(path)
    assert bool(f['reference_stable'])
    with md.Solver(backend=BACKEND,cmin=cmin,accuracy=accuracy) as s:
        out=s(**{k:f[k] for k in FIELDS},periods=f['periods'],nmodes=3)
    # 0.2.0 resolved these seven witnesses on both CPU and CUDA. Preserve that
    # acceptance boundary: returning all UNRESOLVED must not pass a regression.
    # model9384 remains an explicitly retained numerical limitation.
    if path.stem!='witness_model9384_period11':
        assert not np.any(out.trace_flags&md.TRACE_SEARCH_UNCONVERGED)
    if np.any(out.trace_flags&md.TRACE_SEARCH_UNCONVERGED):
        assert np.isnan(out.phase_velocity).all()
        assert np.all(out.status==md.SolveStatus.UNRESOLVED)
    else:
        if accuracy=='high':np.testing.assert_allclose(out.phase_velocity[0,:,0],f['reference_hi'][:3],atol=3e-6,rtol=2e-6)
        else:assert_tolerant(out,f['reference_hi'],accuracy)


ATTRS = ("phase_velocity", "mode_mask", "status", "trace_flags", "search_level",
         "root_count", "search_bounds")


def same(a, b, selection=slice(None)):
    for name in ATTRS:
        np.testing.assert_array_equal(getattr(a, name)[selection], getattr(b, name))


def test_reused_capacity_mutated_inputs_and_profile():
    batch = md.random_model_batch(9, 6, seed=20261028)
    periods = np.array([2., .8, 2., 1.4])
    with md.Solver(backend=BACKEND, threads=4) as solver:
        first = solver(**batch, periods=periods, nmodes=3)
        assert solver.stats()["engine_allocations"] == 1
        assert "timings_s" not in solver.stats()
        tail = solver(**{k: v[:2] for k, v in batch.items()}, periods=periods, nmodes=3)
        assert solver.stats()["engine_allocations"] == 0
        assert solver.stats()["capacity_models"] == 9
        same(first, tail, slice(0, 2))
        # Both model and period objects keep their addresses but change values.
        for key in ("vp", "vs"):
            batch[key] *= 1.03
        periods[:] = [1., 3., 1., .5]
        solver.profile = True
        changed = solver(**batch, periods=periods, nmodes=3)
        stats = solver.stats()
        assert stats["timings_s"]["solver_total"] > 0
        assert stats["cache_hits"] >= 0
        stats["timings_s"]["solver_total"] = -1
        assert solver.stats()["timings_s"]["solver_total"] > 0
        with md.Solver(backend=BACKEND, threads=1) as fresh:
            expected = fresh(**batch, periods=periods, nmodes=3)
        same(changed, expected)
        assert np.any(first.phase_velocity != changed.phase_velocity)
        solver.close()
        again = solver(**batch, periods=periods, nmodes=3)
        same(again, expected)


def test_concurrent_solvers_and_result_ownership():
    batch = md.random_model_batch(3, 5, seed=876)
    def work(profile):
        with md.Solver(backend=BACKEND, threads=2, profile=profile) as s:
            return s(**batch, periods=[.5, 1., 2.], nmodes=3)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(work, [True, False]))
    same(a, b)
    # A subsequent call/close must not mutate already returned arrays.
    snapshot = a.phase_velocity.copy()
    work(False)
    np.testing.assert_array_equal(snapshot, a.phase_velocity)


@pytest.mark.skipif(BACKEND != "cuda", reason="CUDA launch variants require CUDA")
@pytest.mark.parametrize("block", ["32", "64", "128", "256"])
def test_cuda_launch_profile_and_tail(monkeypatch, block):
    batch = md.random_model_batch(5, 7, seed=545)
    with md.Solver(backend="cuda") as s:
        expected = s(**batch, periods=[.7, 2., .7], nmodes=3)
    monkeypatch.setenv("MODISP_CUDA_BLOCK", block)
    with md.Solver(backend="cuda") as s:
        for profile in (False, False, True, False):
            s.profile = profile
            same(expected, s(**batch, periods=[.7, 2., .7], nmodes=3))
        tail = s(**{k: v[:2] for k, v in batch.items()}, periods=[.7, 2., .7], nmodes=3)
        same(expected, tail, slice(0, 2))
        assert s.stats()["engine_allocations"] == 0


def test_maximum_layer_count_shared_memory():
    h=np.zeros(256);vp=np.full(256,6.);vs=np.full(256,3.5);rho=np.full(256,2.7)
    with md.Solver(backend=BACKEND) as s:
        many=s(h,vp,vs,rho,[1.],nmodes=1)
        one=s([0.],[6.],[3.5],[2.7],[1.],nmodes=1)
    assert not np.any(many.trace_flags & md.TRACE_SEARCH_UNCONVERGED)
    np.testing.assert_allclose(many.phase_velocity,one.phase_velocity,atol=3e-6,rtol=2e-6)


def test_backend_secular_defaults_overrides_and_auto_fallback(monkeypatch):
    model=([.2,0.],[2.,5.],[.8,2.8],[2.,2.6])
    def unavailable():raise md.BackendUnavailableError("test unavailable")
    if BACKEND=="cpu":monkeypatch.setattr(md,"_load_cuda",unavailable)
    with md.Solver(backend=BACKEND) as solver:
        default=solver(*model,[.5,1.],nmodes=2)
        expected="fast" if BACKEND=="cuda" else "dunkin"
        assert default.secular==solver.stats()["secular"]==solver.kernel_report()["secular"]==expected
    with md.Solver(backend=BACKEND,secular=expected) as solver:
        explicit=solver(*model,[.5,1.],nmodes=2)
    np.testing.assert_array_equal(default.phase_velocity,explicit.phase_velocity)
    with md.Solver(backend="auto") as solver:
        selected=solver(*model,[1.],nmodes=1)
        assert selected.secular==("fast" if selected.backend_used=="cuda" else "dunkin")
        monkeypatch.setattr(md,"_load_cuda",unavailable)
        fallback=solver(*model,[1.],nmodes=1)
        assert fallback.backend_used=="cpu" and fallback.secular=="dunkin"
        assert solver.stats()["secular"]==solver.kernel_report()["secular"]=="dunkin"
