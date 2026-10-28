import importlib
import os
from pathlib import Path
import sys
import numpy as np
import pytest
import modisp as md
from modisp.cli import main

BACKEND=os.environ.get('MODISP_TEST_BACKEND','cpu')
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


@pytest.mark.parametrize('path',sorted((Path(__file__).parent/'data').glob('B10000_*.npz')),ids=lambda x:x.stem)
def test_previously_unresolved_benchmark_cases(path):
    f=np.load(path)
    with md.Solver(backend=BACKEND,cmin=.2,accuracy='high') as s:
        out=s(**{k:f[k] for k in FIELDS},periods=f['periods'],nmodes=3)
    ref=f['reference'][0,:3];x=out.phase_velocity[0,:,0]
    if np.any(out.trace_flags&4):
        assert np.isnan(x).all()
        assert np.all(out.status==md.SolveStatus.UNRESOLVED)
    else:
        np.testing.assert_allclose(x,ref,atol=3e-6,rtol=2e-6)


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


def test_budget_and_failed_work_diagnostics():
    with md.Solver(backend=BACKEND,workspace_bytes=1,accuracy='high') as s:
        with pytest.raises(MemoryError,match='workspace'):s([0.],[6.],[3.5],[2.7],[1.])
    # Extreme optical thickness exceeds the explicit search workload ceiling.
    with md.Solver(backend=BACKEND,accuracy='high') as s:
        out=s([1e6,0.],[2.,6.],[.8,3.5],[2.,2.7],[.01],nmodes=2)
    assert np.all(out.trace_flags&md.TRACE_WORKSPACE_LIMIT)
    assert np.all(out.trace_flags&md.TRACE_SEARCH_UNCONVERGED)
    assert np.all(out.status==md.SolveStatus.UNRESOLVED)
    assert np.isnan(out.phase_velocity).all()
    assert s.last_result is out


def test_cli_npz(tmp_path,capsys):
    source=tmp_path/'input.npz';target=tmp_path/'output.npz'
    np.savez(source,thickness=[0.],vp=[6.],vs=[3.5],rho=[2.7],periods=[1.,2.])
    assert main(['doctor'])==0
    assert 'cpu_module' in capsys.readouterr().out
    assert main(['forward',str(source),'--output',str(target),'--backend',BACKEND,'--nmodes','1'])==0
    out=np.load(target)
    assert out['phase_velocity'].shape==(1,1,2)
    assert out['status'].dtype==np.uint8


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
    assert result.backend_used=='cuda'


def test_finite_unrepresentable_model_is_flagged():
    with md.Solver(backend=BACKEND,accuracy='high') as s:
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
def test_independent_witness_regressions(path):
    f=np.load(path)
    assert bool(f['reference_stable'])
    with md.Solver(backend=BACKEND,cmin=.2,accuracy='high') as s:
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
        np.testing.assert_allclose(out.phase_velocity[0,:,0],f['reference_hi'][:3],atol=3e-6,rtol=2e-6)
