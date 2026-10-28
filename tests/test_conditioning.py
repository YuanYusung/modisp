import os
import numpy as np
import pytest
from scipy.optimize import brentq
import modisp as md
from modisp import native
BACKEND=os.environ.get('MODISP_TEST_BACKEND','cpu')


def scan(model,period,normalize=True,nc=131073):
    fields=(model.thickness,model.vp,model.vs,model.rho)
    cs=np.linspace(.1,np.nextafter(model.vs[-1],0),nc)
    ys=native.secular_grid(*fields,[period],cs[0],cs[-1],nc,normalize=normalize)[0]
    ii=np.flatnonzero(np.signbit(ys[:-1])!=np.signbit(ys[1:]))
    def f(c):
        if normalize:return md._cpu.secular(*fields,period,np.array([c]),0)[0]
        # The public diagnostic grid requires two nodes; the first is c.
        return native.secular_grid(*fields,[period],c,np.nextafter(c,np.inf),2,normalize=False)[0,0]
    return np.array([brentq(f,cs[i],cs[i+1],xtol=2e-13) for i in ii])


@pytest.mark.parametrize('name,period',[
    ('m04_lvz',1.441220),('m06_multi_reversal',1.0749),
    ('m08_near_kissing',3.414549),('m04_lvz',1.5149285729544828),
    ('m03_sediment_basement',5.661625992822728)])
def test_near_roots_and_precision_boundaries(name,period):
    model=next(m for m in md.CANONICAL_MODELS if m.name==name)
    a=scan(model,period,nc=65537);b=scan(model,period)
    np.testing.assert_allclose(a[:7],b[:7],atol=1e-10,rtol=1e-10)
    with md.Solver(backend=BACKEND,cmin=.1,accuracy='high') as s:out=s(model.thickness,model.vp,model.vs,model.rho,[period],nmodes=5)
    actual=out.phase_velocity[0,:,0]
    if np.any(out.trace_flags&4):assert np.isnan(actual).all()
    else:
        expected=np.full(5,np.nan);expected[:min(5,len(b))]=b[:5]
        np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=2e-6)


def test_normalization_preserves_roots_not_amplitudes():
    model=md.CANONICAL_MODELS[1]
    a=scan(model,5.,normalize=True,nc=32769)
    b=scan(model,5.,normalize=False,nc=32769)
    assert len(a)==len(b)
    np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10)


def test_periods_remain_fp64_in_refinement():
    m=md.CANONICAL_MODELS[1];periods=np.array([5.,5.*(1.+2e-8)])
    assert np.float32(periods[0])==np.float32(periods[1])
    with md.Solver(backend=BACKEND,c_tol_rel=1e-11,accuracy="high") as s:
        out=s(m.thickness,m.vp,m.vs,m.rho,periods,nmodes=1)
    assert abs(np.diff(out.phase_velocity[0,0])[0])>1e-10


def test_precision_probe_does_not_change_production_precision():
    m=md.CANONICAL_MODELS[2];fields=(m.thickness,m.vp,m.vs,m.rho)
    fp32=native.secular_grid(*fields,[5.661625992822728],.2,2.,8192,precision='float32')
    fp64=native.secular_grid(*fields,[5.661625992822728],.2,2.,8192)
    assert np.any(fp32!=fp64)
    with md.Solver(backend=BACKEND,accuracy='high') as s:
        s(*fields,[5.661625992822728],nmodes=3)
        assert s.kernel_report()['precision']=='float64'
