#!/Users/yusong/miniforge3/bin/python
"""Independent analytic/reference physics and numerical conditioning."""
import os
import numpy as np
import pytest
from scipy.optimize import brentq
import modisp as md
from modisp import native
BACKEND=os.environ.get("MODISP_TEST_BACKEND","cpu")


@pytest.mark.parametrize("accuracy,rtol",[("high",0.),("balanced",2e-5),("fast",2e-4)])
def test_batch_halfspace_rayleigh_and_period_order(accuracy,rtol):
    vp,vs=6.,3.5
    k=(vs/vp)**2
    root=brentq(lambda x:x**3-8*x*x+(24-16*k)*x-16*(1-k),0.,1.)
    periods=np.array([10.,.1,1.,.1])
    out=md.dispersion(np.array([[0.,vp,vs,2.7]]),periods,nmodes=1,backend=BACKEND,accuracy=accuracy)
    assert isinstance(out,md.DispersionBatch)
    assert out.phase_velocity.shape==(1,1,4)
    np.testing.assert_array_equal(out.period,periods)
    np.testing.assert_array_equal(out.mode,[0])
    assert out.wave=="rayleigh"
    np.testing.assert_allclose(out.phase_velocity[0,0],vs*np.sqrt(root),atol=1e-6,rtol=rtol)


def test_halfspace_has_no_love_guided_modes():
    with md.Solver(backend=BACKEND) as solver:
        out=solver([0.],[6.],[3.5],[2.7],[.1,1.,10.],nmodes=1,wave="love")
        assert solver.last_result is out
    assert out.phase_velocity.shape==(1,1,3)
    assert np.isnan(out.phase_velocity).all()
    assert not out.mode_mask.any()
    assert np.all(out.status==md.SolveStatus.NOT_FOUND)
    assert not np.any(out.trace_flags&md.TRACE_SEARCH_UNCONVERGED)


@pytest.mark.parametrize("accuracy,rtol",[("high",0.),("balanced",2e-5),("fast",2e-4)])
def test_love_two_layer_analytic(accuracy,rtol):
    h,b1,b2,r1,r2=1.,1.5,3.5,2.,2.7
    t=2.;omega=2*np.pi/t
    def secular(c):
        q=omega*np.sqrt(1/b1**2-1/c**2)
        r=omega*np.sqrt(1/c**2-1/b2**2)
        return r2*b2*b2*r*np.cos(q*h)-r1*b1*b1*q*np.sin(q*h)
    cs=np.linspace(np.nextafter(b1,b2),np.nextafter(b2,b1),10000)
    ff=secular(cs);ii=np.flatnonzero(np.signbit(ff[:-1])!=np.signbit(ff[1:]))
    expected=[brentq(secular,cs[i],cs[i+1],xtol=1e-13) for i in ii]
    with md.Solver(backend=BACKEND,accuracy=accuracy) as s:
        out=s([h,0.],[3.,6.],[b1,b2],[r1,r2],[t],nmodes=3,wave="love")
    actual=out.phase_velocity[0,:,0]
    np.testing.assert_allclose(actual[:len(expected)],expected,atol=1e-6,rtol=rtol)
    assert np.isnan(actual[len(expected):]).all()


def test_love_ignores_inviscid_water_depth():
    per=np.logspace(-1,1,24)
    with md.Solver(backend=BACKEND) as s:
        dry=s([1.,0.],[3.,6.],[1.5,3.5],[2.,2.7],per,nmodes=3,wave="love")
        for h in [0.,.025,1.,5.]:
            wet=s([h,1.,0.],[1.5,3.,6.],[0.,1.5,3.5],[1.02,2.,2.7],per,nmodes=3,wave="love")
            np.testing.assert_allclose(wet.phase_velocity,dry.phase_velocity,atol=1e-6,rtol=1e-7)


def test_zero_water_limit_and_slow_seabed():
    per=np.logspace(-1,1,12)
    with md.Solver(backend=BACKEND) as s:
        dry=s([.1,0.],[.5,2.],[.18,.6],[1.8,2.5],per,nmodes=2)
        wet=s([0.,.1,0.],[1.5,.5,2.],[0.,.18,.6],[1.02,1.8,2.5],per,nmodes=2)
    np.testing.assert_allclose(wet.phase_velocity,dry.phase_velocity,atol=2e-6,rtol=0)
    assert np.nanmin(wet.phase_velocity)<.2


def test_thick_water_scholte_limit():
    vf,vp,vs,rf,rs=1.5,6.,3.5,1.02,2.7
    def secular(c):
        x=(c/vs)**2
        A=np.sqrt(1-(c/vp)**2);B=np.sqrt(1-x);C=np.sqrt(1-(c/vf)**2)
        return 4*(1+(vs/vp)**2-(vs/vp)**2*x)/(1+A*B)-4+x+(rf/rs)*x*A/C
    root=brentq(secular,1e-8,vf*(1-1e-10),xtol=1e-13)
    with md.Solver(backend=BACKEND) as s:
        out=s([20.,0.],[vf,vp],[0.,vs],[rf,rs],[1.],nmodes=1)
    assert not out.trace_flags[0,0]&md.TRACE_SEARCH_UNCONVERGED
    assert abs(out.phase_velocity[0,0,0]-root)<2e-6


def test_invalid_internal_fluid_and_backend():
    with md.Solver(backend=BACKEND) as s:
        with pytest.raises(ValueError,match="water"):
            s([1.,1.,0.],[3.,1.5,6.],[1.5,0.,3.5],[2.,1.,2.7],[1.])
    with pytest.raises(ValueError):md.Solver(backend="other")


@pytest.mark.parametrize("wave",["rayleigh","love"])
@pytest.mark.parametrize("accuracy,rtol",[("high",0.),("balanced",2e-5),("fast",2e-4)])
def test_cpu_matches_disba_easy_models(wave,accuracy,rtol):
    disba=pytest.importorskip("disba")
    model=np.array([[1.,3.,1.5,2.],[0.,6.,3.5,2.7]])
    per=np.logspace(-1,1,16)
    ref=disba.PhaseDispersion(*model.T)(per,mode=0,wave=wave)
    actual=md.dispersion(model,per,nmodes=1,wave=wave,backend=BACKEND,accuracy=accuracy)
    np.testing.assert_allclose(actual.period,ref.period)
    np.testing.assert_allclose(actual.phase_velocity[0,0],ref.velocity,atol=4e-6,rtol=rtol)


@pytest.mark.parametrize('vs',[.18,1.5,3.5])
def test_no_sh_guide_is_not_a_cutoff_root(vs):
    with md.Solver(backend=BACKEND) as s:
        for h in [0.,.025,20.]:
            out=s([h,0.],[1.5,2*vs],[0.,vs],[1.02,2.7],[.1,1.,10.],nmodes=3,wave='love')
            assert np.isnan(out.phase_velocity).all()
            assert np.all(out.status==md.SolveStatus.NOT_FOUND)
            assert np.all(out.search_level==-3)
            assert s.stats()['analytical_empty']==3


def test_love_cutoff_is_diagnosed():
    b1,b2,h=1.5,3.5,1.
    cutoff=2*h*np.sqrt(1/b1**2-1/b2**2)
    with md.Solver(backend=BACKEND) as s:
        out=s([h,0.],[3.,6.],[b1,b2],[2.,2.7],[cutoff*(1-1e-10),cutoff,cutoff*(1+1e-10)],nmodes=3,wave='love')
    for j in range(3):
        values=out.phase_velocity[0,:,j]
        if np.any(values>=.99*b2):assert out.trace_flags[0,j]&md.TRACE_NEAR_CUTOFF


@pytest.mark.parametrize('wave',['rayleigh','love'])
def test_long_period_preserves_physical_scale_invariance(wave):
    with md.Solver(backend=BACKEND) as s:
        normal=s([.3,0.],[2.,5.],[.8,2.8],[2.,2.6],[1.],nmodes=3,wave=wave)
        long=s([3e5,0.],[2.,5.],[.8,2.8],[2.,2.6],[1e6],nmodes=3,wave=wave)
    np.testing.assert_allclose(normal.phase_velocity,long.phase_velocity,atol=3e-6,rtol=2e-6)
    assert not np.any(long.trace_flags&4)


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
