import os
import numpy as np
import pytest
from scipy.optimize import brentq
import modisp as md
BACKEND=os.environ.get("MODISP_TEST_BACKEND","cpu")


def test_batch_halfspace_rayleigh_and_period_order():
    vp,vs=6.,3.5
    k=(vs/vp)**2
    root=brentq(lambda x:x**3-8*x*x+(24-16*k)*x-16*(1-k),0.,1.)
    periods=np.array([10.,.1,1.,.1])
    out=md.dispersion(np.array([[0.,vp,vs,2.7]]),periods,nmodes=1,backend=BACKEND)
    assert isinstance(out,md.DispersionBatch)
    assert out.phase_velocity.shape==(1,1,4)
    np.testing.assert_array_equal(out.period,periods)
    np.testing.assert_array_equal(out.mode,[0])
    assert out.wave=="rayleigh"
    np.testing.assert_allclose(out.phase_velocity[0,0],vs*np.sqrt(root),atol=1e-6,rtol=0)


def test_halfspace_has_no_love_guided_modes():
    with md.Solver(backend=BACKEND) as solver:
        out=solver([0.],[6.],[3.5],[2.7],[.1,1.,10.],nmodes=1,wave="love")
        assert solver.last_result is out
    assert out.phase_velocity.shape==(1,1,3)
    assert np.isnan(out.phase_velocity).all()
    assert not out.mode_mask.any()
    assert np.all(out.status==md.SolveStatus.NOT_FOUND)
    assert not np.any(out.trace_flags&md.TRACE_SEARCH_UNCONVERGED)


def test_love_two_layer_analytic():
    h,b1,b2,r1,r2=1.,1.5,3.5,2.,2.7
    t=2.;omega=2*np.pi/t
    def secular(c):
        q=omega*np.sqrt(1/b1**2-1/c**2)
        r=omega*np.sqrt(1/c**2-1/b2**2)
        return r2*b2*b2*r*np.cos(q*h)-r1*b1*b1*q*np.sin(q*h)
    cs=np.linspace(np.nextafter(b1,b2),np.nextafter(b2,b1),10000)
    ff=secular(cs);ii=np.flatnonzero(np.signbit(ff[:-1])!=np.signbit(ff[1:]))
    expected=[brentq(secular,cs[i],cs[i+1],xtol=1e-13) for i in ii]
    with md.Solver(backend=BACKEND) as s:
        out=s([h,0.],[3.,6.],[b1,b2],[r1,r2],[t],nmodes=3,wave="love")
    actual=out.phase_velocity[0,:,0]
    np.testing.assert_allclose(actual[:len(expected)],expected,atol=1e-6,rtol=0)
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


def test_threads_periods_and_chunking():
    batch=md.random_model_batch(5,8,seed=20261027)
    per=np.array([1.,3.,1.,10.])
    values=[]
    for threads in [1,4]:
        with md.Solver(backend=BACKEND,threads=threads,cmin=.2) as s:
            values.append(s(**batch,periods=per,nmodes=3))
    np.testing.assert_array_equal(values[0].phase_velocity,values[1].phase_velocity)
    np.testing.assert_array_equal(values[0].phase_velocity[:,:,0],values[0].phase_velocity[:,:,2])


def test_invalid_internal_fluid_and_backend():
    with md.Solver(backend=BACKEND) as s:
        with pytest.raises(ValueError,match="water"):
            s([1.,1.,0.],[3.,1.5,6.],[1.5,0.,3.5],[2.,1.,2.7],[1.])
    with pytest.raises(ValueError):md.Solver(backend="other")


@pytest.mark.parametrize("wave",["rayleigh","love"])
def test_cpu_matches_disba_easy_models(wave):
    disba=pytest.importorskip("disba")
    model=np.array([[1.,3.,1.5,2.],[0.,6.,3.5,2.7]])
    per=np.logspace(-1,1,16)
    ref=disba.PhaseDispersion(*model.T)(per,mode=0,wave=wave)
    actual=md.dispersion(model,per,nmodes=1,wave=wave,backend=BACKEND)
    np.testing.assert_allclose(actual.period,ref.period)
    np.testing.assert_allclose(actual.phase_velocity[0,0],ref.velocity,atol=4e-6,rtol=0)

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
