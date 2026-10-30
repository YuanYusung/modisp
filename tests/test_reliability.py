#!/Users/yusong/miniforge3/bin/python
"""Analytic roots, independent instance state, and cross-period diagnostics."""
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from scipy.optimize import brentq
import modisp as md

BACKEND=os.environ.get('MODISP_TEST_BACKEND','cpu')
F=md.ModeReliabilityFlag


@pytest.mark.parametrize('refinement',['brent','chandrupatla','legacy'])
def test_two_layer_love_analytic_spectrum_and_cropping(refinement):
    model=np.array([[.3,2.,.8,2.],[0.,5.,2.8,2.6]])
    periods=np.array([.1,.3,1.,3.,10.])
    with md.Solver(backend=BACKEND,accuracy='high',refinement=refinement,c_tol_rel=1e-9) as s:
        result=s(*model.T,periods,nmodes=24,wave='love')
        assert s.stats()['love_phase_accepted']==len(periods)
    for t,p in enumerate(periods):
        omega=2*np.pi/p
        def phase(c):
            q=omega*np.sqrt(max(0.,1/.8**2-1/c**2));r=omega*np.sqrt(max(0.,1/c**2-1/2.8**2))
            return .3*q-np.arctan2(2.6*2.8**2*r,2*.8**2*q)
        lo=np.nextafter(.8,np.inf);hi=np.nextafter(2.8,0.);ref=[]
        for n in range(24):
            if phase(hi)<=n*np.pi:break
            ref.append(brentq(lambda c:phase(c)-n*np.pi,lo,hi,xtol=1e-14))
        finite=result.phase_velocity[0,:,t];finite=finite[np.isfinite(finite)]
        np.testing.assert_allclose(finite,ref,rtol=1e-8,atol=3e-10)
    with md.Solver(backend=BACKEND,accuracy='high',cmin=1.,cmax=2.,refinement=refinement) as s:
        cropped=s(*model.T,[.1],nmodes=24,wave='love')
    expected=result.phase_velocity[0,:,0];expected=expected[(expected>=1.)&(expected<=2.)]
    actual=cropped.phase_velocity[0,:,0];actual=actual[np.isfinite(actual)]
    np.testing.assert_allclose(actual,expected,atol=2e-9)
    assert s.kernel_report()['search_kernel']=='love_phase_with_scan_fallback'


def test_per_solver_refinement_is_independent_under_concurrency():
    model=np.array([[.3,2.,.8,2.],[.1,1.5,.4,1.8],[0.,5.,2.8,2.6]])
    solvers=[md.Solver(backend=BACKEND,accuracy='high',threads=1,refinement=r) for r in ['brent','chandrupatla']]
    try:
        expected=[s(*model.T,[.1,.3,1.],nmodes=4).phase_velocity for s in solvers]
        with ThreadPoolExecutor(2) as pool:
            values=list(pool.map(lambda s:s(*model.T,[.1,.3,1.],nmodes=4).phase_velocity,solvers))
        for a,b in zip(values,expected):np.testing.assert_array_equal(a,b)
        assert [s.stats()['refinement'] for s in solvers]==['brent','chandrupatla']
    finally:
        for s in solvers:s.close()


def test_love_phase_random_reversal_water_and_scan_regression():
    batch=md.random_model_batch(8,8,seed=20261031,lvz_prob=1.)
    for strategy in ['phase','scan']:
        with md.Solver(backend=BACKEND,threads=4,accuracy='high',love_strategy=strategy) as s:
            r=s(**batch,periods=np.logspace(-1,1,12),nmodes=8,wave='love')
        if strategy=='phase':phase=r
        else:scan=r
    np.testing.assert_array_equal(phase.mode_mask,scan.mode_mask)
    np.testing.assert_allclose(phase.phase_velocity,scan.phase_velocity,rtol=2e-6,atol=3e-6,equal_nan=True)


def test_mode_checks_sort_duplicates_gaps_and_do_not_edit_results():
    periods=np.array([1.2,1.,1.1,1.1,4.])
    v=np.array([[[1.02,1.,1.01,1.01,1.2],[2.02,2.,2.01,2.01,2.2]]])
    before=v.copy();r=md.check_mode_reliability(v,periods)
    np.testing.assert_array_equal(v,before)
    assert r.previous_index.tolist()==[2,-1,1,2,0]
    assert np.all(r.flags[:,:,1]&F.UNASSESSED)
    assert np.all(r.flags[:,:,4]&F.PERIOD_GAP)
    assert not r.assessed[:,:,4].any()
    assert r.passes_checks[:,:,0].all() and r.passes_checks[:,:,3].all()
    bad=v.copy();bad[0,0,3]+=1e-4
    check=md.check_mode_reliability(bad,periods)
    assert check.flags[0,0,3]&F.DUPLICATE_MISMATCH
    assert check.flags[0,0,2]&F.DUPLICATE_MISMATCH


def test_mode_checks_birth_rank_ambiguity_and_numerical_failure():
    v=np.array([[[1.,1.3,1.8],[1.4,1.8,np.nan],[2.,2.05,2.1]]])
    flags=np.array([[0,4096,4]],dtype=np.uint32)
    r=md.check_mode_reliability(v,[1.,1.1,1.2],trace_flags=flags)
    assert np.any(r.flags[:,:,1]&F.RANK_AMBIGUOUS)
    assert np.any(r.flags[:,:,1]&F.STEP_JUMP)
    assert np.all(r.flags[:,:,2]&F.ROOT_COUNT_CHANGE)
    assert np.all(r.flags[:,:,2]&F.NUMERICAL_FAILURE)
    assert np.all(r.flags[:,:,1]&F.SEARCH_UNCERTAIN)
    assert not r.assessed[:,:,2].any()


def test_mode_checks_adjacent_representable_periods():
    p=np.array([1e100,np.nextafter(1e100,np.inf),np.nextafter(np.nextafter(1e100,np.inf),np.inf)])
    with np.errstate(all='raise'):
        r=md.check_mode_reliability(np.ones((1,1,3)),p)
    assert r.passes_checks[0,0,1:].all()


def test_cli_standalone_checker_and_forward_metadata(tmp_path):
    from modisp.cli import main
    src=tmp_path/'input.npz';out=tmp_path/'output.npz';check=tmp_path/'check.npz'
    model=np.array([[.3,2.,.8,2.],[0.,5.,2.8,2.6]])
    np.savez(src,**dict(zip(['thickness','vp','vs','rho'],model.T)),periods=[1.,1.1,1.2])
    assert main(['forward',str(src),'--output',str(out),'--backend',BACKEND,'--wave','love',
                 '--refinement','chandrupatla','--close-root-rtol','0.02','--check-continuity'])==0
    assert main(['check-modes',str(out),'--output',str(check)])==0
    with np.load(out) as z,np.load(check) as c:
        assert str(z['refinement'])=='chandrupatla' and str(z['love_strategy'])=='phase'
        for key in c.files:np.testing.assert_array_equal(z[key],c[key])


@pytest.mark.parametrize('option',[dict(refinement='newton'),dict(love_strategy='track')])
def test_invalid_new_options(option):
    with pytest.raises(ValueError):md.Solver(**option)
