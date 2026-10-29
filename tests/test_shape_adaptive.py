#!/Users/yusong/miniforge3/bin/python
"""Independent numerical witnesses and API contracts for shape-adaptive 0.3.1."""
import os
from pathlib import Path
import sys
import numpy as np
import pytest
import modisp as md

BACKEND=os.environ.get('MODISP_TEST_BACKEND','cpu')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


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


def test_reference_union_does_not_merge_two_close_modes():
    from validate_shape_paper import merge_witnesses
    root=np.array([1.,1.0000001,1.1])
    merged=merge_witnesses(root,[root+1e-10,root-1e-10])
    np.testing.assert_array_equal(merged,root)
    assert len(merged)==3


# Table 1 model 1, doi:10.1785/0120260071. These root pairs were checked
# with pinned PyGRT 0.18.0 (two controls) and separately bracketed raw CPS.
# Counts and one-to-one matching are essential near mode kissing.
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


def test_cli_persists_sampling_configuration(tmp_path):
    from modisp.cli import main
    src,dst=tmp_path/'input.npz',tmp_path/'result.npz'
    np.savez(src,thickness=[0.],vp=[6.],vs=[3.5],rho=[2.7],periods=[1.])
    assert main(['forward',str(src),'--output',str(dst),'--backend',BACKEND,
                 '--search-strategy','shape_adaptive','--sampling-rtol','.003','--nmodes','1'])==0
    with np.load(dst,allow_pickle=False) as z:
        assert str(z['search_strategy'])=='shape_adaptive'
        assert float(z['sampling_rtol'])==.003


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
