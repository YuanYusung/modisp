"""Per-cell benchmark evidence must retain validity and modal changes."""
import importlib.util
from pathlib import Path

import numpy as np


def driver():
    spec=importlib.util.spec_from_file_location('group_benchmark',Path(__file__).parents[1]/'tools/benchmark_continuation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def snapshot(values, flags=None):
    u=np.asarray(values,dtype=float)[None,None,:]
    f=np.zeros(u.shape,dtype=np.uint32) if flags is None else np.asarray(flags,dtype=np.uint32)[None,None,:]
    return dict(group_velocity=u,group_velocity_flags=f,group_velocity_valid=np.isfinite(u)&(f==0),
                phase_velocity=np.ones_like(u))


def test_equal_valid_totals_can_hide_two_opposite_transitions():
    a=snapshot([1,np.nan,3,np.nan],[0,4,0,16])
    b=snapshot([np.nan,2,3,np.nan],[4,0,0,16])
    out=driver().compare_groups(a,b)
    assert out['valid_to_invalid']==out['invalid_to_valid']==1
    assert out['flags_changed']==2 and out['review_needed'] and out['passed']
    assert not out['existing_valid_bitwise_equal']
    assert out['flag_transitions']==[dict(from_flag=0,to_flag=4,count=1),dict(from_flag=4,to_flag=0,count=1)]
    assert out['common_valid']['count']==1 and out['common_valid']['absolute']['max']==0


def test_error_quantiles_and_negative_finite_groups():
    a=snapshot([-1,2,3,4]);b=snapshot([-1.0001,2.002,3,4])
    out=driver().compare_groups(a,b)
    assert out['passed'] and out['common_valid']['count']==4
    assert np.isclose(out['common_valid']['absolute']['p50'],.00005)
    assert np.isclose(out['common_valid']['relative']['max'],.001)
    b['group_velocity'][0,0,2]+=.1
    assert not driver().compare_groups(a,b)['passed']


def test_phase_rank_disagreement_is_reported_separately():
    a=snapshot([1,2]);b=snapshot([1,3]);b['phase_velocity'][0,0,1]=2
    out=driver().compare_groups(a,b)
    assert out['common_valid']['over_budget']==1
    assert out['aligned_common_valid']['over_budget']==0
    assert out['unaligned_common_valid']==1 and out['review_needed']


def test_invalid_quality_mask_never_passes():
    a=snapshot([1,2]);b=snapshot([1,np.nan])
    b['group_velocity_valid'][:]=True
    out=driver().compare_groups(a,b)
    assert out['valid_mask_errors']==1 and not out['passed']


def test_private_cache_sentinel_cannot_escape_as_a_public_flag():
    a=snapshot([1]);b=snapshot([np.nan],[0x80000000])
    out=driver().compare_groups(a,b)
    assert out['unknown_flags']==1 and not out['passed']
