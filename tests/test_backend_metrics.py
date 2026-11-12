#!/Users/yusong/miniforge3/bin/python
"""Protect scientific benchmark accounting against silent root-count errors."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from backend_metrics import compare


def result(values,flags=0,count=None):
    v=np.asarray(values,dtype=float).reshape(1,-1,1)
    mask=np.isfinite(v)
    return dict(phase_velocity=v,mode_mask=mask,
                status=np.where(mask,0,2 if flags&4 else 1).astype(np.uint8),
                trace_flags=np.array([[flags]],dtype=np.uint32),
                numerical_flags=np.array([[flags&~256]],dtype=np.uint32),
                root_count=np.array([[len(values) if count is None else count]]),
                search_level=np.array([[8]]),search_bounds=np.array([[.1,4.]]))


def test_missing_close_root_cannot_be_certified_by_one_candidate():
    m=compare(result([1.,2.],flags=2048),result([1.,1.0001,2.]))
    assert m["set_missing"]==m["cluster_unmatched"]==1
    assert m["rank_shifts"]==1
    assert m["ordinal_wrong"]==1
    assert m["matched_cells"]==2


def test_missing_isolated_root_is_not_waived_by_approximate_flag():
    m=compare(result([1.,3.],flags=2048),result([1.,2.,3.]))
    assert m["isolated_missing"]==1
    assert m["rank_shifts"]==1


def test_reference_unresolved_is_unjudged_not_zero_error():
    m=compare(result([1.]),result([np.nan],flags=4,count=-1))
    assert m["judged_traces"]==m["finite_common_cells"]==0
    assert m["unjudged_reference_traces"]==1
    assert m["abs_error_km_s"]["max"] is None


def test_exhausted_prefix_does_not_turn_unknown_tail_into_extra():
    m=compare(result([1.,4.]),result([1.,2.]))
    assert m["set_extra"]==0
    assert m["unjudged_prefix_cells"]==1
    assert m["set_missing"]==1


def test_empty_analytic_family_rejects_spurious_root():
    m=compare(result([1.]),result([np.nan]))
    assert m["set_extra"]==m["isolated_spurious"]==1


def test_guard_count_is_compared_at_candidate_cap():
    m=compare(result([1.],count=3),result([1.,2.,3.,4.],count=6))
    assert m["root_count_mismatches"]==0


def test_new_failure_remains_in_missing_and_failure_counts():
    m=compare(result([np.nan],flags=4,count=-1),result([1.]))
    assert m["ordinal_missing"]==m["set_missing"]==1
    assert m["newly_numerical_failed_cells"]==m["newly_numerical_failed_traces"]==1


def test_strict_and_budgeted_error_are_separate():
    m=compare(result([1.00001]),result([1.]),budget_rtol=2e-5)
    assert m["ordinal_wrong"]==1
    assert m["ordinal_budget_wrong"]==m["set_missing"]==0
