"""Public automatic sampling, fixed tolerances and trace-level review."""
import os
from pathlib import Path

import numpy as np
import pytest

import modisp
from modisp.cli import main


BACKEND = os.environ.get("MODISP_TEST_BACKEND", "cpu")
PRECISIONS = ["float64", "float32"] if BACKEND=="cuda" else ["float64"]
WATER = Path(__file__).parent / "data/accuracy_high_cpu_linux/04_water.npz"
MODEL_KEYS = ("thickness", "vp", "vs", "rho")


@pytest.mark.parametrize("precision", PRECISIONS)
def test_auto_sampling_retains_water_root_and_caller_order(precision):
    with np.load(WATER) as data:
        model = [data[k] for k in MODEL_KEYS]
        periods = data["periods"]
        expected = data["phase_velocity"]
    order = [4, 1, 1, 3, 0, 2]
    with modisp.Solver(backend=BACKEND, precision=precision, accuracy="auto",
                       group_velocity=True, threads=1) as solver:
        first = solver(*model, periods, nmodes=4)
        again = solver(*model, periods[order], nmodes=4)
        stats = solver.stats()
        report = solver.kernel_report()
    assert stats["periods_computed"]==periods.size
    assert stats["accuracy"]==report["accuracy"]==again.accuracy=="auto"
    assert stats["refinement_tolerance"]==1e-5
    assert stats["sampling_rtol_max"]==report["sampling_rtol_max"]==.05
    assert stats["auto_relaxed_traces"]==np.count_nonzero(first.auto_relaxed)>0
    assert stats["auto_relaxed_leaves"]>=stats["auto_relaxed_traces"]
    assert not np.any(again.numerical_flags & modisp.TRACE_AUTO_RELAXED)
    assert again.phase_velocity.dtype==np.dtype(precision)
    assert again.group_velocity.dtype==np.float64
    np.testing.assert_allclose(first.phase_velocity, expected, rtol=2e-5, atol=1e-6)
    for name in ("phase_velocity", "group_velocity", "group_velocity_flags", "auto_relaxed",
                 "trace_flags", "root_count", "status", "search_level", "continuity_flags"):
        np.testing.assert_array_equal(getattr(again,name), getattr(first,name)[...,order])


@pytest.mark.parametrize("strategy,sampling", [("shape_adaptive", .01), ("shape_adaptive", .003),
                                               ("qe_adaptive", None)])
def test_fixed_sampling_and_qe_keep_balanced_work(strategy,sampling):
    with np.load(WATER) as data:
        inputs = [data[k] for k in (*MODEL_KEYS, "periods")]
    options = dict(backend=BACKEND, threads=1, search_strategy=strategy, sampling_rtol=sampling,
                   c_tol_rel=3e-6, close_root_rtol=.008)
    with modisp.Solver(accuracy="balanced", **options) as fixed, modisp.Solver(accuracy="auto", **options) as auto:
        baseline = fixed(*inputs, nmodes=3)
        out = auto(*inputs, nmodes=3)
        stats = auto.stats()
        assert stats["evaluations"]==fixed.stats()["evaluations"]
    for field in ("phase_velocity", "trace_flags", "root_count", "search_level"):
        np.testing.assert_array_equal(getattr(out,field), getattr(baseline,field))
    assert stats["auto_relaxed_leaves"]==0 and not out.auto_relaxed.any()
    assert stats["sampling_rtol_max"]==sampling
    assert stats["refinement_tolerance"]==3e-6


def test_auto_endpoint_failure_receives_independent_review():
    with modisp.Solver(backend=BACKEND, accuracy="auto", continuity_check=False, threads=1) as solver:
        result = solver([1e200,0.], [1e200,2e200], [5e199,1e200], [1e200,1e200], [1.], nmodes=1)
        stats = solver.stats()
    assert result.continuity_flags.item()==int(modisp.ContinuityFlag.SEARCH |
                                             modisp.ContinuityFlag.REVIEWED | modisp.ContinuityFlag.UNRESOLVED)
    assert result.trace_flags.item() & modisp.TRACE_SEARCH_UNCONVERGED
    assert result.mode_index_uncertain.item()
    assert stats["continuity_full_searches"]==stats["continuity_unresolved"]==1


def test_auto_cli_npz_and_default_solver_reuse(tmp_path, monkeypatch):
    target = tmp_path/"solve.npz"
    omitted = tmp_path/"default.npz"
    options = ["forward",str(WATER),"--backend",BACKEND,"--threads","1","--nmodes","2"]
    assert main([*options,"--output",str(target),"--accuracy","auto"])==0
    assert main([*options,"--output",str(omitted)])==0
    with np.load(target) as result, np.load(omitted) as default:
        assert set(default.files)==set(result.files)
        for field in result.files:
            np.testing.assert_array_equal(default[field],result[field])
        assert result["accuracy"].item()=="auto"
        assert result["sampling_rtol_max"].item()==.05
        np.testing.assert_array_equal(result["auto_relaxed"], result["trace_flags"] & modisp.TRACE_AUTO_RELAXED != 0)
    with modisp.Solver(search_strategy="uniform",accuracy="auto") as legacy:
        assert legacy.accuracy=="uniform" and not legacy._auto_sampling
    with np.load(WATER) as data:
        inputs = [data[k] for k in (*MODEL_KEYS, "periods")]
    monkeypatch.setattr(modisp,"_DEFAULT",None)
    if BACKEND=="cpu":
        def unavailable_cuda():
            raise modisp.BackendUnavailableError("CPU test environment")
        monkeypatch.setattr(modisp,"_load_cuda",unavailable_cuda)
    try:
        default = modisp.dispersion_batch(*inputs,nmodes=1)
        engine = modisp._DEFAULT
        modisp.dispersion_batch(*inputs,nmodes=1)
        assert engine is modisp._DEFAULT and engine.calls==2
        assert default.accuracy==modisp.DEFAULT_ACCURACY
    finally:
        modisp.release()
