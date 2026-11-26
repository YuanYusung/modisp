"""Exact reuse and caller-order restoration for independent period searches."""
import os

import numpy as np
import pytest

import modisp as md


BACKEND = os.environ.get("MODISP_TEST_BACKEND", "cpu")
PERIOD_FIELDS = (
    "phase_velocity", "trace_flags", "search_level", "root_count", "status",
    "mode_mask", "close_root_mask", "approximate_search", "mode_index_uncertain",
    "close_root_risk", "potentially_ambiguous", "numerical_flags",
    "group_velocity", "group_velocity_flags", "group_velocity_valid", "prediction_flags",
)


@pytest.mark.parametrize("wave,group,strategy", [
    ("rayleigh", False, "independent"),
    ("love", True, "independent"),
    ("love", True, "anchor_continuation"),
])
def test_repeated_periods_restore_every_field(wave, group, strategy):
    unique = np.array([.2, 1., np.nextafter(1., 2.), 3.])
    order = [3, 1, 0, 1, 2, 0]
    model = md.random_model_batch(3, 5, seed=3492)
    with md.Solver(backend=BACKEND, frequency_strategy=strategy,
                   group_velocity=group, chunk_size=2) as solver:
        reference = solver(**model, periods=unique, nmodes=3, wave=wave)
        work = solver.stats()["evaluations"]
        result = solver(**model, periods=unique[order], nmodes=3, wave=wave)
        stats = solver.stats()

    assert result.frequency_strategy_used == "independent"
    assert stats["periods_requested"] == len(order)
    assert stats["periods_computed"] == len(unique)
    assert stats["evaluations"] == work
    assert stats["chunks"] == 2
    assert stats["engine_allocations"] == 0
    np.testing.assert_array_equal(result.period, unique[order])
    np.testing.assert_array_equal(result.search_bounds, reference.search_bounds)
    for field in PERIOD_FIELDS:
        actual = getattr(result, field)
        original = getattr(reference, field)
        if original is None:
            assert actual is None, field
        else:
            expected = original.take(order, axis=-1)
            assert actual.shape == expected.shape and actual.dtype == expected.dtype, field
            assert actual.flags.c_contiguous, field
            assert actual.tobytes() == expected.tobytes(), field


def test_independent_unique_grid_preserves_native_order(monkeypatch):
    seen = []
    native_run = md._cpu.CpuSolver.run

    def record(engine, h, vp, vs, rho, periods, *args):
        seen.append(periods)
        return native_run(engine, h, vp, vs, rho, periods, *args)

    monkeypatch.setattr(md._cpu.CpuSolver, "run", record)
    model = md.random_model_batch(1, 4, seed=3492)
    with md.Solver(backend="cpu", threads=1) as solver:
        for periods in (np.array([.5, 1., 2.]), np.array([2., .5, 1.])):
            solver(**model, periods=periods, nmodes=1)
            assert seen[-1] is periods


def test_monotonic_independent_grids_skip_sorting(monkeypatch):
    model = md.random_model_batch(1, 4, seed=3492)

    def unexpected_sort(*args, **kwargs):
        pytest.fail("strictly monotonic independent periods need no sorting")

    monkeypatch.setattr(np, "unique", unexpected_sort)
    with md.Solver(backend=BACKEND) as solver:
        for periods in ([.5, 1., 2.], [2., 1., .5]):
            result = solver(**model, periods=periods, nmodes=1)
            assert result.period.tolist() == periods
            assert solver.stats()["periods_computed"] == len(periods)


@pytest.mark.skipif(BACKEND != "cuda", reason="FP32 repair budget requires CUDA")
def test_cuda_high_fp32_preserves_requested_trace_count():
    model = md.random_model_batch(2, 4, seed=3492)
    periods = [.5, .3, .5, .3]
    with md.Solver(backend="cuda", precision="float32", accuracy="high") as solver:
        result = solver(**model, periods=periods, nmodes=1)
        stats = solver.stats()
    np.testing.assert_array_equal(result.period, periods)
    assert stats["periods_computed"] == stats["periods_requested"] == len(periods)
