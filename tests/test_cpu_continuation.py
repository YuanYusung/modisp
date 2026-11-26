"""Exercise the native CPU continuation through the public Solver."""
import numpy as np
import pytest

import modisp as md


MODEL = ([1., 0.], [3., 6.], [1.5, 3.5], [2., 2.7])
P = md.PredictionFlag


@pytest.mark.parametrize("wave,strategy", [("rayleigh", "shape_adaptive"), ("love", "qe_adaptive")])
def test_cpu_continuation_preserves_guarded_roots_and_groups(wave, strategy):
    periods = np.geomspace(.2, .22, 17)
    # Exercise chunking, caller order, and an exact duplicate.
    order = [16, 2, 0, 1, *range(3, 16), 2]
    batch = [np.tile(field, (3, 1)) for field in MODEL]
    options = dict(backend="cpu", threads=2, chunk_size=2, search_strategy=strategy, group_velocity=True)
    with md.Solver(**options, frequency_strategy="anchor_continuation") as solver:
        out = solver(*batch, periods[order], nmodes=2, wave=wave)
        stats = solver.stats()
    with md.Solver(**options, accuracy="high") as reference:
        expected = reference(*batch, periods[order], nmodes=2, wave=wave)
    assert out.frequency_strategy_used == "anchor_continuation"
    accepted = (out.prediction_flags & int(P.ACCEPTED)) != 0
    assert accepted.any()
    assert np.all(out.prediction_flags[accepted] & int(P.AUDIT))
    assert stats["continuation_accepted"] + stats["continuation_fallbacks"] + stats["continuation_anchor_traces"] == 3 * 17
    assert 0 < stats["continuation_derivative_evaluations"] <= stats["continuation_evaluations"] <= stats["evaluations"]
    assert stats["group_reused"] > 0
    np.testing.assert_array_equal(out.root_count, expected.root_count)
    np.testing.assert_array_equal(out.group_velocity_valid, expected.group_velocity_valid)
    np.testing.assert_allclose(out.phase_velocity, expected.phase_velocity, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(out.group_velocity, expected.group_velocity, rtol=2e-3, atol=3e-6)
    np.testing.assert_array_equal(out.prediction_flags[:, 1], out.prediction_flags[:, -1])


def test_cpu_anchor_prescreen_and_reused_solver_state():
    crossing = np.geomspace(1.15, 1.25, 17)
    dense = np.geomspace(.2, .22, 17)
    with md.Solver(backend="cpu", threads=2, frequency_strategy="anchor_continuation") as solver:
        first = solver(*MODEL, dense, nmodes=2, wave="love")
        birth = solver(*MODEL, crossing, nmodes=2, wave="love")
        stats = solver.stats()
        again = solver(*MODEL, dense, nmodes=2, wave="love")
    assert stats["continuation_prescreen_skipped_roots"] > 0
    unsafe = (birth.prediction_flags & int(P.ROOT_COUNT_UNSAFE)) != 0
    assert unsafe.any()
    assert np.all(birth.prediction_flags[unsafe] & int(P.FALLBACK))
    assert not np.any(birth.prediction_flags[unsafe] & int(P.ACCEPTED))
    np.testing.assert_array_equal(first.phase_velocity, again.phase_velocity)
    np.testing.assert_array_equal(first.prediction_flags, again.prediction_flags)
