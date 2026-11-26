"""Group velocities against independent physics and public API invariants."""
import os

import numpy as np
import pytest
from scipy.optimize import brentq

import modisp as md
from modisp.cli import main


BACKEND = os.environ.get("MODISP_TEST_BACKEND", "cpu")
MODEL = ([1., 0.], [3., 6.], [1.5, 3.5], [2., 2.7])


def love_reference(period, nmodes=3):
    """Two-layer Love roots and analytical implicit frequency derivatives.

    This is the traction equation, independent of the production recurrence.
    Factoring out omega keeps its roots and derivative ratio unchanged.
    """
    h, b1, b2, rho1, rho2 = 1., 1.5, 3.5, 2., 2.7
    mu1, mu2 = rho1*b1*b1, rho2*b2*b2
    omega = 2*np.pi/period

    def secular(c):
        a = np.sqrt(1/b1**2-1/c**2)
        b = np.sqrt(1/c**2-1/b2**2)
        z = omega*h*a
        return mu2*b*np.cos(z)-mu1*a*np.sin(z)

    grid = np.linspace(np.nextafter(b1, b2), np.nextafter(b2, b1), 32769)
    values = secular(grid)
    brackets = np.flatnonzero(np.signbit(values[:-1]) != np.signbit(values[1:]))
    roots = np.array([brentq(secular, grid[i], grid[i+1], xtol=2e-14)
                      for i in brackets[:nmodes]])
    a = np.sqrt(1/b1**2-1/roots**2)
    b = np.sqrt(1/roots**2-1/b2**2)
    da, db = 1/(roots**3*a), -1/(roots**3*b)
    z = omega*h*a
    dc = (mu2*db*np.cos(z)-mu2*b*np.sin(z)*omega*h*da
          -mu1*da*np.sin(z)-mu1*a*np.cos(z)*omega*h*da)
    dw = -mu2*b*np.sin(z)*h*a-mu1*a*np.cos(z)*h*a
    group = roots*roots*dc/(roots*dc+omega*dw)
    return roots, group


@pytest.mark.parametrize("secular", ["dunkin", "fast"])
def test_single_period_halfspace_group_equals_phase(secular):
    with md.Solver(backend=BACKEND, accuracy="high", secular=secular,
                   group_velocity=True) as solver:
        out = solver([0.], [6.], [3.5], [2.7], [1.], nmodes=1)
    assert out.group_velocity.shape == (1, 1, 1)
    assert out.group_velocity.dtype == np.float64
    assert out.group_velocity_flags.dtype == np.uint32
    assert out.group_velocity_valid.dtype == np.bool_
    assert out.group_velocity_valid.all()
    np.testing.assert_array_equal(out.group_velocity_valid, out.group_velocity_mask)
    np.testing.assert_allclose(out.group_velocity, out.phase_velocity, rtol=2e-6)


@pytest.mark.parametrize("secular", ["dunkin", "fast"])
def test_love_group_against_analytic_implicit_derivative(secular):
    # The short-period roots lie near Vs_top and need a genuinely local stencil.
    periods = np.array([.1, .5, 1., 1.2, 2., 4., 10.])
    with md.Solver(backend=BACKEND, accuracy="high", secular=secular,
                   group_velocity=True, group_rtol=1e-3) as solver:
        out = solver(*MODEL, periods, nmodes=3, wave="love")
    for i, period in enumerate(periods):
        phase, group = love_reference(period)
        n = len(phase)
        np.testing.assert_allclose(out.phase_velocity[0, :n, i], phase,
                                   rtol=2e-6, atol=3e-6)
        assert np.isfinite(out.group_velocity[0, :n, i]).all()
        np.testing.assert_allclose(out.group_velocity[0, :n, i], group,
                                   rtol=1e-3, atol=2e-6)
        assert np.isnan(out.group_velocity[0, n:, i]).all()
        assert not out.group_velocity_valid[0, n:, i].any()
        assert np.all(out.group_velocity_flags[0, n:, i] & 1)


def test_group_does_not_change_phase_and_is_input_grid_independent():
    periods = np.array([2., .5, 1., .5])
    with md.Solver(backend=BACKEND, accuracy="high") as plain:
        expected = plain(*MODEL, periods, nmodes=2, wave="love")
    with md.Solver(backend=BACKEND, accuracy="high", group_velocity=True) as solver:
        out = solver(*MODEL, periods, nmodes=2, wave="love")
        single = solver(*MODEL, [.5], nmodes=2, wave="love")
    for name in ("phase_velocity", "trace_flags", "root_count", "status", "search_level"):
        np.testing.assert_array_equal(getattr(out, name), getattr(expected, name))
    np.testing.assert_array_equal(out.period, periods)
    np.testing.assert_array_equal(out.group_velocity[:, :, 1], out.group_velocity[:, :, 3])
    np.testing.assert_allclose(out.group_velocity[:, :, 1], single.group_velocity[:, :, 0],
                               rtol=1e-10, atol=1e-10, equal_nan=True)
    assert expected.group_velocity is None
    assert expected.group_velocity_flags is None
    assert expected.group_velocity_valid is None


def test_group_reuse_with_in_place_model_changes_and_physical_scaling():
    arrays = [np.array(x, dtype=float) for x in MODEL]
    periods = np.array([.5, 1., 2.])
    options = dict(backend=BACKEND, accuracy="high", group_velocity=True)
    with md.Solver(**options) as solver:
        original = solver(*arrays, periods, nmodes=2, wave="love")
        arrays[0] *= 1.7
        changed = solver(*arrays, periods, nmodes=2, wave="love")
        scaled = solver(*arrays, periods*1.7, nmodes=2, wave="love")
    with md.Solver(**options) as fresh:
        expected = fresh(*arrays, periods, nmodes=2, wave="love")
    np.testing.assert_array_equal(changed.group_velocity, expected.group_velocity)
    np.testing.assert_array_equal(changed.group_velocity_flags, expected.group_velocity_flags)
    np.testing.assert_allclose(scaled.group_velocity, original.group_velocity,
                               rtol=3e-5, atol=3e-6, equal_nan=True)
    assert not np.allclose(changed.phase_velocity, original.phase_velocity, equal_nan=True)


def test_missing_modes_and_close_pairs_are_not_valid_group_measurements():
    with md.Solver(backend=BACKEND, accuracy="high", group_velocity=True) as solver:
        empty = solver([0.], [6.], [3.5], [2.7], [1., 2.], nmodes=2, wave="love")
        model = next(m for m in md.CANONICAL_MODELS if m.name == "m04_lvz")
        close = solver(model.thickness, model.vp, model.vs, model.rho,
                       [1.441220], nmodes=6)
    assert np.isnan(empty.group_velocity).all()
    assert np.all(empty.group_velocity_flags & 1)
    assert not empty.group_velocity_valid.any()
    assert close.close_root_mask.any()
    assert np.all(close.group_velocity_flags[close.close_root_mask] & 16)
    assert not close.group_velocity_valid[close.close_root_mask].any()
    assert np.isnan(close.group_velocity[close.close_root_mask]).all()


def test_derivative_failure_is_flagged_without_changing_a_valid_phase_root():
    options = dict(backend=BACKEND, accuracy="high", group_velocity=True)
    with md.Solver(**options) as normal, md.Solver(group_rtol=1e-12, **options) as strict:
        reference = normal(*MODEL, [.1], nmodes=1, wave="love")
        out = strict(*MODEL, [.1], nmodes=1, wave="love")
    np.testing.assert_array_equal(out.phase_velocity, reference.phase_velocity)
    np.testing.assert_array_equal(out.status, reference.status)
    assert out.mode_mask.all()
    assert np.all(out.group_velocity_flags & 4)
    assert not out.group_velocity_valid.any()
    assert np.isnan(out.group_velocity).all()


def test_water_group_scholte_limit_and_love_depth_invariance():
    # In a deep water layer the fundamental tends to the nondispersive
    # fluid/solid interface wave, whose group and phase velocities coincide.
    # Acoustic overtones are densely spaced. Tighten only the heuristic close
    # threshold here so this test measures the isolated interface derivative.
    with md.Solver(backend=BACKEND, accuracy="high", group_velocity=True,
                   close_root_rtol=1e-4) as solver:
        wet = solver([20., 0.], [1.5, 6.], [0., 3.5], [1.02, 2.7], [1.], nmodes=1)
        assert wet.group_velocity_valid.all()
        np.testing.assert_allclose(wet.group_velocity, wet.phase_velocity, rtol=2e-5)
        dry = solver(*MODEL, [.5, 1., 2.], nmodes=2, wave="love")
        for depth in (0., .25, 5.):
            sh = solver([depth, 1., 0.], [1.5, 3., 6.], [0., 1.5, 3.5],
                        [1.02, 2., 2.7], [.5, 1., 2.], nmodes=2, wave="love")
            np.testing.assert_allclose(sh.group_velocity, dry.group_velocity,
                                       rtol=2e-8, atol=2e-8, equal_nan=True)
            np.testing.assert_array_equal(sh.group_velocity_flags, dry.group_velocity_flags)


@pytest.mark.parametrize("wave", ["rayleigh", "love"])
def test_group_against_converged_disba_reference(wave):
    disba = pytest.importorskip("disba")
    model = np.array([[.2, 2., .8, 2.], [0., 5., 2.8, 2.6]])
    periods = np.array([.7, 1., 2., 4.])
    coarse = disba.GroupDispersion(*model.T, dt=.005)(periods, mode=0, wave=wave)
    fine = disba.GroupDispersion(*model.T, dt=.0025)(periods, mode=0, wave=wave)
    np.testing.assert_array_equal(fine.period, periods)
    np.testing.assert_allclose(coarse.velocity, fine.velocity, rtol=5e-4, atol=1e-5)
    with md.Solver(backend=BACKEND, accuracy="high", group_velocity=True) as solver:
        out = solver(*model.T, periods, nmodes=1, wave=wave)
    assert out.group_velocity_valid.all()
    np.testing.assert_allclose(out.group_velocity[0, 0], fine.velocity,
                               rtol=2e-3, atol=2e-5)


@pytest.mark.parametrize("option,value", [
    ("group_velocity", 1), ("group_velocity", "yes"),
    ("group_rtol", 0.), ("group_rtol", -1.), ("group_rtol", np.nan),
    ("group_rtol", np.inf), ("group_rtol", True), ("group_rtol", 1.),
])
def test_group_option_validation(option, value):
    with pytest.raises(ValueError, match=option):
        md.Solver(**{option: value})


def test_cli_group_arrays_are_optional_and_have_numeric_types(tmp_path):
    source, plain, grouped = (tmp_path/name for name in ("input.npz", "plain.npz", "group.npz"))
    np.savez(source, thickness=[0.], vp=[6.], vs=[3.5], rho=[2.7], periods=[1.])
    command = ["forward", str(source), "--backend", BACKEND, "--accuracy", "high", "--nmodes", "1"]
    assert main(command+["--output", str(plain)]) == 0
    assert main(command+["--output", str(grouped), "--group-velocity", "--group-rtol", ".002"]) == 0
    with np.load(plain, allow_pickle=False) as a, np.load(grouped, allow_pickle=False) as b:
        assert "group_velocity" not in a
        assert "group_velocity_flags" not in a
        assert "group_velocity_mask" not in a
        assert b["group_velocity"].dtype == np.float64
        assert b["group_velocity_flags"].dtype == np.uint32
        assert b["group_velocity_mask"].dtype == np.bool_
        assert float(b["group_rtol"]) == .002
        np.testing.assert_array_equal(a["phase_velocity"], b["phase_velocity"])


@pytest.mark.skipif(BACKEND != "cuda", reason="FP32 requires CUDA")
def test_fp32_phase_keeps_fp64_group_and_identical_phase_values():
    options = dict(backend="cuda", accuracy="high", precision="float32")
    with md.Solver(**options) as plain, md.Solver(group_velocity=True, **options) as solver:
        phase = plain(*MODEL, [.5, 1., 2.], nmodes=2, wave="love")
        out = solver(*MODEL, [.5, 1., 2.], nmodes=2, wave="love")
    assert out.phase_velocity.dtype == np.float32
    assert out.group_velocity.dtype == np.float64
    np.testing.assert_array_equal(out.phase_velocity, phase.phase_velocity)
    for i, period in enumerate([.5, 1., 2.]):
        _, reference = love_reference(period, 2)
        np.testing.assert_allclose(out.group_velocity[0, :len(reference), i], reference,
                                   rtol=2e-3, atol=3e-6)
