"""Physical witnesses of two narrow Rayleigh pairs missed by independent balanced search."""
import os
from pathlib import Path

import numpy as np
import pytest

import modisp as md


FIXTURE = Path(__file__).parent / "data" / "continuity_rayleigh.npz"
BACKEND = os.environ.get("MODISP_TEST_BACKEND", "cpu")
FIELDS = ("thickness", "vp", "vs", "rho")
pytestmark = pytest.mark.skipif(not FIXTURE.is_file(),
                               reason="requires the local Rayleigh regression fixture")


@pytest.mark.parametrize("index", [0, 1], ids=["M06-1453", "M06-1881"])
def test_independent_dunkin_scan_resolves_archived_pair(index):
    pytest.importorskip("disba")
    from disba._cps._surf96 import dltar
    from scipy.optimize import brentq

    with np.load(FIXTURE) as z:
        arrays = [z[k][index] for k in FIELDS]
        omega = 2 * np.pi / z["periods"][z["period_index"][index]]
        bounds = z["scan_bounds"][index]
        pair = z["reference_pair"][index]
    ca = np.empty((5, 5))

    def secular(c):
        return dltar(omega / c, omega, *arrays, 2, -1, ca)

    for n in (65, 129):
        x = np.linspace(*bounds, n)
        signs = np.array([np.signbit(secular(v)) for v in x])
        brackets = np.flatnonzero(signs[:-1] != signs[1:])
        roots = [brentq(secular, x[j], x[j + 1], xtol=1e-13) for j in brackets]
        np.testing.assert_allclose(roots, pair, rtol=0, atol=1e-9)


@pytest.mark.parametrize("index", [0, 1], ids=["M06-1453", "M06-1881"])
def test_review_recovers_pair_without_assigning_close_group_velocity(index):
    with np.load(FIXTURE) as z:
        arrays = [z[k][index] for k in FIELDS]
        j = z["period_index"][index]
        p = z["periods"][j-3:j+4]
        expected = z["reference_pair"][index]
    with md.Solver(backend=BACKEND, continuity_check=False, group_velocity=True) as baseline:
        old = baseline(*arrays, p, nmodes=5)
    with md.Solver(backend=BACKEND, group_velocity=True) as solver:
        out = solver(*arrays, p, nmodes=5)
        stats = solver.stats()
    assert not np.any(old.continuity_flags)
    assert not np.allclose(old.phase_velocity[0, 3:5, 3], expected, atol=3e-6)
    np.testing.assert_allclose(out.phase_velocity[0, 3:5, 3], expected, atol=3e-6, rtol=2e-5)
    assert not out.group_velocity_valid[0, 3:5, 3].any()
    assert np.all(out.group_velocity_flags[0, 3:5, 3] & int(md.GroupVelocityFlag.CLOSE_ROOT))
    flags = md.ContinuityFlag
    expected_flags = flags.ENVELOPE | flags.REVIEWED | flags.REPAIRED
    assert int(out.continuity_flags[0, 3]) & int(expected_flags) == int(expected_flags)
    assert stats["continuity_repaired"] >= 1
    np.testing.assert_array_equal(out.period, p)
    for name in ("phase_velocity", "group_velocity", "mode_mask", "group_velocity_flags",
                 "status", "trace_flags", "root_count", "search_level"):
        a = getattr(old, name)
        b = getattr(out, name)
        untouched = (out.continuity_flags[0] & int(flags.REPAIRED)) == 0
        assert a[..., untouched].tobytes() == b[..., untouched].tobytes(), name
