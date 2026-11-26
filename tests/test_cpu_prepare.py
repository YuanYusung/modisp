import numpy as np
import pytest

import modisp as md


@pytest.mark.parametrize("cmin", [None, 0.7])
def test_cpu_shape_grid_matches_recomputed_lower_bound(monkeypatch, cmin):
    # Include a water layer, where the automatic bound uses a Scholte root.
    model = np.array([[0.1, 1.5, 0.0, 1.0], [0.2, 2.5, 1.1, 2.2], [0.0, 4.5, 2.4, 2.5]])
    with md.Solver(backend="cpu", threads=1, cmin=cmin, group_velocity=True) as solver:
        args = (*model.T, [0.2, 0.7, 1.0])
        actual = solver(*args)
        monkeypatch.delattr(md._cpu, "SUPPORTS_CANONICAL_LOWER", raising=False)
        expected = solver(*args)
    for name in ("phase_velocity", "group_velocity", "group_velocity_flags", "root_count",
                 "trace_flags", "search_bounds", "search_level"):
        assert getattr(actual, name).tobytes() == getattr(expected, name).tobytes()
