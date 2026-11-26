"""Coverage of the default trigger, compact reviews and caller period order."""
from pathlib import Path

import numpy as np
import pytest

import modisp
from modisp import ContinuityFlag as F
from modisp._cpu import continuity_screen


WITNESS = Path(__file__).parent / "data" / "continuity_rayleigh.npz"
requires_witness = pytest.mark.skipif(not WITNESS.is_file(),
                                     reason="requires the local Rayleigh regression fixture")


def test_bilateral_phase_envelope_and_fp32_ulps():
    periods = np.array([1., 1.01, 1.02, 1.4])
    phases = np.ones((4, 2, 4))
    phases[:, 1] = 2.
    phases[0, 0, 1] = .95
    phases[1, 0, 1] = np.nextafter(.95, 0.)
    phases[2, 1, 1] = np.nan
    phases[3, :, 0] = np.nan
    order = [1, 3, 0, 2]
    flags, suspect = continuity_screen(periods[order], np.ascontiguousarray(phases[:, :, order]), None, True)
    assert suspect
    np.testing.assert_array_equal(flags[:, 0], [0, F.ENVELOPE, F.ENVELOPE, F.NO_BILATERAL])
    assert np.all(flags[:, 1:] == F.NO_BILATERAL)

    phases = np.ones((2, 2, 1), dtype=np.float32)
    phases.view(np.int32)[:, 1, 0] += [16, 17]
    flags, suspect = continuity_screen(periods[:1], phases, None, True)
    assert suspect
    np.testing.assert_array_equal(flags[:, 0], [F.CLOSE_FP32 | F.NO_BILATERAL, F.NO_BILATERAL])


@requires_witness
def test_repaired_trace_survives_shuffling_duplicates_and_workspace_reuse():
    with np.load(WITNESS) as data:
        models = [data[k] for k in ('thickness', 'vp', 'vs', 'rho')]
        periods = data['periods'][13:18].copy()
        order = [4, 2, 1, 3, 0, 2]
    with modisp.Solver(backend='cpu', threads=1, chunk_size=1, group_velocity=True) as engine:
        sorted_result = engine(*models, periods, nmodes=5)
        shuffled = engine(*models, periods[order], nmodes=5)
        assert engine.stats()['engine_allocations'] == 0
        assert engine.stats()['continuity_suspected'] == 1
    assert shuffled.continuity_flags[0, 1] & F.REPAIRED
    for field in ('phase_velocity', 'group_velocity', 'group_velocity_flags', 'trace_flags',
                  'search_level', 'root_count', 'status', 'continuity_flags'):
        np.testing.assert_array_equal(getattr(sorted_result, field)[..., order], getattr(shuffled, field))


@requires_witness
def test_failed_confirmation_retains_original_trace(monkeypatch):
    with np.load(WITNESS) as data:
        model = [data[k][0] for k in ('thickness', 'vp', 'vs', 'rho')]
        periods = data['periods'][14:17]
    original = modisp.Solver._solve
    def unavailable(self, *args, **kwargs):
        if self.accuracy == 'high':
            raise MemoryError("insufficient review workspace")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(modisp.Solver, '_solve', unavailable)
    with modisp.Solver(backend='cpu', threads=1, group_velocity=True) as checked, \
         modisp.Solver(backend='cpu', threads=1, group_velocity=True, continuity_check=False) as untouched:
        expected = untouched(*model, periods, nmodes=5)
        result = checked(*model, periods, nmodes=5)
    assert result.continuity_flags[0, 1] == F.ENVELOPE | F.REVIEWED | F.UNRESOLVED
    assert result.mode_index_uncertain[0, 1]
    for field in ('phase_velocity', 'group_velocity', 'group_velocity_flags', 'trace_flags',
                  'search_level', 'root_count', 'status'):
        np.testing.assert_array_equal(getattr(result, field), getattr(expected, field))


@pytest.mark.parametrize("setting", [None, 1, "true"])
def test_continuity_option_requires_boolean(setting):
    with pytest.raises(ValueError, match="continuity_check"):
        modisp.Solver(continuity_check=setting)


def test_guard_cutoff_and_group_counts_after_repair():
    # The central period has a close Rayleigh pair and a guard near half-space Vs.
    # None of the 30 requested modes is near the cutoff.
    layer = np.array([[.01, 1.55, .18, 1.78], [.01, 1.8, .35, 1.85],
                      [.02, 1., .12, 1.5], [0., 2., .6, 1.93]])
    layer[-1, 1:3] *= 1.1600000000000001
    periods = 1 / np.array([60.55, 60.6, 60.65])
    with modisp.Solver(backend='cpu', threads=1, group_velocity=True) as engine:
        out = engine(*layer.T, periods, nmodes=30)
        assert engine.stats()['group_invalid'] == np.count_nonzero(out.group_velocity_flags)
    assert out.continuity_flags[0, 1] & F.REPAIRED
    assert np.nanmax(out.phase_velocity[0, :, 1]) < .99 * layer[-1, 2]
    assert not (out.trace_flags[0, 1] & modisp.TRACE_NEAR_CUTOFF)


@pytest.mark.parametrize('backend', ['cpu', 'cuda'])
@requires_witness
def test_group_review_uses_final_fp64_roots(backend, monkeypatch):
    import os
    if backend == 'cuda' and os.environ.get('MODISP_TEST_BACKEND') != 'cuda':
        pytest.skip('requires CUDA test run')
    native = modisp._cpu if backend == 'cpu' else modisp._load_cuda()
    with np.load(WITNESS) as data:
        models = [data[k] for k in ('thickness', 'vp', 'vs', 'rho')]
        periods = data['periods'][13:18]
    opts = dict(backend=backend, threads=1, group_velocity=True,
                precision='float32' if backend == 'cuda' else 'float64')
    with modisp.Solver(**opts) as solver:
        actual = solver(*models, periods)
        stats = solver.stats()
    assert stats['continuity_repaired'] > 0
    # Exercise the compatible eager-group implementation on exactly the same
    # repaired roots and guard bounds; it also differentiates discarded guards.
    monkeypatch.delattr(native, '_review_groups')
    with modisp.Solver(**opts) as solver:
        expected = solver(*models, periods)
        eager = solver.stats()
        assert 0 < stats['continuity_group_evaluations'] < eager['continuity_group_evaluations']
        saved = eager['continuity_group_evaluations'] - stats['continuity_group_evaluations']
        for key in ('evaluations', 'continuity_evaluations', 'group_evaluations'):
            assert eager[key] - stats[key] == saved
    for key in ('phase_velocity', 'group_velocity', 'group_velocity_flags',
                'root_count', 'trace_flags', 'continuity_flags'):
        assert getattr(actual, key).tobytes() == getattr(expected, key).tobytes()


@pytest.mark.parametrize('chunk_size', [1, 3])
def test_cuda_compacts_only_requested_model_period_pairs(chunk_size, monkeypatch):
    import os
    if os.environ.get('MODISP_TEST_BACKEND') != 'cuda':
        pytest.skip('requires CUDA test run')
    models = modisp.random_model_batch(3, 10, seed=164)
    periods = np.array([.43, .31, .58])
    # Inject one suspicion per model at a different original period.
    flags = np.eye(3, dtype=np.uint8)*int(F.ENVELOPE)
    original = modisp._cpu.continuity_screen
    def screen(per, phase, prediction, check):
        if phase.shape[0]==3 and phase.shape[2]==3:
            return flags.copy(), True
        return original(per, phase, prediction, check)
    monkeypatch.setattr(modisp._cpu, 'continuity_screen', screen)
    opts = dict(backend='cuda', precision='float32', accuracy='balanced',
                chunk_size=chunk_size, group_velocity=True)
    with modisp.Solver(**opts) as solver:
        packed = solver(**models, periods=periods, nmodes=4)
        stats = solver.stats()
    assert stats['continuity_suspected']==3
    assert 0 < stats['continuity_cuda_batches'] <= stats['continuity_full_searches']+stats['continuity_local_searches']
    mod = modisp._load_cuda()
    monkeypatch.setattr(mod, 'SUPPORTS_ROW_PERIODS', False)
    with modisp.Solver(**opts) as solver:
        separate = solver(**models, periods=periods, nmodes=4)
        for key in ('continuity_local_searches','continuity_full_searches','continuity_evaluations'):
            assert solver.stats()[key]==stats[key]
    for field in ('phase_velocity','group_velocity','group_velocity_flags','root_count',
                  'status','trace_flags','search_level','continuity_flags'):
        np.testing.assert_array_equal(getattr(packed, field), getattr(separate, field))
