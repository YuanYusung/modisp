"""Opt-in CUDA frequency prediction, fallback, and period-index restoration."""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import modisp as md
from modisp.cli import main


BACKEND = os.environ.get("MODISP_TEST_BACKEND", "cpu")
MODEL = ([1., 0.], [3., 6.], [1.5, 3.5], [2., 2.7])
ANCHOR, ACCEPTED, FALLBACK, BYPASSED, AUDIT, AUDIT_FAILED = 1, 2, 4, 8, 16, 32
SOURCE_SEARCH_UNCERTAIN = int(md.PredictionFlag.SOURCE_SEARCH_UNCERTAIN)


@pytest.mark.parametrize("option,value", [
    ("frequency_strategy", "sequential"), ("anchor_stride", 1),
    ("anchor_stride", True), ("anchor_stride", 2.5),
])
def test_continuation_option_validation(option, value):
    with pytest.raises(ValueError, match=option):
        md.Solver(**{option: value})


@pytest.mark.parametrize("option,value", [
    ("group_velocity", True), ("frequency_strategy", "anchor_continuation"),
])
def test_legacy_uniform_rejects_new_features(option, value):
    with pytest.raises(NotImplementedError, match="adaptive"):
        md.Solver(search_strategy="uniform", **{option: value})


def test_cpu_high_continuation_request_bypasses_without_changing_results():
    periods = np.geomspace(.2, .4, 17)
    with md.Solver(backend="cpu", accuracy="high") as reference, md.Solver(
        backend="cpu", accuracy="high", frequency_strategy="anchor_continuation"
    ) as solver:
        expected = reference(*MODEL, periods, nmodes=2, wave="love")
        out = solver(*MODEL, periods, nmodes=2, wave="love")
        stats = solver.stats()
    np.testing.assert_array_equal(out.phase_velocity, expected.phase_velocity)
    np.testing.assert_array_equal(out.trace_flags, expected.trace_flags)
    assert expected.prediction_flags is None
    assert out.prediction_flags.shape == (1, len(periods))
    assert out.prediction_flags.dtype == np.uint32
    assert np.all(out.prediction_flags & BYPASSED)
    assert out.frequency_strategy == "anchor_continuation"
    assert out.frequency_strategy_used == stats["frequency_strategy_used"] == "independent"
    assert "high" in out.frequency_strategy_reason.lower()


def test_wrapper_restores_every_period_axis_with_a_native_test_double(monkeypatch):
    """Exercise real wrapper routing without requiring or simulating GPU math."""
    seen = []

    class Native:
        def __init__(self, capacity, layers, nf, nmodes, *config):
            self.nmodes = nmodes

        def run(self, h, vp, vs, rho, periods, bounds, *options):
            seen.append(periods.copy())
            B, T, K = len(h), len(periods), self.nmodes
            velocity = np.broadcast_to(.8+.01*periods[None, None, :]
                                       +.3*np.arange(K)[None, :, None], (B, K, T)).copy()
            rank = np.arange(T)
            flags = np.broadcast_to(np.where(rank % 2, md.TRACE_CONTINUATION_APPROXIMATE, 0),
                                    (B, T)).astype(np.uint32).copy()
            levels = np.broadcast_to(rank, (B, T)).astype(np.int32).copy()
            counts = np.full((B, T), K, dtype=np.int32)
            group_flags = np.broadcast_to((rank % 2)[None, None, :]*4,
                                         (B, K, T)).astype(np.uint32).copy()
            prediction = np.broadcast_to(np.where(rank % 2, ACCEPTED | AUDIT, ANCHOR),
                                        (B, T)).astype(np.uint32).copy()
            return velocity, flags, {}, levels, counts, dict(
                group_velocity=velocity+1., group_velocity_flags=group_flags,
                prediction_flags=prediction)

    module = SimpleNamespace(QeSolver=Native, SUPPORTS_AUTO_ACCURACY=True, available=lambda device: True,
                             set_device=lambda device: None)
    monkeypatch.setattr(md, "_load_cuda", lambda: module)
    unique = np.geomspace(.2, .4, 9)
    order = np.array([8, 2, 4, 0, 3, 5, 1, 7, 6, 2, 2])
    with md.Solver(backend="cuda", group_velocity=True,
                   frequency_strategy="anchor_continuation") as solver:
        out = solver(*MODEL, unique[order], nmodes=2)
        stats = solver.stats()
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], unique)
    phase = .8+.01*unique[None, :]+.3*np.arange(2)[:, None]
    np.testing.assert_array_equal(out.phase_velocity[0], phase[:, order])
    np.testing.assert_array_equal(out.group_velocity[0], (phase+1.)[:, order])
    np.testing.assert_array_equal(out.group_velocity_flags[0],
                                  np.broadcast_to((np.arange(9) % 2)*4, (2, 9))[:, order])
    np.testing.assert_array_equal(out.search_level[0], np.arange(9)[order])
    np.testing.assert_array_equal(out.trace_flags[0],
                                  np.where(np.arange(9) % 2, md.TRACE_CONTINUATION_APPROXIMATE, 0)[order])
    np.testing.assert_array_equal(out.prediction_flags[0],
                                  np.where(np.arange(9) % 2, ACCEPTED | AUDIT, ANCHOR)[order])
    np.testing.assert_array_equal(out.root_count, np.full((1, len(order)), 2))
    assert stats["periods_requested"] == len(order)
    assert stats["periods_computed"] == len(unique)


@pytest.mark.skipif(BACKEND != "cuda", reason="continuation acceleration requires CUDA")
@pytest.mark.parametrize("accuracy,periods", [
    ("high", np.geomspace(.2, .4, 17)), ("balanced", np.array([.2, .3, .2, .4])),
    ("balanced", np.geomspace(.1, 10., 16)),
])
def test_cuda_high_and_small_unique_grids_bypass(accuracy, periods):
    options = dict(backend="cuda", accuracy=accuracy)
    with md.Solver(**options) as reference, md.Solver(
        frequency_strategy="anchor_continuation", **options
    ) as solver:
        expected = reference(*MODEL, periods, nmodes=1, wave="love")
        out = solver(*MODEL, periods, nmodes=1, wave="love")
    np.testing.assert_array_equal(out.phase_velocity, expected.phase_velocity)
    np.testing.assert_array_equal(out.trace_flags, expected.trace_flags)
    assert np.all(out.prediction_flags & BYPASSED)
    assert out.frequency_strategy_used == "independent"
    assert out.frequency_strategy_reason


@pytest.mark.skipif(BACKEND != "cuda", reason="continuation acceleration requires CUDA")
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_cuda_dense_prediction_audits_and_preserves_root_prefix(precision):
    periods = np.geomspace(.2, .35, 65)
    options = dict(backend="cuda", precision=precision, accuracy="balanced")
    with md.Solver(group_velocity=True, **options) as reference, md.Solver(
        frequency_strategy="anchor_continuation", group_velocity=True, **options
    ) as solver:
        expected = reference(*MODEL, periods, nmodes=1, wave="love")
        out = solver(*MODEL, periods, nmodes=1, wave="love")
        stats = solver.stats()
    np.testing.assert_array_equal(out.mode_mask, expected.mode_mask)
    np.testing.assert_array_equal(out.root_count, expected.root_count)
    np.testing.assert_allclose(out.phase_velocity, expected.phase_velocity, rtol=3e-5, atol=3e-6)
    flags = out.prediction_flags
    assert flags[0, 0] & ANCHOR and flags[0, -1] & ANCHOR
    accepted = (flags & ACCEPTED) != 0
    assert accepted.any(), stats
    assert np.all(flags[accepted] & AUDIT)
    assert not np.any(flags[accepted] & (FALLBACK | AUDIT_FAILED))
    assert np.all(out.trace_flags[accepted] & md.TRACE_CONTINUATION_APPROXIMATE)
    assert np.all(out.mode_index_uncertain[accepted])
    assert out.frequency_strategy_used == "anchor_continuation"
    assert stats["continuation_accepted"] > 0
    assert stats["continuation_evaluations"] > 0
    assert stats["continuation_candidates"] >= stats["continuation_accepted"]
    assert stats["group_reused"] > 0
    np.testing.assert_allclose(out.group_velocity, expected.group_velocity,
                               rtol=1e-3, atol=3e-6, equal_nan=True)


@pytest.mark.skipif(BACKEND != "cuda", reason="continuation acceleration requires CUDA")
def test_cuda_source_sampling_hint_allows_audited_isolated_root_prediction():
    # This layer has a hidden-pair sampling hint although the complete K=4
    # recovered prefix is well separated. A hint is not an observed close pair.
    arrays = ([5., 0.], [3., 6.], [1.5, 3.5], [2.2, 2.8])
    periods = .5*np.exp(np.linspace(-.004, .004, 17))
    options = dict(backend="cuda", secular="dunkin")
    with md.Solver(accuracy="high", group_velocity=True, **options) as high, md.Solver(
        accuracy="balanced", **options
    ) as independent, md.Solver(
        accuracy="balanced", frequency_strategy="anchor_continuation", **options
    ) as solver:
        reference = high(*arrays, periods, nmodes=4)
        source = independent(*arrays, periods, nmodes=2)
        out = solver(*arrays, periods, nmodes=2)
    assert np.all(source.trace_flags & md.TRACE_CLOSE_ROOT_RISK)
    assert np.all(source.root_count == 4)
    gaps = np.diff(reference.phase_velocity, axis=1)
    assert np.all(gaps > .005*reference.phase_velocity[:, 1:])
    assert reference.group_velocity_valid.all()
    flags = out.prediction_flags
    anchors = (flags & ANCHOR) != 0
    assert not np.any(flags[anchors] & SOURCE_SEARCH_UNCERTAIN)
    assert np.all(flags[~anchors] & SOURCE_SEARCH_UNCERTAIN)
    accepted = (flags & ACCEPTED) != 0
    assert accepted.any(), flags
    assert np.all(flags[accepted] & AUDIT)
    assert not np.any(flags[accepted] & (AUDIT_FAILED | FALLBACK))
    assert np.all(out.trace_flags[accepted] & md.TRACE_CONTINUATION_APPROXIMATE)
    np.testing.assert_array_equal(out.mode_mask, source.mode_mask)
    np.testing.assert_allclose(out.phase_velocity, reference.phase_velocity[:, :2],
                               rtol=3e-5, atol=3e-6)


@pytest.mark.skipif(BACKEND != "cuda", reason="continuation acceleration requires CUDA")
def test_cuda_observed_close_guard_roots_still_force_fallback():
    arrays = ([5., 0.], [3., 6.], [1.5, 3.5], [2.2, 2.8])
    periods = .3*np.exp(np.linspace(-.004, .004, 17))
    options = dict(backend="cuda", secular="dunkin")
    with md.Solver(accuracy="high", **options) as high, md.Solver(
        accuracy="balanced", frequency_strategy="anchor_continuation", **options
    ) as solver:
        reference = high(*arrays, periods, nmodes=4)
        out = solver(*arrays, periods, nmodes=2)
    gaps = np.diff(reference.phase_velocity, axis=1)
    assert np.all(np.any(gaps <= .005*reference.phase_velocity[:, 1:], axis=1))
    nonanchors = (out.prediction_flags & ANCHOR) == 0
    assert nonanchors.any()
    assert np.all(out.prediction_flags[nonanchors] & FALLBACK)
    assert np.all(out.prediction_flags[nonanchors] & int(md.PredictionFlag.ANCHOR_UNSAFE))
    assert not np.any(out.prediction_flags & ACCEPTED)
    np.testing.assert_allclose(out.phase_velocity, reference.phase_velocity[:, :2],
                               rtol=3e-5, atol=3e-6)


@pytest.mark.skipif(BACKEND != "cuda", reason="continuation acceleration requires CUDA")
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_cuda_partial_root_prefix_preserves_missing_modes_and_group_output(precision):
    # Only one Rayleigh mode exists on this long-period interval although the
    # caller requests five. A partial prefix needs an audit of the upper tail.
    periods = np.geomspace(4., 6., 33)
    options = dict(backend="cuda", secular="dunkin", group_velocity=True)
    with md.Solver(accuracy="high", **options) as high:
        reference = high(*MODEL, periods, nmodes=5)
    with md.Solver(accuracy="balanced", precision=precision,
                   frequency_strategy="anchor_continuation", **options) as solver:
        out = solver(*MODEL, periods, nmodes=5)
        # Reuse the allocation with many roots, then return to a one-root
        # model. Unused root/group slots must not retain the earlier prefix.
        rich = solver([50., 0.], *MODEL[1:], periods, nmodes=5)
        again = solver(*MODEL, periods, nmodes=5)
    assert rich.mode_mask[:, 1:].any()
    accepted = (out.prediction_flags & ACCEPTED) != 0
    assert accepted.any(), out.prediction_flags
    assert np.all(out.prediction_flags[accepted] & AUDIT)
    np.testing.assert_array_equal(reference.root_count, np.ones((1, len(periods)), dtype=np.int32))
    np.testing.assert_array_equal(out.root_count, reference.root_count)
    np.testing.assert_array_equal(out.mode_mask, reference.mode_mask)
    np.testing.assert_allclose(out.phase_velocity, reference.phase_velocity,
                               rtol=3e-5, atol=3e-6, equal_nan=True)
    assert out.group_velocity.dtype == np.float64
    assert out.group_velocity_valid[:, 0].all()
    np.testing.assert_allclose(out.group_velocity[:, 0], reference.group_velocity[:, 0],
                               rtol=2e-3, atol=3e-6)
    assert np.isnan(out.group_velocity[:, 1:]).all()
    assert np.all(out.group_velocity_flags[:, 1:] & 1)
    assert np.all(out.status[:, 1:] == md.SolveStatus.NOT_FOUND)
    assert np.all(out.trace_flags[accepted] & md.TRACE_CONTINUATION_APPROXIMATE)
    for name in ("phase_velocity", "root_count", "group_velocity", "group_velocity_flags"):
        np.testing.assert_array_equal(getattr(out, name), getattr(again, name))


@pytest.mark.skipif(BACKEND != "cuda", reason="continuation acceleration requires CUDA")
def test_cuda_root_birth_between_different_anchor_counts_forces_fallback():
    # The first higher Love mode reaches the half-space cutoff inside this
    # interval: neighboring anchors contain two roots and one root.
    periods = np.geomspace(1.15, 1.25, 33)
    options = dict(backend="cuda", secular="dunkin")
    with md.Solver(accuracy="high", **options) as high, md.Solver(
        accuracy="balanced", anchor_stride=4,
        frequency_strategy="anchor_continuation", **options
    ) as solver:
        reference = high(*MODEL, periods, nmodes=5, wave="love")
        out = solver(*MODEL, periods, nmodes=5, wave="love")
    np.testing.assert_array_equal(np.unique(reference.root_count), [1, 2])
    anchors = np.flatnonzero(out.prediction_flags[0] & ANCHOR)
    crossing = []
    for left, right in zip(anchors[:-1], anchors[1:]):
        if reference.root_count[0, left] != reference.root_count[0, right]:
            crossing.extend(range(left+1, right))
    assert crossing, out.prediction_flags
    assert np.all(out.prediction_flags[0, crossing] & FALLBACK)
    assert np.all(out.prediction_flags[0, crossing] & int(md.PredictionFlag.ROOT_COUNT_UNSAFE))
    assert not np.any(out.prediction_flags[0, crossing] & ACCEPTED)
    np.testing.assert_array_equal(out.root_count, reference.root_count)
    np.testing.assert_array_equal(out.mode_mask, reference.mode_mask)
    np.testing.assert_allclose(out.phase_velocity, reference.phase_velocity,
                               rtol=3e-5, atol=3e-6, equal_nan=True)


@pytest.mark.skipif(BACKEND != "cuda", reason="continuation acceleration requires CUDA")
def test_cuda_fp32_alias_witness_accepted_prefixes_match_independent_high():
    # The stored balanced FP32 path contains nine over-budget values caused
    # by repeated roots and shifted ranks. Current GPU aliases can vary with
    # architecture; retain the historical fact without requiring exactly nine
    # differences from today's independent solve.
    path = Path(__file__).parent/"data"/"continuation_fp32_aliases.npz"
    if not path.is_file():
        pytest.skip("requires the local FP32 regression fixture")
    with np.load(path, allow_pickle=False) as fixture:
        batch = {name: fixture[name].copy() for name in ("thickness", "vp", "vs", "rho")}
        periods = fixture["periods"].copy()
        stored_high = fixture["high_fp64_phase_velocity"].copy()
        stored_plain = fixture["full_independent_fp32_phase_velocity"].copy()
        stored_continuation = fixture["full_continuation_fp32_phase_velocity"].copy()
        bad_cells = fixture["bad_cells_local"].copy()

    def changed_traces(candidate, reference):
        both = np.isfinite(candidate) & np.isfinite(reference)
        changed = np.isfinite(candidate) != np.isfinite(reference)
        changed |= both & (np.abs(candidate-reference) >
                           np.maximum(3e-6, 3e-5*np.abs(reference)))
        return changed, np.any(changed, axis=1)

    frozen_bad, frozen_traces = changed_traces(stored_plain, stored_high)
    assert int(frozen_bad.sum()) == 9
    assert int(frozen_traces.sum()) == 4
    np.testing.assert_array_equal(np.argwhere(frozen_bad), bad_cells)
    np.testing.assert_array_equal(np.isfinite(stored_continuation), np.isfinite(stored_high))
    np.testing.assert_allclose(stored_continuation, stored_high, rtol=3e-5, atol=3e-6)

    options = dict(backend="cuda", precision="float32", accuracy="balanced")
    with md.Solver(**options) as independent, md.Solver(
        frequency_strategy="anchor_continuation", **options
    ) as solver, md.Solver(
        backend="cuda", precision="float64", accuracy="high", c_tol_rel=1e-9
    ) as high:
        plain = independent(**batch, periods=periods, nmodes=5)
        out = solver(**batch, periods=periods, nmodes=5)
        reference = high(**batch, periods=periods, nmodes=5)

    def assert_full_prefix(traces):
        # A trace-level mask selects every requested rank, including finite
        # masks and guard-aware root counts. Checking only changed cells could
        # conceal an additional duplicate or a shifted prefix.
        actual = out.phase_velocity.transpose(0, 2, 1)[traces]
        expected = reference.phase_velocity.transpose(0, 2, 1)[traces]
        np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
        np.testing.assert_array_equal(out.root_count[traces], reference.root_count[traces])
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6, equal_nan=True)

    flags = out.prediction_flags
    accepted = (flags & ACCEPTED) != 0
    assert accepted.any(), flags
    assert_full_prefix(accepted)
    assert np.all(flags[accepted] & AUDIT)
    assert not np.any(flags[accepted] & (FALLBACK | AUDIT_FAILED))
    assert np.all(out.trace_flags[accepted] & md.TRACE_CONTINUATION_APPROXIMATE)
    assert np.all(out.mode_index_uncertain[accepted])

    # Source-risk hints remain visible on their dependent targets, even when
    # a coarse audit accepts a physically supported, isolated prefix.
    assert np.any(flags[accepted] & SOURCE_SEARCH_UNCERTAIN)
    for model in range(len(batch["thickness"])):
        anchors = np.flatnonzero(flags[model] & ANCHOR)
        for left, right in zip(anchors[:-1], anchors[1:]):
            hinted = bool((out.trace_flags[model, left] | out.trace_flags[model, right])
                          & md.TRACE_CLOSE_ROOT_RISK)
            np.testing.assert_array_equal(
                (flags[model, left+1:right] & SOURCE_SEARCH_UNCERTAIN) != 0,
                np.full(right-left-1, hinted, dtype=bool))

    _, current_changed = changed_traces(out.phase_velocity, plain.phase_velocity)
    if current_changed.any():
        assert_full_prefix(current_changed)
    # Anchors and unchanged global fallbacks retain their existing FP32
    # behavior; no blanket high-accuracy assertion is made for those traces.


@pytest.mark.skipif(BACKEND != "cuda", reason="continuation acceleration requires CUDA")
def test_cuda_unsorted_duplicates_chunking_and_reuse_restore_input_order():
    unique = np.geomspace(.2, .35, 33)
    order = np.r_[np.arange(32, -1, -1), [4, 12, 4]]
    periods = unique[order]
    batch = {name: np.repeat(np.asarray(value)[None, :], 3, axis=0)
             for name, value in zip(("thickness", "vp", "vs", "rho"), MODEL)}
    options = dict(backend="cuda", accuracy="balanced", group_velocity=True,
                   frequency_strategy="anchor_continuation")
    with md.Solver(chunk_size=1, **options) as solver:
        out = solver(**batch, periods=periods, nmodes=1, wave="love")
        solver([0.], [6.], [3.5], [2.7], [1.], nmodes=1)
        again = solver(**batch, periods=periods, nmodes=1, wave="love")
    with md.Solver(chunk_size=3, **options) as solver:
        sorted_out = solver(**batch, periods=unique, nmodes=1, wave="love")
    np.testing.assert_array_equal(out.period, periods)
    for name in ("phase_velocity", "group_velocity", "group_velocity_flags"):
        np.testing.assert_array_equal(getattr(out, name), getattr(again, name))
        np.testing.assert_array_equal(getattr(out, name), getattr(sorted_out, name)[:, :, order])
    np.testing.assert_array_equal(out.prediction_flags, sorted_out.prediction_flags[:, order])


@pytest.mark.skipif(BACKEND != "cuda", reason="continuation acceleration requires CUDA")
def test_cuda_unsafe_anchor_interval_falls_back_to_independent_search():
    # No Love guide provides no prefix from which a local predictor can start.
    # This deterministic unsafe case must not invent roots or claim acceptance.
    periods = np.geomspace(.1, 2., 33)
    with md.Solver(backend="cuda", frequency_strategy="anchor_continuation") as solver:
        out = solver([0.], [6.], [3.5], [2.7], periods, nmodes=2, wave="love")
        stats = solver.stats()
    assert np.isnan(out.phase_velocity).all()
    assert not np.any(out.prediction_flags & ACCEPTED)
    nonanchors = (out.prediction_flags & ANCHOR) == 0
    assert nonanchors.any()
    assert np.all(out.prediction_flags[nonanchors] & (FALLBACK | BYPASSED))


@pytest.mark.skipif(BACKEND != "cuda", reason="continuation acceleration requires CUDA")
@pytest.mark.parametrize("case", ["lvz", "witness", "water"])
def test_cuda_rayleigh_difficult_cases_against_independent_high(case, monkeypatch):
    if case == "witness":
        with np.load(Path(__file__).parent/"data"/"witness_model9951_period23.npz") as z:
            arrays = [z[k].copy() for k in ("thickness", "vp", "vs", "rho")]
            center = float(z["periods"][0])
    elif case == "lvz":
        model = next(m for m in md.CANONICAL_MODELS if m.name == "m04_lvz")
        arrays = [model.thickness, model.vp, model.vs, model.rho]
        center = 1.441220
    else:
        arrays = ([.2, 1., 0.], [1.5, 3., 6.], [0., 1.5, 3.5], [1.02, 2., 2.7])
        center = .5
    periods = center*np.exp(np.linspace(-.004, .004, 17))
    common = dict(backend="cuda", cmin=.1)
    with md.Solver(accuracy="high", **common) as high, md.Solver(
        accuracy="balanced", **common
    ) as independent, md.Solver(
        accuracy="balanced", frequency_strategy="anchor_continuation", **common
    ) as solver:
        reference = high(*arrays, periods, nmodes=7)
        expected = independent(*arrays, periods, nmodes=5)
        out = solver(*arrays, periods, nmodes=5)
    # A tolerant preset can report a close cluster; it must still retain every
    # isolated high-reference root and flag any resulting rank uncertainty.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"tools"))
    from tolerant_metrics import compare
    metrics = compare(out.phase_velocity, reference.phase_velocity, out.trace_flags,
                      reference_flags=reference.trace_flags, error_rtol=3e-5)
    assert metrics["accepted"], metrics
    np.testing.assert_array_equal(out.mode_mask, expected.mode_mask)
    np.testing.assert_allclose(out.phase_velocity, expected.phase_velocity,
                               rtol=3e-5, atol=3e-6, equal_nan=True)
    newly_unresolved = ((out.trace_flags & md.TRACE_SEARCH_UNCONVERGED) != 0) & (
        (expected.trace_flags & md.TRACE_SEARCH_UNCONVERGED) == 0)
    assert not newly_unresolved.any()
    if case == "lvz":
        assert np.any(out.prediction_flags & FALLBACK)


def test_cli_continuation_metadata_and_bypass_flags(tmp_path):
    source, target = tmp_path/"input.npz", tmp_path/"output.npz"
    np.savez(source, thickness=[0.], vp=[6.], vs=[3.5], rho=[2.7], periods=[1., 2.])
    assert main(["forward", str(source), "--output", str(target), "--backend", "cpu",
                 "--nmodes", "1", "--frequency-strategy", "anchor_continuation",
                 "--anchor-stride", "3"]) == 0
    with np.load(target, allow_pickle=False) as out:
        assert str(out["frequency_strategy"]) == "anchor_continuation"
        assert str(out["frequency_strategy_used"]) == "independent"
        assert str(out["frequency_strategy_reason"])
        assert int(out["anchor_stride"]) == 3
        assert out["prediction_flags"].dtype == np.uint32
        assert np.all(out["prediction_flags"] & BYPASSED)
