"""Screen nearby periods and confirm suspicious mode sequences in FP64."""
from enum import IntFlag
import numpy as np


class ContinuityFlag(IntFlag):
    REVIEWED = 1
    REPAIRED = 2
    UNRESOLVED = 4
    NO_BILATERAL = 8
    ENVELOPE = 16
    CLOSE_FP32 = 32
    PREDICTION = 64
    SEARCH = 128


def review(solver, backend, models, periods, wave, bounds, velocity, trace, levels, counts,
           group, group_flags, flags, stats, occupied_bytes):
    """Confirm the complete prefix plus two guard roots, compacting only suspect rows."""
    from . import _cpu, Solver, TRACE_SEARCH_UNCONVERGED, TRACE_UNRESOLVED_BRACKET, TRACE_F_REJECTED, TRACE_FP64_REPAIR, TRACE_NEAR_CUTOFF
    reason_bits = int(ContinuityFlag.ENVELOPE | ContinuityFlag.CLOSE_FP32 | ContinuityFlag.PREDICTION | ContinuityFlag.SEARCH)
    model_ids, freq_ids = np.nonzero(flags & reason_bits)
    n = len(model_ids)
    stats['continuity_suspected'] = n
    if not n:
        return
    modes = velocity.shape[1]
    guards = min(256, modes + 2)
    # Models with the same suspect periods can share a native batch. On CPU this
    # also lets one prepared model serve every flagged period in that batch.
    batches = {}
    paired = backend == 'cuda' and getattr(solver._module, 'SUPPORTS_ROW_PERIODS', False)
    if backend == 'cpu':
        for ids in np.split(np.arange(n), np.flatnonzero(np.diff(model_ids)) + 1):
            batches.setdefault(tuple(freq_ids[ids]), []).append(model_ids[ids[0]])
    elif not paired:
        for f in np.unique(freq_ids):
            batches[(f,)] = model_ids[freq_ids == f]

    # At the public 256-mode limit the last mode's right neighbor is private
    # to the native search. Older optional plugins likewise use native output.
    native = _cpu if backend == 'cpu' else solver._module
    group_from_roots = getattr(native, '_review_groups', None)
    defer_groups = group is not None and modes < 256 and group_from_roots is not None
    # GroupResult holds two doubles and two 32-bit fields. At most one
    # accepted trace's scratch is live beside the verifier's native buffers.
    scratch = 24*modes if defer_groups else 0
    if defer_groups and backend == 'cuda':
        scratch += 24*modes + 8*(4*models[0].shape[1] + guards)
    settings = dict(backend=backend, device=solver.device, threads=solver.threads, accuracy='high',
                    cmin=solver.cmin, cmax=solver.cmax, c_tol_rel=min(solver.c_tol_rel, 1e-6),
                    f_accept=solver.f_accept, secular=solver.secular, refinement=solver.refinement,
                    group_velocity=solver.group_velocity and not defer_groups, group_rtol=solver.group_rtol,
                    close_root_rtol=solver.close_root_rtol, continuity_check=False,
                    workspace_bytes=max(1, solver.workspace_bytes-scratch), chunk_size=solver.chunk_size)

    with Solver(**settings) as verifier:
        def solve(rows, frequencies, limits, local):
            # GPU verification accepts one original period per row. This avoids
            # evaluating the Cartesian product or launching each frequency alone.
            selected_periods = periods[frequencies]
            per_row = {'_row_periods': selected_periods} if paired else {}
            traces = len(rows) if paired else len(rows)*len(frequencies)
            stats['continuity_' + ('local_searches' if local else 'full_searches')] += traces
            try:
                result = verifier._solve(*(v[rows] for v in models), selected_periods[:1] if paired else selected_periods, guards,
                                         wave, _bounds=limits, _reserved=occupied_bytes, _owner=solver, **per_row)
            except MemoryError:
                # A user budget sufficient for the original solve may be too
                # small for high precision and its two additional guard roots.
                return None
            counters = verifier.stats()
            if paired:
                stats['continuity_cuda_batches'] = stats.get('continuity_cuda_batches', 0) + counters['chunks']
            evaluations = counters.get('evaluations', 0)
            derivative_evals = counters.get('group_evaluations', 0)
            if group is not None:
                stats['group_evaluations'] += derivative_evals
                stats['continuity_group_evaluations'] = stats.get('continuity_group_evaluations', 0) + derivative_evals
            stats['continuity_evaluations'] += evaluations
            stats['evaluations'] = stats.get('evaluations', 0) + evaluations
            return result

        def adopt(result, i, j, row, f, local):
            if result is None:
                return False
            count = int(result.root_count[i, j])
            roots = result.phase_velocity[i, :, j]
            known = min(guards, count)
            if (count < 0 or result.search_level[i, j] < 0 or
                    result.trace_flags[i, j] & (TRACE_SEARCH_UNCONVERGED | TRACE_UNRESOLVED_BRACKET | TRACE_F_REJECTED)):
                return False
            if local and count < modes + 2:
                return False
            recovered = roots[:known].astype(velocity.dtype)
            if not np.isfinite(recovered).all() or np.any(recovered[1:] <= recovered[:-1]):
                return False
            if local and (not known or roots[known-1] >= result.search_bounds[i, 1]*(1-4*verifier.c_tol_rel)):
                return False
            old = velocity[row, :, f]
            new = roots[:modes].astype(old.dtype)
            count = min(count, modes + 2)
            alias_changed = (flags[row, f] & int(ContinuityFlag.CLOSE_FP32)
                             and not np.array_equal(old, new, equal_nan=True))
            search_failed = bool(flags[row, f] & int(ContinuityFlag.SEARCH))
            if (search_failed or counts[row, f] != count or alias_changed or
                    not np.allclose(old, new, rtol=solver.c_tol_rel, atol=0, equal_nan=True)):
                if defer_groups:
                    packed = np.concatenate((*[v[row] for v in models], roots[:known]))
                    try:
                        new_group, new_flags, evaluations = group_from_roots(
                            packed, models[0].shape[1], modes, int(wave == 'love'),
                            int(result.secular == 'fast'), periods[f], *result.search_bounds[i],
                            solver.group_rtol, solver.close_root_rtol)
                    except MemoryError:
                        return False
                    stats['evaluations'] += evaluations
                    stats['continuity_evaluations'] += evaluations
                    stats['group_evaluations'] += evaluations
                    stats['continuity_group_evaluations'] = stats.get('continuity_group_evaluations', 0) + evaluations
                elif group is not None:
                    new_group = result.group_velocity[i, :modes, j]
                    new_flags = result.group_velocity_flags[i, :modes, j]
                old[:] = new
                counts[row, f] = count
                word = int(result.trace_flags[i, j])
                cutoff = .99 * models[2][row, -1]
                if (word & TRACE_NEAR_CUTOFF and np.any(roots[modes:] >= cutoff)
                        and not np.any(roots[:modes] >= cutoff)):
                    word &= ~TRACE_NEAR_CUTOFF
                trace[row, f] = word
                if old.dtype == np.float32:
                    trace[row, f] |= TRACE_FP64_REPAIR
                levels[row, f] = result.search_level[i, j]
                if group is not None:
                    stats['group_invalid'] += np.count_nonzero(new_flags) - np.count_nonzero(group_flags[row, :, f])
                    group[row, :, f] = new_group
                    group_flags[row, :, f] = new_flags
                flags[row, f] |= int(ContinuityFlag.REPAIRED)
                stats['continuity_repaired'] += 1
            return True

        def local_window(roots, counts_in_batch, limits):
            # Preserve rank by starting at the original lower bound. Only try
            # a truncated upper bound when there is evidence for two guards.
            observed = np.where(np.isfinite(roots), roots, 0.).max(axis=(1, 2))
            if modes > 1:
                gaps = np.diff(roots, axis=1)
                margin = np.where(np.isfinite(gaps), gaps, 0.).max(axis=(1, 2))
            else:
                margin = .1 * observed
            upper = observed + np.maximum(2.5*margin, .02*observed)
            have_guards = np.all(counts_in_batch >= modes + 2, axis=1)
            eligible = have_guards & (guards == modes + 2) & (upper > limits[:, 0]) & (upper < limits[:, 1]*.98)
            return eligible, upper

        if paired:
            limits = bounds[model_ids]
            eligible, upper = local_window(velocity[model_ids, :, freq_ids, None],
                                           counts[model_ids, freq_ids, None], limits)
            pending = np.ones(n, dtype=bool)
            for local in (True, False):
                ids = np.flatnonzero(eligible if local else pending)
                if not ids.size:
                    continue
                cut = limits[ids].copy()
                if local:
                    cut[:, 1] = upper[ids]
                result = solve(model_ids[ids], freq_ids[ids], cut, local)
                for i, k in enumerate(ids):
                    pending[k] = not adopt(result, i, 0, model_ids[k], freq_ids[k], local)
            flags[model_ids[pending], freq_ids[pending]] |= int(ContinuityFlag.UNRESOLVED)
            stats['continuity_unresolved'] += np.count_nonzero(pending)

        for indices, selected in batches.items():
            rows = np.asarray(selected, dtype=np.intp)
            frequencies = np.array(indices, dtype=np.intp)
            limits = bounds[rows].copy()
            roots = velocity[np.ix_(rows, np.arange(modes), frequencies)]
            local_rows, upper = local_window(roots, counts[np.ix_(rows, frequencies)], limits)
            pending = np.ones((len(rows), len(frequencies)), dtype=bool)
            if np.any(local_rows):
                ids = np.flatnonzero(local_rows)
                local_bounds = limits[ids].copy()
                local_bounds[:, 1] = upper[ids]
                candidate = solve(rows[ids], frequencies, local_bounds, True)
                for i, k in enumerate(ids):
                    for j, f in enumerate(frequencies):
                        pending[k, j] = not adopt(candidate, i, j, rows[k], f, True)
            # Group equal remaining CPU period lists to share layer preparation
            # during full review too. CUDA batches already contain a single period.
            remaining = {}
            for k, mask in enumerate(pending):
                cols = tuple(np.flatnonzero(mask))
                if cols:
                    remaining.setdefault(cols, []).append(k)
            for cols, ids in remaining.items():
                requested = frequencies[list(cols)]
                full = solve(rows[ids], requested, limits[ids], False)
                for i, k in enumerate(ids):
                    for j, f in enumerate(requested):
                        if not adopt(full, i, j, rows[k], f, False):
                            flags[rows[k], f] |= int(ContinuityFlag.UNRESOLVED)
                            stats['continuity_unresolved'] += 1
    flags[model_ids, freq_ids] |= int(ContinuityFlag.REVIEWED)
