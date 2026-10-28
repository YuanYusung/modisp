#!/Users/yusong/miniforge3/bin/python
"""Keep ordinary-root errors separate from flagged close-cluster ambiguity.

References may contain guard roots beyond the requested output prefix. Values
above a truncated reference prefix are counted as unknown, never as validated.
"""
import numpy as np

APPROXIMATE = 2048
CLOSE_RISK = 4096


def compare(velocity, reference, flags, close_rtol=.005, error_rtol=2e-5, reference_flags=None, reference_complete=None):
    velocity, reference, flags = map(np.asarray, (velocity, reference, flags))
    if velocity.ndim != 3 or reference.ndim != 3 or velocity.shape[0] != reference.shape[0] or velocity.shape[2] != reference.shape[2]:
        raise ValueError('expected compatible [B,M,T] arrays')
    metrics = {k: 0 for k in (
        'traces', 'reference_close_traces', 'approximate_traces', 'close_risk_traces',
        'ordinal_wrong', 'ordinal_missing', 'ordinary_missing', 'ordinary_spurious',
        'close_cluster_missing', 'cluster_approximations', 'mode_index_shifts',
        'beyond_reference_prefix', 'unflagged_uncertainty', 'unresolved_traces',
        'reference_unresolved_traces', 'unjudged_finite_cells', 'unjudged_reference_boundary')}
    metrics['max_matched_relative_error'] = 0.
    for b in range(velocity.shape[0]):
        for t in range(velocity.shape[2]):
            x = velocity[b, :, t]
            r = reference[b, :, t]
            full = r[np.isfinite(r)]
            finite = x[np.isfinite(x)]
            word = int(flags[b, t])
            marked = bool(word & (APPROXIMATE | CLOSE_RISK))
            metrics['traces'] += 1
            metrics['approximate_traces'] += bool(word & APPROXIMATE)
            metrics['close_risk_traces'] += bool(word & CLOSE_RISK)
            metrics['unresolved_traces'] += bool(word & 4)
            if reference_flags is not None and int(reference_flags[b,t])&4:
                metrics['reference_unresolved_traces'] += 1
                metrics['unjudged_finite_cells'] += len(finite)
                metrics['unflagged_uncertainty'] += bool(len(finite) and not marked)
                continue
            groups = []
            for value in full:
                if groups and value-groups[-1][-1] <= close_rtol*value:
                    groups[-1].append(value)
                else:
                    groups.append([value])
            metrics['reference_close_traces'] += any(len(g) > 1 for g in groups)
            complete=(len(full)<reference.shape[1]) if reference_complete is None else bool(reference_complete[b,t])
            # Every isolated reference root in the requested prefix must exist
            # somewhere in the returned set, even on a flagged trace.
            prefix = 0
            for group in groups:
                if prefix >= len(x):
                    break
                hit = any(np.any(np.abs(finite-value) <= error_rtol*value) for value in group)
                if len(group) == 1 and not hit:
                    if prefix==len(full)-1 and not complete:
                        metrics['unjudged_reference_boundary'] += 1
                        metrics['unflagged_uncertainty'] += not marked
                    else:metrics['ordinary_missing'] += 1
                elif len(group) > 1 and not hit:
                    metrics['close_cluster_missing'] += 1
                    metrics['unflagged_uncertainty'] += not marked
                prefix += len(group)
            for value in finite:
                if not len(full):
                    metrics['ordinary_spurious'] += 1
                    continue
                nearest = int(np.argmin(np.abs(full-value)))
                relative = abs(value-full[nearest])/full[nearest]
                if relative <= error_rtol:
                    metrics['max_matched_relative_error'] = max(metrics['max_matched_relative_error'], float(relative))
                    continue
                clustered = any(len(g)>1 and g[0]*(1-close_rtol) <= value <= g[-1]*(1+close_rtol) for g in groups)
                if clustered:
                    metrics['cluster_approximations'] += 1
                    metrics['unflagged_uncertainty'] += not marked
                elif not complete and value > full[-1]:
                    metrics['beyond_reference_prefix'] += 1
                    metrics['unflagged_uncertainty'] += not marked
                else:
                    metrics['ordinary_spurious'] += 1
            for j in range(min(len(x), len(r))):
                if np.isfinite(r[j]) and not np.isfinite(x[j]):
                    metrics['ordinal_missing'] += 1
                if np.isfinite(r[j]) and np.isfinite(x[j]) and abs(x[j]-r[j]) > error_rtol*r[j]:
                    metrics['ordinal_wrong'] += 1
                    if len(full) and np.any(np.abs(full-x[j]) <= error_rtol*full):
                        metrics['mode_index_shifts'] += 1
                    metrics['unflagged_uncertainty'] += not marked
    metrics['ordinary_failures'] = metrics['ordinary_missing'] + metrics['ordinary_spurious']
    metrics['accepted'] = metrics['ordinary_failures'] == 0 and metrics['unflagged_uncertainty'] == 0
    return metrics
