#!/Users/yusong/miniforge3/bin/python
"""Numerical accounting for backend benchmarks; no solver or timing code."""
from __future__ import annotations

import numpy as np


def quantiles(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not a.size:
        return dict.fromkeys(("p50", "p95", "p99", "max"))
    return dict(zip(("p50", "p95", "p99", "max"),
                    map(float, np.quantile(a, [.5, .95, .99, 1.], method="linear"))))


def compare(actual, reference, *, budget_rtol=2e-6, stable=None):
    """Retain ordinal errors AND one-to-one root matches on a known prefix.

    reference can have guard roots beyond the requested K. A finite candidate
    past an exhausted reference prefix is unjudged, never automatically extra.
    Rows marked reference-unresolved are excluded and explicitly counted.
    """
    v = np.asarray(actual["phase_velocity"], dtype=np.float64)
    r = np.asarray(reference["phase_velocity"], dtype=np.float64)
    if v.shape[0] != r.shape[0] or v.shape[2] != r.shape[2] or r.shape[1] < v.shape[1]:
        raise ValueError("reference must cover the same models/periods and at least K roots")
    B, K, T = v.shape
    ref_flags = np.asarray(reference.get("trace_flags", np.zeros((B, T), dtype=np.uint32)))
    judged = (ref_flags & (4 | 1)) == 0
    if stable is not None:
        judged &= np.asarray(stable, dtype=bool)
    finite_v, finite_r = np.isfinite(v), np.isfinite(r[:, :K])
    common = finite_v & finite_r & judged[:, None, :]
    err = np.abs(v[common] - r[:, :K][common])
    rel = err / np.abs(r[:, :K][common])
    strict = np.maximum(3e-6, 2e-6 * np.abs(r[:, :K][common]))
    metrics = dict(
        total_cells=int(v.size), total_traces=B*T,
        judged_traces=int(judged.sum()), unjudged_reference_traces=int((~judged).sum()),
        reference_valid_cells=int((finite_r & judged[:, None, :]).sum()),
        finite_common_cells=int(common.sum()),
        ordinal_missing=int((finite_r & ~finite_v & judged[:, None, :]).sum()),
        ordinal_extra=int((~finite_r & finite_v & judged[:, None, :]).sum()),
        ordinal_wrong=int((err > strict).sum()),
        ordinal_budget_wrong=int((err > np.maximum(3e-6, budget_rtol*np.abs(r[:, :K][common]))).sum()),
        abs_error_km_s=quantiles(err), relative_error=quantiles(rel),
        set_missing=0, set_extra=0, unjudged_prefix_cells=0,
        cluster_unmatched=0, isolated_missing=0, isolated_spurious=0,
        rank_shifts=0, unflagged_uncertainty=0, matched_cells=0,
        per_mode=[],
    )
    matched_errors, matched_relative = [], []
    for k in range(K):
        mask = common[:, k]
        e = np.abs(v[:, k][mask] - r[:, k][mask])
        metrics["per_mode"].append(dict(mode=k, common=int(mask.sum()),
            abs_error_km_s=quantiles(e), relative_error=quantiles(e / np.abs(r[:, k][mask]))))
    for b, t in np.argwhere(judged):
        ref_all = r[b, :, t]
        ref_roots = ref_all[np.isfinite(ref_all)]
        wanted = ref_roots[:K]
        vals = v[b, :, t]
        indices = np.flatnonzero(np.isfinite(vals))
        xs = vals[indices]
        close = np.zeros(len(ref_roots), dtype=bool)
        if len(ref_roots) > 1:
            gaps = np.diff(ref_roots) <= .005 * ref_roots[1:]
            close[:-1] |= gaps
            close[1:] |= gaps
        # Ascending, one-to-one merge. Matching can use the two reference guards
        # to identify a shifted prefix rather than spuriously calling it extra.
        i = j = 0
        trace_shift = False
        matched_ref = set()
        matched_actual = set()
        while i < len(ref_roots) and j < len(xs):
            e = abs(xs[j] - ref_roots[i])
            if e <= max(3e-6, budget_rtol*abs(ref_roots[i])):
                matched_ref.add(i)
                matched_actual.add(j)
                matched_errors.append(e)
                matched_relative.append(e/abs(ref_roots[i]))
                metrics["rank_shifts"] += int(indices[j] != i)
                trace_shift |= indices[j] != i
                i += 1
                j += 1
            elif xs[j] < ref_roots[i]:
                j += 1
            else:
                i += 1
        for i in range(len(wanted)):
            if i not in matched_ref:
                metrics["set_missing"] += 1
                metrics["cluster_unmatched" if close[i] else "isolated_missing"] += 1
        prefix_exhausted = len(ref_roots) == r.shape[1]
        for j, x in enumerate(xs):
            if j in matched_actual:
                continue
            if prefix_exhausted and x > ref_roots[-1]:
                metrics["unjudged_prefix_cells"] += 1
            else:
                metrics["set_extra"] += 1
                in_cluster = any(close[i] and abs(x-y) <= .005*abs(y)
                                 for i, y in enumerate(ref_roots))
                metrics["isolated_spurious"] += int(not in_cluster)
        changed = trace_shift or any(i not in matched_ref for i in range(len(wanted)))
        if changed and not (int(actual.get("trace_flags", np.zeros((B,T), dtype=np.uint32))[b,t]) & (2048|4096)):
            metrics["unflagged_uncertainty"] += 1
    metrics["matched_cells"] = len(matched_errors)
    metrics["matched_abs_error_km_s"] = quantiles(matched_errors)
    metrics["matched_relative_error"] = quantiles(matched_relative)
    for name in ("mode_mask", "status", "root_count", "search_level", "trace_flags", "numerical_flags"):
        if name not in actual or name not in reference:
            continue
        x, y = np.asarray(actual[name]), np.asarray(reference[name])
        if x.ndim == 3:
            y = y[:, :K]
        if name == "root_count":
            y = np.where(y < 0, y, np.minimum(y, K+2))
        metrics[name + "_mismatches"] = int(np.count_nonzero(x != y))
    if "trace_flags" in actual and "trace_flags" in reference:
        xor = np.asarray(actual["trace_flags"], dtype=np.uint32) ^ ref_flags
        metrics["per_bit_xor"] = {str(1 << bit): int(np.count_nonzero(xor & (1 << bit)))
                                  for bit in range(16)}
        candidate_bad = (np.asarray(actual["trace_flags"]) & (4|1)) != 0
        ref_bad = (ref_flags & (4|1)) != 0
        metrics["newly_unresolved_traces"] = int(np.count_nonzero(candidate_bad & ~ref_bad))
    if "status" in actual and "status" in reference:
        a_bad = np.isin(actual["status"], [2,3])
        r_bad = np.isin(reference["status"][:, :K], [2,3])
        metrics["newly_numerical_failed_cells"] = int(np.count_nonzero(a_bad & ~r_bad))
        metrics["newly_numerical_failed_traces"] = int(np.count_nonzero(
            np.any(a_bad,axis=1) & ~np.any(r_bad,axis=1)))
    if "search_bounds" in actual and "search_bounds" in reference:
        d = np.abs(actual["search_bounds"] - reference["search_bounds"])
        metrics["bounds_max_abs_km_s"] = float(d.max())
    return metrics
