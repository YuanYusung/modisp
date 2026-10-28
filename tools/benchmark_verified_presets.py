#!/Users/yusong/miniforge3/bin/python
"""Check independent references, then time interleaved equal-accuracy presets.

Never waive wrong/missing/extra/unresolved failures for performance results.
Use PYTHONPATH or an isolated environment to select the wheel under test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import modisp as md

from benchmark_modephys import convert, RESULTS

FIELDS = ('thickness', 'vp', 'vs', 'rho')
PRESETS = ('high', 'balanced', 'fast')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024**2), b''):
            h.update(part)
    return h.hexdigest()


def identical(a, b):
    for name in RESULTS:
        x, y = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        if x.dtype.kind == 'f':
            x, y = x.view(np.uint64), y.view(np.uint64)
        if not np.array_equal(x, y):
            raise AssertionError(f'{name} differs from high')


def external(args):
    reports = []
    for path in sorted(args.references.glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            model = {k: data[k] for k in FIELDS}
            per, reference = data['periods'], data['reference']
            stable = np.asarray(data['reference_stable'], dtype=bool)
            if not np.all(stable):
                raise ValueError(f'{path}: reference instability requires explicit treatment')
            low = float(min(.2, .75*np.min(model['vs']))) if path.stem.startswith('qed_') else .2
            case = dict(suite=path.stem, sha256=digest(path), models=len(reference),
                        traces=len(reference)*len(per), presets={})
            for accuracy in PRESETS:
                with md.Solver(backend=args.backend, threads=args.threads, accuracy=accuracy,
                               cmin=low, workspace_bytes=1024**3) as solver:
                    result = solver(**model, periods=per, nmodes=reference.shape[1])
                a, b = result.phase_velocity, reference
                fa, fb = np.isfinite(a), np.isfinite(b)
                common = fa & fb
                err = np.abs(a[common]-b[common])
                metrics = dict(wrong=int(np.count_nonzero(err > np.maximum(3e-6, 2e-6*np.abs(b[common])))),
                               missing=int(np.count_nonzero(fb & ~fa)), extra=int(np.count_nonzero(fa & ~fb)),
                               unresolved=int(np.count_nonzero(result.trace_flags & md.TRACE_SEARCH_UNCONVERGED)),
                               max_common_error=float(err.max()) if err.size else None)
                case['presets'][accuracy] = metrics
                if any(metrics[k] for k in ('wrong', 'missing', 'extra', 'unresolved')):
                    raise AssertionError((path, accuracy, metrics))
                np.savez_compressed(args.output/f'{path.stem}.{accuracy}.npz', **{k:getattr(result,k) for k in RESULTS})
            reports.append(case)
            print('reference', path.stem, 'clean for all presets', flush=True)
    if len(reports) != 11:
        raise ValueError('expected all 11 independent reference suites')
    return reports


def performance(args):
    raw = np.load(args.input, mmap_mode='r')
    if not 0 < args.models <= len(raw):
        raise ValueError("requested models exceed available independent inputs")
    model = convert(raw[:args.models])
    per = np.load(args.periods)[::args.period_stride]
    solvers = {p:md.Solver(backend=args.backend, threads=args.threads, accuracy=p,
                          workspace_bytes=1024**3) for p in PRESETS}
    reports = {p:dict(first=None, samples=[], stats=None) for p in PRESETS}
    try:
        # Cold calls and an independent warmup are excluded from steady medians.
        reference = None
        for p in PRESETS:
            start = time.perf_counter()
            result = solvers[p](**model, periods=per, nmodes=5)
            reports[p]['first'] = time.perf_counter()-start
            if reference is None:
                reference = result
            identical(reference, result)
            np.savez_compressed(args.output/f'performance.{p}.first.npz', **{k:getattr(result,k) for k in RESULTS})
            solvers[p](**model, periods=per, nmodes=5)
        for i in range(args.repeats):
            order = PRESETS[i % 3:] + PRESETS[:i % 3]
            for p in order:
                start = time.perf_counter()
                result = solvers[p](**model, periods=per, nmodes=5)
                reports[p]['samples'].append(time.perf_counter()-start)
                identical(reference, result)
                reports[p]['stats'] = solvers[p].stats()
                if i == args.repeats-1:
                    np.savez_compressed(args.output/f'performance.{p}.last.npz', **{k:getattr(result,k) for k in RESULTS})
            print('steady', i+1, {p:round(reports[p]['samples'][-1], 4) for p in PRESETS}, flush=True)
        baseline = statistics.median(reports['high']['samples'])
        for p in PRESETS:
            median = statistics.median(reports[p]['samples'])
            reports[p].update(median=median, speedup_vs_high=baseline/median,
                              bitwise_equal_to_high=True)
        unique=len({b''.join(np.ascontiguousarray(model[k][i]).tobytes() for k in FIELDS) for i in range(args.models)})
        return dict(models=args.models, unique_models=unique, layers=model['thickness'].shape[1], periods=len(per),
                    nmodes=5, input_sha256=digest(args.input), periods_sha256=digest(args.periods),
                    period_stride=args.period_stride, presets=reports,
                    baseline='same-build high; 0.2.1 schedule, full diagnostics verified')
    finally:
        for solver in solvers.values():
            solver.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--periods', type=Path, required=True)
    parser.add_argument('--models', type=int, default=128)
    parser.add_argument('--period-stride', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.models, args.period_stride, args.repeats, args.threads) < 1:
        parser.error('counts must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(version=md.__version__, python=sys.version, platform=platform.platform(),
                  numpy=np.__version__, backend=args.backend, threads=args.threads,
                  cpu_extension_sha256=digest(md._cpu.__file__))
    if args.backend == 'cuda':
        report['gpu'] = md.device_name()
        from modisp import _load_cuda
        report['cuda_extension_sha256'] = digest(_load_cuda().__file__)
    try:
        if args.references:
            report['references'] = external(args)
        report['performance'] = performance(args)
        report['passed'] = True
    except BaseException as exc:
        report['passed'] = False
        report['error'] = repr(exc)
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['performance']['presets'], indent=2), flush=True)


if __name__ == '__main__':
    main()
