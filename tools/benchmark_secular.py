#!/Users/yusong/miniforge3/bin/python
"""Benchmark the experimental secular evaluator separately from root search.

Stage 1 evaluates the signed dispersion function on one fixed million-node
velocity grid for both implementations (same model, period and FP64 nodes).
Stage 2 compares full solver throughput and root sets with identical inputs.
The two stages are reported separately because a faster function does not imply
a proportionally faster adaptive sampler or root refiner.
"""
import argparse
from contextlib import ExitStack
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import modisp as md


def _model_arrays(model):
    return (np.asarray(model.thickness, dtype=np.float64).ravel(),
            np.asarray(model.vp, dtype=np.float64).ravel(),
            np.asarray(model.vs, dtype=np.float64).ravel(),
            np.asarray(model.rho, dtype=np.float64).ravel())


def representative_models():
    out = []
    for index in (0, 3, 7):
        out.append((f"canonical_{index}",) + _model_arrays(md.CANONICAL_MODELS[index]))
    # surface acoustic water layer over a two-layer solid
    out.append(("water_two_layer",
                np.array([.2, 1., 0.]), np.array([1.5, 3., 6.]),
                np.array([0., 1.5, 3.5]), np.array([1.02, 2., 2.7])))
    # 70-layer solid and 70-layer water-backed models
    batch = md.random_model_batch(1, 70, seed=20261029)
    out.append(("random_70_solid",
                np.asarray(batch["thickness"][0], dtype=np.float64).ravel(),
                np.asarray(batch["vp"][0], dtype=np.float64).ravel(),
                np.asarray(batch["vs"][0], dtype=np.float64).ravel(),
                np.asarray(batch["rho"][0], dtype=np.float64).ravel()))
    h = np.asarray(batch["thickness"][0], dtype=np.float64).copy()
    vp = np.asarray(batch["vp"][0], dtype=np.float64).copy()
    vs = np.asarray(batch["vs"][0], dtype=np.float64).copy()
    rho = np.asarray(batch["rho"][0], dtype=np.float64).copy()
    h[0], vp[0], vs[0], rho[0] = .2, 1.5, 0., 1.02
    out.append(("random_70_water", h, vp, vs, rho))
    return out


def compare_pointwise(c, f0, f1):
    finite0, finite1 = np.isfinite(f0), np.isfinite(f1)
    all_finite = bool(finite0.all() and finite1.all())
    scale = max(1., float(np.max(np.abs(f0[finite0])))) if finite0.any() else 1.
    common = finite0 & finite1
    diff = np.abs(f0[common] - f1[common])
    away = common & (np.abs(f0) > 1e-9 * scale)
    crossings = []
    for f, finite in ((f0, finite0), (f1, finite1)):
        crossed = finite[:-1] & finite[1:] & (np.signbit(f[:-1]) != np.signbit(f[1:]))
        crossings.append(c[np.flatnonzero(crossed)])
    same_count = crossings[0].size == crossings[1].size
    crossing_error = (float(np.max(np.abs(crossings[0] - crossings[1])))
                      if same_count and crossings[0].size else 0. if same_count else None)
    max_diff = float(diff.max()) if diff.size else None
    signs = int(np.count_nonzero(np.signbit(f0[away]) != np.signbit(f1[away])))
    return dict(all_finite=all_finite,
                finite_mask_equal=bool(np.array_equal(finite0, finite1)),
                max_abs_difference=max_diff,
                max_scaled_difference=max_diff / scale if max_diff is not None else None,
                sign_mismatches_away_from_zero=signs,
                crossings={str(k): int(v.size) for k, v in enumerate(crossings)},
                crossing_count_equal=same_count,
                max_crossing_node_difference=crossing_error,
                accepted=bool(all_finite and max_diff <= 2e-10 * scale and signs == 0
                              and same_count and crossing_error == 0.))


def pointwise(model, period, evaluations, repeats):
    name, h, vp, vs, rho = model
    lo = max(.05, .1 * float(vs[-1]))
    hi = float(np.nextafter(vs[-1], 0.))
    # A non-uniform but deterministic grid: denser near both ends and near the
    # half-space cutoff, where the fundamental and highest modes concentrate.
    u = np.linspace(0., 1., evaluations, dtype=np.float64)
    c = lo + (hi - lo) * (u + .02 * np.sin(2. * np.pi * u))
    c[0], c[-1] = lo, hi
    samples = {0: [], 1: []}
    values = {}
    for method in (0, 1):
        md._cpu.secular(h, vp, vs, rho, period, c, 0, method)  # untimed warmup
    for repeat in range(repeats):
        for method in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            start = time.perf_counter()
            values[method] = np.asarray(md._cpu.secular(h, vp, vs, rho, period, c, 0, method))
            samples[method].append(time.perf_counter() - start)
    medians = {str(k): float(np.median(v)) for k, v in samples.items()}
    return dict(model=name, period=float(period), evaluations=int(c.size),
                samples_seconds={str(k): v for k, v in samples.items()},
                median_seconds=medians,
                ns_per_eval={k: v / c.size * 1e9 for k, v in medians.items()},
                mevals_per_second={k: c.size / v / 1e6 for k, v in medians.items()},
                speedup=medians["0"] / medians["1"],
                **compare_pointwise(c, values[0], values[1]))


def compare_results(slow, fast):
    finite_slow = np.isfinite(slow.phase_velocity)
    finite_fast = np.isfinite(fast.phase_velocity)
    both = finite_slow & finite_fast
    diff = np.abs(slow.phase_velocity[both] - fast.phase_velocity[both])
    report = dict(missing=int(np.count_nonzero(finite_slow & ~finite_fast)),
                extra=int(np.count_nonzero(~finite_slow & finite_fast)),
                finite_common=int(np.count_nonzero(both)),
                max_abs_difference=float(diff.max()) if diff.size else None,
                wrong=int(np.count_nonzero(diff > np.maximum(3e-6, 2e-6 * np.abs(slow.phase_velocity[both])))) if diff.size else 0,
                root_count_equal=bool(np.array_equal(slow.root_count, fast.root_count)),
                finite_mask_equal=bool(np.array_equal(finite_slow, finite_fast)))
    report["accepted"] = (report["missing"] == report["extra"] == report["wrong"] == 0
                          and report["root_count_equal"] and report["finite_mask_equal"])
    return report


def solver_stage(args, stack):
    batch = md.random_model_batch(args.models, 70, seed=args.seed)
    periods = np.logspace(-1, 2, args.periods)
    solvers = {}
    for secular in ("dunkin", "fast"):
        solvers[secular] = stack.enter_context(md.Solver(backend=args.backend, threads=args.threads,
                                     accuracy=args.accuracy, secular=secular,
                                     profile=args.profile,
                                     workspace_bytes=args.workspace_bytes))
    for secular in ("dunkin", "fast"):
        solvers[secular](**batch, periods=periods, nmodes=args.nmodes)
    samples = {key: [] for key in solvers}
    results = {}
    for repeat in range(args.repeats):
        order = ("dunkin", "fast") if repeat % 2 == 0 else ("fast", "dunkin")
        for secular in order:
            start = time.perf_counter()
            results[secular] = solvers[secular](**batch, periods=periods, nmodes=args.nmodes)
            samples[secular].append(time.perf_counter() - start)
    out = dict(models=int(args.models), periods=int(args.periods), nmodes=int(args.nmodes),
               layers=70, backend=args.backend, accuracy=args.accuracy,
               threads=int(args.threads), samples_seconds=samples,
               median_seconds={k: float(np.median(v)) for k, v in samples.items()},
               models_per_second={k: float(args.models / np.median(v)) for k, v in samples.items()},
               evaluations={k: int(solvers[k].stats()["evaluations"]) for k in solvers},
               speedup=float(np.median(samples["dunkin"]) / np.median(samples["fast"])))
    out["comparison"] = compare_results(results["dunkin"], results["fast"])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("evals", "solver", "both"), default="both")
    parser.add_argument("--evaluations", type=int, default=1_000_000)
    parser.add_argument("--periods-eval", type=float, nargs="+", default=[.1, 1., 10., 100.])
    parser.add_argument("--eval-repeats", type=int, default=3)
    parser.add_argument("--models", type=int, default=64)
    parser.add_argument("--periods", type=int, default=100)
    parser.add_argument("--nmodes", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--accuracy", choices=("high", "balanced", "fast"), default="high")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--workspace-bytes", type=int, default=256 * 1024 ** 2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if min(args.evaluations, args.eval_repeats, args.models, args.periods,
           args.nmodes, args.threads, args.repeats) < 1:
        parser.error("all counts must be positive")
    if args.evaluations < 2 or not np.isfinite(args.periods_eval).all() or min(args.periods_eval) <= 0:
        parser.error("evaluation grid needs at least two nodes and finite positive periods")
    report = dict(mode=args.mode, backend=args.backend,
                  modisp=md.__version__, python=os.sys.executable,
                  protocol="warmup then alternating paired samples; crossing locations are grid nodes",
                  pointwise_backend="cpu", seed=args.seed, profile=args.profile,
                  environment=dict(platform=platform.platform(), machine=platform.machine(),
                                   python=platform.python_version(), cpu_count=os.cpu_count()))
    if args.mode in ("evals", "both"):
        report["pointwise"] = []
        for model in representative_models():
            for period in args.periods_eval:
                row = pointwise(model, period, args.evaluations, args.eval_repeats)
                report["pointwise"].append(row)
                print(json.dumps(row), flush=True)
    if args.mode in ("solver", "both"):
        with ExitStack() as stack:
            report["solver"] = solver_stage(args, stack)
        print(json.dumps(report["solver"]), flush=True)
    report["accepted"] = (all(row["accepted"] for row in report.get("pointwise", []))
                          and report.get("solver", {}).get("comparison", {}).get("accepted", True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
