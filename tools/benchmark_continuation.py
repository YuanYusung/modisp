#!/usr/bin/env python3
"""Measure group velocity and CUDA continuation on fixed large-batch cases.

Examples::

    python tools/benchmark_continuation.py --backend cuda --output report.json
    python tools/benchmark_continuation.py --backend cpu --smoke --case batch1024 \
        --waves love --precisions float64 --output smoke.json

Timings use persistent solvers with profiling disabled, three warmups and seven
measurements. A separate profiled call records stage counters and workspace.
An optional 0.4.0 interpreter supplies an actual same-host plain-phase baseline.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np


CASES = {
    "batch1024": (1024, 16, 5, 1),
    "batch10000_t32": (10000, 32, 20, 5),
    "batch10000_t64": (10000, 64, 50, 5),
}
VARIANTS = {
    "independent": {},
    "group": {"group_velocity": True},
    "continuation": {"frequency_strategy": "anchor_continuation"},
    "combined": {"group_velocity": True, "frequency_strategy": "anchor_continuation"},
}


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(json_value(report), indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def environment(md):
    result = dict(version=md.__version__, python=sys.executable,
                  platform=platform.platform(), numpy=np.__version__,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    for name, module in (("python", md), ("cpu", md._cpu)):
        path = Path(module.__file__)
        result[name+"_module"] = str(path)
        result[name+"_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        path = Path(md._load_cuda().__file__)
        result["cuda_module"] = str(path)
        result["cuda_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    except (RuntimeError, ImportError):
        pass
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                              "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=10)
        if gpu.returncode == 0:
            result["gpu_name_and_driver"] = gpu.stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return result


def timings(samples):
    samples = np.asarray(samples, dtype=float)
    median = float(np.median(samples))
    q1, q3 = np.percentile(samples, [25, 75])
    spread = float((q3-q1)/median)
    return dict(samples_seconds=samples.tolist(), median_seconds=median,
                min_seconds=float(samples.min()), max_seconds=float(samples.max()),
                iqr_over_median=spread, noisy=bool(spread > .05))


def compare_phase(reference, candidate):
    x, y = reference["phase_velocity"], candidate["phase_velocity"]
    fx, fy = np.isfinite(x), np.isfinite(y)
    both = fx & fy
    error = np.abs(x[both]-y[both])
    relative = error/np.maximum(np.abs(x[both]), np.finfo(float).tiny)
    missing, extra = int(np.count_nonzero(fx & ~fy)), int(np.count_nonzero(~fx & fy))
    wrong = int(np.count_nonzero(error > np.maximum(3e-6, 3e-5*np.abs(x[both]))))
    count_equal = bool(np.array_equal(reference["root_count"], candidate["root_count"]))
    return dict(missing=missing, extra=extra, wrong=wrong,
                root_count_equal=count_equal, finite_common=int(both.sum()),
                max_abs_error=float(error.max()) if error.size else None,
                max_relative_error=float(relative.max()) if relative.size else None,
                passed=missing == extra == wrong == 0 and count_equal)


def snapshot(out):
    value=dict(phase_velocity=out.phase_velocity.copy(),root_count=out.root_count.copy(),
               trace_flags=out.trace_flags.copy())
    prediction=getattr(out,'prediction_flags',None)
    if prediction is not None:value['prediction_flags']=prediction.copy()
    if getattr(out,'group_velocity',None) is not None:
        for name in ('group_velocity','group_velocity_valid','group_velocity_flags'):
            value[name]=getattr(out,name).copy()
    return value


def compare_groups(reference, candidate, group_rtol=1e-3):
    """Compare group values and quality transitions after throughput timing.

    Values at different recovered phase ranks are reported separately. A mask
    transition requires review; neither an old NaN nor a larger valid count
    establishes which result is physically correct.
    """
    if 'group_velocity' not in reference or 'group_velocity' not in candidate:
        return None
    a,b=reference['group_velocity'],candidate['group_velocity']
    fa,fb=reference['group_velocity_flags'],candidate['group_velocity_flags']
    va,vb=reference['group_velocity_valid'],candidate['group_velocity_valid']
    common=va&vb
    ca,cb=reference['phase_velocity'],candidate['phase_velocity']
    aligned=np.isfinite(ca)&np.isfinite(cb)&(np.abs(ca-cb)<=np.maximum(3e-6,3e-5*np.abs(ca)))
    valid_mask_errors=int(np.count_nonzero(va!=(np.isfinite(a)&(fa==0))))
    valid_mask_errors+=int(np.count_nonzero(vb!=(np.isfinite(b)&(fb==0))))
    valid_mask_errors+=int(np.count_nonzero(va&~np.isfinite(ca)))+int(np.count_nonzero(vb&~np.isfinite(cb)))

    def errors(mask):
        finite=mask&np.isfinite(a)&np.isfinite(b)
        absolute=np.abs(a[finite]-b[finite])
        relative=absolute/np.maximum(np.abs(a[finite]),np.finfo(float).tiny)
        def quantiles(values):
            return dict(zip(('p50','p95','p99','max'),
                            np.percentile(values,[50,95,99,100]).tolist())) if values.size else None
        budget=2*group_rtol*np.maximum.reduce((np.abs(a[finite]),np.abs(b[finite]),np.abs(ca[finite])))
        return dict(count=int(mask.sum()),finite_count=int(finite.sum()),absolute=quantiles(absolute),relative=quantiles(relative),
                    over_budget=int(np.count_nonzero(absolute>np.maximum(1e-6,budget))))

    codes=(fa.astype(np.uint64)<<np.uint64(32))|fb.astype(np.uint64)
    unique,counts=np.unique(codes,return_counts=True)
    transitions=[dict(from_flag=int(code>>np.uint64(32)),to_flag=int(code&np.uint64(0xffffffff)),count=int(n))
                 for code,n in zip(unique,counts) if int(code>>np.uint64(32))!=int(code&np.uint64(0xffffffff))]
    lost=int(np.count_nonzero(va&~vb));gained=int(np.count_nonzero(~va&vb))
    unknown_flags=int(np.count_nonzero((fa|fb)&np.uint32(0xffffffc0)))
    aligned_errors=errors(common&aligned)
    return dict(valid_to_invalid=lost,invalid_to_valid=gained,flags_changed=int(np.count_nonzero(fa!=fb)),
                flag_transitions=transitions,common_valid=errors(common),aligned_common_valid=aligned_errors,
                unaligned_common_valid=int(np.count_nonzero(common&~aligned)),valid_mask_errors=valid_mask_errors,
                unknown_flags=unknown_flags,
                existing_valid_bitwise_equal=bool(np.array_equal(a[va].view(np.uint64),b[va].view(np.uint64))),
                tolerance=dict(absolute_floor=1e-6,group_rtol=group_rtol,multiplier=2,
                               scale='max(abs(U_reference), abs(U_candidate), abs(c_reference))'),
                review_needed=bool(lost or gained or transitions or np.any(common&~aligned)),
                passed=valid_mask_errors==0 and unknown_flags==0 and aligned_errors['over_budget']==0)


def verify_phase_differences(md, batch, periods, nmodes, wave, options, snapshots):
    """Adjudicate every changed trace against a separate high FP64 solve.

    This runs after timing. Strict comparisons are never overwritten. A
    reference disagreement is acceptable only if the candidate's entire
    requested prefix, masks, counts and ordering pass the high comparison.
    """
    reference=snapshots['independent']
    x=reference['phase_velocity']
    changed={}
    for name,value in snapshots.items():
        if compare_phase(reference,value)['passed']:
            continue
        y=value['phase_velocity']
        finite=np.isfinite(x)&np.isfinite(y)
        different=(np.isfinite(x)!=np.isfinite(y))|(finite&(
            np.abs(x-y)>np.maximum(3e-6,3e-5*np.abs(x))))
        changed[name]=np.any(different,axis=1)|(reference['root_count']!=value['root_count'])
    if not changed:
        return {}
    models=np.flatnonzero(np.any(np.logical_or.reduce(list(changed.values())),axis=1))
    subset={name:values[models] for name,values in batch.items()}
    config=dict(options,accuracy='high',precision='float64',profile=False,
                group_velocity=False,frequency_strategy='independent')
    started=time.perf_counter()
    with md.Solver(**config) as solver:
        high=snapshot(solver(**subset,periods=periods,nmodes=nmodes,wave=wave))
    elapsed=time.perf_counter()-started

    def masked(value,trace):
        return dict(phase_velocity=np.where(trace[:,None,:],value['phase_velocity'],np.nan),
                    root_count=np.where(trace,value['root_count'],0))

    def ordered(prefixes):
        if np.isinf(prefixes).any():return False
        finite=np.isfinite(prefixes)
        pairs=finite[:,1:]&finite[:,:-1]
        return bool(np.all(~pairs|(np.diff(prefixes,axis=1)>0)) and
                    not np.any(finite[:,1:]&~finite[:,:-1]))

    results={}
    # These high-reference flags cannot certify a changed recovered prefix.
    invalid_reference=1|4|8|32|64|128|512|1024
    for name,full_trace in changed.items():
        trace=full_trace[models]
        candidate={k:v[models] for k,v in snapshots[name].items()}
        old={k:v[models] for k,v in reference.items()}
        comparison=compare_phase(masked(high,trace),masked(candidate,trace))
        old_comparison=compare_phase(masked(high,trace),masked(old,trace))
        prefixes=candidate['phase_velocity'].transpose(0,2,1)[trace]
        ascending=ordered(prefixes)
        resolved=bool(np.all(high['root_count'][trace]>=0) and
                      not np.any(high['trace_flags'][trace]&invalid_reference) and
                      ordered(high['phase_velocity'].transpose(0,2,1)[trace]) and
                      np.all(np.isfinite(high['phase_velocity']).sum(axis=1)[trace]==
                             np.minimum(nmodes,high['root_count'][trace])))
        witnesses=[]
        for local,f in np.argwhere(trace):
            witnesses.append(dict(model_index=int(models[local]),period_index=int(f),
                period_seconds=float(periods[f]),
                high_phase_velocity=high['phase_velocity'][local,:,f],
                independent_phase_velocity=old['phase_velocity'][local,:,f],
                candidate_phase_velocity=candidate['phase_velocity'][local,:,f],
                high_root_count=int(high['root_count'][local,f]),
                independent_root_count=int(old['root_count'][local,f]),
                candidate_root_count=int(candidate['root_count'][local,f]),
                high_trace_flags=int(high['trace_flags'][local,f]),
                independent_trace_flags=int(old['trace_flags'][local,f]),
                candidate_trace_flags=int(candidate['trace_flags'][local,f])))
            if 'prediction_flags' in candidate:
                witnesses[-1]['candidate_prediction_flags']=int(candidate['prediction_flags'][local,f])
        results[name]=dict(passed=bool(comparison['passed'] and ascending and resolved),
            changed_traces=int(trace.sum()),model_indices=models,reference_config=config,
            reference_seconds=elapsed,reference_resolved=resolved,candidate_ascending=ascending,
            candidate_comparison=comparison,independent_comparison=old_comparison,
            model_inputs=subset,periods=periods,witnesses=witnesses)
    return results


def group_summary(out):
    values = getattr(out, "group_velocity", None)
    if values is None:
        return None
    flags = out.group_velocity_flags
    return dict(finite=int(np.isfinite(values).sum()),
                valid=int(out.group_velocity_valid.sum()),
                cells=int(values.size),
                flags={name: int(np.count_nonzero(flags & bit))
                       for name, bit in (("missing_root", 1), ("polish_failed", 2),
                                         ("derivative_unstable", 4), ("ill_conditioned", 8),
                                         ("close_root", 16), ("nonfinite", 32))})


def prediction_summary(out):
    flags = getattr(out, "prediction_flags", None)
    if flags is None:
        return None
    counts = {name: int(np.count_nonzero(flags & bit))
              for name, bit in (("anchor", 1), ("accepted", 2), ("fallback", 4),
                                ("bypassed", 8), ("audit", 16), ("audit_failed", 32),
                                ("source_search_uncertain", 64), ("root_count_unsafe",128),
                                ("anchor_unsafe",256),("window_unsafe",512),
                                ("bracket_failed",1024),("correction_failed",2048))}
    attempts = counts["accepted"]+counts["fallback"]
    counts["accepted_source_search_uncertain"] = int(np.count_nonzero(
        ((flags & 2) != 0) & ((flags & 64) != 0)))
    return dict(cells=int(flags.size), **counts,
                local_success_rate=counts["accepted"]/attempts if attempts else None,
                fallback_rate=counts["fallback"]/attempts if attempts else None)


class Baseline:
    """A persistent, idle-between-requests 0.4.0 process."""
    def __init__(self, python, source, options, wave, nmodes, monitor=None):
        env = os.environ.copy()
        # The chosen interpreter must load its installed 0.4.0, not this tree.
        env.pop("PYTHONPATH", None)
        self.process = subprocess.Popen(
            [str(python), "-u", str(Path(__file__).resolve()), "--worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env)
        self.monitor=monitor
        if monitor:
            # CUDA module initialization can already consume GPU time during
            # setup; register the worker before waiting for that response.
            monitor.allowed.add(self.process.pid)
        try:
            self.metadata = self.request(dict(action="setup", source=str(source),
                                             options=options, wave=wave, nmodes=nmodes))
            if self.metadata["version"] != "0.4.0":
                raise ValueError("--baseline-python must load modisp 0.4.0")
        except BaseException:
            self.close()
            raise

    def request(self, command):
        self.process.stdin.write(json.dumps(command)+"\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        if not response:
            stderr = self.process.stderr.read()
            raise RuntimeError("baseline worker exited: "+stderr[-4000:])
        result = json.loads(response)
        if "error" in result:
            raise RuntimeError("baseline worker: "+result["error"])
        return result

    def run(self, save=None):
        return self.request(dict(action="run", save=None if save is None else str(save)))

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"action":"close"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=30)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=30)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()
        if self.monitor:
            self.monitor.allowed.discard(self.process.pid)


def worker():
    import modisp as md
    solver = None
    try:
        for line in sys.stdin:
            command = json.loads(line)
            try:
                if command["action"] == "close":
                    break
                if command["action"] == "setup":
                    with np.load(command["source"], allow_pickle=False) as source:
                        batch = {k: source[k].copy() for k in ("thickness", "vp", "vs", "rho")}
                        periods = source["periods"].copy()
                    solver = md.Solver(**command["options"])
                    wave, nmodes = command["wave"], command["nmodes"]
                    result = environment(md)
                else:
                    started = time.perf_counter()
                    out = solver(**batch, periods=periods, nmodes=nmodes, wave=wave)
                    elapsed = time.perf_counter()-started
                    if command.get("save"):
                        np.savez(command["save"], **snapshot(out))
                    result = dict(seconds=elapsed, stats=solver.stats())
            except Exception as exc:
                result = dict(error=f"{type(exc).__name__}: {exc}")
            print(json.dumps(json_value(result), allow_nan=False), flush=True)
    finally:
        if solver is not None:
            solver.close()


def run_case(md, args, name, shape, wave, precision, directory, monitor):
    models, count, layers, nmodes = shape
    batch = md.random_model_batch(models, layers, seed=args.seed+list(CASES).index(name),
                                  h_min=args.h_min, h_max=args.h_max)
    periods = np.geomspace(args.period_min, args.period_max, count)
    source = directory/f"{name}-{wave}-{precision}.npz"
    np.savez(source, **batch, periods=periods)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    options = dict(backend=args.backend, accuracy=args.accuracy, precision=precision,
                   workspace_bytes=args.workspace_bytes, threads=args.threads, profile=False)
    result = dict(case=name, shape=dict(B=models, T=count, L=layers, K=nmodes),
                  wave=wave, precision=precision, input_sha256=digest,
                  model_seed=args.seed+list(CASES).index(name),
                  model_generation=dict(h_min_km=args.h_min, h_max_km=args.h_max),
                  period_range_seconds=[float(periods[0]), float(periods[-1])],
                  options=options, variants={}, order=[])
    snapshots, group_outputs, prediction_outputs = {}, {}, {}
    variants = {key: dict(extra, **({"anchor_stride": args.anchor_stride}
                                  if extra.get("frequency_strategy") == "anchor_continuation" else {}))
                for key, extra in VARIANTS.items()}
    with ExitStack() as stack:
        solvers = {key: stack.enter_context(md.Solver(**options, **extra))
                   for key, extra in variants.items()}
        baseline = None
        if args.baseline_python is not None:
            baseline = Baseline(args.baseline_python, source, options, wave, nmodes, monitor)
            stack.callback(baseline.close)
            result["baseline_environment"] = baseline.metadata
        names = list(solvers)+( ["baseline_0.4.0"] if baseline is not None else [] )
        samples = {key: [] for key in names}

        def call(key, save=False):
            if monitor:
                monitor.check()
            if key == "baseline_0.4.0":
                target = directory/"baseline-result.npz" if save else None
                data = baseline.run(target)
                if target is not None:
                    with np.load(target, allow_pickle=False) as stored:
                        snapshots[key] = {k: stored[k].copy() for k in stored.files}
                return data
            started = time.perf_counter()
            out = solvers[key](**batch, periods=periods, nmodes=nmodes, wave=wave)
            elapsed = time.perf_counter()-started
            if save:
                snapshots[key] = snapshot(out)
                group_outputs[key] = group_summary(out)
                prediction_outputs[key] = prediction_summary(out)
            return dict(seconds=elapsed, stats=solvers[key].stats())

        for key in names:
            result["variants"][key] = dict(cold_seconds=call(key)["seconds"])
        for iteration in range(args.warmups+args.repeats):
            # Rotate all variants, including the baseline, to limit load drift.
            order = names[iteration % len(names):]+names[:iteration % len(names)]
            for key in order:
                data = call(key, save=iteration == args.warmups+args.repeats-1)
                if iteration >= args.warmups:
                    samples[key].append(data["seconds"])
                    result["order"].append([iteration-args.warmups, key, data["seconds"]])
                    result["variants"][key]["stats"] = data["stats"]
        for key in names:
            result["variants"][key].update(timings(samples[key]))
            result["variants"][key]["models_per_second"] = models/result["variants"][key]["median_seconds"]
            if key in group_outputs:
                result["variants"][key]["group"] = group_outputs[key]
                result["variants"][key]["prediction"] = prediction_outputs[key]
            result["variants"][key]["phase_comparison"] = compare_phase(snapshots["independent"], snapshots[key])
        reference_seconds = result["variants"]["independent"]["median_seconds"]
        for key in names:
            result["variants"][key]["relative_wall_time"] = result["variants"][key]["median_seconds"]/reference_seconds
        if baseline is not None:
            result["plain_phase_ratio_0_4_to_current"] = result["variants"]["baseline_0.4.0"]["median_seconds"]/reference_seconds
        # Instrumentation is separate from every throughput sample.
        for key, extra in variants.items():
            with md.Solver(**dict(options, profile=True), **extra) as instrumented:
                instrumented(**batch, periods=periods, nmodes=nmodes, wave=wave)
                instrumented(**batch, periods=periods, nmodes=nmodes, wave=wave)
                result["variants"][key]["profiled_stats"] = instrumented.stats()
                result["variants"][key]["kernel_report"] = instrumented.kernel_report()
        verification=verify_phase_differences(md,batch,periods,nmodes,wave,options,snapshots)
        for key,value in verification.items():
            result['variants'][key]['physical_verification']=value
        result['group_comparison']=compare_groups(snapshots.get('group',{}),snapshots.get('combined',{}))
        if monitor:
            monitor.check()
    result['strict_passed']=all(value['phase_comparison']['passed'] for value in result['variants'].values())
    result["passed"] = all(value['phase_comparison']['passed'] or
                           value.get('physical_verification',{}).get('passed',False)
                           for value in result['variants'].values())
    if result['group_comparison'] is not None:
        # Changed validity or modal alignment needs a separate physical review;
        # a larger valid count alone cannot certify a release benchmark.
        result['group_review_required']=result['group_comparison']['review_needed']
        result['passed']=(result['passed'] and result['group_comparison']['passed']
                          and not result['group_review_required'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--case", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--waves", nargs="+", choices=("rayleigh", "love"), default=["rayleigh", "love"])
    parser.add_argument("--precisions", nargs="+", choices=("float64", "float32"), default=["float64", "float32"])
    parser.add_argument("--accuracy", choices=("balanced", "fast"), default="balanced")
    parser.add_argument("--anchor-stride", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="replace each selected shape by B=2,T=16,L=5,K=1")
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--h-min", type=float, default=.5,
                        help="minimum finite-layer thickness in km (default: .5)")
    parser.add_argument("--h-max", type=float, default=6.,
                        help="maximum finite-layer thickness in km; use .04/.04 for a 40m grid")
    parser.add_argument("--period-min", type=float, default=.1,
                        help="minimum period in seconds (default: .1)")
    parser.add_argument("--period-max", type=float, default=10.,
                        help="maximum period in seconds (default: 10)")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--workspace-bytes", type=int, default=256*1024**2)
    parser.add_argument("--baseline-python", type=Path)
    parser.add_argument("--background-pids", nargs="+", type=int, default=[],
                        help="record and permit known background GPU processes; results are not uncontended")
    parser.add_argument("--background-processes", nargs="+", default=[],
                        help="permit recorded background process names even when their PIDs change")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    if args.output is None:
        parser.error("--output is required")
    if args.repeats < 1 or args.warmups < 0:
        parser.error("--repeats must be positive and --warmups nonnegative")
    if args.anchor_stride < 2:
        parser.error("--anchor-stride must be at least 2")
    if not np.isfinite([args.h_min, args.h_max]).all() or not 0 < args.h_min <= args.h_max:
        parser.error("thickness limits must be finite and satisfy 0 < --h-min <= --h-max")
    if (not np.isfinite([args.period_min, args.period_max]).all()
            or not 0 < args.period_min < args.period_max):
        parser.error("period limits must be finite and satisfy 0 < --period-min < --period-max")
    import modisp as md
    report = dict(environment=environment(md), warmups=args.warmups, repeats=args.repeats,
                  smoke=args.smoke, seed=args.seed, h_min_km=args.h_min, h_max_km=args.h_max,
                  period_min_seconds=args.period_min, period_max_seconds=args.period_max,
                  anchor_stride=args.anchor_stride, results=[], skipped=[], passed=False,
                  comparison_policy='Strict independent comparisons are retained. Changed traces require their entire requested prefix to pass a separate resolved high FP64 solve after timing.',
                  permitted_background_pids=args.background_pids,
                  permitted_background_processes=args.background_processes,
                  uncontended=not bool(args.background_pids or args.background_processes))
    monitor = None
    try:
        if args.backend == "cuda":
            from benchmark_modephys import GpuMonitor
            monitor = GpuMonitor({os.getpid(), *args.background_pids}, args.background_processes)
            monitor.__enter__()
        with tempfile.TemporaryDirectory(prefix="modisp-continuation-bench-") as temporary:
            for name in args.case:
                shape = (2, 16, 5, 1) if args.smoke else CASES[name]
                for wave in args.waves:
                    for precision in args.precisions:
                        if args.backend == "cpu" and precision == "float32":
                            report["skipped"].append(dict(case=name, wave=wave, precision=precision,
                                                          reason="CPU supports float64 only"))
                            continue
                        row = run_case(md, args, name, shape, wave, precision, Path(temporary), monitor)
                        report["results"].append(row)
                        save_report(args.output, report)
                        print(json.dumps(dict(case=name, wave=wave, precision=precision,
                                              passed=row["passed"],strict_passed=row['strict_passed'],
                                              reference_verified=[key for key,value in row['variants'].items()
                                                  if value.get('physical_verification',{}).get('passed')])), flush=True)
        report["passed"] = bool(report["results"]) and all(row["passed"] for row in report["results"])
        report['strict_passed']=bool(report['results']) and all(row['strict_passed'] for row in report['results'])
        if monitor:
            report["gpu_process_samples"] = monitor.samples
            report["gpu_peak_process_mib"] = monitor.peak_mib
            report["permitted_background_samples"] = monitor.background_samples
            report["concurrent_compute"] = monitor.competitors
        save_report(args.output, report)
        return 0 if report["passed"] else 2
    except BaseException as exc:
        report["invalid_reason"] = f"{type(exc).__name__}: {exc}"
        if monitor:
            report["gpu_process_samples"] = monitor.samples
            report["permitted_background_samples"] = monitor.background_samples
            report["concurrent_compute"] = monitor.competitors
        save_report(args.output, report)
        raise
    finally:
        if monitor:
            monitor.close()


if __name__ == "__main__":
    raise SystemExit(main())
