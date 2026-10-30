#!/Users/yusong/miniforge3/bin/python
"""Alternate calls to persistent before/after installations on identical inputs.

Each child waits for an explicit command; only one solver computes at a time.
Cold call, warmup and five steady samples are separate. Results from both first
and last calls retain every diagnostic for regression checking.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from benchmark_modephys import (GpuMonitor, RESULTS, atomic_json, convert, digest_file,
                               environment, make_solver)




def compare(x, y):
    fx, fy = np.isfinite(x["phase_velocity"]), np.isfinite(y["phase_velocity"])
    both = fx & fy
    error = np.abs(x["phase_velocity"][both]-y["phase_velocity"][both])
    diagnostics = {k: int(np.count_nonzero(x[k] != y[k])) for k in RESULTS if k != "phase_velocity"}
    thresholds = {str(t): int(np.count_nonzero(error > t)) for t in (1e-6, 1e-5, 1e-4, 1e-3)}
    return dict(wrong=int(np.count_nonzero(error > np.maximum(3e-6, 2e-6*np.abs(x["phase_velocity"][both])))),
                missing=int(np.count_nonzero(fx & ~fy)), extra=int(np.count_nonzero(~fx & fy)),
                newly_unresolved=int(np.count_nonzero(((y["trace_flags"] & 4) != 0) & ((x["trace_flags"] & 4) == 0))),
                max_common_abs_error=float(error.max()) if error.size else None,
                finite_common=int(np.count_nonzero(both)), error_counts_gt=thresholds,
                diagnostics_changed=diagnostics)


def worker(args):
    import modisp as md
    model = np.load(args.input, mmap_mode="r")
    if args.models > len(model):
        raise ValueError("requested model count exceeds the input; no repeated tasks are substituted")
    batch = convert(model[:args.models])
    periods = np.ascontiguousarray(np.load(args.periods)[::args.period_stride], dtype=np.float64)
    with make_solver(md, args) as solver:
        print(json.dumps(dict(ready=True, environment=environment(md))), flush=True)
        for line in sys.stdin:
            command = json.loads(line)
            if command["action"] == "close":
                break
            started = time.perf_counter()
            result = solver(**batch, periods=periods, nmodes=5)
            elapsed = time.perf_counter()-started
            if command.get("save"):
                np.savez(command["save"], **{k: getattr(result, k) for k in RESULTS}, periods=periods)
            print(json.dumps(dict(seconds=elapsed, stats=solver.stats(),
                                  kernel_report=solver.kernel_report())), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--before-python", type=Path)
    p.add_argument("--after-python", type=Path)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--periods", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--models", type=int, default=1024)
    p.add_argument("--period-stride", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--workspace-bytes", type=int, default=256*1024**2)
    p.add_argument("--native-chunk", type=int)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--accuracy", choices=("high", "balanced", "fast"))
    p.add_argument("--search-strategy",choices=("shape_adaptive","qe_adaptive","uniform"))
    p.add_argument("--secular",choices=("dunkin","fast"))
    p.add_argument("--refinement",choices=("legacy","brent","chandrupatla"))
    p.add_argument("--love-strategy",choices=("phase","scan"))
    p.add_argument("--allow-accuracy-change", action="store_true",
                   help="do not fail when outputs differ; still report the differences")
    args = p.parse_args()
    if min(args.models, args.repeats, args.period_stride) < 1:
        p.error("model/sample/stride counts must be positive")
    if args.worker:
        worker(args)
        return
    if not args.before_python or not args.after_python:
        p.error("before and after interpreters are required")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    children = {}
    monitor = None
    report = {"input_sha256": digest_file(args.input), "models": args.models,
              "period_stride": args.period_stride, "backend": args.backend,
              "threads": args.threads, "workspace_bytes": args.workspace_bytes,
              "accuracy":args.accuracy,"search_strategy":args.search_strategy,"secular":args.secular,"refinement":args.refinement,"love_strategy":args.love_strategy,
              "order": [], "variants": {}}
    try:
        for name, python in (("before", args.before_python), ("after", args.after_python)):
            command = [str(python), "-u", str(Path(__file__).resolve()), "--worker",
                       "--input", str(args.input), "--periods", str(args.periods),
                       "--output", str(out), "--backend", args.backend,
                       "--threads", str(args.threads), "--models", str(args.models),
                       "--period-stride", str(args.period_stride),
                       "--workspace-bytes", str(args.workspace_bytes)]
            if args.profile:
                command.append("--profile")
            if args.accuracy:
                command.extend(["--accuracy", args.accuracy])
            for option in ("search_strategy","secular","refinement","love_strategy"):
                if getattr(args,option):command.extend(["--"+option.replace("_","-"),getattr(args,option)])
            if args.native_chunk:
                command.extend(["--native-chunk", str(args.native_chunk)])
            stderr = (out / f"{name}.stderr.log").open("w")
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=stderr, text=True, bufsize=1)
            children[name] = proc, stderr
            ready = proc.stdout.readline()
            if not ready:
                raise RuntimeError(f"{name} did not start; see stderr log")
            report["variants"][name] = dict(environment=json.loads(ready)["environment"], calls=[])
        if args.backend == "cuda":
            monitor = GpuMonitor({proc.pid for proc, _ in children.values()})
            time.sleep(2)
            monitor.check()
        for call in range(args.repeats+2):
            # Reverse order every round to limit monotonic thermal/load drift.
            for name in (("before", "after") if call % 2 == 0 else ("after", "before")):
                proc = children[name][0]
                if monitor:
                    monitor.check()
                command = dict(action="run")
                if call in (0, args.repeats+1):
                    command["save"] = str(out / f"{name}.{'first' if call == 0 else 'last'}.npz")
                proc.stdin.write(json.dumps(command)+"\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError(f"{name} exited during call {call}; see stderr log")
                result = json.loads(line)
                if monitor:
                    monitor.check()
                report["variants"][name]["calls"].append(result)
                report["order"].append([call, name, result["seconds"]])
                atomic_json(out / "progress.json", report)
                print(json.dumps(dict(call=call, variant=name, seconds=result["seconds"])), flush=True)
        for name in children:
            variant = report["variants"][name]
            variant["steady_samples_seconds"] = [c["seconds"] for c in variant["calls"][2:]]
            variant["steady_median_seconds"] = float(np.median(variant["steady_samples_seconds"]))
            variant["models_per_second"] = args.models/variant["steady_median_seconds"]
        report["speedup"] = report["variants"]["before"]["steady_median_seconds"]/report["variants"]["after"]["steady_median_seconds"]
        report["regression"] = {}
        if monitor:
            report["gpu_process_samples"] = monitor.samples
            report["gpu_peak_process_mib"] = {name: monitor.peak_mib.get(proc.pid) for name, (proc, _) in children.items()}
        for call in ("first", "last"):
            with np.load(out / f"before.{call}.npz") as a, np.load(out / f"after.{call}.npz") as b:
                report["regression"][call] = compare(a, b)
        report["exact_equal"] = all(not any(m[k] for k in ("wrong", "missing", "extra", "newly_unresolved")) and not any(m["diagnostics_changed"].values()) for m in report["regression"].values())
        report["passed"] = report["exact_equal"] or bool(args.allow_accuracy_change)
        atomic_json(out / "report.json", report)
        if not report["passed"]:
            raise SystemExit(2)
    except BaseException as exc:
        report["invalid_reason"] = str(exc)
        if monitor:
            report["gpu_process_samples"] = monitor.samples
            report["concurrent_compute"] = monitor.competitors
        atomic_json(out / "invalid.json", report)
        raise
    finally:
        if monitor:
            monitor.close()
        for proc, stderr in children.values():
            if proc.poll() is None:
                try:
                    proc.stdin.write('{"action":"close"}\n')
                    proc.stdin.flush()
                    proc.wait(timeout=30)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    proc.terminate()
                    proc.wait(timeout=30)
            stderr.close()


if __name__ == "__main__":
    main()
