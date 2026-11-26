"""Paired forward timings and byte-exact comparisons on the frozen backend inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from benchmark_backends import ARRAYS, digest, load_case, shell, write_json


# The first three cases use identical B, L, T and solver settings on every host.
CASES = {
    "dry": ("solid24", "rayleigh", 64, 64),
    "wet": ("wet24", "rayleigh", 64, 64),
    "love": ("solid24", "love", 64, 64),
    "large": ("solid24", "rayleigh", 2048, 64),
    "short-wet": ("wet24", "rayleigh", 64, 1),
    "qe": ("solid24", "rayleigh", 64, 1),
    "qe-love": ("solid24", "love", 256, 1),
    "repeat": ("solid24", "rayleigh", 16, 16),
}
FIELDS = (*ARRAYS, "period", "mode", "group_velocity", "group_velocity_flags",
          "group_velocity_valid", "group_velocity_mask", "prediction_flags")


def setup(runtime, inputs, case, backend, threads):
    global solver, solve_args
    sys.path.insert(0, str(runtime))
    import modisp

    if Path(modisp.__file__).parent.resolve().parent != runtime:
        raise RuntimeError(f"unexpected modisp installation {modisp.__file__}")
    family, wave, models, periods = CASES[case]
    grid = "periods1" if periods == 1 else "periods64"
    batch, values = load_case(inputs, dict(input_family=family, B=models, periods=grid))
    if case == "repeat":
        sample = values[::4]
        values = np.concatenate((sample, sample[::-1]))
    solver = modisp.Solver(backend=backend, threads=threads, group_velocity=case == "repeat",
                           search_strategy="qe_adaptive" if case in ("qe", "qe-love") else "shape_adaptive")
    solve_args = dict(**batch, periods=values, wave=wave, nmodes=5)


def timed_call():
    start = time.perf_counter_ns()
    result = solver(**solve_args)
    elapsed_ms = (time.perf_counter_ns() - start) * 1e-6
    fields = {}
    for name in FIELDS:
        arr = getattr(result, name)
        fields[name] = None if arr is None else (
            arr.dtype.str, arr.shape, hashlib.sha256(arr.tobytes(order="C")).hexdigest())
    fields["strategy"] = (result.frequency_strategy_used, result.frequency_strategy_reason)
    stats = solver.stats()
    return elapsed_ms, fields, {k: stats[k] for k in
        ("evaluations", "periods_requested", "periods_computed", "chunks")}


def paired(args, case):
    signatures, stats = {}, {}
    timings = {"base": [], "candidate": []}
    mismatches = set()
    with ExitStack() as stack:
        workers = {key: stack.enter_context(ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn"), initializer=setup,
            initargs=(runtime, args.inputs, case, args.backend, args.threads)))
            for key, runtime in (("base", args.base), ("candidate", args.candidate))}
        for i in range(1 + 3 + args.repeats):
            for key in (("candidate", "base") if (i + int(args.reverse)) % 2 else ("base", "candidate")):
                elapsed, fields, stats[key] = workers[key].submit(timed_call).result()
                timings[key].append(elapsed)
                first = signatures.setdefault(key, fields)
                mismatches.update(k for k in first if fields[k] != first[k])
    mismatches.update(k for k in signatures["base"] if signatures["base"][k] != signatures["candidate"][k])
    variants = {}
    for key, times in timings.items():
        steady = times[4:]
        q1, median, q3 = np.percentile(steady, [25, 50, 75])
        variants[key] = dict(first_ms=times[0], warmup_ms=times[1:4], samples_ms=steady,
                             median_ms=median, iqr_ms=q3-q1, stats=stats[key])
    return dict(variants=variants, speedup=variants["base"]["median_ms"]/variants["candidate"]["median_ms"],
                differing_fields=sorted(mismatches), base_fields=signatures["base"],
                candidate_changed={k:v for k,v in signatures["candidate"].items()
                                   if v != signatures["base"][k]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "base", "candidate", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--backend", choices=("cuda", "cpu"), required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--reverse", action="store_true", help="start with the candidate in the first pair")
    p.add_argument("--cases", nargs="+", choices=CASES, default=["dry", "wet", "love", "repeat"])
    args = p.parse_args()
    if args.repeats < 7:
        p.error("at least seven timed pairs are required")
    args.base, args.candidate = args.base.resolve(), args.candidate.resolve()
    pinned = Path(__file__).resolve().parents[1]/"docs/benchmarks/backend-0.4.0-inputs.json"
    expected = json.loads(pinned.read_text())["files"]
    names = {CASES[c][0]+".npz" for c in args.cases} | {"periods64.npy", "periods1.npy"}
    inputs = {name: digest(args.inputs/name) for name in sorted(names)}
    if any(value != expected[name] for name, value in inputs.items()):
        raise ValueError("benchmark inputs differ from the published 0.4.0 checksums")
    if args.output.exists():
        p.error(f"output already exists: {args.output}")
    gpu = ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,driver_version", "--format=csv,noheader"]
    report = dict(backend=args.backend, threads=args.threads, reverse=args.reverse, python=platform.python_version(),
                  numpy=np.__version__, affinity=sorted(os.sched_getaffinity(0)), input_sha256=inputs,
                  native_sha256={k:{str(v.relative_to(root)):digest(v) for v in sorted(root.glob("modisp*/*.so"))}
                                 for k, root in (("base",args.base),("candidate",args.candidate))}, rows={})
    for case in args.cases:
        before = shell(gpu)["stdout"] if args.backend == "cuda" else None
        row = paired(args, case)
        if before is not None:
            row["gpu_start_end"] = [before.strip(), shell(gpu)["stdout"].strip()]
        report["rows"][case] = row
        write_json(args.output, report)
        print(f"{case} {row['speedup']:.3f}x, differing fields {row['differing_fields']}", flush=True)
    if any(row["differing_fields"] for row in report["rows"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
