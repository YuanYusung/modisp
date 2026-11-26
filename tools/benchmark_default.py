"""Paired default-solver benchmarks on two isolated builds of modisp."""
import argparse
from concurrent.futures import Future, ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import platform
import sys

import numpy as np

import benchmark_continuity as timing

CASES = timing.TIMING_CASES + ["large", "large-love", "small-dry", "small-wet"]


def setup(runtime, args, case, group, variant):
    sys.path.insert(0, str(runtime))
    import modisp
    if Path(modisp.__file__).parent.parent.resolve() != runtime:
        raise RuntimeError("unexpected modisp runtime")
    if case == "large-love":
        data = timing.case_input("large", args.inputs)
        data["wave"] = "love"
    elif case.startswith("small-"):
        data = timing.case_input(case.replace("small-", "wide-"), args.inputs)
        data.update({name: data[name][:1] for name in timing.MODEL_FIELDS})
        data["periods"] = data["periods"][:4]
    else:
        data = timing.case_input(case, args.inputs)
    method = args.candidate_secular if variant == "candidate" else args.base_secular
    timing.solver = modisp.Solver(backend=args.backend, threads=args.threads, group_velocity=group,
                                  frequency_strategy=args.strategy, secular=method, workspace_bytes=1024**3)
    timing.call_args = data


def make_worker(stack, args, case, group, variant):
    runtime = getattr(args, variant).resolve()
    if args.same_process:
        setup(runtime, args, case, group, variant)
        solver = stack.enter_context(timing.solver)
        data = timing.call_args

        class LocalWorker:
            def submit(self, function, *params):
                timing.solver, timing.call_args = solver, data
                completed = Future()
                completed.set_result(function(*params))
                return completed

        return LocalWorker()
    return stack.enter_context(ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"),
        initializer=setup, initargs=(runtime, args, case, group, variant)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "candidate", "inputs", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--pairs", type=int, default=7)
    p.add_argument("--cases", nargs="+", choices=CASES, default=timing.TIMING_CASES)
    p.add_argument("--outputs", choices=("phase", "group", "both"), default="both")
    p.add_argument("--strategy", choices=("independent", "anchor_continuation"), default="independent")
    for key in ("base", "candidate"):
        p.add_argument("--"+key+"-secular", choices=("dunkin", "fast"), default=None)
    p.add_argument("--diagnostic", action="store_true", help="take a separate profiled call for each variant")
    p.add_argument("--same-process", action="store_true",
                   help="alternate two warmed solver instances in one process on the same build")
    args = p.parse_args()
    if args.pairs < 7:
        p.error("at least seven steady pairs are required")
    if args.same_process and args.base.resolve() != args.candidate.resolve():
        p.error("--same-process requires identical base and candidate runtime paths")
    args.inputs = args.inputs.resolve()
    out = dict(backend=args.backend, threads=args.threads, version=platform.python_version(), numpy=np.__version__,
               affinity=sorted(os.sched_getaffinity(0)), strategy=args.strategy, warmups=3, pairs=args.pairs,
               base_secular=args.base_secular, candidate_secular=args.candidate_secular, same_process=args.same_process,
               input_sha256={name:timing.sha(args.inputs/name) for name in ("solid24.npz","wet24.npz","periods64.npy")},
               source_sha256=timing.sha(__file__), timing_source_sha256=timing.sha(timing.__file__),
               libraries={k:timing.snapshot(getattr(args,k)) for k in ("base","candidate")}, cases={})
    if args.backend == "cuda":
        out["gpu_start"] = timing.gpu_state()
    groups = {"phase": (False,), "group": (True,), "both": (False, True)}[args.outputs]
    for case in args.cases:
        row = out["cases"][case] = {}
        for group in groups:
            row["phase+group" if group else "phase"] = timing.time_case(args, case, group, ["base","candidate"], make_worker)
            timing.save(args.output, out)
        print(case, flush=True)
    if args.backend == "cuda":
        out["gpu_end"] = timing.gpu_state()
        timing.save(args.output, out)


if __name__ == "__main__":
    main()
