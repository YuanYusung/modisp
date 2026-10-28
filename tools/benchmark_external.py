#!/usr/bin/env python3
"""Same-host solid-Rayleigh accuracy and performance comparison.

The NPZ ``reference`` is the independent judge for reported correctness.  disba
and QEDispInv are additional solvers, not arbiters.  Third-party sources and
libraries are neither bundled nor built by this script.
"""
import os

# These must be set before importing numpy (and before native solver imports).
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import concurrent.futures
import csv
import ctypes
import hashlib
import importlib.metadata
import json
import multiprocessing
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

FIELDS = ("thickness", "vp", "vs", "rho")
NMODES = 4
SOLVERS = (
    "modisp-cpu1",
    "modisp-cpu8",
    "modisp-cuda",
    "disba-serial",
    "disba-process8",
    "qed-serial",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_output(path):
    path = path.resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise SystemExit(f"output directory exists and is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_suites(baseline):
    paths = sorted(baseline.glob("*.npz"))
    if len(paths) != 11:
        raise SystemExit(f"expected 11 NPZ suites in {baseline}, found {len(paths)}")
    suites = []
    for path in paths:
        with np.load(path, allow_pickle=False) as source:
            data = {key: source[key].copy() for key in source.files}
        missing = set(FIELDS + ("periods", "reference", "reference_stable")) - set(data)
        if missing:
            raise SystemExit(f"{path}: missing fields {sorted(missing)}")
        shape = data["reference"].shape
        if len(shape) != 3 or shape[1] != NMODES:
            raise SystemExit(f"{path}: reference must have shape [B,4,T], got {shape}")
        b = shape[0]
        if any(data[key].shape != (b, data[FIELDS[0]].shape[1]) for key in FIELDS):
            raise SystemExit(f"{path}: inconsistent model array shapes")
        if data["periods"].ndim != 1 or data["periods"].size != shape[2]:
            raise SystemExit(f"{path}: periods do not match reference")
        for key in FIELDS + ("periods", "reference"):
            data[key] = np.ascontiguousarray(data[key], dtype=np.float64)
        suites.append({"name": path.stem, "path": path, "data": data})
    return suites


def stable_mask(suite):
    stable = np.asarray(suite["data"]["reference_stable"], dtype=bool)
    reference = suite["data"]["reference"]
    if stable.shape == (reference.shape[0], reference.shape[2]):
        stable = stable[:, None, :]
    elif stable.shape == (reference.shape[0],):
        stable = stable[:, None, None]
    try:
        return np.broadcast_to(stable, reference.shape)
    except ValueError as exc:
        raise SystemExit(
            f"{suite['name']}: reference_stable shape {stable.shape} cannot "
            f"broadcast to {reference.shape}"
        ) from exc


def selected_indices(suite, subset):
    name = suite["name"]
    count = suite["data"]["reference"].shape[0]
    if not subset or name == "canonical" or name.startswith("qed_"):
        return list(range(count))
    if name.startswith("random_L"):
        return list(range(min(4, count)))
    if name == "seed20260916":
        return list(range(min(8, count)))
    raise SystemExit(f"no subset rule for suite {name}")


def take_models(suite, indices):
    data = suite["data"]
    return {
        key: np.ascontiguousarray(data[key][indices])
        for key in FIELDS + ("reference",)
    }


def unique_model_count(suite):
    """Count distinct layer-parameter vectors, independently of period outputs."""
    data = suite["data"]
    seen = set()
    for index in range(data["reference"].shape[0]):
        digest = hashlib.sha256()
        for key in FIELDS:
            values = np.ascontiguousarray(data[key][index], dtype=np.float64)
            digest.update(key.encode("ascii"))
            digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
            digest.update(values.tobytes())
        seen.add(digest.digest())
    return len(seen)


def make_manifest(suites, subset=False, performance_batches=None):
    entries = []
    for suite in suites:
        if performance_batches is not None:
            if suite["name"] != "seed20260916":
                continue
            available = suite["data"]["reference"].shape[0]
            needed = max(performance_batches)
            unique = unique_model_count(suite)
            if unique != available:
                raise SystemExit(
                    f"seed20260916 contains {unique} unique model vectors, "
                    f"expected {available}"
                )
            indices = list(range(min(needed, available)))
            task_count = needed
        else:
            indices = selected_indices(suite, subset)
            task_count = len(indices)
        entries.append({
            "suite": suite["name"],
            "indices": indices,
            "input_sha256": sha256(suite["path"]),
            "models": task_count,
            "unique_models": len(indices),
            "sampling": (
                "prefix" if task_count <= len(indices)
                else f"cyclic modulo {len(indices)} source models"
            ),
            "unique_trajectories": len(indices) * int(suite["data"]["periods"].size),
            "trajectories": task_count * int(suite["data"]["periods"].size),
        })
    manifest = {
        "subset": bool(subset),
        "suites": entries,
        "models": sum(item["models"] for item in entries),
        "unique_models": sum(item["unique_models"] for item in entries),
        "trajectories": sum(item["trajectories"] for item in entries),
    }
    if performance_batches is not None:
        source_models = entries[0]["unique_models"] if entries else 0
        manifest["performance_batches"] = list(performance_batches)
        manifest["performance_model_mapping"] = (
            f"For B <= {source_models}, use the first B distinct models; for "
            f"B > {source_models}, task i uses source model i % {source_models}."
        )
    if subset and performance_batches is None:
        if (manifest["models"], manifest["trajectories"]) != (43, 1347):
            raise SystemExit(
                "subset does not match the predetermined 43 models / 1347 "
                f"trajectories: {manifest['models']} / {manifest['trajectories']}"
            )
    return manifest


def cmin_for(suite):
    data = suite["data"]
    if suite["name"].startswith("qed_"):
        return float(min(0.2, 0.75 * np.min(data["vs"])))
    return 0.2


def compare(output, reference, stable, unresolved=0, unresolved_mask=None):
    output = np.asarray(output)
    finite_output = np.isfinite(output)
    finite_reference = np.isfinite(reference)
    both = finite_output & finite_reference & stable
    difference = np.abs(output - reference)
    tolerance = np.maximum(3e-6, 2e-6 * np.abs(reference))
    wrong = both & (difference > tolerance)
    missing = finite_reference & ~finite_output & stable
    extra = finite_output & ~finite_reference & stable
    metrics = {
        "wrong": int(np.count_nonzero(wrong)),
        "missing": int(np.count_nonzero(missing)),
        "extra": int(np.count_nonzero(extra)),
        "unresolved": int(unresolved),
        "max_common_abs_error": (
            float(np.max(difference[both])) if np.any(both) else None
        ),
    }
    if unresolved_mask is not None:
        metrics["unresolved"] = int(np.count_nonzero(unresolved_mask))
        metrics["unflagged_missing"] = int(
            np.count_nonzero(missing & ~unresolved_mask[:, None, :])
        )
    return metrics


def run_modisp_accuracy(suite, indices, backend, threads=1):
    import modisp

    selected = take_models(suite, indices)
    with modisp.Solver(backend=backend, threads=threads, cmin=cmin_for(suite)) as solver:
        result = solver(
            *(selected[key] for key in FIELDS),
            suite["data"]["periods"], nmodes=NMODES, wave="rayleigh",
        )
    if result.backend_used != backend:
        raise RuntimeError(f"requested {backend}, got {result.backend_used}")
    unresolved = (result.trace_flags & modisp.TRACE_SEARCH_UNCONVERGED) != 0
    return result.phase_velocity, result.status, unresolved


def map_disba_curve(curve, periods):
    """Expand disba's compressed curve, including duplicate input periods."""
    returned_periods = np.asarray(curve.period, dtype=np.float64).reshape(-1)
    velocities = np.asarray(curve.velocity, dtype=np.float64).reshape(-1)
    if returned_periods.size != velocities.size:
        raise ValueError("disba returned different period and velocity lengths")
    expanded = np.full(periods.size, np.nan, dtype=np.float64)
    for period, velocity in zip(returned_periods, velocities):
        locations = np.flatnonzero(periods == period)
        if locations.size == 0:
            raise ValueError(f"disba returned an unrequested period: {period!r}")
        expanded[locations] = velocity
    return expanded


def disba_model(model, periods, cmin, cmax):
    import disba

    output = np.full((NMODES, periods.size), np.nan, dtype=np.float64)
    unresolved = 0
    try:
        dispersion = disba.PhaseDispersion(*model.T)
    except Exception:
        return output, 1
    for mode in range(NMODES):
        try:
            curve = dispersion(periods, mode=mode, wave="rayleigh")
            values = map_disba_curve(curve, periods)
            valid = np.isfinite(values)
            values[valid & ~((values >= cmin) & (values < cmax) & (values < model[-1, 2]))] = np.nan
            output[mode] = values
        except Exception:
            unresolved += 1
    return output, unresolved


def run_disba_accuracy(suite, indices):
    selected = take_models(suite, indices)
    data = suite["data"]
    periods = data["periods"]
    cmin = cmin_for(suite)
    output = np.full(selected["reference"].shape, np.nan, dtype=np.float64)
    unresolved = 0
    for local, model_index in enumerate(indices):
        model = np.column_stack([selected[key][local] for key in FIELDS])
        output[local], errors = disba_model(
            model, periods, cmin, float(data["vs"][model_index, -1])
        )
        unresolved += errors
    return output, unresolved


def load_qed(path, benchmark=False):
    library = ctypes.CDLL(str(path))
    function = library.qed_benchmark_roots if benchmark else library.qed_reference_roots
    pointer = ctypes.POINTER(ctypes.c_double)
    function.argtypes = [
        ctypes.c_int, pointer, pointer, pointer, pointer,
        ctypes.c_int, pointer, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, pointer,
    ]
    function.restype = ctypes.c_int
    library.qed_reference_error.argtypes = []
    library.qed_reference_error.restype = ctypes.c_char_p
    return library, function


def qed_call(function, model, periods, cmin, cmax):
    arrays = [np.ascontiguousarray(column, dtype=np.float64) for column in model.T]
    periods = np.ascontiguousarray(periods, dtype=np.float64)
    # Native layout is [T,modes]; public layout is [modes,T].
    native = np.empty((periods.size, NMODES), dtype=np.float64)
    pointer = ctypes.POINTER(ctypes.c_double)
    arguments = [array.ctypes.data_as(pointer) for array in arrays]
    code = function(
        ctypes.c_int(model.shape[0]), *arguments,
        ctypes.c_int(periods.size), periods.ctypes.data_as(pointer),
        ctypes.c_int(NMODES), ctypes.c_double(cmin), ctypes.c_double(cmax),
        native.ctypes.data_as(pointer),
    )
    if code != 0:
        raise RuntimeError(f"QED adapter returned {code}")
    return native.T.copy()


def qed_error(library):
    message = library.qed_reference_error()
    return message.decode("utf-8", errors="replace") if message else "unknown QED error"


def run_qed_accuracy(suite, indices, library, function):
    selected = take_models(suite, indices)
    data = suite["data"]
    periods = data["periods"]
    cmin = cmin_for(suite)
    output = np.full(selected["reference"].shape, np.nan, dtype=np.float64)
    unresolved = 0
    messages = []
    for local, model_index in enumerate(indices):
        model = np.column_stack([selected[key][local] for key in FIELDS])
        try:
            output[local] = qed_call(
                function, model, periods, cmin, float(data["vs"][model_index, -1])
            )
        except Exception as exc:
            unresolved += 1
            messages.append(f"model {model_index}: {exc}; {qed_error(library)}")
    return output, unresolved, messages


def accuracy(args, suites, environment):
    import modisp  # Fail before producing a partial report if modisp is absent.

    del modisp
    library, qed_function = load_qed(args.qed_library, benchmark=False)
    summary = {
        "environment": environment,
        "reference": "NPZ independent reference",
        "suites": {},
        "overall": {},
    }
    for suite in suites:
        indices = selected_indices(suite, args.subset)
        selected = take_models(suite, indices)
        reference = selected["reference"]
        full_stable = stable_mask(suite)
        stable = full_stable[indices]
        suite_report = {"indices": indices, "solvers": {}}
        for solver_name in args.solvers:
            if solver_name.startswith("modisp"):
                backend = solver_name.split("-")[1]
                output, status, unresolved_mask = run_modisp_accuracy(
                    suite, indices, backend, args.threads
                )
                metrics = compare(
                    output, reference, stable, unresolved_mask=unresolved_mask
                )
                extra = {"status": status, "unresolved": unresolved_mask}
                messages = []
            elif solver_name == "disba":
                output, unresolved = run_disba_accuracy(suite, indices)
                metrics = compare(output, reference, stable, unresolved=unresolved)
                extra = {}
                messages = []
            else:
                output, unresolved, messages = run_qed_accuracy(
                    suite, indices, library, qed_function
                )
                metrics = compare(output, reference, stable, unresolved=unresolved)
                extra = {}
            stem = f"{suite['name']}.{solver_name}"
            np.savez_compressed(
                args.out / f"{stem}.npz", phase_velocity=output,
                reference=reference, reference_stable=stable, **extra,
            )
            report = dict(metrics)
            if messages:
                report["errors"] = messages
            write_json(args.out / f"{stem}.json", report)
            suite_report["solvers"][solver_name] = report
            overall = summary["overall"].setdefault(solver_name, {
                "wrong": 0, "missing": 0, "extra": 0, "unresolved": 0,
                "unflagged_missing": 0, "max_common_abs_error": None,
            })
            for key in ("wrong", "missing", "extra", "unresolved", "unflagged_missing"):
                overall[key] += metrics.get(key, 0)
            error = metrics["max_common_abs_error"]
            if error is not None:
                overall["max_common_abs_error"] = max(
                    overall["max_common_abs_error"] or 0.0, error
                )
        summary["suites"][suite["name"]] = suite_report
    write_json(args.out / "summary.json", summary)
    write_accuracy_readme(args.out / "README.md", summary)
    if args.require_clean_modisp:
        for solver, values in summary["overall"].items():
            if solver.startswith("modisp") and any(values[k] for k in ("wrong", "missing", "extra", "unresolved")):
                raise SystemExit(f"numerical acceptance failed for {solver}; see summary.json")


def write_accuracy_readme(path, summary):
    lines = [
        "# External accuracy results", "",
        "The NPZ reference is the judge; disba and QED are compared solvers.", "",
        "| Suite | Solver | Wrong | Missing | Extra | Unresolved | Unflagged missing | Max common abs error |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for suite, suite_report in summary["suites"].items():
        for solver, report in suite_report["solvers"].items():
            lines.append(
                f"| {suite} | {solver} | {report['wrong']} | {report['missing']} | "
                f"{report['extra']} | {report['unresolved']} | "
                f"{report.get('unflagged_missing', 'n/a')} | {report['max_common_abs_error']} |"
            )
    path.write_text("\n".join(lines) + "\n")


def disba_worker(model, periods, cmin, cmax):
    """Top-level and therefore pickleable by the Linux fork process pool."""
    return disba_model(model, periods, cmin, cmax)


def disba_indexed_worker(index, model, periods, cmin, cmax):
    output, unresolved = disba_worker(model, periods, cmin, cmax)
    return index, output, unresolved


def disba_warm_worker(model, periods, cmin, cmax):
    # Return the PID as well as exercising every mode's JIT in this worker.
    disba_worker(model, periods, cmin, cmax)
    return os.getpid()


def run_disba_batch(models, periods, cmin, cmaxes, executor=None):
    output = np.full((models.shape[0], NMODES, periods.size), np.nan, dtype=np.float64)
    unresolved = 0
    if executor is None:
        results = (
            disba_indexed_worker(i, models[i], periods, cmin, float(cmaxes[i]))
            for i in range(models.shape[0])
        )
    else:
        results = executor.map(
            disba_indexed_worker, range(models.shape[0]), models,
            [periods] * models.shape[0], [cmin] * models.shape[0],
            list(map(float, cmaxes)),
        )
    for index, values, errors in results:
        output[index] = values
        unresolved += errors
    return output, unresolved


def make_modisp_runner(models, periods, cmin, cmaxes, backend, threads):
    import modisp

    del cmaxes
    solver = modisp.Solver(backend=backend, threads=threads, cmin=cmin)

    def call():
        result = solver(
            *(models[:, :, field] for field in range(4)),
            periods, nmodes=NMODES, wave="rayleigh",
        )
        if result.backend_used != backend:
            raise RuntimeError(f"requested {backend}, got {result.backend_used}")
        unresolved = (result.trace_flags & modisp.TRACE_SEARCH_UNCONVERGED) != 0
        return result.phase_velocity, unresolved

    return solver, call


def make_qed_runner(models, periods, cmin, cmaxes, function):
    def call():
        output = np.full((models.shape[0], NMODES, periods.size), np.nan, dtype=np.float64)
        unresolved = 0
        for index, model in enumerate(models):
            try:
                output[index] = qed_call(
                    function, model, periods, cmin, float(cmaxes[index])
                )
            except Exception:
                unresolved += 1
        return output, unresolved

    return call


def measure(call, repeats):
    start = time.perf_counter()
    first, first_status = call()
    first_seconds = time.perf_counter() - start
    start = time.perf_counter()
    call()
    warmup_seconds = time.perf_counter() - start
    steady = []
    last = last_status = None
    for _ in range(repeats):
        start = time.perf_counter()
        last, last_status = call()
        steady.append(time.perf_counter() - start)
    return first, first_status, last, last_status, {
        "first_call_seconds": first_seconds,
        "warmup_seconds": warmup_seconds,
        "steady_seconds": steady,
        "median_seconds": statistics.median(steady),
        "min_seconds": min(steady),
        "max_seconds": max(steady),
    }


def correctness(first, last, first_status, last_status, reference, stable, unresolved_kind):
    if unresolved_kind == "modisp":
        first_metrics = compare(first, reference, stable, unresolved_mask=first_status)
        last_metrics = compare(last, reference, stable, unresolved_mask=last_status)
    else:
        first_metrics = compare(first, reference, stable, unresolved=first_status)
        last_metrics = compare(last, reference, stable, unresolved=last_status)
    return {"first": first_metrics, "last": last_metrics}


def performance(args, suites, environment):
    suite = next((item for item in suites if item["name"] == "seed20260916"), None)
    if suite is None:
        raise SystemExit("performance requires seed20260916.npz")
    data = suite["data"]
    available = data["reference"].shape[0]
    unique = unique_model_count(suite)
    if unique != available:
        raise SystemExit(
            f"seed20260916 contains {unique} unique model vectors, expected {available}"
        )
    models = np.stack(
        [np.column_stack([data[key][i] for key in FIELDS]) for i in range(available)]
    )
    periods = data["periods"]
    reference = data["reference"]
    stable = stable_mask(suite)
    if models.shape[1:] != (10, 4) or periods.size != 64 or reference.shape[1] != 4:
        raise SystemExit(
            f"performance requires L=10,T=64,nmodes=4; got {models.shape}, T={periods.size}"
        )
    cmin = 0.2
    cmaxes = data["vs"][:, -1]
    rows = []
    for solver_name in args.solvers:
        for batch_size in args.batches:
            model_indices = np.arange(batch_size)
            if batch_size > available:
                model_indices %= available
            batch = np.ascontiguousarray(models[model_indices])
            cmaxes_batch = cmaxes[model_indices]
            ref = reference[model_indices]
            stable_batch = stable[model_indices]
            unique_models = int(np.unique(model_indices).size)
            sampling = (
                "prefix" if batch_size <= available
                else f"cyclic modulo {available} source models"
            )
            executor = None
            solvers = None
            pool_create_seconds = None
            worker_warmup_seconds = None
            init_start = time.perf_counter()
            if solver_name.startswith("modisp"):
                backend = "cuda" if solver_name == "modisp-cuda" else "cpu"
                threads = 1 if solver_name == "modisp-cpu1" else 8
                solvers, call = make_modisp_runner(
                    batch, periods, cmin, cmaxes_batch, backend, threads
                )
                kind = "modisp"
            elif solver_name == "disba-serial":
                call = lambda: run_disba_batch(batch, periods, cmin, cmaxes_batch)
                kind = "external"
            elif solver_name == "disba-process8":
                if sys.platform != "linux":
                    raise RuntimeError("disba-process8 requires Linux fork")
                context = multiprocessing.get_context("fork")
                pool_start = time.perf_counter()
                executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=8, mp_context=context
                )
                pool_create_seconds = time.perf_counter() - pool_start
                # Force creation and JIT of every worker, independently of B.
                warm_start = time.perf_counter()
                futures = [
                    executor.submit(
                        disba_warm_worker, batch[0], periods, cmin, float(cmaxes_batch[0])
                    ) for _ in range(64)
                ]
                worker_pids = {future.result() for future in futures}
                worker_warmup_seconds = time.perf_counter() - warm_start
                if len(worker_pids) != 8:
                    raise RuntimeError(f"only {len(worker_pids)} pool workers were warmed")
                call = lambda: run_disba_batch(
                    batch, periods, cmin, cmaxes_batch, executor=executor
                )
                kind = "external"
            else:
                library, function = load_qed(args.qed_library, benchmark=True)
                call = make_qed_runner(batch, periods, cmin, cmaxes_batch, function)
                kind = "external"
            initialization_seconds = time.perf_counter() - init_start
            try:
                first, first_status, last, last_status, timing = measure(call, args.repeats)
                checks = correctness(
                    first, last, first_status, last_status, ref, stable_batch, kind
                )
                if kind == "modisp":
                    bad = sum(checks[end][key] for end in ("first", "last")
                              for key in ("wrong", "missing", "extra", "unresolved"))
                    if bad:
                        raise RuntimeError(
                            f"{solver_name} B={batch_size} failed output validation: {checks}"
                        )
                np.savez_compressed(
                    args.out / f"{solver_name}.B{batch_size}.npz",
                    first=first, last=last, reference=ref,
                    reference_stable=stable_batch,
                    first_unresolved=first_status, last_unresolved=last_status,
                )
                row = {
                    "solver": solver_name,
                    "B": batch_size,
                    "unique_models": unique_models,
                    "sampling": sampling,
                    "trajectories": batch_size * int(periods.size),
                    "initialization_seconds": initialization_seconds,
                    "pool_create_seconds": pool_create_seconds,
                    "worker_warmup_seconds": worker_warmup_seconds,
                    **timing,
                    "tasks_per_second": batch_size / timing["median_seconds"],
                    "checks": checks,
                }
                rows.append(row)
            finally:
                if solvers is not None:
                    solvers.close()
                if executor is not None:
                    executor.shutdown(wait=True)
    summary = {
        "environment": environment,
        "suite": "seed20260916",
        "unique_source_models": unique,
        "performance_batches": args.batches,
        "performance_solvers": args.solvers,
        "batch_mapping": (
            f"B <= {available}: first B distinct models; B > {available}: "
            f"task i uses source model i % {available}"
        ),
        "reference": "NPZ independent reference",
        "cmin": cmin,
        "nmodes": NMODES,
        "results": rows,
    }
    write_json(args.out / "summary.json", summary)
    write_performance_csv(args.out / "summary.csv", rows)
    write_performance_readme(args.out / "README.md", rows)


def write_performance_csv(path, rows):
    fields = [
        "solver", "B", "unique_models", "sampling", "trajectories",
        "initialization_seconds", "pool_create_seconds",
        "worker_warmup_seconds", "first_call_seconds", "warmup_seconds",
        "median_seconds", "min_seconds", "max_seconds", "tasks_per_second",
        "first_wrong", "first_missing", "first_extra", "first_unresolved",
        "last_wrong", "last_missing", "last_extra", "last_unresolved",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fields}
            for end in ("first", "last"):
                for metric in ("wrong", "missing", "extra", "unresolved"):
                    flat[f"{end}_{metric}"] = row["checks"][end][metric]
            writer.writerow(flat)


def write_performance_readme(path, rows):
    lines = [
        "# External performance results", "",
        "Times are seconds; tasks/s uses the median steady-state batch time. "
        "The NPZ reference is the correctness judge for first and last outputs.", "",
        "For batches larger than the 64-model source suite, task index `i` maps "
        "to source model `i % 64`; `Independent models` reports the number of distinct "
        "parameter vectors actually represented. Issue counts are per task, so "
        "repeated source models are counted on each occurrence.", "",
        "| Solver | Task count | Independent models | Input mapping | Period traces | Init | Pool create | Worker warmup | First call | Warmup | Median | Min | Max | Tasks/s | First issues | Last issues |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        issue_text = {}
        for end in ("first", "last"):
            check = row["checks"][end]
            issue_text[end] = ", ".join(
                f"{key}={check[key]}" for key in ("wrong", "missing", "extra", "unresolved")
            )
        def number(value):
            return "n/a" if value is None else f"{value:.6g}"
        lines.append(
            f"| {row['solver']} | {row['B']} | {row['unique_models']} | "
            f"{row['sampling']} | {row['trajectories']} | "
            f"{number(row['initialization_seconds'])} | "
            f"{number(row['pool_create_seconds'])} | {number(row['worker_warmup_seconds'])} | "
            f"{number(row['first_call_seconds'])} | {number(row['warmup_seconds'])} | "
            f"{number(row['median_seconds'])} | {number(row['min_seconds'])} | "
            f"{number(row['max_seconds'])} | {number(row['tasks_per_second'])} | "
            f"{issue_text['first']} | {issue_text['last']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def readable_cpu():
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or "unknown"


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment(qed_library):
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    environment = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": readable_cpu(),
        "affinity": affinity,
        "load_average_at_start": (
            list(os.getloadavg()) if hasattr(os, "getloadavg") else None
        ),
        "python": sys.version,
        "versions": {
            name: package_version(name)
            for name in ("numpy", "numba", "disba", "modisp")
        },
        "thread_environment": {
            key: os.environ[key]
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "qed_library": str(qed_library),
        "qed_library_sha256": sha256(qed_library),
    }
    try:
        import modisp
        start = time.perf_counter()
        device_name = modisp.device_name()
        environment["cuda"] = {
            "device_name": device_name,
            "initialization_seconds": time.perf_counter() - start,
        }
    except Exception as exc:
        environment["cuda"] = {"unavailable": str(exc)}
    return environment


def parse_args():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--baseline", type=Path, required=True,
                        help="directory containing the 11 baseline NPZ suites")
    common.add_argument("--qed-library", type=Path, required=True,
                        help="externally built libqed_reference shared library")
    common.add_argument("--out", type=Path, required=True,
                        help="new or empty raw-output directory")
    common.add_argument("--repeats", type=int, default=5)
    common.add_argument(
        "--batches", type=int, nargs="+", default=[1, 8, 64, 1000],
        help="performance task counts; values above the 64-model suite cycle through its fixed models",
    )
    common.add_argument("--subset", action="store_true",
                        help="accuracy: canonical/all QED, 4 per random_L, 8 seed models")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    accuracy_parser = subparsers.add_parser("accuracy", parents=[common], help="compare complete outputs")
    accuracy_parser.add_argument("--solvers", nargs="+", choices=("modisp-cpu", "modisp-cuda", "disba", "qed"), default=["modisp-cpu", "modisp-cuda", "disba", "qed"])
    accuracy_parser.add_argument("--threads", type=int, default=1)
    accuracy_parser.add_argument("--require-clean-modisp", action="store_true")
    performance_parser = subparsers.add_parser(
        "performance", parents=[common], help="measure fixed seed batches"
    )
    performance_parser.add_argument(
        "--solvers", nargs="+", choices=SOLVERS, default=list(SOLVERS),
        help="solver configurations to measure (default: all)",
    )
    args = parser.parse_args()
    if args.repeats < 1 or any(batch < 1 for batch in args.batches):
        parser.error("--repeats and every --batches value must be positive")
    if not 1 <= getattr(args, "threads", 1) <= 256:
        parser.error("--threads must be 1..256")
    if len(set(args.batches)) != len(args.batches):
        parser.error("--batches values must be unique")
    if args.command == "performance" and len(set(args.solvers)) != len(args.solvers):
        parser.error("--solvers values must be unique")
    args.qed_library = args.qed_library.resolve()
    if not args.qed_library.is_file():
        parser.error(f"QED library does not exist: {args.qed_library}")
    return args


def main():
    args = parse_args()
    args.out = prepare_output(args.out)
    suites = load_suites(args.baseline.resolve())
    environment = collect_environment(args.qed_library)
    manifest = make_manifest(
        suites, subset=args.subset,
        performance_batches=args.batches if args.command == "performance" else None,
    )
    write_json(args.out / "manifest.json", manifest)
    if args.command == "accuracy":
        accuracy(args, suites, environment)
    else:
        performance(args, suites, environment)


if __name__ == "__main__":
    main()
