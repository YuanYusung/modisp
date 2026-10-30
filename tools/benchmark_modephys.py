#!/Users/yusong/miniforge3/bin/python
"""Reproducible ModePhys preparation, streaming FP64 runs and comparisons.

No source dataset or existing generated product is modified. Uncompressed NPZ
is not treated as memory mapped: prepare extracts its NPY member once.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
from contextlib import nullcontext
import json
import os
import platform
import resource
import shutil
import sys
import subprocess
import threading
import time
import zipfile
from pathlib import Path

import numpy as np

FIELDS = ("thickness", "vp", "vs", "rho")
RESULTS = {"phase_velocity": (np.float64, "modes"),
           "mode_mask": (np.bool_, "modes"), "status": (np.uint8, "modes"),
           "trace_flags": (np.uint32, "traces"),
           "search_level": (np.int32, "traces"),
           "root_count": (np.int32, "traces"),
           "search_bounds": (np.float64, "bounds")}


class GpuMonitor:
    """Observe actual compute, not resident contexts, without touching peers."""
    def __init__(self, allowed):
        self.allowed = allowed
        self.competitors = []
        self.peak_mib = {}
        self.samples = []
        index = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        if not index.isdigit():
            raise ValueError("strict benchmark monitoring requires a numeric CUDA_VISIBLE_DEVICES")
        self.proc = subprocess.Popen(["nvidia-smi", "pmon", "-i", index, "-s", "um", "-d", "1"],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            fields = line.split()
            if len(fields) < 10 or not fields[0].isdigit() or not fields[1].isdigit():
                continue
            pid = int(fields[1])
            self.samples.append(line.strip())
            if pid in self.allowed and fields[9].isdigit():
                self.peak_mib[pid] = max(self.peak_mib.get(pid, 0), int(fields[9]))
            if pid not in self.allowed and fields[2] == "C" and fields[3].isdigit() and int(fields[3]) > 0:
                self.competitors.append(line.strip())

    def check(self):
        if self.proc.poll() is not None:
            raise RuntimeError("GPU process monitor ended; cannot certify contention-free timing")
        if self.competitors:
            raise RuntimeError("concurrent GPU compute observed: " + self.competitors[0])

    def __enter__(self):
        try:
            time.sleep(2)
            self.check()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.thread.join(timeout=5)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def extract_model(path, target):
    started = time.perf_counter()
    if target.exists():
        raise FileExistsError(target)
    with zipfile.ZipFile(path) as archive:
        with archive.open("data.npy") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst, 4 * 1024**2)
    model = np.load(target, mmap_mode="r", allow_pickle=False)
    if model.ndim != 3 or model.shape[1:] != (70, 4):
        raise ValueError(f"expected [N,70,4], got {model.shape}")
    max_depth_error = 0.
    for start in range(0, len(model), 16384):
        depth = np.asarray(model[start:start+16384, :, 0], dtype=np.float64)
        if not np.isfinite(depth).all():
            raise ValueError("nonfinite depth grid")
        max_depth_error = max(max_depth_error, float(np.max(np.abs(depth-np.arange(70)*.04))))
    if max_depth_error > 5e-6:
        raise ValueError("depth grid differs from the fixed 0.04 km Convention A")
    return {"source": str(path.resolve()), "source_sha256": digest_file(path),
            "npy": str(target.resolve()), "npy_sha256": digest_file(target),
            "shape": list(model.shape), "dtype": str(model.dtype), "max_depth_grid_error_km": max_depth_error,
            "unpack_and_hash_seconds": time.perf_counter() - started}


def independent_rows(model, candidates, count, seen=None):
    seen = set() if seen is None else seen
    selected = []
    for i in candidates:
        # Physical parameters, not depth or row ID, define an independent model.
        key = hashlib.sha256(np.ascontiguousarray(model[int(i), :, 1:]).tobytes()).digest()
        if key not in seen:
            seen.add(key)
            selected.append(int(i))
            if len(selected) == count:
                return selected
    raise ValueError(f"only {len(selected)} distinct models available; requested {count}")


def prepare(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {"seed": args.seed, "convention": "A", "datasets": {}}
    for name, folder in (("base", "extracted"), ("aug", "extracted_aug")):
        manifest["datasets"][name] = extract_model(
            args.root / folder / "FlatVelA_model.npz", out / f"{name}.npy")
    base = np.load(out / "base.npy", mmap_mode="r")
    aug = np.load(out / "aug.npy", mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    metrics = np.empty((4, len(aug)), dtype=np.float64)
    for start in range(0, len(aug), 16384):
        stop = min(start + 16384, len(aug))
        vs = np.asarray(aug[start:stop, :, 2], dtype=np.float64)
        metrics[0, start:stop] = vs.max(axis=1) / vs[:, -1]
        metrics[1, start:stop] = np.maximum(0, -np.diff(vs, axis=1)).max(axis=1)
        metrics[2, start:stop] = vs.max(axis=1) / vs.min(axis=1)
        metrics[3, start:stop] = (.04 / vs[:, :-1]).sum(axis=1)
    # 2,000 stress models (500 per metric) plus 8,000 uniform independent rows.
    seen = set()
    stress = []
    for metric in metrics:
        stress.extend(independent_rows(aug, np.argsort(-metric, kind="stable"), 500, seen))
    random = independent_rows(aug, rng.permutation(len(aug)), 8000, seen)
    rows = np.array(stress + random, dtype=np.int64)
    rows = rows[rng.permutation(len(rows))]
    np.save(out / "aug_rows.npy", rows)
    np.save(out / "aug10000.npy", aug[rows])
    base_rows = np.array(independent_rows(base, rng.permutation(len(base)), 512))
    # Interleave both sources: the first 256 rows remain balanced.
    tuning = np.empty((1024, 70, 4), dtype=base.dtype)
    tuning[0::2] = base[base_rows]
    tuning[1::2] = aug[rows[:512]]
    np.save(out / "tuning1024.npy", tuning)
    np.save(out / "tuning_source_rows.npy", np.column_stack((base_rows, rows[:512])))
    np.save(out / "periods.npy", np.logspace(np.log10(.2), np.log10(10.), 100))
    manifest.update(aug_independent_models=len(seen), aug_stress_models=len(stress),
                    metric_names=["max_vs_over_halfspace", "largest_downward_vs_jump",
                                  "max_vs_over_min_vs", "shear_optical_thickness"],
                    total_prepare_seconds=time.perf_counter() - started)
    manifest["files"] = {p.name: digest_file(p) for p in sorted(out.glob("*.npy"))}
    atomic_json(out / "manifest.json", manifest)
    print(json.dumps({"output": str(out), "prepare_seconds": manifest["total_prepare_seconds"]}))


def convert(model):
    h = np.full(model.shape[:2], .04, dtype=np.float64)
    h[:, -1] = 0.
    return dict(thickness=h, vp=np.ascontiguousarray(model[:, :, 1], dtype=np.float64),
                vs=np.ascontiguousarray(model[:, :, 2], dtype=np.float64),
                rho=np.ascontiguousarray(model[:, :, 3], dtype=np.float64))


def environment(md):
    return dict(python=sys.executable, python_version=platform.python_version(),
                platform=platform.platform(), numpy=np.__version__, modisp=md.__version__,
                cpu_module=md._cpu.__file__, cpu_sha256=digest_file(md._cpu.__file__),
                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                cuda_options={k: v for k, v in os.environ.items() if k.startswith("MODISP_CUDA_")})


def make_solver(md, args):
    options = dict(backend=args.backend, threads=args.threads,
                   workspace_bytes=args.workspace_bytes, chunk_size=args.native_chunk)
    if "profile" in inspect.signature(md.Solver).parameters:
        options["profile"] = args.profile
    elif args.profile:
        raise ValueError("this installation does not implement profile=True")
    accuracy = getattr(args, "accuracy", None)
    if accuracy is not None and "accuracy" in inspect.signature(md.Solver).parameters:
        options["accuracy"] = accuracy
    strategy = getattr(args, "search_strategy", None)
    if strategy is not None and "search_strategy" in inspect.signature(md.Solver).parameters:
        options["search_strategy"] = strategy
    secular = getattr(args, "secular", None)
    if secular is not None and "secular" in inspect.signature(md.Solver).parameters:
        options["secular"] = secular
    for key in ("refinement","love_strategy"):
        value=getattr(args,key,None)
        if value is not None:
            if key not in inspect.signature(md.Solver).parameters:raise ValueError(f"installation does not support {key}")
            options[key]=value
    # A pre-0.3.0 installation has only its exact default path; passing a
    # preset to it is intentionally ignored so the same A/B command can pair
    # the old and new packages.
    return md.Solver(**options)


def add_stats(total, stats):
    maxima = {"workspace_bytes", "workspace_budget_bytes", "capacity_models",
              "pinned_host_bytes", "device_workspace_bytes"}
    constant = {"density_table_knots", "cache_capacity", "threads", "capacity"}
    for key, value in stats.items():
        if key == "timings_s":
            dst = total.setdefault(key, {})
            for stage, seconds in value.items():
                dst[stage] = dst.get(stage, 0.) + float(seconds)
        elif key == "active_traces_by_stage":
            total[key] = [a + int(b) for a, b in zip(total.get(key, [0]*len(value)), value)]
        elif isinstance(value, (int, float)):
            total[key] = max(total.get(key, 0), value) if key in maxima else (
                value if key in constant else total.get(key, 0) + value)
        else:
            total[key] = value


def block_digest(arrays, start, stop):
    h = hashlib.sha256()
    for name in RESULTS:
        h.update(np.ascontiguousarray(arrays[name][start:stop]).tobytes())
    return h.hexdigest()


def run(args):
    import modisp as md
    started = time.perf_counter()
    model = np.load(args.input, mmap_mode="r", allow_pickle=False)
    n = min(len(model), args.models or len(model))
    periods = np.ascontiguousarray(np.load(args.periods, allow_pickle=False), dtype=np.float64)
    env = environment(md)
    if args.backend == "cuda":
        module = md._load_cuda()
        env.update(cuda_module=module.__file__, cuda_sha256=digest_file(module.__file__))
    config = dict(input=str(args.input.resolve()), input_sha256=digest_file(args.input),
                  models=n, periods=periods.tolist(), nmodes=5, convention="A",
                  backend=args.backend, threads=args.threads, workspace_bytes=args.workspace_bytes,
                  chunk=args.chunk, native_chunk=args.native_chunk, profile=args.profile,
                  accuracy=getattr(args, "accuracy", None),
                  search_strategy=getattr(args, "search_strategy", None),
                  secular=getattr(args, "secular", None), refinement=getattr(args,"refinement",None),love_strategy=getattr(args,"love_strategy",None),environment=env)
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    out = args.output.resolve()
    checkpoint = out / "checkpoint.json"
    if args.resume:
        state = json.loads(checkpoint.read_text())
        if state["identity"] != identity:
            raise ValueError("resume input/configuration/installed extension differs")
    else:
        out.mkdir(parents=True, exist_ok=False)
        state = dict(identity=identity, config=config, blocks=[], complete=False)
        np.save(out / "periods.npy", periods)
        atomic_json(checkpoint, state)
    arrays = {}
    for name, (dtype, shape) in RESULTS.items():
        dims = (n, 5, len(periods)) if shape == "modes" else ((n, 2) if shape == "bounds" else (n, len(periods)))
        arrays[name] = np.lib.format.open_memmap(out / f"{name}.npy", mode="r+" if args.resume else "w+", dtype=dtype, shape=dims)
    for block in state["blocks"]:
        if block_digest(arrays, block["start"], block["stop"]) != block["sha256"]:
            raise ValueError("completed output block checksum differs")
    first = state["blocks"][-1]["stop"] if state["blocks"] else 0
    setup_s = time.perf_counter() - started
    monitor = GpuMonitor({os.getpid()}) if args.backend == "cuda" and getattr(args, "guard_gpu", False) else None
    with (monitor if monitor else nullcontext()), make_solver(md, args) as solver:
        for start in range(first, n, args.chunk):
            if monitor:
                monitor.check()
            stop = min(start + args.chunk, n)
            t = time.perf_counter()
            block = np.array(model[start:stop], copy=True)
            read_s = time.perf_counter() - t
            t = time.perf_counter()
            batch = convert(block)
            convert_s = time.perf_counter() - t
            t = time.perf_counter()
            result = solver(**batch, periods=periods, nmodes=5)
            solve_s = time.perf_counter() - t
            if monitor:
                monitor.check()
            stats = solver.stats()
            t = time.perf_counter()
            for name in RESULTS:
                arrays[name][start:stop] = getattr(result, name)
                arrays[name].flush()
            sha = block_digest(arrays, start, stop)
            write_s = time.perf_counter() - t
            state["blocks"].append(dict(start=start, stop=stop, sha256=sha,
                                        read_s=read_s, convert_s=convert_s, solve_s=solve_s,
                                        write_flush_hash_s=write_s, stats=stats))
            atomic_json(checkpoint, state)
            print(json.dumps(dict(rows=[start, stop], solve_s=solve_s,
                                  models_per_s=(stop-start)/solve_s)), flush=True)
        kernel_report = solver.kernel_report() if first < n else state.get("kernel_report", {})
    session_s = time.perf_counter() - started
    if monitor:
        state["gpu_process_samples"] = list(monitor.samples)
        state["peak_gpu_process_mib"] = monitor.peak_mib.get(os.getpid())
    stats = {}
    for block in state["blocks"]:
        add_stats(stats, block["stats"])
    state.update(complete=True, stats=stats, kernel_report=kernel_report,
                 latest_session_seconds=session_s, latest_setup_seconds=setup_s,
                 total_solve_seconds=sum(b["solve_s"] for b in state["blocks"]),
                 resumed=args.resume, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024))
    # A resumed run reports per-session time; never label it uninterrupted E2E.
    if not args.resume:
        state.update(end_to_end_seconds=session_s, end_to_end_models_per_s=n/session_s)
    else:
        state.pop("end_to_end_seconds", None)
        state.pop("end_to_end_models_per_s", None)
    state["unresolved_traces"] = int(np.count_nonzero(arrays["trace_flags"] & md.TRACE_SEARCH_UNCONVERGED))
    state["status_counts"] = {str(int(s)): int(np.count_nonzero(arrays["status"] == s)) for s in md.SolveStatus}
    state["level_counts"] = {str(int(s)): int(np.count_nonzero(arrays["search_level"] == s)) for s in np.unique(arrays["search_level"])}
    atomic_json(checkpoint, state)
    atomic_json(out / "report.json", state)


def compare(args):
    a = {k: np.load(args.before / f"{k}.npy", mmap_mode="r") for k in RESULTS}
    b = {k: np.load(args.after / f"{k}.npy", mmap_mode="r") for k in RESULTS}
    if a["phase_velocity"].shape != b["phase_velocity"].shape:
        raise ValueError("result shapes differ")
    report = dict(wrong=0, missing=0, extra=0, newly_unresolved=0, max_common_error=0.,
                  finite_common=0, error_counts_gt={str(t): 0 for t in (1e-6, 1e-5, 1e-4, 1e-3)},
                  diagnostics_changed={k: 0 for k in RESULTS if k != "phase_velocity"})
    for start in range(0, len(a["phase_velocity"]), 512):
        end = start + 512
        x, y = a["phase_velocity"][start:end], b["phase_velocity"][start:end]
        fx, fy = np.isfinite(x), np.isfinite(y)
        common = fx & fy
        err = np.abs(x[common] - y[common])
        report["wrong"] += int(np.count_nonzero(err > np.maximum(3e-6, 2e-6*np.abs(x[common]))))
        report["missing"] += int(np.count_nonzero(fx & ~fy))
        report["extra"] += int(np.count_nonzero(~fx & fy))
        if err.size:
            report["max_common_error"] = max(report["max_common_error"], float(err.max()))
        report["finite_common"] += int(np.count_nonzero(common))
        for threshold in report["error_counts_gt"]:
            report["error_counts_gt"][threshold] += int(np.count_nonzero(err > float(threshold)))
        report["newly_unresolved"] += int(np.count_nonzero(((b["trace_flags"][start:end] & 4) != 0) & ((a["trace_flags"][start:end] & 4) == 0)))
        for name in report["diagnostics_changed"]:
            report["diagnostics_changed"][name] += int(np.count_nonzero(a[name][start:end] != b[name][start:end]))
    report["reference_kind"] = "before/after regression; not an independent physical reference"
    report["exact_equal"] = not any(report[k] for k in ("wrong", "missing", "extra", "newly_unresolved")) and not any(report["diagnostics_changed"].values())
    report["passed"] = report["exact_equal"] or bool(getattr(args, "allow_accuracy_change", False))
    atomic_json(args.output, report)
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit(2)


def bench(args):
    import modisp as md
    model = np.load(args.input, mmap_mode="r")[:args.models or 1024]
    periods = np.ascontiguousarray(np.load(args.periods)[::args.period_stride], dtype=np.float64)
    batch = convert(model)
    samples, outputs = [], []
    monitor = GpuMonitor({os.getpid()}) if args.backend == "cuda" and getattr(args, "guard_gpu", False) else None
    with (monitor if monitor else nullcontext()), make_solver(md, args) as solver:
        for i in range(args.repeats + 2):
            if monitor:
                monitor.check()
            t = time.perf_counter()
            result = solver(**batch, periods=periods, nmodes=5)
            elapsed = time.perf_counter() - t
            if monitor:
                monitor.check()
            if i == 0:
                cold = elapsed
                outputs.append({k: getattr(result, k).copy() for k in RESULTS})
            if i >= 2:
                samples.append(elapsed)
            print(json.dumps(dict(call=i, seconds=elapsed)), flush=True)
        outputs.append({k: getattr(result, k).copy() for k in RESULTS})
        report = dict(environment=environment(md), models=len(model), periods=len(periods),
                      backend=args.backend, threads=args.threads, workspace_bytes=args.workspace_bytes,
                      profile=args.profile, first_call_seconds=cold, samples_seconds=samples,
                      steady_median_seconds=float(np.median(samples)), stats=solver.stats(),
                      kernel_report=solver.kernel_report())
    report["models_per_second"] = len(model)/report["steady_median_seconds"]
    if monitor:
        report["gpu_process_samples"] = list(monitor.samples)
        report["peak_gpu_process_mib"] = monitor.peak_mib.get(os.getpid())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, values in zip(("first", "last"), outputs):
        np.savez(args.output.with_suffix(f".{name}.npz"), **values, periods=periods)
    atomic_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261028)
    p = sub.add_parser("compare")
    for flag in ("before", "after", "output"):
        p.add_argument(f"--{flag}", type=Path, required=True)
    p.add_argument("--allow-accuracy-change", action="store_true")
    for command in ("run", "bench"):
        p = sub.add_parser(command)
        p.add_argument("--input", type=Path, required=True)
        p.add_argument("--periods", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--backend", choices=("cpu", "cuda"), required=True)
        p.add_argument("--threads", type=int, default=8)
        p.add_argument("--workspace-bytes", type=int, default=256*1024**2)
        p.add_argument("--native-chunk", type=int)
        p.add_argument("--models", type=int, default=0)
        p.add_argument("--profile", action="store_true")
        p.add_argument("--accuracy", choices=("high", "balanced", "fast"))
        p.add_argument("--search-strategy", choices=("shape_adaptive", "qe_adaptive", "uniform"))
        p.add_argument("--secular", choices=("dunkin", "fast"))
        p.add_argument("--refinement",choices=("legacy","brent","chandrupatla"))
        p.add_argument("--love-strategy",choices=("phase","scan"))
        p.add_argument("--guard-gpu", action="store_true", help="reject observed competing GPU compute")
        if command == "run":
            p.add_argument("--chunk", type=int, default=1024)
            p.add_argument("--resume", action="store_true")
        else:
            p.add_argument("--repeats", type=int, default=5)
            p.add_argument("--period-stride", type=int, default=1)
    args = parser.parse_args()
    if getattr(args, "models", 0) < 0 or getattr(args, "chunk", 1) < 1 or getattr(args, "repeats", 1) < 1 or getattr(args, "period_stride", 1) < 1:
        parser.error("counts and strides must be positive")
    globals()[args.command](args)


if __name__ == "__main__":
    main()
