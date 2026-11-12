#!/Users/yusong/miniforge3/bin/python
"""Same-input CPU FP64 / CUDA FP64 / CUDA FP32 benchmark and diagnostics.

Use the pinned matrix in docs/benchmarks/backend-0.4.0-matrix.json. The run
command checkpoints each workload and can resume without pooling samples.
No peer process is stopped, no device setting is modified, and no output
array is filtered to improve the reported numerical result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

from backend_metrics import compare, quantiles

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("thickness", "vp", "vs", "rho")
ARRAYS = ("phase_velocity", "mode_mask", "status", "trace_flags", "numerical_flags",
          "search_level", "root_count", "search_bounds", "approximate_search",
          "mode_index_uncertain", "close_root_mask", "close_root_risk", "potentially_ambiguous")
PATHS = {
    "C64": dict(backend="cpu", precision="float64", secular="dunkin"),
    "G64": dict(backend="cuda", precision="float64", secular="fast", device=0),
    "G32": dict(backend="cuda", precision="float32", secular="fast", device=0),
}
ORDERS = (("C64","G64","G32"), ("G64","G32","C64"), ("G32","C64","G64"),
          ("C64","G32","G64"), ("G32","G64","C64"), ("G64","C64","G32"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def shell(command):
    try:
        p = subprocess.run(command, text=True, capture_output=True, timeout=20)
        return dict(command=command, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except Exception as exc:
        return dict(command=command, error=str(exc))


def environment(md):
    import modisp_cuda
    packages = {name: importlib.metadata.version(name) for name in
                ("modisp", "modisp-cuda12", "numpy", "scipy", "nvidia-ml-py")}
    modules = {}
    for parent in (Path(md.__file__).parent, Path(modisp_cuda.__file__).parent):
        for p in sorted(parent.glob("*.so")):
            modules[p.name] = digest(p)
    return dict(packages=packages, native_sha256=modules,
        python=platform.python_version(), platform=platform.platform(),
        affinity=sorted(os.sched_getaffinity(0)), load_average=list(os.getloadavg()),
        env={k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "MODISP_CUDA_BLOCK", "MODISP_CUDA_BLOCKS_PER_SM")},
        lscpu=shell(["lscpu"]), topology=shell(["lscpu", "-e=CPU,CORE,SOCKET,NODE,ONLINE"]),
        gpu=shell(["nvidia-smi", "-q"]), nvcc=shell(["nvcc", "--version"]),
        doctor=shell([sys.executable, "-m", "modisp.cli", "doctor"]),
        tool_sha256=digest(__file__), metrics_sha256=digest(Path(__file__).with_name("backend_metrics.py")))


def prepare(args):
    import modisp as md
    if md.__version__ != "0.4.0":
        raise RuntimeError("input preparation requires modisp 0.4.0")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    matrix = json.loads(args.matrix.read_text())
    common = matrix["inputs"]["common_options"]
    batches = {}
    for spec in matrix["inputs"]["families"]:
        if spec["id"] == "wet24":
            batch = {k: v[:64].copy() for k, v in batches["solid24"].items()}
            for k, value in spec["replace_first_layer"].items():
                batch[k][:, 0] = value
        else:
            batch = md.random_model_batch(spec["B"], spec["L"], seed=spec["seed"], **common)
        batches[spec["id"]] = batch
        np.savez(out / (spec["id"] + ".npz"), **batch)
        physical = np.stack([batch[k] for k in FIELDS], axis=-1)
        hashes = [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in physical]
        if len(set(hashes)) != len(hashes):
            raise ValueError("duplicate physical models")
        write_json(out / (spec["id"] + "-models.json"), hashes)
    np.save(out / "periods1.npy", np.array([3.], dtype=np.float64))
    np.save(out / "periods64.npy", np.geomspace(.15, 10., 64))
    write_json(out / "matrix.json", matrix)
    manifest = dict(modisp=md.__version__, numpy=np.__version__,
                    files={p.name:digest(p) for p in sorted(out.iterdir())})
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


def load_case(inputs, case):
    with np.load(inputs / (case["input_family"] + ".npz"), allow_pickle=False) as z:
        batch = {k:np.ascontiguousarray(z[k][:case["B"]], dtype=np.float64) for k in FIELDS}
    return batch, np.ascontiguousarray(np.load(inputs / (case["periods"] + ".npy")), dtype=np.float64)


def result_arrays(result):
    return {k:np.asarray(getattr(result, k)) for k in ARRAYS}


def signature(arrays):
    h = hashlib.sha256()
    for key in ARRAYS:
        a = arrays[key]
        if a.dtype.kind == "f" and np.isnan(a).any():
            a = a.copy()
            a[np.isnan(a)] = np.nan  # disregard meaningless NaN payload bits
        h.update(key.encode())
        h.update(a.dtype.str.encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def worker(args):
    started = time.perf_counter()
    import modisp as md
    import_s = time.perf_counter() - started
    case = json.loads(args.case)
    batch, periods = load_case(args.input, case)
    opts = dict(PATHS[args.path], threads=args.threads, accuracy=case["accuracy"],
                search_strategy="shape_adaptive", refinement="brent", love_strategy="phase",
                close_root_rtol=.005, f_accept=1., workspace_bytes=1024**3,
                chunk_size=args.chunk_size, profile=args.profile)
    started = time.perf_counter()
    solver = md.Solver(**opts)
    ctor_s = time.perf_counter() - started
    print(json.dumps(dict(ready=True, pid=os.getpid(), import_s=import_s, constructor_s=ctor_s,
                          options=opts)), flush=True)
    first_arrays = None
    first_signature = None
    for line in sys.stdin:
        command = json.loads(line)
        if command["action"] == "close":
            break
        try:
            start = time.perf_counter_ns()
            result = solver(**batch, periods=periods, nmodes=case["nmodes"], wave=case["wave"])
            seconds = (time.perf_counter_ns() - start)*1e-9
            if result.backend_used != opts["backend"] or result.precision != opts["precision"]:
                raise RuntimeError("unexpected backend or precision")
            arrays = result_arrays(result)
            sig = signature(arrays)
            if first_arrays is None:
                first_arrays = {k:v.copy() for k,v in arrays.items()}
                first_signature = sig
            differences = {}
            if sig != first_signature:
                for k, x in arrays.items():
                    y = first_arrays[k]
                    eq = (x == y)
                    if x.dtype.kind == "f":
                        eq |= np.isnan(x) & np.isnan(y)
                    if not eq.all():
                        differences[k] = int(np.count_nonzero(~eq))
            stats = solver.stats()
            counts = Counter(map(int, arrays["status"].ravel()))
            flags = arrays["trace_flags"]
            repair = int(np.count_nonzero(flags & md.TRACE_FP64_REPAIR))
            if repair != stats.get("fp64_repair_traces", 0):
                raise RuntimeError("repair flag/counter disagreement")
            summary = dict(seconds=seconds, signature=sig, repeated_differences=differences,
                status_counts={str(i):counts.get(i,0) for i in range(4)},
                numerical_failure_traces=int(np.count_nonzero(np.any((arrays["status"]==2)|(arrays["status"]==3),axis=1))),
                flag_counts={str(1<<i):int(np.count_nonzero(flags & (1<<i))) for i in range(16)},
                search_level_counts={str(int(i)):int(n) for i,n in zip(*np.unique(arrays["search_level"],return_counts=True))},
                stats=stats, kernel_report=solver.kernel_report())
            save = command.get("save")
            if differences and not save:
                save = str(args.output / (command["tag"] + "-changed.npz"))
            if save:
                dest = Path(save)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if len(periods) > 1:
                    check = result.check_modes()
                    arrays.update(reliability_flags=check.flags, reliability_assessed=check.assessed,
                                  reliability_passes_checks=check.passes_checks)
                    summary["reliability"] = dict(assessed=int(check.assessed.sum()),
                        passes=int(check.passes_checks.sum()),
                        flags={str(1<<i):int(np.count_nonzero(check.flags & (1<<i))) for i in range(11)})
                np.savez_compressed(dest, **arrays, periods=periods)
                summary["saved"] = dest.name
                summary["output_sha256"] = digest(dest)
            print(json.dumps(summary, allow_nan=False), flush=True)
        except Exception as exc:
            print(json.dumps(dict(error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())), flush=True)
    solver.close()


class Worker:
    def __init__(self, args, case, path, directory, *, profile=False, chunk_size=None):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.stderr = (directory / "worker.stderr").open("w")
        cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--input", str(args.input),
               "--output", str(directory), "--case", json.dumps(case), "--path", path,
               "--threads", str(args.threads)]
        if profile:
            cmd.append("--profile")
        if chunk_size:
            cmd.extend(("--chunk-size",str(chunk_size)))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.stderr, text=True, bufsize=1)
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"worker failed to start: {directory}")
        self.ready = json.loads(line)

    def solve(self, tag, save=False):
        command = dict(action="solve", tag=tag)
        if save:
            command["save"] = str(self.directory / (tag+".npz"))
        self.proc.stdin.write(json.dumps(command)+"\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"worker exited: {self.directory}")
        return json.loads(line)

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write('{"action":"close"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.terminate()
                self.proc.wait(timeout=10)
        self.stderr.close()


class Monitor:
    """NVML sampling in the parent, with explicit PID and activity scope."""
    def __init__(self, gpu_uuid, output, interval=.25, allowed=()):
        import pynvml as nv
        self.nv = nv
        nv.nvmlInit()
        self.handle = nv.nvmlDeviceGetHandleByUUID(gpu_uuid)
        self.output, self.interval, self.allowed = output, interval, set(allowed)
        self.stop = threading.Event()
        self.samples = []
        self.errors = []
        self.peaks = {}
        self.activity = set()
        self.foreign_residents = set()
        self.last_util = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self):
        nv, h = self.nv, self.handle
        values = dict(time_ns=time.time_ns(), load_average=list(os.getloadavg()))
        mem = nv.nvmlDeviceGetMemoryInfo(h)
        util = nv.nvmlDeviceGetUtilizationRates(h)
        values.update(device_used_bytes=int(mem.used), gpu_util=int(util.gpu), memory_util=int(util.memory),
                      temperature=int(nv.nvmlDeviceGetTemperature(h,nv.NVML_TEMPERATURE_GPU)),
                      sm_clock_mhz=int(nv.nvmlDeviceGetClockInfo(h,nv.NVML_CLOCK_SM)))
        processes = nv.nvmlDeviceGetComputeRunningProcesses(h)
        values["pid_memory_bytes"] = {}
        for p in processes:
            amount = int(p.usedGpuMemory)
            if 0 <= amount < 2**63:
                values["pid_memory_bytes"][str(p.pid)] = amount
                self.peaks[str(p.pid)] = max(self.peaks.get(str(p.pid),0), amount)
            if p.pid not in self.allowed:
                self.foreign_residents.add(p.pid)
        try:
            activity = nv.nvmlDeviceGetProcessUtilization(h,self.last_util)
            for u in activity:
                self.last_util = max(self.last_util,int(u.timeStamp))
                if u.pid not in self.allowed and int(u.smUtil)>0:
                    self.activity.add(int(u.pid))
            values["foreign_active_process_count"] = len(self.activity)
        except nv.NVMLError as exc:
            if type(exc).__name__ not in ("NVMLError_NotFound",):
                label = type(exc).__name__
                if label not in self.errors:
                    self.errors.append(label)
        self.samples.append(values)

    def _loop(self):
        while not self.stop.is_set():
            started=time.monotonic()
            try:
                self._sample()
            except Exception as exc:
                label=type(exc).__name__+": "+str(exc)
                if label not in self.errors:
                    self.errors.append(label)
            self.stop.wait(max(0.,self.interval-(time.monotonic()-started)))

    def __enter__(self):
        # Set the activity cursor now; old process-utilization samples do not
        # establish contention during a new measurement interval.
        self.last_util = int(time.time()*1e6)
        self.thread.start()
        return self

    def __exit__(self,*exc):
        self.stop.set()
        self.thread.join(timeout=10)
        self._sample()
        self.output.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(self.output,"wt") as f:
            for row in self.samples:
                f.write(json.dumps(row)+"\n")
        self.nv.nvmlShutdown()

    def summary(self):
        intervals=np.diff([s["time_ns"] for s in self.samples])*1e-6
        loads=[s["load_average"][0] for s in self.samples]
        return dict(requested_interval_ms=self.interval*1000,sample_count=len(self.samples),
            observed_interval_ms=quantiles(intervals),pid_peak_bytes=self.peaks,
            device_sampled_peak_bytes=max((s["device_used_bytes"] for s in self.samples),default=None),
            foreign_active_process_count=len(self.activity), foreign_resident_compute_count=len(self.foreign_residents),
            contamination_observed=bool(self.activity),monitor_errors=self.errors,
            load_average_1min_range=[min(loads),max(loads)] if loads else [],
            gpu_util_range=[min(s["gpu_util"] for s in self.samples),max(s["gpu_util"] for s in self.samples)] if self.samples else [],
            temperature_range=[min(s["temperature"] for s in self.samples),max(s["temperature"] for s in self.samples)] if self.samples else [])


def timed_group(args, case, group_id):
    directory=args.output/"timing"/group_id
    record_path=directory/"record.json"
    if record_path.exists():
        return json.loads(record_path.read_text())
    workers={}
    records={p:dict(first=None,warmup=[],steady=[]) for p in PATHS}
    started=time.time()
    try:
        for path in PATHS:
            workers[path]=Worker(args,case,path,directory/path)
            records[path]["worker"]=workers[path].ready
        with Monitor(args.gpu_uuid,directory/"monitor.jsonl.gz",allowed=[w.proc.pid for w in workers.values()]) as monitor:
            for path in PATHS:
                records[path]["first"]=workers[path].solve("first",save=True)
            for warmup in range(3):
                for path in ORDERS[warmup]:
                    records[path]["warmup"].append(workers[path].solve(f"warm{warmup}"))
            for round_index in range(12):
                for position,path in enumerate(ORDERS[round_index%6]):
                    result=workers[path].solve(f"steady{round_index:02}",save=round_index in (0,11))
                    result.update(round=round_index,position=position)
                    records[path]["steady"].append(result)
                print(json.dumps(dict(event="round",case=group_id,round=round_index,
                    seconds={p:records[p]["steady"][-1].get("seconds") for p in PATHS})),flush=True)
        for path,r in records.items():
            good=[x for x in r["steady"] if "seconds" in x]
            times=[x["seconds"] for x in good]
            r["summary"]=dict(completed=len(good)==12,steady_seconds=times,
                median_s=float(np.median(times)) if times else None,
                q25_s=float(np.quantile(times,.25)) if times else None,
                q75_s=float(np.quantile(times,.75)) if times else None,
                min_s=min(times) if times else None,max_s=max(times) if times else None,
                valid_root_counts=[x["status_counts"]["0"] for x in good],
                repeated_changes=sum(bool(x.get("repeated_differences")) for x in r["warmup"]+r["steady"]),
                failed_calls=sum("error" in x for x in [r["first"]]+r["warmup"]+r["steady"]))
        record=dict(case=case,group_id=group_id,started_unix=started,finished_unix=time.time(),
                    monitor=monitor.summary(),paths=records)
        write_json(record_path,record)
        return record
    finally:
        for w in workers.values():
            w.close()


def memory_group(args, case, group_id):
    for path in ("G64","G32"):
        directory=args.output/"memory"/group_id/path
        target=directory/"record.json"
        if target.exists():
            continue
        w=Worker(args,case,path,directory)
        try:
            with Monitor(args.gpu_uuid,directory/"monitor.jsonl.gz",interval=.01,allowed=[w.proc.pid]) as mon:
                samples=[w.solve("first")]
                samples += [w.solve(f"warm{i}") for i in range(3)]
                samples += [w.solve(f"steady{i}") for i in range(12)]
            write_json(target,dict(worker=w.ready,monitor=mon.summary(),last=samples[-1],
                errors=[s for s in samples if "error" in s],
                sampled_pid_peak_bytes=mon.peaks.get(str(w.proc.pid)),
                label="10 ms sampled PID VRAM peak, lower bound on instantaneous peak"))
        finally:
            w.close()


def references(args, cases):
    for case,group_id in cases:
        directory=args.output/"references"/group_id
        if (directory/"record.json").exists():
            continue
        spec=dict(case,accuracy="high",nmodes=7)
        w=Worker(args,spec,"C64",directory)
        try:
            record=w.solve("reference",save=True)
            write_json(directory/"record.json",record)
            print(json.dumps(dict(event="reference",case=group_id,seconds=record.get("seconds"))),flush=True)
        finally:
            w.close()


def profile_and_chunk(args, cases):
    targets={"M05-R","M06-R","M04-L","S03-R"}
    for case,group_id in cases:
        if group_id not in targets:
            continue
        for path in PATHS:
            directory=args.output/"profiling"/group_id/path
            if (directory/"record.json").exists():
                continue
            w=Worker(args,case,path,directory,profile=True)
            try:
                warm=w.solve("warmup")
                rows=[w.solve(f"profile{i}") for i in range(3)]
                write_json(directory/"record.json",dict(warmup=warm,calls=rows))
            finally:
                w.close()
        if group_id=="S03-R":
            directory=args.output/"chunk_check"/group_id
            if not (directory/"record.json").exists():
                w=Worker(args,case,"G32",directory,chunk_size=8)
                try:
                    write_json(directory/"record.json",w.solve("chunk8",save=True))
                finally:
                    w.close()


def independent(args):
    import modisp as md
    from scipy.optimize import brentq
    out=args.output/"independent"
    if (out/"record.json").exists():
        return
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    specs=[]
    for p in sorted((ROOT/"tests/data").glob("witness_*.npz"))+sorted((ROOT/"tests/data").glob("B10000_*.npz")):
        cropped=p.name.startswith("B10000")
        with np.load(p,allow_pickle=False) as z:
            expected=z["reference" if cropped else "reference_hi"].ravel()[:3 if cropped else 5]
            specs.append(dict(id=p.stem,batch={k:z[k].copy() for k in FIELDS},periods=z["periods"].copy(),
                wave="rayleigh",reference=expected.reshape(1,-1,1),options=dict(cmin=.2) if cropped else {},sha256=digest(p)))
    h,b1,b2,r1,r2=1.,1.5,3.5,2.,2.7
    def love_roots(period,n):
        omega=2*np.pi/period
        def f(c):
            q=omega*np.sqrt(1/b1**2-1/c**2)
            r=omega*np.sqrt(1/c**2-1/b2**2)
            return r2*b2*b2*r*np.cos(q*h)-r1*b1*b1*q*np.sin(q*h)
        cs=np.linspace(np.nextafter(b1,b2),np.nextafter(b2,b1),n)
        ff=f(cs)
        ii=np.flatnonzero(np.signbit(ff[:-1])!=np.signbit(ff[1:]))
        return np.array([brentq(f,cs[i],cs[i+1],xtol=1e-13) for i in ii])
    root=love_roots(2.,10000)
    np.testing.assert_allclose(root,love_roots(2.,20000),atol=1e-11,rtol=1e-11)
    expected=np.full((1,3,1),np.nan)
    expected[0,:len(root),0]=root
    specs.append(dict(id="analytic_love_two_layer",batch=dict(thickness=[1.,0.],vp=[3.,6.],vs=[1.5,3.5],rho=[2.,2.7]),
                      periods=np.array([2.]),wave="love",reference=expected,options={}))
    specs.append(dict(id="analytic_love_empty",batch=dict(thickness=[0.],vp=[6.],vs=[3.5],rho=[2.7]),
                      periods=np.array([.1,1.,10.]),wave="love",reference=np.full((1,1,3),np.nan),options={}))
    cutoff=2*h*np.sqrt(1/b1**2-1/b2**2)
    periods=cutoff*np.array([1-1e-10,1,1+1e-10])
    expected=np.full((1,3,3),np.nan)
    for j,period in enumerate(periods):
        a,b=love_roots(period,10000),love_roots(period,20000)
        np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10)
        expected[0,:min(len(a),3),j]=a[:3]
    specs.append(dict(id="analytic_love_cutoff",batch=dict(thickness=[1.,0.],vp=[3.,6.],vs=[1.5,3.5],rho=[2.,2.7]),
                      periods=periods,wave="love",reference=expected,options={},stable=np.array([[True,False,True]])))
    ratio=(.18/.34)**2
    rayleigh=.18*np.sqrt(brentq(lambda x:x**3-8*x*x+(24-16*ratio)*x-16*(1-ratio),0.,1.))
    specs.append(dict(id="analytic_low_velocity_rayleigh",batch=dict(thickness=[5.,0.],vp=[.34,12.],vs=[.18,6.],rho=[1.8,3.3]),
                      periods=np.array([.5,1.]),wave="rayleigh",reference=np.full((1,1,2),rayleigh),options=dict(cmin=.005,cmax=.22)))
    for spec in specs:
        for accuracy in ("high","balanced"):
            for path,opts in PATHS.items():
                row=dict(case=spec["id"],path=path,accuracy=accuracy,reference_kind="independent_checked_prefix_or_analytic",input_sha256=spec.get("sha256"))
                try:
                    with md.Solver(**opts,threads=args.threads,accuracy=accuracy,workspace_bytes=1024**3,**spec["options"]) as solver:
                        result=solver(**spec["batch"],periods=spec["periods"],nmodes=spec["reference"].shape[1],wave=spec["wave"])
                        arrays=result_arrays(result)
                        row.update(metrics=compare(arrays,dict(phase_velocity=spec["reference"]),
                            budget_rtol=2e-6 if accuracy=="high" else 2e-5,stable=spec.get("stable")),stats=solver.stats(),
                            status_counts={str(i):int(np.count_nonzero(result.status==i)) for i in range(4)},
                            repair_traces=int(np.count_nonzero(result.trace_flags&32768)))
                        np.savez_compressed(out/f'{spec["id"]}-{accuracy}-{path}.npz',**arrays,reference=spec["reference"],periods=spec["periods"])
                except Exception as exc:
                    row.update(error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
                rows.append(row)
    # Independent water invariance check for SH (not new branch truth).
    for accuracy in ("high","balanced"):
        for path,opts in PATHS.items():
            with md.Solver(**opts,threads=args.threads,accuracy=accuracy) as solver:
                dry=solver([1.,0.],[3.,6.],[1.5,3.5],[2.,2.7],np.geomspace(.1,10.,24),nmodes=3,wave="love")
                wet=solver([.08,1.,0.],[1.5,3.,6.],[0.,1.5,3.5],[1.02,2.,2.7],np.geomspace(.1,10.,24),nmodes=3,wave="love")
                rows.append(dict(case="love_water_invariance",path=path,accuracy=accuracy,
                    reference_kind="physical_invariant",metrics=compare(result_arrays(wet),result_arrays(dry),budget_rtol=2e-6 if accuracy=="high" else 2e-5)))
    write_json(out/"record.json",rows)
    print(json.dumps(dict(event="independent",rows=len(rows))),flush=True)


def analyze(args, cases):
    rows=[]
    for case,group_id in cases:
        record_path=args.output/"timing"/group_id/"record.json"
        if not record_path.exists():
            continue
        record=json.loads(record_path.read_text())
        snapshots={}
        for path in PATHS:
            p=args.output/"timing"/group_id/path/"steady11.npz"
            if p.exists():
                with np.load(p,allow_pickle=False) as z:
                    snapshots[path]={k:z[k].copy() for k in ARRAYS}
        ref_file=args.output/"references"/group_id/"reference.npz"
        reference=None
        if ref_file.exists():
            with np.load(ref_file,allow_pickle=False) as z:
                reference={k:z[k].copy() for k in ARRAYS}
        for path in PATHS:
            r=record["paths"][path]
            summary=r["summary"]
            row=dict(case=group_id,path=path,geometry={k:case[k] for k in ("B","L","T","nmodes","wave","accuracy")},
                     timing=summary,monitor=record["monitor"],first_solve_s=r["first"].get("seconds"))
            if summary["median_s"]:
                t=summary["median_s"]
                row["models_per_s"]=case["B"]/t
                row["traces_per_s"]=case["B"]*case["T"]/t
                good=[s for s in r["steady"] if "seconds" in s]
                valid_counts=[s["status_counts"]["0"] for s in good]
                row["valid_roots_per_s"]=(valid_counts[0]/t if len(set(valid_counts))==1 else
                    float(np.median([s["status_counts"]["0"]/s["seconds"] for s in good])))
            if path in snapshots:
                arr=snapshots[path]
                row["status_counts"]={str(i):int(np.count_nonzero(arr["status"]==i)) for i in range(4)}
                row["flag_counts"]=r["steady"][-1]["flag_counts"]
                row["stats"]=r["steady"][-1]["stats"]
                row["reliability"]=r["steady"][-1].get("reliability")
                if reference is not None:
                    row["against_cpu_high"]=compare(arr,reference,budget_rtol=2e-6 if case["accuracy"]=="high" else 2e-5)
                other={"G64":"C64","G32":"G64"}.get(path)
                if other in snapshots:
                    row["paired_reference"]=other
                    row["paired"]=compare(arr,snapshots[other],budget_rtol=2e-6)
            memory=args.output/"memory"/group_id/path/"record.json"
            if memory.exists():
                row["memory"]=json.loads(memory.read_text())
            rows.append(row)
    write_json(args.output/"summary.json",dict(rows=rows,case_count=len(cases),path_rows=len(rows)))
    return rows


def run(args):
    import modisp as md
    if md.__version__!="0.4.0":
        raise RuntimeError("benchmark requires modisp 0.4.0")
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
        if os.environ.get(key)!="1":
            raise RuntimeError(f"set {key}=1 before starting the process")
    if args.affinity:
        cpus=set(map(int,args.affinity.split(",")))
        os.sched_setaffinity(0,cpus)
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.input/"manifest.json").read_text())
    for name,expected in manifest["files"].items():
        if digest(args.input/name)!=expected:
            raise ValueError(f"frozen input hash changed: {name}")
    matrix=json.loads((args.input/"matrix.json").read_text())
    seen=set()
    cases=[]
    for case in matrix["cases"]:
        group_id=case["id"].rsplit("-",1)[0]
        if group_id in seen or (args.cases and group_id not in args.cases):
            continue
        seen.add(group_id)
        cases.append((case,group_id))
    config=dict(threads=args.threads,gpu_uuid=args.gpu_uuid,affinity=sorted(os.sched_getaffinity(0)),
                input_manifest_sha256=digest(args.input/"manifest.json"),tool_sha256=digest(__file__))
    target=args.output/"configuration.json"
    if target.exists() and json.loads(target.read_text())!=config:
        raise ValueError("resume configuration changed; use another output directory")
    write_json(target,config)
    if not (args.output/"environment.json").exists():
        write_json(args.output/"environment.json",environment(md))
    for phase in args.phases:
        print(json.dumps(dict(event="phase",phase=phase,time_unix=time.time())),flush=True)
        if phase=="timing":
            for case,group_id in cases:
                timed_group(args,case,group_id)
                analyze(args,cases)
        elif phase=="memory":
            for case,group_id in cases:
                memory_group(args,case,group_id)
                print(json.dumps(dict(event="memory",case=group_id)),flush=True)
        elif phase=="reference":
            references(args,cases)
        elif phase=="independent":
            independent(args)
        elif phase=="profile":
            profile_and_chunk(args,cases)
        elif phase=="analyze":
            analyze(args,cases)
    analyze(args,cases)
    print(json.dumps(dict(event="complete",output=str(args.output))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("prepare")
    q.add_argument("--matrix",type=Path,default=ROOT/"docs/benchmarks/backend-0.4.0-matrix.json")
    q.add_argument("--output",type=Path,required=True)
    q=sub.add_parser("worker")
    q.add_argument("--input",type=Path,required=True)
    q.add_argument("--output",type=Path,required=True)
    q.add_argument("--case",required=True)
    q.add_argument("--path",choices=PATHS,required=True)
    q.add_argument("--threads",type=int,required=True)
    q.add_argument("--profile",action="store_true")
    q.add_argument("--chunk-size",type=int)
    q=sub.add_parser("run")
    q.add_argument("--input",type=Path,required=True)
    q.add_argument("--output",type=Path,required=True)
    q.add_argument("--threads",type=int,required=True)
    q.add_argument("--affinity")
    q.add_argument("--gpu-uuid",required=True)
    q.add_argument("--cases",nargs="*")
    q.add_argument("--phases",nargs="+",choices=("timing","memory","reference","independent","profile","analyze"),
                   default=["timing","memory","reference","independent","profile","analyze"])
    args=p.parse_args()
    if args.command=="prepare":
        prepare(args)
    elif args.command=="worker":
        worker(args)
    else:
        run(args)


if __name__=="__main__":
    main()
