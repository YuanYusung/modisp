"""Compare same-host forward solves, and judge changed traces against independent searches.

Use --mode accuracy to check the retained missed-root witnesses and new model seeds,
or --mode time for the wide/dense, dry/wet Rayleigh and Love timing matrix.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from backend_metrics import compare

MODEL_FIELDS = ("thickness", "vp", "vs", "rho")
TIMING_CASES = [f"{grid}-{wave}" for grid in ("wide", "dense") for wave in ("dry", "wet", "love")]
CHECK_CASES = ["m06-1453", "m06-1881", "holdout-dry", "holdout-wet", "holdout-love", "love-cutoff",
               "s02-70", "m06-endpoints", "fp32-alias"]
DATA = Path(__file__).resolve().parents[1] / "tests/data"
WITNESSES = Path(__file__).resolve().parents[1] / "docs/benchmarks/backend-0.4.0-witnesses-rtx5050.npz"


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(runtime):
    return {str(p.relative_to(runtime)): sha(p) for pkg in ["modisp", "modisp_cuda"]
            for p in sorted((runtime/pkg).glob("*")) if p.suffix == ".so" or p.name in ("__init__.py","_continuity.py")}


def gpu_state():
    def query(fields):
        result = subprocess.run(["nvidia-smi", fields, "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True)
        return [row.split(", ") for row in result.stdout.splitlines() if row.strip()]

    processes = query("--query-compute-apps=gpu_uuid,used_memory")
    devices = []
    for index, uuid, name, driver, memory, utilization in query(
            "--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu"):
        external = [used for device, used in processes if device == uuid]
        devices.append(dict(index=int(index), name=name, driver=driver, used_mib=int(memory),
                            utilization_percent=int(utilization), external_processes=len(external),
                            external_mib=sum(int(used) for used in external if used.isdigit())))
    return devices


def case_input(name, inputs):
    group = "rayleigh"
    count = 5
    if name == "large":
        import modisp
        model = modisp.random_model_batch(128,50,seed=20261114,h_min=.04,h_max=.04)
        periods = np.geomspace(.1,.15,64)
    elif name in TIMING_CASES:
        grid, wave = name.split("-")
        file = inputs / ("wet24.npz" if wave == "wet" else "solid24.npz")
        with np.load(file) as z:
            model = {k: np.ascontiguousarray(z[k][:12]) for k in MODEL_FIELDS}
        periods = np.geomspace(.1, .15, 40) if grid == "dense" else np.geomspace(.15, 10., 40)
        if grid == "dense":
            model["thickness"][:, :-1] = .04
            if wave == "wet":
                model["thickness"][:, 0] = .08
        group = "love" if wave == "love" else "rayleigh"
    elif name.startswith("m06-") and name != "m06-endpoints":
        with np.load(DATA / "continuity_rayleigh.npz") as z:
            i = list(z["model_index"]).index(int(name.split("-")[1]))
            j = z["period_index"][i]
            model = {k: z[k][i:i+1] for k in MODEL_FIELDS}
            periods = z["periods"][j-3:j+4]
    elif name in ("s02-70", "m06-endpoints"):
        prefix, models = ("S02_R", [0]) if name == "s02-70" else ("M06_R", [1193,1914])
        j = 1 if name == "s02-70" else 0
        with np.load(WITNESSES) as z:
            ids = [np.flatnonzero((z[prefix+"__model_index"] == b)&(z[prefix+"__period_index"] == j)).item() for b in models]
            model = {k:z[prefix+"__"+k][ids] for k in MODEL_FIELDS}
        periods = np.load(inputs / "periods64.npy")[:8]
    elif name == "love-cutoff":
        model = dict(zip(MODEL_FIELDS, [np.array([x]) for x in ([1.,0.],[3.,6.],[1.5,3.5],[2.,2.7])]))
        group = "love"
        periods = np.geomspace(1.15,1.25,33)
    elif name == "fp32-alias":
        with np.load(DATA / "continuation_fp32_aliases.npz") as z:
            model = {k:z[k].copy() for k in MODEL_FIELDS}
            periods = z["periods"].copy()
    else:
        import modisp
        model=modisp.random_model_batch(10, 14, seed=20261127, h_min=.035, h_max=.3, lvz_prob=.5)
        periods=np.geomspace(.14,3.,40)
        group="love" if name=="holdout-love" else "rayleigh"
        if name=="holdout-wet":
            for k, value in zip(MODEL_FIELDS, (.1,1.5,0.,1.02)):
                model[k][:,0] = value
    return dict(**model, periods=periods, nmodes=count, wave=group)


def start(runtime, inputs, case, backend, group, variant, threads):
    global solver, call_args
    sys.path.insert(0, str(runtime))
    import modisp
    assert Path(modisp.__file__).parent.parent == runtime
    call_args = case_input(case, inputs)
    options = dict(backend=backend, threads=threads, group_velocity=group)
    if case == "fp32-alias":
        options["precision"] = "float32"
    if variant == "high":
        options.update(backend="cpu", accuracy="high", precision="float64", c_tol_rel=1e-9)
        call_args["nmodes"] += 2
    if variant in ("base-predict", "new-predict"):
        options["frequency_strategy"] = "anchor_continuation"
    if variant == "disabled":
        options["continuity_check"] = False
    solver = modisp.Solver(**options)


def run(timing=False):
    t = time.perf_counter_ns()
    obj = solver(**call_args)
    elapsed = (time.perf_counter_ns()-t)*1e-6
    if timing:
        return elapsed
    data = {k:v for k,v in vars(obj).items() if isinstance(v,np.ndarray)}
    stats = solver.stats()
    return data, {k:int(v) for k,v in stats.items()
                  if isinstance(v,(np.integer,int)) and (k.startswith(("continu", "prediction"))
                  or k in ("group_evaluations","evaluations","periods_computed","chunks"))}, obj.frequency_strategy_used


def profiled_call():
    solver.profile = True
    solver(**call_args)
    return solver.stats()["timings_s"]


def worker(stack,args,case,group,variant):
    runtime = (args.root / ("base" if variant in ("high","base","base-predict") else "candidate")).resolve()
    return stack.enter_context(ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                initializer=start, initargs=(runtime, args.inputs, case,args.backend, group,variant,args.threads)))


def time_case(args,case,group,variants,make_worker=worker):
    rounds=[]
    for reverse in (False,True):
        with ExitStack() as stack:
            pool={k:make_worker(stack,args,case,group,k) for k in variants}
            times={v:[] for v in variants}
            for i in range(4 + args.pairs):
                order = variants[::-1] if (i + int(reverse)) % 2 else variants
                for v in order:
                    times[v].append(pool[v].submit(run, True).result())
            samples={}
            for v,all_times in times.items():
                data,stats,used = pool[v].submit(run).result()
                q1,median,q3 = np.quantile(all_times[4:], [.25,.5,.75])
                samples[v] = dict(first_ms=all_times[0],warmup_ms=all_times[1:4],
                                  samples_ms=all_times[4:],median_ms=float(median),
                                  iqr_ms=float(q3-q1), frequency_strategy_used=used, stats=stats)
                if args.diagnostic:
                    samples[v]["profiled_timings_s"] = pool[v].submit(profiled_call).result()
            rounds.append(samples)
    return rounds


def toggle_run(pairs, reverse):
    samples = {"default": [], "disabled": []}
    for i in range(4 + pairs):
        for check in ((True, False) if (i + reverse) % 2 else (False, True)):
            solver.continuity_check = check
            samples["default" if check else "disabled"].append(run(timing=True))
    summary = {}
    for key, seq in samples.items():
        q1, median, q3 = np.quantile(seq[4:], [.25, .5, .75])
        summary[key] = dict(first_ms=seq[0], warmup_ms=seq[1:4], samples_ms=seq[4:],
                            median_ms=float(median), iqr_ms=float(q3-q1))
    return summary


def toggle_case(args, case, group):
    rounds = []
    for reverse in (False, True):
        with ExitStack() as stack:
            pool = worker(stack, args, case, group, "default")
            rounds.append(pool.submit(toggle_run, args.pairs, reverse).result())
    return rounds


def metric(data,ref):
    m=compare(data,ref,budget_rtol=2e-5)
    return {k:m[k] for k in ("total_traces","judged_traces","unjudged_reference_traces",
            "ordinal_missing","ordinal_extra","ordinal_budget_wrong","set_missing","set_extra",
            "rank_shifts","isolated_missing","isolated_spurious","unjudged_prefix_cells")}


def bad_slots(data,ref):
    x=data["phase_velocity"]
    r=ref["phase_velocity"][:,:x.shape[1]]
    return ((np.isfinite(x)!=np.isfinite(r)) |
            (np.abs(x-r)>np.maximum(3e-6,2e-5*np.abs(r)))) & ((ref["trace_flags"] & 5)==0)[:,None,:]


def compare_groups(actual, reference, slot_errors):
    k = actual["phase_velocity"].shape[1]
    reference_phase = reference["phase_velocity"][:, :k]
    matched = (np.isfinite(reference_phase) & ~slot_errors
               & (((reference["trace_flags"] & 5) == 0)[:, None, :]))
    valid = actual["group_velocity_valid"]
    reference_valid = reference["group_velocity_valid"][:, :k]
    common = matched & valid & reference_valid
    new = actual["group_velocity"][common]
    old = reference["group_velocity"][:, :k][common]
    error = np.abs(new - old)
    scale = np.maximum(np.maximum(np.abs(new), np.abs(old)), np.abs(reference_phase[common]))
    return dict(common_aligned=int(common.sum()), invalid_to_valid=int((matched & ~reference_valid & valid).sum()),
                valid_to_invalid=int((matched & reference_valid & ~valid).sum()),
                above_budget=int((error > np.maximum(1e-6, .002 * scale)).sum()),
                max_aligned_abs_error_km_s=float(error.max()) if error.size else None)


def accuracy_case(args,case,variants):
    variants = ["high"] + variants
    full = {}
    rows = {}
    for v in variants:
        with ExitStack() as stack:
            full[v], stats, used = worker(stack,args,case,True,v).submit(run).result()
        if v == "high":
            continue
        current, base, high = full[v], full["base"], full["high"]
        errors, old_errors = bad_slots(current,high),bad_slots(base,high)
        unresolved=(current["trace_flags"] & 4)!=0
        old_unresolved=(base["trace_flags"] & 4)!=0
        raw_changes = [k for k,val in base.items() if k not in current or val.dtype!=current[k].dtype
                       or val.shape!=current[k].shape or val.tobytes()!=current[k].tobytes()]
        phase_equal = np.all(np.isclose(current["phase_velocity"],base["phase_velocity"],rtol=0,atol=0,equal_nan=True),axis=1)
        same = np.broadcast_to(phase_equal[:,None,:],current["group_velocity"].shape)
        group_bytes = current["group_velocity"].view(np.uint64)
        base_bytes = base["group_velocity"].view(np.uint64)
        cont_flags = current.get("continuity_flags")
        codes,counts = np.unique(cont_flags,return_counts=True) if cont_flags is not None else ([],[])
        oldbad, newbad = old_errors.any(axis=1), errors.any(axis=1)
        rows[v] = dict(metrics=metric(current, high), groups_vs_high=compare_groups(current,high,errors),
                       cells_recovered=int((old_errors&~errors).sum()),
                       cells_introduced=int((~old_errors&errors).sum()),
                       traces_corrected=int((oldbad&~newbad).sum()), traces_newly_wrong=int((~oldbad&newbad).sum()),
                       traces_still_wrong=int((oldbad&newbad).sum()), newly_unresolved=int((unresolved&~old_unresolved).sum()),
                       unresolved=int(unresolved.sum()), changed_fields=raw_changes,
                       same_phase_changed_group=int(np.count_nonzero(same&(group_bytes!=base_bytes))),
                       same_phase_changed_group_flags=int(np.count_nonzero(same&(current["group_velocity_flags"]!=base["group_velocity_flags"]))),
                       group_invalid=int((~current["group_velocity_valid"]).sum()), stats=stats,
                       continuity_flag_histogram={str(int(k)):int(n) for k,n in zip(codes,counts)},frequency_strategy_used=used)
        if case.startswith('m06-') and case != 'm06-endpoints':
            with np.load(DATA / "continuity_rayleigh.npz") as z:
                i=list(z["model_index"]).index(int(case[4:]))
                phase=current["phase_velocity"][0,:,3]
                rows[v]["cps_pair_recovered"]=[bool(np.any(np.abs(phase-r)<=max(3e-6,2e-5*abs(r)))) for r in z["reference_pair"][i]]
    return rows


def census_solve(runtime, inputs, backend, precision, threads):
    sys.path.insert(0, str(runtime))
    import modisp
    assert Path(modisp.__file__).parent.parent == runtime
    with np.load(inputs/"solid24.npz") as z:
        model = {k:z[k] for k in MODEL_FIELDS}
    periods = np.load(inputs/"periods64.npy")
    with modisp.Solver(backend=backend, precision=precision, threads=threads,
                       workspace_bytes=1024**3) as engine:
        out = engine(**model, periods=periods, nmodes=5)
        fields = {k:getattr(out,k) for k in ("phase_velocity","trace_flags","root_count")}
        if getattr(out,"continuity_flags",None) is not None:
            fields["continuity_flags"] = out.continuity_flags
        stats = {k:v for k,v in engine.stats().items() if k.startswith("continuity_")}
    return fields, stats


def census(args):
    """Check the entire original M06 batch, including its original GPU chunk/precision context."""
    with np.load(args.reference) as z:
        ref = {k:z[k] for k in ("phase_velocity","trace_flags","root_count")}
    rows = {}
    for precision in (("float64","float32") if args.backend=="cuda" else ("float64",)):
        before = None
        rows[precision] = {}
        for variant in ("base","candidate"):
            with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context("spawn")) as executor:
                data,stats = executor.submit(census_solve, (args.root/variant).resolve(),args.inputs,
                                             args.backend,precision,args.threads).result()
            assert data["phase_velocity"].shape == (2048,5,64)
            bad = bad_slots(data,ref)
            if before is None:
                before = bad
            indices = np.argwhere(bad.any(axis=1))
            reviewed = np.asarray(data.get("continuity_flags",np.zeros((2048,64),dtype=np.uint8)))
            repaired=(reviewed&2)!=0
            rank_bad=bad.any(axis=1)
            old_bad=before.any(axis=1)
            rows[precision][variant]=dict(metrics=metric(data,ref), stats=stats,
                wrong_interior=int(rank_bad[:,1:-1].sum()), wrong_edges=int(rank_bad[:,[0,-1]].sum()),
                recovered=int((old_bad&~rank_bad).sum()), newly_wrong=int((~old_bad&rank_bad).sum()),
                new_wrong_slots=int((~before&bad).sum()), old_wrong_slots_corrected=int((before&~bad).sum()),
                confirmed_without_repair=int(np.count_nonzero(((reviewed&1)!=0)&~repaired&((reviewed&4)==0))),
                repaired_previously_correct=int((repaired&~old_bad).sum()), remaining_wrong=indices.tolist(),
                unresolved=int(np.count_nonzero(data["trace_flags"]&4)))
    return rows


def micro_worker(runtime):
    """Alternate the check on a single warmed solver to isolate wrapper and screen cost."""
    sys.path.insert(0, str(runtime))
    import modisp
    assert Path(modisp.__file__).parent.parent == runtime
    result = {}
    model = ([1.,0.],[3.,6.],[1.5,3.5],[2.,2.7])
    for n in (1,2,8):
        periods = np.geomspace(.2,.35,n)
        rounds = []
        for reverse in (False,True):
            samples = {"disabled":[], "default":[]}
            signatures = {}
            with modisp.Solver(backend="cpu",threads=1) as engine:
                for i in range(65):
                    order = (True,False) if (i+int(reverse))%2 else (False,True)
                    for check in order:
                        engine.continuity_check = check
                        start = time.perf_counter_ns()
                        out = engine(*model,periods,nmodes=2,wave="love")
                        us = (time.perf_counter_ns()-start)*1e-3
                        key = "default" if check else "disabled"
                        samples[key].append(us)
                        if i == 64:
                            signatures[key] = out.phase_velocity.tobytes()
            assert signatures["disabled"] == signatures["default"]
            rounds.append({k:dict(first_us=v[0],warmup_us=v[1:4],samples_us=v[4:],
                                  median_us=float(np.median(v[4:])),
                                  iqr_us=float(np.subtract(*np.percentile(v[4:],[75,25]))))
                           for k,v in samples.items()})
        result[str(n)] = rounds
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True,help="directory containing base/ and candidate/ installations")
    p.add_argument("--inputs",type=Path,required=True,help="frozen backend-0.4.0 inputs directory")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--backend",choices=["cpu","cuda"],required=True)
    p.add_argument("--mode",choices=["accuracy","time","census","micro","toggle"],required=True)
    p.add_argument("--reference",type=Path,help="archived M06-R CPU FP64 high reference.npz for census")
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--pairs",type=int,default=7,help="steady samples per variant per order, at least seven")
    p.add_argument("--cases",nargs="+",choices=TIMING_CASES+CHECK_CASES+["large"])
    p.add_argument("--baseline-only",action="store_true")
    p.add_argument("--diagnostic", action="store_true",
                   help="time an additional candidate with the check disabled and profile each variant separately")
    p.add_argument("--check-overhead", action="store_true",
                   help="compare just old independent, new check disabled and new default")
    args=p.parse_args()
    if args.pairs < 7:
        p.error("at least seven steady paired samples are required")
    args.inputs=args.inputs.resolve()
    cases=args.cases or (TIMING_CASES if args.mode in ("time","toggle") else CHECK_CASES)
    if args.backend=="cpu":
        cases=[c for c in cases if c!="fp32-alias"]
    variants=["base"]+(["base-predict"] if args.backend=="cuda" else [])
    if not args.baseline_only:
        variants+=(["disabled"] if args.mode=="accuracy" or args.diagnostic else [])+["default","new-predict"]
    if args.check_overhead:
        if args.mode != "time" or args.baseline_only:
            p.error("--check-overhead needs --mode time and both installations")
        variants = ["base","disabled","default"]
    inputs={f:sha(args.inputs/f) for f in ("solid24.npz","wet24.npz","periods64.npy")}
    manifest=json.loads((Path(__file__).resolve().parents[1]/"docs/benchmarks/backend-0.4.0-inputs.json").read_text())["files"]
    assert all(manifest[k]==v for k,v in inputs.items()),"input hash mismatch"
    report=dict(mode=args.mode, backend=args.backend, version=platform.python_version(), numpy=np.__version__,
                affinity=sorted(os.sched_getaffinity(0)), threads=args.threads, input_sha256=inputs,
                source_sha256=sha(__file__), libraries={k:snapshot(args.root/k)
                  for k in (("base",) if args.baseline_only else ("base","candidate"))}, cases={})
    if args.backend=="cuda":report["gpu_start"]=gpu_state()
    if args.mode=="micro":
        if args.backend != "cpu" or args.baseline_only:
            p.error("micro needs the CPU candidate")
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context("spawn")) as ex:
            report["cases"] = ex.submit(micro_worker,(args.root/"candidate").resolve()).result()
        save(args.output,report)
        return
    if args.mode=="census":
        if args.reference is None or args.baseline_only:
            p.error("census requires --reference and both installations")
        report["reference_sha256"]=sha(args.reference)
        report["cases"]["M06-R"]=census(args)
        if args.backend=="cuda":report["gpu_end"]=gpu_state()
        save(args.output,report)
        return
    for case in cases:
        if args.mode=="accuracy":
            report["cases"][case]=accuracy_case(args,case,variants)
        elif args.mode=="toggle":
            report["cases"][case]={str(g):toggle_case(args,case,g) for g in (False,True)}
        else:
            report["cases"][case]={str(g):time_case(args,case,g,variants) for g in (False,True)}
        save(args.output, report)
        print(case,flush=True)
    if args.backend=="cuda":
        report["gpu_end"]=gpu_state()
        save(args.output,report)


if __name__=="__main__":
    main()
