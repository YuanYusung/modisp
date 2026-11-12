#!/Users/yusong/miniforge3/bin/python
"""Build public Markdown/JSON receipts from completed backend runs.

Raw worker outputs and monitors remain in the run directories. Public records
keep numerical counts and hashes, omitting other users' process identifiers,
private installation paths and the verbose nvidia-smi process inventory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from backend_metrics import compare, quantiles


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strip_paths(value):
    if isinstance(value,dict):
        return {k:strip_paths(v) for k,v in value.items()}
    if isinstance(value,list):
        return [strip_paths(v) for v in value]
    if isinstance(value,str):
        return re.sub(r"/(?:home|Users)/[^\s'\"\]]+", "<private-installation-path>", value)
    return value


def clean_monitor(value):
    return {k:v for k,v in value.items() if k!="pid_peak_bytes"}


def write_witnesses(label,directory,destination):
    """Save all discrepancy traces, not a hand-picked set of favorable roots."""
    matrix=read(directory.parent/"inputs/matrix.json")
    arrays={}
    manifest=dict(platform=label,selection="union of any path's strict ordinal error, finite-mask difference, normalized root-count difference, or numerical failure against C64 high",cases={})
    for case_id in dict.fromkeys(x["id"].rsplit("-",1)[0] for x in matrix["cases"]):
        spec=next(x for x in matrix["cases"] if x["id"].rsplit("-",1)[0]==case_id)
        ref_path=directory/"references"/case_id/"reference.npz"
        with np.load(ref_path,allow_pickle=False) as z:
            ref={k:z[k].copy() for k in z.files}
        results={}
        selected=np.zeros((spec["B"],spec["T"]),dtype=bool)
        for path in ("C64","G64","G32"):
            with np.load(directory/"timing"/case_id/path/"steady11.npz",allow_pickle=False) as z:
                result={k:z[k].copy() for k in z.files}
            results[path]=result
            x=result["phase_velocity"]
            y=ref["phase_velocity"][:,:spec["nmodes"]]
            common=np.isfinite(x)&np.isfinite(y)
            selected|=np.any(common&(np.abs(x-y)>np.maximum(3e-6,2e-6*np.abs(y))),axis=1)
            selected|=np.any(np.isfinite(x)!=np.isfinite(y),axis=1)
            selected|=np.any(np.isin(result["status"],[2,3]),axis=1)
            cap=np.where(ref["root_count"]<0,ref["root_count"],np.minimum(ref["root_count"],spec["nmodes"]+2))
            selected|=result["root_count"]!=cap
        index=np.argwhere(selected)
        if not len(index):
            continue
        b,t=index.T
        prefix=case_id.replace("-","_")
        arrays[prefix+"__model_index"]=b
        arrays[prefix+"__period_index"]=t
        arrays[prefix+"__period_seconds"]=results["C64"]["periods"][t]
        with np.load(directory.parent/"inputs"/(spec["input_family"]+".npz"),allow_pickle=False) as z:
            for name in ("thickness","vp","vs","rho"):
                arrays[prefix+"__"+name]=z[name][b]
        for path,result in list(results.items())+[("reference",ref)]:
            for name in ("phase_velocity","mode_mask","status","trace_flags","numerical_flags","search_level","root_count",
                         "search_bounds","approximate_search","mode_index_uncertain","close_root_mask","close_root_risk"):
                x=result[name]
                arrays[prefix+"__"+path+"__"+name]=(x[b] if name=="search_bounds" else
                    x.transpose(0,2,1)[b,t] if x.ndim==3 else x[b,t])
        manifest["cases"][case_id]=dict(trace_count=len(index),geometry={k:spec[k] for k in ("B","L","T","nmodes","wave","accuracy")})
    manifest["note"]="Snapshots retain original full-batch context. Singleton replays can change the per-chunk FP64 repair quota. Counts in benchmark tables use full outputs, not this selected archive."
    arrays["manifest_json"]=np.array(json.dumps(manifest,sort_keys=True))
    np.savez_compressed(destination,**arrays)
    return dict(file=destination.name,sha256=sha(destination),cases=manifest["cases"])


def one_field(text,name):
    m=re.search(r"^\s*"+re.escape(name)+r"\s*:\s*(.*?)\s*$",text,re.M)
    return m.group(1) if m else None


def cross_platform_snapshots(directories):
    if "a800" not in directories or "rtx5050" not in directories:
        return None
    rows=[]
    for p in sorted((directories["a800"]/"timing").glob("*/*/steady11.npz")):
        case,path=p.parent.parent.name,p.parent.name
        q=directories["rtx5050"]/"timing"/case/path/"steady11.npz"
        with np.load(p,allow_pickle=False) as z:
            a={k:z[k].copy() for k in z.files}
        with np.load(q,allow_pickle=False) as z:
            b={k:z[k].copy() for k in z.files}
        x=np.asarray(a["phase_velocity"],dtype=np.float64)
        y=np.asarray(b["phase_velocity"],dtype=np.float64)
        common=np.isfinite(x)&np.isfinite(y)
        error=np.abs(y[common]-x[common])
        row=dict(case=case,path=path,comparison="rtx5050 versus a800; neither is independent truth",
            common_finite_cells=int(common.sum()),strict_ordinal_differences=int(np.count_nonzero(error>np.maximum(3e-6,2e-6*np.abs(x[common])))),
            abs_error_km_s=quantiles(error),relative_error=quantiles(error/np.abs(x[common])))
        for name in ("mode_mask","status","root_count","search_level","trace_flags","numerical_flags"):
            row[name+"_mismatches"]=int(np.count_nonzero(a[name]!=b[name]))
        row["a800_unresolved_cells"]=int(np.count_nonzero(a["status"]==2))
        row["rtx5050_unresolved_cells"]=int(np.count_nonzero(b["status"]==2))
        rows.append(row)
    return dict(scope="same frozen inputs and preset/path, different CPU/Python/native builds/CUDA toolkit/GPU architecture; not a hardware-only causal experiment",rows=rows)


def make_receipt(label,directory,retry_directory=None):
    summary=read(directory/"summary.json")
    if summary["path_rows"]!=57:
        raise ValueError(f"{label}: expected 57 completed path rows, got {summary['path_rows']}")
    env=read(directory/"environment.json")
    config=read(directory/"configuration.json")
    rows=[]
    starts=[]
    ends=[]
    for stored_row in summary["rows"]:
        row=dict(stored_row)
        if not row["timing"]["completed"]:
            raise ValueError(f"incomplete row: {label}/{row['case']}/{row['path']}")
        case=row["case"]
        raw=read(directory/"timing"/case/"record.json")
        attempts=[]
        retry_file=(retry_directory/"timing"/case/"record.json") if retry_directory else None
        if retry_file and retry_file.exists():
            retry_config=read(retry_directory/"configuration.json")
            if retry_config!=config:
                raise ValueError("paired retry changed input, tool, GPU or thread configuration")
            retry_env=read(retry_directory/"environment.json")
            if any(retry_env[k]!=env[k] for k in ("native_sha256","packages","env")):
                raise ValueError("paired retry changed installed code or runtime settings")
            candidate=read(retry_file)
            if not raw["monitor"]["contamination_observed"]:
                raise ValueError("do not replace an unflagged timing group")
            for path in ("C64","G64","G32"):
                old=raw["paths"][path]["steady"][-1]
                new=candidate["paths"][path]["steady"][-1]
                if old["signature"]!=new["signature"]:
                    raise ValueError("paired retry changed numerical snapshots; recompute quality before substitution")
            accepted=not candidate["monitor"]["contamination_observed"]
            attempts.append(dict(role="original_flagged_attempt",monitor=clean_monitor(raw["monitor"]),
                timing=raw["paths"][row["path"]]["summary"],started_unix=raw["started_unix"],finished_unix=raw["finished_unix"]))
            attempts.append(dict(role="full_paired_retry",accepted=accepted,monitor=clean_monitor(candidate["monitor"]),
                timing=candidate["paths"][row["path"]]["summary"],started_unix=candidate["started_unix"],finished_unix=candidate["finished_unix"]))
            if accepted:
                raw=candidate
                chosen=raw["paths"][row["path"]]
                row["timing"]=chosen["summary"]
                row["monitor"]=raw["monitor"]
                row["first_solve_s"]=chosen["first"]["seconds"]
                row["stats"]=chosen["steady"][-1]["stats"]
                t=row["timing"]["median_s"]
                row["models_per_s"]=row["geometry"]["B"]/t
                row["traces_per_s"]=row["geometry"]["B"]*row["geometry"]["T"]/t
                counts=chosen["summary"]["valid_root_counts"]
                row["valid_roots_per_s"]=(counts[0]/t if len(set(counts))==1 else
                    float(np.median([s["status_counts"]["0"]/s["seconds"] for s in chosen["steady"]])))
        starts.append(raw["started_unix"])
        ends.append(raw["finished_unix"])
        result=dict(row)
        worker=raw["paths"][row["path"]]
        result["setup_seconds"]=dict(modisp_import=worker["worker"]["import_s"],
            solver_constructor=worker["worker"]["constructor_s"],first_solve=worker["first"].get("seconds"))
        result["call_counts"]=dict(first=1,warmup=len(worker["warmup"]),steady=len(worker["steady"]),
            attempted=1+len(worker["warmup"])+len(worker["steady"]),failed=row["timing"]["failed_calls"])
        if attempts:
            result["timing_attempts"]=attempts
            result["paired_retry_accepted"]=attempts[-1]["accepted"]
        result["monitor"]=clean_monitor(result["monitor"])
        memory=result.get("memory")
        if memory:
            result["memory"]=dict(sampled_pid_peak_bytes=memory["sampled_pid_peak_bytes"],
                label=memory["label"],monitor=clean_monitor(memory["monitor"]),
                stats=memory["last"].get("stats"),errors=memory["errors"])
        result["snapshot_sha256"]={tag:raw["paths"][row["path"]][key].get("output_sha256")
            for tag,key in (("first","first"),) }
        result["snapshot_sha256"].update({f"steady{r['round']:02}":r["output_sha256"]
            for r in raw["paths"][row["path"]]["steady"] if "output_sha256" in r})
        with np.load(directory/"timing"/case/row["path"]/"steady11.npz",allow_pickle=False) as z:
            v=z["phase_velocity"]
            geometry=row["geometry"]
            if v.shape!=(geometry["B"],geometry["nmodes"],geometry["T"]):
                raise ValueError("snapshot geometry mismatch")
            expected_dtype=np.float32 if row["path"]=="G32" else np.float64
            if v.dtype!=np.dtype(expected_dtype):
                raise ValueError("precision/output dtype mismatch")
            frozen=np.load(directory.parent/"inputs"/("periods1.npy" if geometry["T"]==1 else "periods64.npy"))
            if not np.array_equal(z["periods"],frozen):
                raise ValueError("snapshot period grid changed")
            finite=np.isfinite(v)
            pairs=finite[:,1:] & finite[:,:-1]
            gaps=np.diff(v,axis=1)
            invariants=dict(mask_finite_mismatch=int(np.count_nonzero(z["mode_mask"]!=finite)),
                valid_finite_mismatch=int(np.count_nonzero((z["status"]==0)!=finite)),
                infinite_velocity_cells=int(np.isinf(v).sum()),
                duplicate_adjacent_roots=int(np.count_nonzero(pairs & (gaps==0))),
                descending_adjacent_roots=int(np.count_nonzero(pairs & (gaps<0))),
                invalid_root_count_traces=int(np.count_nonzero((z["root_count"] < -1)|(z["root_count"]>row["geometry"]["nmodes"]+2))))
            result["output_invariants"]=invariants
            if int(finite.sum())!=row["status_counts"]["0"]:
                raise ValueError("snapshot/status-count mismatch")
            with np.load(directory/"references"/case/"reference.npz",allow_pickle=False) as rz:
                reference=np.asarray(rz["phase_velocity"][:,:geometry["nmodes"]],dtype=np.float64)
                actual=np.asarray(v,dtype=np.float64)
                judged=(rz["trace_flags"]&(1|4))==0
                judged&=~np.any(np.isin(rz["status"],[2,3]),axis=1)
                common=finite&np.isfinite(reference)&judged[:,None,:]
                e=np.abs(actual[common]-reference[common])
                count=int(np.count_nonzero(e>np.maximum(3e-6,2e-6*np.abs(reference[common]))))
                if count!=row["against_cpu_high"]["ordinal_wrong"]:
                    raise ValueError("raw-array strict wrong count does not match receipt")
                if int(common.sum())!=row["against_cpu_high"]["finite_common_cells"]:
                    raise ValueError("finite-common denominator mismatch")
                if e.size and not np.isclose(np.percentile(e,99),row["against_cpu_high"]["abs_error_km_s"]["p99"],rtol=1e-12,atol=1e-15):
                    raise ValueError("raw-array P99 error does not match receipt")
        # Independent recomputation of every reported time and rate from raw
        # samples. Do not accept agreement between two formatted tables alone.
        times=np.array([s["seconds"] for s in raw["paths"][row["path"]]["steady"]])
        if len(times)!=12 or not np.isclose(np.median(times),row["timing"]["median_s"],rtol=0,atol=1e-15):
            raise ValueError("raw sample/median mismatch")
        if not np.isclose(row["models_per_s"],row["geometry"]["B"]/np.median(times),rtol=1e-12):
            raise ValueError("model throughput mismatch")
        if not np.isclose(row["traces_per_s"],row["geometry"]["B"]*row["geometry"]["T"]/np.median(times),rtol=1e-12):
            raise ValueError("trace throughput mismatch")
        quality=row["against_cpu_high"]
        new_fail=quality.get("newly_numerical_failed_cells",quality.get("newly_unresolved_traces",0))
        strict=not any(quality[k] for k in ("ordinal_wrong","ordinal_missing","set_missing","set_extra"))
        strict &= new_fail==0 and row["timing"]["repeated_changes"]==0
        strict &= not any(invariants.values())
        budget=not any(quality[k] for k in ("isolated_missing","isolated_spurious","unflagged_uncertainty"))
        budget &= new_fail==0 and row["timing"]["repeated_changes"]==0
        budget &= not any(invariants.values())
        result["strict_judged_prefix_qualified"]=bool(strict)
        result["isolated_root_budget_qualified"]=bool(budget)
        rows.append(strip_paths(result))
    gpu_text=env["gpu"].get("stdout","")
    cpu_text=env["lscpu"].get("stdout","")
    hardware=dict(cpu_model=one_field(cpu_text,"Model name"),threads=config["threads"],
        affinity=config["affinity"],logical_cpu_count=one_field(cpu_text,"CPU(s)"),
        numa_nodes=one_field(cpu_text,"NUMA node(s)"),
        gpu_product=one_field(gpu_text,"Product Name"),driver=one_field(gpu_text,"Driver Version"),
        build_toolkit_available_on_host=strip_paths(env["nvcc"]),
        python=env["python"],platform=env["platform"],packages=env["packages"],
        native_sha256=env["native_sha256"],cuda_visible_device_sha256=hashlib.sha256(config["gpu_uuid"].encode()).hexdigest(),
        initial_load_average=env["load_average"],environment_variables=env["env"])
    hardware["environment_variables"]["CUDA_VISIBLE_DEVICES"]="UUID fingerprint recorded above"
    wheel_receipt=directory/"wheel_sha256.json"
    if wheel_receipt.exists():
        hardware["installed_wheel_sha256"]=read(wheel_receipt)
        published=read(Path(__file__).resolve().parents[1]/"docs/benchmarks/backend-0.4.0-release-wheels.json")
        digests={a["name"]:a["sha256"] for a in published["assets"]}
        if any(digests.get(name)!=value for name,value in hardware["installed_wheel_sha256"].items()):
            raise ValueError("installed wheel does not match the published release digest")
        hardware["wheel_hashes_verified_against_release_asset_digests"]=True
    chunk={}
    p=directory/"chunk_check/S03-R/chunk8.npz"
    if p.exists():
        with np.load(p,allow_pickle=False) as z:
            a={k:z[k].copy() for k in z.files}
        with np.load(directory/"timing/S03-R/G32/steady11.npz",allow_pickle=False) as z:
            b={k:z[k].copy() for k in z.files}
        chunk=dict(metrics=compare(a,b),chunk8=read(directory/"chunk_check/S03-R/record.json"),
                   chunk8_output_sha256=sha(p))
    profiles={}
    for p in sorted((directory/"profiling").glob("*/*/record.json")):
        profile=read(p)
        timings=[c["stats"].get("timings_s",{}) for c in profile["calls"] if "stats" in c]
        keys=set().union(*(t.keys() for t in timings))
        profiles[f"{p.parent.parent.name}/{p.parent.name}"]=dict(
            calls=len(timings),stage_median_s={k:float(np.median([t[k] for t in timings if k in t])) for k in sorted(keys)},
            raw_calls=profile["calls"])
    receipt=dict(schema_version=1,label=label,source_commit="b27734f7198ea469663e1c0dc809cdd25c19c6ef",
        measurement_tool_commit="6259e53cbf0edd681646c789d34878773575a6e6",
        measurement_started_utc=datetime.fromtimestamp(min(starts),timezone.utc).isoformat(),
        measurement_finished_utc=datetime.fromtimestamp(max(ends),timezone.utc).isoformat(),
        measurement_timezone="Asia/Shanghai",hardware=hardware,
        input_manifest_sha256=config["input_manifest_sha256"],
        tool_sha256=env["tool_sha256"],metrics_sha256=env["metrics_sha256"],
        report_tool_sha256=sha(__file__),
        frozen_settings=dict(search_strategy="shape_adaptive",refinement="brent",love_strategy="phase",
            cmin=None,cmax=None,workspace_bytes=1024**3,chunk_size=None,
            first_calls=1,warmup_calls=3,steady_calls=12,quantile_method="linear"),
        path_rows=57,workload_wave_rows=19,rows=rows,
        independent=read(directory/"independent/record.json"),
        chunk_check=chunk,profiling=profiles,
        monitor_clarification=read(directory/"monitor_clarification.json") if (directory/"monitor_clarification.json").exists() else None,
        wet_chunk_followup=read(directory/"wet_chunk_followup/record.json") if (directory/"wet_chunk_followup/record.json").exists() else None,
        limitations=["C64 high is same-implementation regression, not complete physical truth",
            "root_count caps at K+2; prefix and unstable reference domains remain unjudged",
            "NVML 10 ms requested sampling gives an observed PID peak, not exact instantaneous allocation peak",
            "CPU uses Dunkin while CUDA uses fast; backend comparison includes evaluator defaults",
            "G32 high may attempt bounded FP64 repair",
            "shared-load results are diagnostic records, not uncontended speed rankings"])
    return strip_paths(receipt)


def number(value,digits=3):
    return "N/A" if value is None else f"{value:.{digits}g}"


def ms(row):
    t=row["timing"]
    return f"{t['median_s']*1000:.3f} [{t['q25_s']*1000:.3f},{t['q75_s']*1000:.3f}]"


def raw_ratio(a,b):
    return f"{a['timing']['median_s']/b['timing']['median_s']:.2f}"


def readme_section(receipts):
    if set(receipts)!={"a800","rtx5050"}:
        raise ValueError("README summary requires both completed platform runs")
    lines=["<!-- backend-benchmark-0.4.0:start -->",
        "## Same-input backend comparison (0.4.0)","",
        "Measured on **2026-11-12** with matching released 0.4.0 / ABI 8 wheels: "
        "57 configurations per host, covering Rayleigh/Love, B=1/64/2048, T=1/64, "
        "K=1/5, L=24/70, balanced/high and surface water. "
        "The same frozen inputs were copied between hosts. CPU uses Dunkin/FP64; CUDA uses fast/FP64 or FP32.","",
        "Selected rows below request five modes. Times are **complete-call steady medians in ms**, "
        "after one separate first call and three warmups, from 12 interleaved calls per path. "
        "All rows use shape_adaptive, Brent, Love phase, automatic bounds and a 1 GiB workspace budget. "
        "The full report includes IQRs, all raw samples, throughput, sampled PID VRAM peaks and numerical accounting.","",
        "**† marks rows with a GPU activity flag. A800 is a shared-load diagnostic run:** foreign GPU activity was observed in every workload group, "
        "with heavy host CPU load. Its ratios are not uncontended speed rankings. "
        "RTX 5050 uses an 8-physical-core-thread CPU baseline, versus 32 on A800. "
        "Its initially flagged M06-L group was repeated in full; the receipt retains both attempts and records whether the retry was accepted.","",
        "| Host | Case; B × L × T | Preset | CPU FP64 ms | CUDA FP64 ms | CUDA FP32 ms | FP64 / FP32 time ratio |",
        "|---|---|---|---:|---:|---:|---:|"]
    by_host={}
    selected=("M01-R","M05-R","M06-R","M06-L","S03-R","W01-WR")
    for label in ("a800","rtx5050"):
        by_case={}
        for row in receipts[label]["rows"]:
            by_case.setdefault(row["case"],{})[row["path"]]=row
        by_host[label]=by_case
        if label=="a800" and not all(row["monitor"]["contamination_observed"] for row in receipts[label]["rows"]):
            raise ValueError("revise A800 contention wording for changed monitoring data")
        for case,wrong,count in (("M06-R",104,16),("M06-L",13,1)):
            row=by_case[case]["G32"]
            if row["paired"]["ordinal_wrong"]!=wrong or row["paired"]["root_count_mismatches"]!=count:
                raise ValueError("revise README paired numerical claims")
            if row["isolated_root_budget_qualified"]:
                raise ValueError("revise README large-batch qualification claim")
        dry=by_case["S03-R"]["G32"]
        if not dry["strict_judged_prefix_qualified"] or dry["flag_counts"]["32768"]!=20:
            raise ValueError("revise README high-prefix/repair claim")
        for path in ("C64","G64","G32"):
            independent=[x for x in receipts[label]["independent"] if x["accuracy"]=="high" and x["path"]==path and x["case"].startswith(("witness_","B10000_"))]
            if len(independent)!=11 or sum(x["metrics"]["reference_valid_cells"] for x in independent)!=49:
                raise ValueError("independent-reference coverage changed")
            if any(any(x["metrics"][k] for k in ("ordinal_wrong","set_missing","set_extra")) for x in independent):
                raise ValueError("independent-reference qualification changed")
        for case in selected:
            group=by_case[case]
            a,b,c=(group[p] for p in ("C64","G64","G32"))
            g=a["geometry"]
            host="A800†" if label=="a800" else "RTX 5050"+("†" if a["monitor"]["contamination_observed"] else "")
            times=[f"{x['timing']['median_s']*1000:.3f}" for x in (a,b,c)]
            lines.append(f"| {host} | {case}; {g['B']} × {g['L']} × {g['T']} | {g['accuracy']} | {' | '.join(times)} | {raw_ratio(b,c)} |")
    lines += ["","R=dry Rayleigh, L=Love, WR=wet Rayleigh. Ratios are raw wall-time ratios; "
        "they do not imply equivalent root completeness. Single-model latency can favor CPU, and FP32 does not win on every workload.","",
        "**Read speed with root and failure accounting.** On both platforms, M06-R G32 differed from same-preset G64 in "
        "104 strict ordinal cells and 16 capped-count traces; M06-L had 13 strict ordinal differences and one capped-count difference. "
        "Balanced FP64 itself has close-cluster omissions/rank shifts against high. "
        "Neither large-batch G32 row passes the isolated-root-budget gate against the CPU-high regression witness. "
        "The full tables retain the large ordinal errors rather than hiding them behind small common-root percentiles."]
    wet=[]
    for label in ("a800","rtx5050"):
        row=by_host[label]["W01-WR"]["G32"]
        cells=row["status_counts"]["2"]
        wet.append(f"{label}: {cells//5} traces / {cells} cells")
    lines += ["","Wet-high G32 retained explicit UNRESOLVED results ("+"; ".join(wet)+
        "). In S03-R dry/high, G32 passed the judged-prefix check on both hosts, with 20 attempted FP64 repairs each. "
        "The 11 difficult independent Rayleigh fixtures recovered all 49 requested reference roots "
        "in high on all three paths on both hosts; this does not extend that validation to every benchmark input.","",
        "See the [full 0.4.0 report](docs/BACKEND_BENCHMARK_0_4_0.md) for all 114 rows, "
        "per-bit diagnostics, one-to-one matching, explicit unjudged domains, cross-platform comparisons and discrepancy snapshots. "
        "The [frozen matrix](docs/benchmarks/backend-0.4.0-matrix.json) and "
        "[driver](tools/benchmark_backends.py) reproduce the experiment. "
        "These are backend/precision comparisons within 0.4.0, not version-to-version speedups.",
        "<!-- backend-benchmark-0.4.0:end -->",""]
    return "\n".join(lines)


def report(receipts):
    out=["# modisp 0.4.0: A800 and RTX 5050 backend benchmark", "",
         "Measured on 2026-11-12 (Asia/Shanghai), using the released 0.4.0 CPU/CUDA wheels and source commit `b27734f`; measurement tools are frozen at `6259e53`. "
         "Both platforms used the same frozen input files, not merely the same random seed. Each completed 57 path configurations / 19 workload-wave rows.", "",
         "All four installed wheel hashes match the [published release asset digests](benchmarks/backend-0.4.0-release-wheels.json).", "",
         "The [prepared-input manifest](benchmarks/backend-0.4.0-inputs.json) records every frozen file hash, including the normalized matrix JSON written by prepare.", "",
         "**The A800 host and selected GPU had competing workloads. Its times are retained as shared-load diagnostic observations; "
         "they are not an uncontended CPU/GPU or cross-platform ranking. RTX 5050 results retain their own monitoring record and an 8-thread CPU baseline.**", "",
         "## Protocol", "",
         "The matrix is [machine readable](benchmarks/backend-0.4.0-matrix.json). B=1/64/2048 and T=1/64 form the core 24-layer, "
         "five-mode balanced grid for dry Rayleigh and Love. Additional rows isolate K=1, L=70 and high accuracy at B=64/T=64, plus wet Rayleigh/high. "
         "L includes the half-space. Modes are ascending recovered roots, not tracked physical branches.", "",
         "Inputs: 2,048 distinct 24-layer models (seed 20261112), 64 distinct 70-layer models (20261113), h=.04–.25 km, LVZ probability .5, "
         "Vs=.6–3.6 km/s and the fully specified generator options in the matrix. Smaller B values are master-batch prefixes. "
         "Wet replaces the first layer with (.08,1.5,0,1.02). T=1 is [3] s; T=64 is geometric .15–10 s. Both frozen arrays and hashes are retained.", "",
         "Settings: shape_adaptive, Brent, Love phase, automatic velocity bounds, close_root_rtol=.005, f_accept=1, 1 GiB workspace budget, "
         "no explicit chunk cap. CPU uses Dunkin/FP64; CUDA uses fast with FP64 or FP32. Thus CPU/GPU comparison includes evaluator defaults. "
         "Inputs remain contiguous host FP64 for all paths. G32 high includes any FP64 repair attempts.", "",
         "One first call and three warmups precede 12 calls per path, using all six C64/G64/G32 permutations twice. "
         "Values below are complete synchronous Solver-call wall times: automatic bounds, input conversion, transfers, search/repair, native chunks and output packaging. "
         "File IO, hashing, continuity checks and profile=True are excluded. The 12 raw samples are in the receipts. "
         "Cells show median [Q25,Q75] ms, with linear quantiles. Ratios divide the medians in the same row.", "",
         "## Hardware and provenance", "",
         "| Platform | CPU / threads | GPU | Python / NumPy / SciPy | Driver | Timing receipt |",
         "|---|---|---|---|---|---|"]
    for label,r in receipts.items():
        h=r["hardware"]
        p=h["packages"]
        out.append(f"| {label} | {h['cpu_model']} / {h['threads']} | {h['gpu_product']} | {h['python']} / {p['numpy']} / {p['scipy']} | {h['driver']} | [JSON](benchmarks/backend-0.4.0-20261112-{label}.json) |")
    out += ["","Compact discrepancy snapshots are available for " + ", ".join(
        f"[{label}](benchmarks/{r['discrepancy_witnesses']['file']})" for label,r in receipts.items()) +
        ". They contain the actual selected model/period inputs, three-path outputs, high-FP64 reference prefixes, status and trace diagnostics. "
        "Selection is for inspection; all benchmark counts use complete outputs. Singleton replays may alter FP64 repair quotas, so reproduce the full matrix for the original batch context."]
    for label,r in receipts.items():
        by_case={}
        for row in r["rows"]:
            by_case.setdefault(row["case"],{})[row["path"]]=row
        contaminated=sum(group["C64"]["monitor"]["contamination_observed"] for group in by_case.values())
        loads=[x for row in r["rows"] for x in row["monitor"]["load_average_1min_range"]]
        out += ["",f"## {label}: timing", "",
            f"Foreign GPU activity was observed in {contaminated}/19 workload groups. Host one-minute load ranged from {min(loads):.2f} to {max(loads):.2f}. "
            + ("Rows marked † are diagnostic observations, not accepted uncontended speedups; unmarked rows retain their own monitoring record." if contaminated else
               "No competing compute activity was observed on the selected GPU; resident desktop/graphics memory remains included in device-wide monitoring. This is not proof of a wholly isolated host."),"",
            "| Case | B × L × T; K / accuracy | CPU FP64 ms | CUDA FP64 ms | CUDA FP32 ms | GPU64 / CPU × | GPU32 / CPU × | FP32 / FP64 × |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
        for case,group in by_case.items():
            a,b,c=(group[p] for p in ("C64","G64","G32"))
            g=a["geometry"]
            display_case=case+("†" if a["monitor"]["contamination_observed"] else "")
            out.append(f"| {display_case} | {g['B']} × {g['L']} × {g['T']}; {g['nmodes']} / {g['accuracy']} | {ms(a)} | {ms(b)} | {ms(c)} | {raw_ratio(a,b)} | {raw_ratio(a,c)} | {raw_ratio(b,c)} |")
        out += ["","R=dry Rayleigh, L=Love, WR=wet Rayleigh. M01–M06 are the B/T grid, S01 is K=1, S02 is L=70 and S03 is high. "
            "Raw timing ratios do not imply equivalent root completeness. The numerical tables below retain mismatches and failed qualification.","",
            f"### {label}: throughput and sampled memory", "",
            "Rates use the steady median; traces/s=B*T/t and do not multiply by K. PID memory is measured separately with one CUDA worker, "
            "requested 10 ms NVML sampling. It includes process context overhead and is a sampled lower bound on instantaneous peak. "
            "Actual sampling intervals and native chunk capacities are in the receipt; native device bytes and workspace budgets are not total VRAM peaks.","",
            "| Case / path | models/s | traces/s | VALID roots/call | valid roots/s | PID sampled peak MiB | Native device MiB | Chunks / capacity |",
            "|---|---:|---:|---:|---:|---:|---:|---|"]
        retried=sorted({row["case"] for row in r["rows"] if row.get("timing_attempts")})
        if retried:
            # Insert the note above this section's throughput table, keeping
            # the original and replacement sample sets visible in the receipt.
            note="Full paired retries were made for "+", ".join(retried)+" after monitoring flags. "
            note+="Each replacement requires identical numerical snapshot signatures on all three paths and an unflagged retry; samples are not pooled. Both attempts remain in JSON."
            if not any(row.get("paired_retry_accepted") for row in r["rows"] if row.get("timing_attempts")):
                note+=" The repeat also had an activity flag, so no replacement was accepted and the original flagged medians remain in the table. No competing compute process was resident in either group; the activity source was not retained, so the flag is not reclassified."
            out[-2:-2]=[note,""]
        for row in r["rows"]:
            st=row["stats"]
            peak=row.get("memory",{}).get("sampled_pid_peak_bytes")
            native=st.get("device_workspace_bytes")
            out.append(f"| {row['case']} / {row['path']} | {row['models_per_s']:.2f} | {row['traces_per_s']:.2f} | {row['status_counts']['0']} | {row['valid_roots_per_s']:.2f} | {number(None if peak is None else peak/2**20,4)} | {number(None if native is None else native/2**20,4)} | {st['chunks']} / {st['capacity_models']} |")
        out += ["",f"### {label}: numerical results against CPU FP64 high", "",
            "CPU high/K=7 is a same-implementation regression witness, not independent physical truth. "
            "Strict wrong means abs error > max(3e-6,2e-6*abs(reference)) km/s on common finite ordinal slots. "
            "Budget wrong uses the same threshold for high and max(3e-6,2e-5*abs(reference)) for balanced. "
            "Missing/extra use one-to-one matches within the judged prefix, at that preset's budget. "
            "Rank shifts and unmatched close roots remain separate; matching coverage is retained in JSON. All counts below are per case, not independent totals across repeated B prefixes.","",
            "| Case / path | VALID / NOT_FOUND / UNRESOLVED / REJECTED | Strict / budget ordinal wrong | Set missing / extra | Rank shifts / cluster unmatched | Abs P99 / max km/s | Unjudged traces / prefix cells | Strict judged prefix / isolated budget |",
            "|---|---|---|---|---|---|---|---|"]
        for row in r["rows"]:
            q=row["against_cpu_high"]
            s=row["status_counts"]
            error=q["abs_error_km_s"]
            out.append(f"| {row['case']} / {row['path']} | {' / '.join(str(s[str(i)]) for i in range(4))} | {q['ordinal_wrong']} / {q['ordinal_budget_wrong']} | {q['set_missing']} / {q['set_extra']} | {q['rank_shifts']} / {q['cluster_unmatched']} | {number(error['p99'])} / {number(error['max'])} | {q['unjudged_reference_traces']} / {q['unjudged_prefix_cells']} | {'pass' if row['strict_judged_prefix_qualified'] else 'fail'} / {'pass' if row['isolated_root_budget_qualified'] else 'fail'} |")
        out += ["","NOT_FOUND is not automatically a numerical failure or evidence of physical absence. Numerical failed cells are UNRESOLVED+REJECTED; "
            "their denominator is B*K*T. Failed traces have any such cell and use B*T; exceptions/OOM are separate failed-call counts. "
            "A budget pass remains approximate if cluster/rank differences exist. Unjudged domains are not counted as passes.","",
            f"### {label}: paired diagnostics", "",
            "Each G64 row compares with C64 at the same preset; G32 compares with G64. Mismatch tuple is mask cells / status cells / capped-count traces / "
            "search-level traces / full-flag traces. Root counts cap at K+2 and include guards; equal saturated counts do not certify completeness. "
            "Flags are cumulative and precision-dependent. A successful fallback or attempted repair is not by itself a numerical failure.","",
            "| Case / path | Diagnostic mismatch tuple | Repair / shape fallback / Love phase fallback traces | Approximate / close-risk traces | Repeated output changes | Failed calls |",
            "|---|---|---|---|---:|---:|"]
        for row in r["rows"]:
            if row["path"]=="C64":
                continue
            q=row["paired"]
            flags=row["flag_counts"]
            mismatch=" / ".join(str(q.get(k+"_mismatches","N/A")) for k in ("mode_mask","status","root_count","search_level","trace_flags"))
            out.append(f"| {row['case']} / {row['path']} | {mismatch} | {flags['32768']} / {flags['8192']} / {flags['16384']} | {flags['2048']} / {flags['4096']} | {row['timing']['repeated_changes']} | {row['timing']['failed_calls']} |")
        out += ["","Per-bit XOR, numerical_flags differences, status transitions via preserved arrays, reliability assessed/pass counts, first-call costs "
            "and 12 raw samples remain in the receipt/raw results. CPU/GPU bitwise identity is not an API guarantee.","",
            f"### {label}: velocity differences at the same preset", "",
            "These isolate path/precision changes from the CPU-high preset comparison. Ordinal errors include rank shifts; "
            "no rematching is used to reduce their reported maxima. The matching counts here use the strict tolerance, even for balanced.","",
            "| Case / candidate → reference | Common / reference-valid cells | Strict ordinal wrong | Ordinal missing / extra | One-to-one missing / extra | Rank shifts | Abs P99 / max km/s | Relative P99 / max |",
            "|---|---|---:|---|---|---:|---|---|"]
        for row in r["rows"]:
            if "paired" not in row:
                continue
            q=row["paired"]
            a,b=q["abs_error_km_s"],q["relative_error"]
            out.append(f"| {row['case']} / {row['path']} → {row['paired_reference']} | {q['finite_common_cells']} / {q['reference_valid_cells']} | {q['ordinal_wrong']} | {q['ordinal_missing']} / {q['ordinal_extra']} | {q['set_missing']} / {q['set_extra']} | {q['rank_shifts']} | {number(a['p99'])} / {number(a['max'])} | {number(b['p99'])} / {number(b['max'])} |")
        out += ["",
            f"### {label}: independent checks", "",
            "The 11 difficult Rayleigh cases retain the original expected root prefixes: eight witness files (reference_hi first five) and three "
            "B10000 files (reference first three, cmin=.2). Analytic Love and the factored low-velocity Rayleigh case are separate checks. "
            "Exact Love cutoff is unjudged. Wet Love is a physical invariance check, not new independent full-spectrum truth.","",
            "| Check | Path / preset | Reference-valid / common cells | Strict wrong | Missing / extra | UNRESOLVED / REJECTED cells | Repair traces |",
            "|---|---|---|---:|---|---|---:|"]
        suites={"Rayleigh difficult (11)":lambda x:x.startswith(("witness_","B10000_")),
                "Analytic Love":lambda x:x.startswith("analytic_love"),
                "Low-velocity Rayleigh":lambda x:x=="analytic_low_velocity_rayleigh"}
        for suite,predicate in suites.items():
            for accuracy in ("high","balanced"):
                for path in ("C64","G64","G32"):
                    rr=[x for x in r["independent"] if predicate(x["case"]) and x["path"]==path and x["accuracy"]==accuracy]
                    if any("error" in x for x in rr):
                        out.append(f"| {suite} | {path} / {accuracy} | ERROR; see receipt | N/A | N/A | N/A | N/A |")
                        continue
                    sums={k:sum(x["metrics"][k] for x in rr) for k in ("reference_valid_cells","finite_common_cells","ordinal_wrong","set_missing","set_extra")}
                    failed=[sum(x["status_counts"][str(i)] for x in rr) for i in (2,3)]
                    repairs=sum(x.get("repair_traces",0) for x in rr)
                    out.append(f"| {suite} | {path} / {accuracy} | {sums['reference_valid_cells']} / {sums['finite_common_cells']} | {sums['ordinal_wrong']} | {sums['set_missing']} / {sums['set_extra']} | {failed[0]} / {failed[1]} | {repairs} |")
        chunk=r["chunk_check"]
        if chunk:
            m=chunk["metrics"]
            out += ["",f"The S03-R G32 chunk_size=8 recheck had {m['ordinal_wrong']} strict ordinal differences, "
                    f"{m['set_missing']}/{m['set_extra']} missing/extra and {m.get('trace_flags_mismatches',0)} flag differences "
                    "relative to the uncapped run. Its full stats and per-bit differences are retained. "
                    "The high FP32 repair quota is per native chunk; changing chunking can affect the repair path even when velocities agree."]
        followup=r.get("wet_chunk_followup")
        if followup:
            out += ["","An additional **untimed wet-high follow-up** retains the original W01 timing/failure row. " + "; ".join(
                f"chunk_size={x['chunk_size']}: {x['result']['stats'].get('fp64_repair_traces',0)} attempted FP64 repairs, "
                f"{x['result']['status_counts']['2']} UNRESOLVED cells, {x['against_cpu_high']['set_missing']} missing reference roots"
                for x in followup["rows"]) + ". These diagnostic calls do not replace the original benchmark row."]
    out += ["","## Reproduction", "",
        "Install the matching released 0.4.0 CPU/CUDA wheels for each host; the two runs used NumPy 2.2.6. "
        "Install scipy and nvidia-ml-py for the auxiliary validation and monitoring, using the versions recorded per platform. "
        "Invoke scripts with the selected environment's Python; the source shebang is the maintainer's local convention.","",
        "```sh\npython tools/benchmark_backends.py prepare --output /new/inputs\n# Copy the frozen input directory unchanged to each host.\nOMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\\nCUDA_VISIBLE_DEVICES=GPU-UUID python tools/benchmark_backends.py run \\\n  --input /new/inputs --output /new/run --gpu-uuid GPU-UUID \\\n  --threads 8 --affinity 0,1,2,3,4,5,6,7\n# A800: use 32 physical cores on the GPU-local NUMA node instead.\npython tools/report_backends.py --run a800=/path/to/a800/run \\\n  --run rtx5050=/path/to/rtx5050/run --output docs\n```", "",
        "A completed group is checkpointed. Reusing a run directory requires the same input manifest, tool hash, threads, affinity and GPU. "
        "A resumed run retains per-group dates and samples; it does not fabricate a single uninterrupted end-to-end pass. "
        "Memory runs and profile=True calls are separate from the main timing. Complete per-call NPZ outputs and monitor logs stay outside the source tree; "
        "the public receipts retain result hashes, scalar counters, all raw steady samples, numerical accounting and stage profiles.", "",
        "The matrix tests backends/precision within 0.4.0. It does not measure a version-to-version speedup or certify physical branch identity."]
    return "\n".join(out)+"\n"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",action="append",required=True,help="LABEL=/path/to/completed/run")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--retry",action="append",default=[],help="LABEL=/path/to/full-paired-retry (flagged groups only)")
    p.add_argument("--update-readme",action="store_true",help="update README.md next to the docs directory")
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"benchmarks").mkdir(exist_ok=True)
    receipts={}
    directories={}
    retries={label:Path(path) for label,path in (item.split("=",1) for item in args.retry)}
    for item in args.run:
        label,path=item.split("=",1)
        directories[label]=Path(path)
        receipt=make_receipt(label,Path(path),retries.get(label))
        receipt["discrepancy_witnesses"]=write_witnesses(label,Path(path),args.output/"benchmarks"/f"backend-0.4.0-witnesses-{label}.npz")
        receipts[label]=receipt
        write(args.output/"benchmarks"/f"backend-0.4.0-20261112-{label}.json",receipt)
    input_manifests=[directory.parent/"inputs/manifest.json" for directory in directories.values()]
    if len({sha(path) for path in input_manifests})!=1:
        raise ValueError("platform input manifests differ")
    (args.output/"benchmarks/backend-0.4.0-inputs.json").write_bytes(input_manifests[0].read_bytes())
    text=report(receipts)
    cross=cross_platform_snapshots(directories)
    if cross:
        write(args.output/"benchmarks/backend-0.4.0-20261112-cross-platform.json",cross)
        text += "\n## Cross-platform numerical consistency\n\nBoth hosts used byte-identical frozen inputs. This comparison changes GPU architecture, CPU, Python ABI and CUDA/native builds together; neither host is independent truth. "
        text += "All 57 comparisons are in the [cross-platform receipt](benchmarks/backend-0.4.0-20261112-cross-platform.json). Selected rows compare the same path and preset:\n\n"
        text += "| Case / path | Strict ordinal differences | Mask / status / root-count mismatches | UNRESOLVED cells A800 / RTX 5050 | Max common abs difference km/s |\n|---|---:|---|---|---:|\n"
        for row in cross["rows"]:
            if row["case"] in ("M06-R","M06-L","S03-R","W01-WR") and row["path"] in ("G64","G32"):
                text += f"| {row['case']} / {row['path']} | {row['strict_ordinal_differences']} | {row['mode_mask_mismatches']} / {row['status_mismatches']} / {row['root_count_mismatches']} | {row['a800_unresolved_cells']} / {row['rtx5050_unresolved_cells']} | {number(row['abs_error_km_s']['max'])} |\n"
    (args.output/"BACKEND_BENCHMARK_0_4_0.md").write_text(text)
    if args.update_readme:
        target=args.output.parent/"README.md"
        original=target.read_text()
        section=readme_section(receipts)
        pattern=r"<!-- backend-benchmark-0.4.0:start -->.*?<!-- backend-benchmark-0.4.0:end -->\n?"
        if re.search(pattern,original,re.S):
            updated=re.sub(pattern,lambda match:section,original,flags=re.S)
        else:
            heading="## Refinement, Love phase search and mode diagnostics"
            if heading not in original:
                raise ValueError("README insertion point missing")
            updated=original.replace(heading,section+"\n"+heading,1)
        target.write_text(updated)
    print(json.dumps(dict(platforms=list(receipts),rows=sum(r["path_rows"] for r in receipts.values()))))


if __name__=="__main__":
    main()
