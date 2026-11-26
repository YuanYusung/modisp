"""Same-host public Solver timings and independent root-order checks for FP32 and auto."""
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

import benchmark_continuity as bc
from backend_metrics import compare

PRIMARY = ("wide-dry", "wide-wet", "dense-dry", "dense-wet")
CONTROL = ("wide-love", "dense-love", "small-dry", "small-wet", "small-love", "tiny", "shallow", "thick70", "wet-high")
HELD = ("fresh-dry", "fresh-wet", "fresh-love", "fresh2-dry", "fresh2-wet", "fresh2-love")
ARCHIVE = ("m06-1453", "m06-1881", "m06-endpoints", "s02-70", "fp32-alias", "love-cutoff", "paper")
ALL = PRIMARY + CONTROL + HELD + ARCHIVE + ("m06",)
ROOT = Path(__file__).resolve().parents[1]


def inputs(md, name, directory):
    if name in PRIMARY or name in ("wide-love", "dense-love"):
        data = bc.case_input("large", directory)
        if name.startswith("wide"):
            data["periods"] = np.geomspace(.15, 10., 64)
        if name.endswith("love"):
            data["wave"] = "love"
    elif name.startswith("small-"):
        data = bc.case_input(name.replace("small-", "wide-"), directory)
    elif name in HELD:
        seed = 20261203 if name.startswith("fresh2") else 20261201
        data = md.random_model_batch(9, 19, seed=seed, h_min=.018, h_max=.38, lvz_prob=.64)
        data.update(periods=np.geomspace(.115, 5., 43), nmodes=5, wave="love" if name.endswith("love") else "rayleigh")
    elif name == "shallow":
        data = md.random_model_batch(1024, 5, seed=20261114, h_min=.5, h_max=6.)
        data.update(periods=np.geomspace(.1, 10., 16), nmodes=1, wave="rayleigh")
    elif name in ("m06", "thick70", "wet-high", "tiny"):
        family = {"m06": "solid24", "tiny": "solid24", "thick70": "solid70", "wet-high": "wet24"}[name]
        n = {"m06": 2048, "tiny": 1, "thick70": 64, "wet-high": 64}[name]
        with np.load(directory / (family + ".npz")) as z:
            data = {k: z[k][:n].copy() for k in bc.MODEL_FIELDS}
        periods = np.load(directory / "periods64.npy")
        data.update(periods=periods[:4] if name == "tiny" else periods, nmodes=5, wave="rayleigh")
    elif name == "paper":
        model = np.array([[.01, 1.55, .18, 1.78], [.01, 1.8, .35, 1.85],
                          [.02, 1., .12, 1.5], [0., 2., .6, 1.93]])
        data = dict(zip(bc.MODEL_FIELDS, model.T))
        data.update(periods=1. / np.linspace(60.2, 61., 17), nmodes=64, wave="rayleigh")
    else:
        data = bc.case_input(name, directory)
    if name.endswith("-wet") and (name in PRIMARY or name in HELD):
        for k, value in zip(bc.MODEL_FIELDS, (.08, 1.5, 0., 1.02)):
            data[k][:, 0] = value
    return data


def setup(runtime, args, name, tier):
    global solver, data, input_hash
    sys.path.insert(0, str(runtime))
    import modisp as md
    if Path(md.__file__).resolve().parent.parent != runtime:
        raise RuntimeError(f"Unexpected runtime {md.__file__}")
    data = inputs(md, name, args.inputs)
    input_hash = hashlib.sha256(b"".join(np.ascontiguousarray(data[k]).tobytes()
        for k in (*bc.MODEL_FIELDS, "periods"))).hexdigest()
    opts = dict(backend=args.backend, precision=args.precision, threads=args.threads,
                group_velocity=args.group, workspace_bytes=1024**3, frequency_strategy=args.strategy)
    if tier == "reference":
        opts.update(backend="cpu", precision="float64", accuracy="high", c_tol_rel=1e-9,
                    frequency_strategy="independent")
        data["nmodes"] += 2
    elif tier != "default":
        opts["accuracy"] = tier
    if args.unchecked:
        # Capture the raw search in an isolated accuracy worker. Auto can review
        # native failures even when continuity_check=False.
        from modisp import _continuity
        _continuity.review = lambda *unused: None
    solver = md.Solver(**opts)


def pool(stack, runtime, args, case, tier):
    return stack.enter_context(ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"),
        initializer=setup, initargs=(runtime.resolve(), args, case, tier)))


def solve():
    t = time.perf_counter_ns()
    solver(**data)
    return (time.perf_counter_ns() - t) / 1e6


def snapshot(profile=False):
    if profile:
        solver.profile = True
        solver(**data)
    stats = solver.stats()
    counters = {k: v for k, v in stats.items() if isinstance(v, (int, bool, str, float)) or v is None}
    return dict(input_sha256=input_hash, traces=int(solver.last_result.trace_flags.size), counters=counters,
                stages_s=stats.get("timings_s") if profile else None)


def paired(args, case):
    rounds = []
    for reverse in (False, True):
        with ExitStack() as stack:
            baseline = argparse.Namespace(**vars(args))
            baseline.precision = args.base_precision or args.precision
            workers = {
                'base': pool(stack, args.base, baseline, case, args.base_tier),
                'candidate': pool(stack, args.candidate, args, case, args.candidate_tier),
            }
            samples = {k: [] for k in workers}
            for i in range(4 + args.pairs):
                order = list(workers)
                if (i + reverse) % 2:
                    order.reverse()
                for k in order:
                    samples[k].append(workers[k].submit(solve).result())
            row = {}
            for k, v in samples.items():
                q1, median, q3 = np.quantile(v[4:], [.25, .5, .75])
                row[k] = dict(first_ms=v[0], warmup_ms=v[1:4], samples_ms=v[4:],
                              median_ms=float(median), iqr_ms=float(q3-q1),
                              **workers[k].submit(snapshot, args.profile).result())
            if row["base"]["input_sha256"] != row["candidate"]["input_sha256"]:
                raise RuntimeError("A/B input mismatch")
            ratios = np.array(samples["base"][4:]) / np.array(samples["candidate"][4:])
            row["paired_speedup"] = float(np.median(ratios))
            row["median_speedup"] = row["base"]["median_ms"] / row["candidate"]["median_ms"]
            rounds.append(row)
    return rounds


def capture(directory, name):
    solve()
    result = solver.last_result
    arrays = {k: v for k, v in vars(result).items() if isinstance(v, np.ndarray)}
    dest = directory / (name + ".npz")
    np.savez_compressed(dest, **arrays)
    return dict(**snapshot(), result_sha256=bc.sha(dest),
                phase_dtype=str(result.phase_velocity.dtype), shape=list(result.phase_velocity.shape))


def get_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def byte_check(args, case):
    old = get_arrays(args.base / (case + ".npz"))
    new = get_arrays(args.candidate / (case + ".npz"))
    signatures = {}
    changed = {}
    for name in sorted(set(old) | set(new)):
        before, after = old.get(name), new.get(name)
        signatures[name] = [None if a is None else dict(shape=list(a.shape), dtype=a.dtype.str,
            sha256=hashlib.sha256(a.tobytes()).hexdigest()) for a in (before, after)]
        if signatures[name][0] == signatures[name][1]:
            continue
        if before is None or after is None or before.shape != after.shape or before.dtype != after.dtype:
            changed[name] = {"incompatible": True}
            continue
        b = np.ascontiguousarray(before).view(np.uint8).reshape(before.shape + (before.itemsize,))
        a = np.ascontiguousarray(after).view(np.uint8).reshape(after.shape + (after.itemsize,))
        mask = (b != a).any(axis=-1)
        changed[name] = dict(cells=int(mask.sum()), first_indices=np.argwhere(mask)[:12].tolist())
    return dict(arrays=len(signatures), changed=changed, signatures=signatures)


def ordered_errors(values, reference, rtol):
    """Return missing reference ranks, spurious output ranks, and displaced (output, reference) ranks."""
    k = values.size
    actual = np.flatnonzero(np.isfinite(values))
    roots = reference[np.isfinite(reference)]
    i = j = 0
    matched = {}
    while i < roots.size and j < actual.size:
        rank, target = int(actual[j]), roots[i]
        if abs(values[rank] - target) <= max(3e-6, rtol*abs(target)):
            matched[rank] = i
            i += 1
            j += 1
        elif values[rank] < target:
            j += 1
        else:
            i += 1
    missing = set(range(min(k, roots.size))) - set(matched.values())
    extra = {int(n) for n in actual if n not in matched and
             (roots.size < reference.size or values[n] <= roots[-1])}
    shifted = {(a, b) for a, b in matched.items() if a != b}
    return missing, extra, shifted


def evaluate(args, case):
    base = get_arrays(args.base / (case + ".npz"))
    candidate = get_arrays(args.candidate / (case + ".npz"))
    ref = get_arrays(args.archived_m06 if case == "m06" else args.reference / (case + ".npz"))
    left = json.loads((args.base / "captures.json").read_text())["cases"][case]
    right = json.loads((args.candidate / "captures.json").read_text())["cases"][case]
    if left["input_sha256"] != right["input_sha256"]:
        raise RuntimeError(f"{case} input mismatch")
    if case != "m06":
        high = json.loads((args.reference / "captures.json").read_text())["cases"][case]
        if right["input_sha256"] != high["input_sha256"]:
            raise RuntimeError(f"{case} reference input mismatch")
    group_ref = get_arrays(args.m06_groups) if case == "m06" and args.m06_groups else ref
    unchecked = get_arrays(args.before_review / (case + ".npz")) if args.before_review else None
    k = candidate["phase_velocity"].shape[1]
    if ref["phase_velocity"].shape[1] < k+2:
        raise RuntimeError(f"{case} reference lacks guard roots")
    unchanged = base.keys() == candidate.keys() and all(base[n].dtype == candidate[n].dtype
        and base[n].shape == candidate[n].shape and base[n].tobytes() == candidate[n].tobytes() for n in base)
    row = dict(input_sha256=right["input_sha256"], all_fields_identical=unchanged,
               changed_fields=sorted(n for n in set(base) | set(candidate) if n not in base or n not in candidate
                   or base[n].dtype != candidate[n].dtype or base[n].shape != candidate[n].shape
                   or base[n].tobytes() != candidate[n].tobytes()),
               candidate_counters=right["counters"], base_counters=left["counters"])
    def unresolved(current):
        return ((current["trace_flags"] & 5) != 0) | ((current.get("continuity_flags", 0) & 4) != 0)

    hi = ref["phase_velocity"][:, :k]
    judged = ((ref["trace_flags"] & 5) == 0)[:, None, :]
    for rtol in (2e-5, 2e-6):
        wrong = {}
        vrows = {}
        for key, current in (("base", base), ("candidate", candidate)):
            x = current["phase_velocity"]
            wrong[key] = ((np.isfinite(x) != np.isfinite(hi)) | (np.abs(x-hi) > np.maximum(3e-6,rtol*np.abs(hi)))) & judged
            metrics = compare(current, ref, budget_rtol=rtol)
            bad = wrong[key].any(axis=1)
            vrows[key] = dict(**metrics, bad_traces=int(bad.sum()), bad_slots=int(wrong[key].sum()),
                              unresolved=int(np.count_nonzero(current["trace_flags"] & 5)),
                              bad_endpoints=int(np.count_nonzero(bad[:, [0,-1] if bad.shape[1] > 1 else [0]])))
        if "group_velocity_valid" in group_ref and "group_velocity_valid" in candidate:
            group_errors = {}
            for key, current in (("base", base), ("candidate", candidate)):
                group_high = group_ref["phase_velocity"][:, :k]
                gx = current["phase_velocity"]
                group_errors[key] = ((np.isfinite(gx) != np.isfinite(group_high)) |
                    (np.abs(gx-group_high) > np.maximum(3e-6,rtol*np.abs(group_high))))
                vrows[key]["aligned_groups"] = bc.compare_groups(current, group_ref, group_errors[key])
            aligned = ~group_errors["base"] & ~group_errors["candidate"] & np.isfinite(group_high)
            aligned &= ((group_ref["trace_flags"] & 5) == 0)[:, None, :]
            valid = aligned & group_ref["group_velocity_valid"][:, :k] & base["group_velocity_valid"]
            vrows["previously_valid_reference_groups_lost"] = int(np.count_nonzero(valid & ~candidate["group_velocity_valid"]))
        if "group_velocity_valid" in candidate:
            vrows["all_previous_valid_groups_lost"] = int(np.count_nonzero(base["group_velocity_valid"] & ~candidate["group_velocity_valid"]))
        if unchecked is not None:
            x = unchecked["phase_velocity"]
            prior = ((np.isfinite(x) != np.isfinite(hi)) | (np.abs(x-hi)>np.maximum(3e-6,rtol*np.abs(hi)))) & judged
            repaired = (candidate["continuity_flags"] & 2) != 0
            vrows["false_repair_slots"] = int(np.count_nonzero(repaired[:,None,:] & wrong["candidate"] & ~prior))
            vrows["fixed_by_repair_slots"] = int(np.count_nonzero(repaired[:,None,:] & ~wrong["candidate"] & prior))
        delta = [0, 0, 0]
        for b, t in np.argwhere((wrong["base"] | wrong["candidate"]).any(axis=1)):
            ref_trace = ref["phase_velocity"][b, :, t]
            old = ordered_errors(base["phase_velocity"][b,:,t], ref_trace, rtol)
            new = ordered_errors(candidate["phase_velocity"][b,:,t], ref_trace, rtol)
            for i, (earlier, later) in enumerate(zip(old, new)):
                delta[i] += len(later - earlier)
        vrows.update(new_missing_roots=delta[0], new_extra_roots=delta[1], new_displaced_ranks=delta[2])
        introduced = wrong["candidate"] & ~wrong["base"]
        fixed = wrong["base"] & ~wrong["candidate"]
        vrows.update(new_wrong_slots=int(introduced.sum()), corrected_slots=int(fixed.sum()),
            new_wrong_traces=int(np.count_nonzero(wrong["candidate"].any(axis=1) & ~wrong["base"].any(axis=1))),
            new_unresolved=int(np.count_nonzero(unresolved(candidate) & ~unresolved(base))),
            base_unresolved=int(np.count_nonzero(unresolved(base))), candidate_unresolved=int(np.count_nonzero(unresolved(candidate))),
            first_new_wrong_indices=np.argwhere(introduced)[:16].tolist())
        row[str(rtol)] = vrows
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("time", "capture", "judge", "bytes"))
    for name in ("base", "candidate", "runtime", "reference", "archived-m06", "m06-groups", "before-review", "inputs", "output"):
        p.add_argument("--" + name, type=Path, required=name == "output")
    p.add_argument("--cases", nargs="+", choices=ALL, default=PRIMARY)
    for n in ("base-tier", "candidate-tier", "tier"):
        p.add_argument("--" + n, choices=("default", "balanced", "auto", "fast", "high", "reference"), default="balanced")
    p.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--base-precision", choices=("float32", "float64"),
                   help="Optional precision of the baseline in a paired timing run")
    p.add_argument("--precision", choices=("float32", "float64"), default="float32")
    p.add_argument("--strategy", choices=("independent", "anchor_continuation"), default="independent")
    p.add_argument("--group", action="store_true")
    p.add_argument("--unchecked", action="store_true", help="capture the candidate before continuity review")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--pairs", type=int, default=11)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    required = {"time": ("base", "candidate", "inputs"), "capture": ("runtime", "inputs"),
                "bytes": ("base", "candidate"), "judge": ("base", "candidate", "reference")}[args.action]
    if any(getattr(args, n) is None for n in required) or (args.action == "judge" and "m06" in args.cases and args.archived_m06 is None):
        p.error(f"missing paths for {args.action}")
    if args.unchecked and args.action != "capture":
        p.error("--unchecked is an untimed diagnostic capture")
    if args.pairs < 1:
        p.error("--pairs must be positive")
    output = args.output / "captures.json" if args.action == "capture" else args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(options={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  python=platform.python_version(), numpy=np.__version__, affinity=sorted(os.sched_getaffinity(0)),
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  tool_sha256=bc.sha(__file__), libraries={}, cases={})
    if args.inputs:
        manifest = json.loads((ROOT / "docs/benchmarks/backend-0.4.0-inputs.json").read_text())["files"]
        for name in ("solid24.npz", "wet24.npz", "solid70.npz", "periods64.npy"):
            if bc.sha(args.inputs / name) != manifest[name]:
                raise RuntimeError(f"wrong input hash for {name}")
    for n in (("runtime",) if args.action == "capture" else ("base", "candidate") if args.action == "time" else ()):
        report["libraries"][n] = bc.snapshot(getattr(args, n))
    if args.action in ("judge", "bytes"):
        captures = {}
        for variant in ("base", "candidate", "before_review", "reference"):
            directory = getattr(args, variant)
            if directory is not None:
                rec = json.loads((directory / "captures.json").read_text())
                captures[variant] = dict(runtime_sha256=rec["libraries"]["runtime"], options=rec["options"])
        report["captures"] = captures
    if args.action == "judge" and args.archived_m06:
        report["m06_ref_sha256"] = bc.sha(args.archived_m06)
    use_gpu = args.action in ("capture", "time") and args.backend == "cuda" and args.tier != "reference"
    if use_gpu:
        report["gpu_start"] = bc.gpu_state()
    for name in args.cases:
        if args.action == "time":
            val = paired(args, name)
        elif args.action == "capture":
            with ExitStack() as stack:
                w = pool(stack, args.runtime, args, name, args.tier)
                val = w.submit(capture, args.output, name).result()
        elif args.action == "bytes":
            val = byte_check(args, name)
        else:
            val = evaluate(args, name)
        report["cases"][name] = val
        bc.save(output, report)
        print(name, flush=True)
    if use_gpu:
        report["gpu_end"] = bc.gpu_state()
    bc.save(output, report)


if __name__ == "__main__":
    main()
