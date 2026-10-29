"""Small portable command-line interface."""
import argparse
import json
import sys
import numpy as np
from . import Solver, __version__, _cpu, _load_cuda, BackendUnavailableError


def main(argv=None):
    parser=argparse.ArgumentParser(prog="modisp")
    parser.add_argument("--version",action="version",version=__version__)
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("doctor",help="show installed modules and backend status")
    forward=sub.add_parser("forward",help="compute phase velocities from NPZ model/period arrays")
    forward.add_argument("input")
    forward.add_argument("--output",required=True)
    forward.add_argument("--wave",choices=["rayleigh","love"],default="rayleigh")
    forward.add_argument("--backend",choices=["auto","cpu","cuda"],default="auto")
    forward.add_argument("--nmodes",type=int,default=3)
    forward.add_argument("--threads",type=int)
    forward.add_argument("--device",type=int,default=0)
    forward.add_argument("--cmin",type=float)
    forward.add_argument("--cmax",type=float)
    forward.add_argument("--accuracy",choices=["high","balanced","fast"],default=None,
                         help="search precision preset (default: balanced)")
    forward.add_argument("--search-strategy",choices=["shape_adaptive","qe_adaptive","uniform"],default="shape_adaptive")
    forward.add_argument("--sampling-rtol",type=float)
    forward.add_argument("--close-root-rtol",type=float,default=.005)
    forward.add_argument("--c-tol-rel",type=float)
    forward.add_argument("--secular",choices=["dunkin","fast"],default=None,
                         help="secular evaluator (default: CUDA fast, CPU/uniform dunkin)")
    args=parser.parse_args(argv)
    if args.command=="doctor":
        data=dict(version=__version__,python=sys.executable,cpu_module=_cpu.__file__,
                  default_accuracy="balanced",default_search_strategy="shape_adaptive",
                  default_secular_by_backend={"cpu":"dunkin","cuda":"fast","uniform":"dunkin"},secular_options=["dunkin","fast"],
                  private_abi=int(_cpu.ABI_VERSION),sampling_capacity=2048)
        try:
            module=_load_cuda();available=bool(module.available(0));data.update(cuda_module=module.__file__,cuda_available=available,auto_backend="cuda" if available else "cpu")
            if available:
                module.set_device(0);free,total=module.memory_info();data.update(cuda_device=module.device_name(),free_device_bytes=free,total_device_bytes=total)
        except BackendUnavailableError as exc:data.update(cuda_available=False,cuda_reason=str(exc),auto_backend="cpu")
        except RuntimeError as exc:data.update(cuda_available=False,cuda_reason=str(exc),auto_backend="error")
        data["default_secular"]=data["default_secular_by_backend"].get(data["auto_backend"])
        print(json.dumps(data,indent=2));return 0
    with np.load(args.input,allow_pickle=False) as src:
        model={key:src[key] for key in ("thickness","vp","vs","rho")};periods=src["periods"]
    with Solver(backend=args.backend,threads=args.threads,device=args.device,cmin=args.cmin,cmax=args.cmax,
                accuracy=args.accuracy,close_root_rtol=args.close_root_rtol,c_tol_rel=args.c_tol_rel,
                search_strategy=args.search_strategy,sampling_rtol=args.sampling_rtol,
                secular=args.secular) as solver:
        out=solver(**model,periods=periods,nmodes=args.nmodes,wave=args.wave)
        np.savez_compressed(args.output,phase_velocity=out.phase_velocity,periods=out.period,
                            mode_mask=out.mode_mask,status=out.status,trace_flags=out.trace_flags,
                            search_bounds=out.search_bounds,search_level=out.search_level,
                            root_count=out.root_count,backend=out.backend_used,wave=out.wave,
                            accuracy=np.asarray(out.accuracy),numerical_flags=out.numerical_flags,
                            approximate_search=out.approximate_search,close_root_mask=out.close_root_mask,
                            potentially_ambiguous=out.potentially_ambiguous,mode_index_uncertain=out.mode_index_uncertain,
                            close_root_risk=out.close_root_risk,
                            close_root_rtol=np.asarray(out.close_root_rtol),search_strategy=np.asarray(out.search_strategy),
                            sampling_rtol=np.asarray(np.nan if out.sampling_rtol is None else out.sampling_rtol),
                            secular=np.asarray(out.secular))
    return 2 if np.any(out.trace_flags & 4) else 0


if __name__=="__main__":
    raise SystemExit(main())
