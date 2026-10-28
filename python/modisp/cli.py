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
                         help="search precision preset for qe_adaptive (default: balanced)")
    args=parser.parse_args(argv)
    if args.command=="doctor":
        data=dict(version=__version__,python=sys.executable,cpu_module=_cpu.__file__,
                  default_accuracy="balanced",sampling_capacity=2048)
        try:
            module=_load_cuda();available=bool(module.available(0));data.update(cuda_module=module.__file__,cuda_available=available,auto_backend="cuda" if available else "cpu")
            if available:
                module.set_device(0);free,total=module.memory_info();data.update(cuda_device=module.device_name(),free_device_bytes=free,total_device_bytes=total)
        except BackendUnavailableError as exc:data.update(cuda_available=False,cuda_reason=str(exc),auto_backend="cpu")
        except RuntimeError as exc:data.update(cuda_available=False,cuda_reason=str(exc),auto_backend="error")
        print(json.dumps(data,indent=2));return 0
    with np.load(args.input,allow_pickle=False) as src:
        model={key:src[key] for key in ("thickness","vp","vs","rho")};periods=src["periods"]
    with Solver(backend=args.backend,threads=args.threads,device=args.device,cmin=args.cmin,cmax=args.cmax,
                accuracy=args.accuracy) as solver:
        out=solver(**model,periods=periods,nmodes=args.nmodes,wave=args.wave)
        np.savez_compressed(args.output,phase_velocity=out.phase_velocity,periods=out.period,
                            mode_mask=out.mode_mask,status=out.status,trace_flags=out.trace_flags,
                            search_bounds=out.search_bounds,search_level=out.search_level,
                            root_count=out.root_count,backend=out.backend_used,wave=out.wave,
                            accuracy=np.asarray(out.accuracy),numerical_flags=out.numerical_flags)
    return 2 if np.any(out.trace_flags & 4) else 0


if __name__=="__main__":
    raise SystemExit(main())
