"""Installed FP64 evaluator and solid-only precision probes.

Diagnostic grids share the production physics and are not independent physical
references. Production CPU/CUDA solvers always use normalized FP64 evaluation.
"""
import numpy as np
from . import _cpu
NativeUnavailable=RuntimeError


def load():
    return _cpu


def secular_grid(thickness,vp,vs,rho,periods,cmin,cmax,nc,normalize=True,precision="float64",wave="rayleigh"):
    from . import _models
    arrays=_models(thickness,vp,vs,rho)
    if arrays[0].shape[0]!=1:raise ValueError("secular_grid expects one model")
    arrays=[a[0] for a in arrays]
    t=np.ascontiguousarray(periods,dtype=np.float64)
    if t.ndim!=1 or not t.size or not np.isfinite(t).all() or np.any(t<=0):raise ValueError("periods must be finite and positive")
    if not isinstance(nc,(int,np.integer)) or nc<1 or not np.isfinite([cmin,cmax]).all() or not 0<cmin<=cmax:raise ValueError("invalid grid")
    if precision not in ("float32","float64"):raise ValueError("precision must be float32 or float64")
    if wave not in ("rayleigh","love"):raise ValueError("invalid wave")
    if precision=="float64" and normalize:
        c=np.linspace(cmin,cmax,nc)
        return np.array([_cpu.secular(*arrays,p,c,int(wave=="love")) for p in t])
    if wave!="rayleigh":raise ValueError("raw precision probes support solid Rayleigh only")
    return _cpu.probe_grid(*arrays,t,float(cmin),float(cmax),int(nc),bool(normalize),precision=="float32")
