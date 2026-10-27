"""Layered-earth multimodal phase dispersion. Units: km, km/s, g/cm³, s."""
from __future__ import annotations
from enum import IntEnum
import os
import threading
import numpy as np
from . import _cpu
from ._loader import BackendUnavailableError, CudaUnavailable, load as _load_cuda
from .models import LayeredModel, CANONICAL_MODELS, random_model_batch, pad_batch, to_batch

__version__ = "0.2.0"
if _cpu.VERSION!=__version__ or _cpu.ABI_VERSION!=1:raise RuntimeError("CPU package and extension versions differ; reinstall a matching wheel")
TRACE_UNRESOLVED_BRACKET=1
TRACE_RELOCATED=2
TRACE_SEARCH_UNCONVERGED=4
TRACE_WORKSPACE_LIMIT=8
TRACE_NEAR_CUTOFF=16
TRACE_NONFINITE=32
TRACE_REFINEMENT_FAILED=64
TRACE_F_REJECTED=128
TRACE_RANGE_TRUNCATED=256
TRACE_SAMPLING_UNCERTAIN=512
TRACE_ROOT_CONFLICT=1024


class SolveStatus(IntEnum):
    VALID=0
    NOT_FOUND=1
    UNRESOLVED=2
    REJECTED=3


class DispersionBatch:
    def __init__(self, velocity, periods, nmodes, wave, flags, levels, counts, bounds, backend, reason):
        self.phase_velocity=velocity
        self.period=periods.copy()
        self.mode=np.arange(nmodes)
        self.wave=wave
        self.mode_mask=np.isfinite(velocity)
        self.trace_flags=flags
        self.search_level=levels
        self.root_count=counts
        self.search_bounds=bounds.copy()
        self.backend_used=backend
        self.backend_reason=reason
        self.status=np.where(self.mode_mask, SolveStatus.VALID,
                             np.where((flags[:,None,:]&(TRACE_SEARCH_UNCONVERGED|TRACE_UNRESOLVED_BRACKET))!=0,
                                      SolveStatus.UNRESOLVED,SolveStatus.NOT_FOUND)).astype(np.uint8)
        rejected=np.isnan(velocity)&((flags[:,None,:]&TRACE_F_REJECTED)!=0)&(np.arange(nmodes)[None,:,None]<counts[:,None,:])
        self.status[rejected]=SolveStatus.REJECTED
        self.mode_convention="ascending roots in the reported search interval"

    @property
    def n_models(self): return self.phase_velocity.shape[0]
    @property
    def n_modes(self): return self.phase_velocity.shape[1]
    def __len__(self): return self.n_models
    def __repr__(self): return f"<DispersionBatch {self.wave} backend={self.backend_used} B={self.n_models} nmodes={self.n_modes} T={self.period.size}>"


def _models(thickness,vp,vs,rho):
    arrays=[]
    for name,value in zip(("thickness","vp","vs","rho"),(thickness,vp,vs,rho)):
        if np.iscomplexobj(value): raise ValueError(f"{name} must be real")
        a=np.ascontiguousarray(value,dtype=np.float64)
        if a.ndim==1:a=a[None,:]
        if a.ndim!=2 or not a.size or not np.isfinite(a).all(): raise ValueError(f"{name} must be a finite nonempty [B,L] or [L] array")
        arrays.append(a)
    h,a,b,r=arrays
    if any(v.shape!=h.shape for v in arrays): raise ValueError("model array shapes must agree")
    if h.shape[1]>256: raise ValueError("at most 256 layers are supported")
    if np.any(h<0) or np.any(a<=0) or np.any(r<=0) or np.any(b<0): raise ValueError("thickness/vs must be non-negative and vp/rho positive")
    if np.any(b[:,-1]<=0) or np.any(b[:,1:]==0): raise ValueError("only a single first water layer may have vs=0; the half-space must be solid")
    solid=b>0
    if np.any(a[solid]/b[solid]<=np.sqrt(4./3.)): raise ValueError("solid vp/vs must exceed sqrt(4/3) for positive bulk modulus")
    return arrays


class Solver:
    """Reusable FP64 solver; an instance must not be called concurrently.

    backend='auto' selects an available CUDA plugin, otherwise CPU. threads
    controls the persistent CPU pool. Explicit cmin/cmax define a search window
    and consequently the ascending root labels within that window.
    """
    def __init__(self,nc=512,cmin=None,cmax=None,c_tol_rel=1e-6,f_accept=1.,block=256,
                 search_strategy="qe_adaptive",workspace_bytes=256*1024**2,chunk_size=None,
                 *,backend="auto",device=0,threads=None):
        if backend not in ("auto","cpu","cuda"):raise ValueError("backend must be auto, cpu or cuda")
        if search_strategy not in ("qe_adaptive","uniform"):raise ValueError("invalid search_strategy")
        for name,v in (("cmin",cmin),("cmax",cmax)):
            if v is not None and (not np.isfinite(v) or v<=0):raise ValueError(f"{name} must be finite and positive")
        if cmin is not None and cmax is not None and cmax<=cmin:raise ValueError("cmax must exceed cmin")
        if not np.isfinite(c_tol_rel) or c_tol_rel<=0:raise ValueError("c_tol_rel must be positive")
        if not np.isfinite(f_accept) or f_accept<0:raise ValueError("f_accept must be non-negative")
        if nc<3:raise ValueError("nc must be at least 3")
        if not isinstance(workspace_bytes,(int,np.integer)) or workspace_bytes<1:raise ValueError("workspace_bytes must be a positive integer")
        if chunk_size is not None and (not isinstance(chunk_size,(int,np.integer)) or chunk_size<1):raise ValueError("chunk_size must be positive")
        if not isinstance(device,(int,np.integer)) or device<0:raise ValueError("device must be a non-negative integer")
        available=len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else os.cpu_count() or 1
        threads=min(8,available) if threads is None else threads
        if not isinstance(threads,(int,np.integer)) or not 1<=threads<=256:raise ValueError("threads must be 1..256")
        self.nc=int(nc);self.cmin=cmin;self.cmax=cmax;self.c_tol_rel=float(c_tol_rel);self.f_accept=float(f_accept)
        self.block=int(block);self.search_strategy=search_strategy;self.workspace_bytes=int(workspace_bytes);self.chunk_size=chunk_size
        self.backend=backend;self.device=int(device);self.threads=int(threads)
        self._engine=None;self._engine_key=None;self._key=None;self._module=None;self._legacy=None;self._calls=0;self._last_stats=None
        self.last_result=None
        self._lock=threading.RLock()

    def _select(self):
        if self.backend=="cpu":return "cpu","explicit CPU selection"
        try:
            module=_load_cuda()
            if not module.available(self.device):raise BackendUnavailableError(f"CUDA device {self.device} is unavailable")
            try:module.set_device(self.device)
            except RuntimeError as exc:raise BackendUnavailableError(f"CUDA device {self.device} initialization failed: {exc}") from exc
            self._module=module
            return "cuda","CUDA plugin and device available"
        except BackendUnavailableError as exc:
            if self.backend=="cuda":raise
            return "cpu",str(exc)

    def __call__(self,thickness,vp,vs,rho,periods,nmodes=5,wave="rayleigh",names=None):
        if not self._lock.acquire(blocking=False):raise RuntimeError("a Solver instance cannot be used concurrently; create one instance per caller")
        try:return self._solve(thickness,vp,vs,rho,periods,nmodes,wave,names)
        finally:self._lock.release()

    def _solve(self,thickness,vp,vs,rho,periods,nmodes=5,wave="rayleigh",names=None):
        del names
        if wave not in ("rayleigh","love"):raise ValueError("wave must be rayleigh or love")
        if not isinstance(nmodes,(int,np.integer)) or not 1<=nmodes<=256:raise ValueError("nmodes must be 1..256")
        h,a,b,r=_models(thickness,vp,vs,rho);B,L=h.shape
        if np.iscomplexobj(periods):raise ValueError("periods must be real")
        per=np.ascontiguousarray(periods,dtype=np.float64)
        if per.ndim!=1 or not per.size or not np.isfinite(per).all() or np.any(per<=0):raise ValueError("periods must be a finite positive 1-D array")
        backend,reason=self._select()
        if self.search_strategy=="uniform":
            if backend!="cuda" or wave!="rayleigh" or np.any(b==0) or L<2:raise NotImplementedError("uniform is the legacy CUDA solid-Rayleigh option; use qe_adaptive for this combination")
            if self._legacy is None:
                from ._legacy import Solver as LegacySolver
                self._legacy=LegacySolver(self.nc,.2 if self.cmin is None else self.cmin,self.cmax,self.c_tol_rel,self.f_accept,self.block,search_strategy="uniform")
            old=self._legacy(h,a,b,r,per,nmodes,wave)
            bounds=_cpu.bounds(h,a,b,r,0,.2 if self.cmin is None else self.cmin,self.cmax)
            flags=old.trace_flags.copy()|TRACE_RANGE_TRUNCATED
            result=DispersionBatch(old.phase_velocity,per,nmodes,wave,flags,np.full((B,per.size),-2,dtype=np.int32),np.sum(old.mode_mask,axis=1,dtype=np.int32),bounds,backend,reason)
            self._last_stats=self._legacy.stats();self._calls+=1;self.last_result=result
            return result
        bounds=_cpu.bounds(h,a,b,r,int(wave=="love"),self.cmin,self.cmax)
        if not np.isfinite(bounds).all() or np.any(bounds[:,0]<=0):raise ValueError("automatic search range could not be constructed")
        if backend=="cpu":
            fixed=self.threads*(3*2048*16+4*(nmodes+2)*8)+4096
            cost=8*(6*L+2*(129+2*L))+per.size*(8*nmodes+12)
            capacity=min(B,(self.workspace_bytes-fixed)//cost,self.chunk_size or B)
        else:
            fixed=4096+8*per.size
            cost=8*(6*L+2*1024+4)+4+per.size*(64*(nmodes+2)+8*nmodes+48+2*256*8)
            capacity=min(B,(self.workspace_bytes-4096-8*per.size)//cost,self.chunk_size or B)
        if capacity<1:raise MemoryError("workspace_bytes is too small for one model and the backend's fixed workspace; increase the budget or reduce threads")
        key=(backend,self.device,self.threads,capacity,L,per.size,int(nmodes))
        if key!=self._engine_key:
            self.close()
            self._engine=_cpu.CpuSolver(self.threads) if backend=="cpu" else self._module.QeSolver(capacity,L,per.size,int(nmodes))
            self._engine_key=key
        velocity=np.empty((B,nmodes,per.size));flags=np.empty((B,per.size),dtype=np.uint32)
        levels=np.empty((B,per.size),dtype=np.int32);counts=np.empty_like(levels);stats={};chunks=0
        for start in range(0,B,capacity):
            end=min(B,start+capacity)
            args=(h[start:end],a[start:end],b[start:end],r[start:end],per,bounds[start:end])
            if backend=="cpu":out=self._engine.run(*args,int(nmodes),int(wave=="love"),self.cmin is None,self.c_tol_rel,self.f_accept)
            else:out=self._engine.run(*args,self.c_tol_rel,self.f_accept,int(wave=="love"),self.cmin is None)
            velocity[start:end],flags[start:end],part,levels[start:end],counts[start:end]=out
            for k,v in part.items():
                if k in ("density_table_knots","cache_capacity"):stats[k]=int(v)
                elif isinstance(v,(int,np.integer)) and k not in ("threads","capacity"):stats[k]=stats.get(k,0)+int(v)
            chunks+=1
        if self.cmin is not None or self.cmax is not None:flags|=TRACE_RANGE_TRUNCATED
        counts[(flags&TRACE_SEARCH_UNCONVERGED)!=0]=-1
        stats.update(backend=backend,strategy="qe_adaptive",chunks=chunks,threads=self.threads if backend=="cpu" else 0)
        stats.update(workspace_bytes=int(fixed+capacity*cost),workspace_budget_bytes=self.workspace_bytes)
        for k in ("rejected_sign","troughs","hidden_roots","relocated"):stats.setdefault(k,0)
        self._last_stats=stats;self._key=(B,L,per.size,self.nc,nmodes);self._calls+=1
        result=DispersionBatch(velocity,per,nmodes,wave,flags,levels,counts,bounds,backend,reason)
        self.last_result=result
        return result

    @property
    def calls(self):return self._calls
    def stats(self):
        if self._last_stats is None:raise RuntimeError("call the solver first")
        return dict(self._last_stats)
    def kernel_report(self):
        backend=self.last_result.backend_used if self.last_result is not None else self.backend
        if backend=="cuda" and self.search_strategy=="uniform":return {"backend":"cuda","precision":"float32 scan / float64 refinement","legacy":self._legacy.kernel_report()}
        if backend=="cuda" and self._key is not None:return dict(self._module.qe_kernel_report(self._key[1]),backend=backend,precision="float64")
        return {"backend":backend,"precision":"float64","threads":self.threads,"module":_cpu.__file__}
    def close(self):
        with self._lock:
            if self._engine is not None and hasattr(self._engine,"close"):self._engine.close()
            self._engine=None;self._engine_key=None;self._key=None
            if self._legacy is not None:self._legacy.close();self._legacy=None
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()


_DEFAULT=None
def release():
    global _DEFAULT
    if _DEFAULT is not None:_DEFAULT.close();_DEFAULT=None


def dispersion_batch(thickness,vp,vs,rho,periods,nmodes=5,wave="rayleigh",solver=None,names=None,search_strategy=None,**options):
    global _DEFAULT
    if solver is not None:
        if options or (search_strategy is not None and search_strategy!=solver.search_strategy):raise ValueError("options conflict with the supplied Solver")
        return solver(thickness,vp,vs,rho,periods,nmodes,wave,names)
    if options:
        with Solver(search_strategy="qe_adaptive" if search_strategy is None else search_strategy,**options) as s:return s(thickness,vp,vs,rho,periods,nmodes,wave,names)
    strategy="qe_adaptive" if search_strategy is None else search_strategy
    if _DEFAULT is None or _DEFAULT.search_strategy!=strategy:
        release();_DEFAULT=Solver(search_strategy=strategy)
    return _DEFAULT(thickness,vp,vs,rho,periods,nmodes,wave,names)


def dispersion(model,periods,nmodes=5,wave="rayleigh",**options):
    batch=to_batch([model]) if isinstance(model,LayeredModel) else model
    if not isinstance(batch,dict):
        a=np.asarray(batch)
        if a.ndim!=2 or a.shape[1]!=4:raise ValueError("model must be LayeredModel, an array with columns h/vp/vs/rho, or a batch dict")
        batch=dict(zip(("thickness","vp","vs","rho"),a.T))
    return dispersion_batch(**batch,periods=periods,nmodes=nmodes,wave=wave,**options)


def device_name():return _load_cuda().device_name()

from . import cuda, models
