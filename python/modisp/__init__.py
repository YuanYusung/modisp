"""Layered-earth multimodal phase dispersion. Units: km, km/s, g/cm³, s."""
from __future__ import annotations
from enum import IntEnum
import os
import threading
import time
import numpy as np
from . import _cpu
from ._loader import BackendUnavailableError, CudaUnavailable, load as _load_cuda
from .models import LayeredModel, CANONICAL_MODELS, random_model_batch, pad_batch, to_batch

__version__ = "0.3.0"
if _cpu.VERSION!=__version__ or _cpu.ABI_VERSION!=4:raise RuntimeError("CPU package and extension versions differ; reinstall a matching wheel")
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
TRACE_APPROXIMATE_SEARCH=2048
TRACE_CLOSE_ROOT_RISK=4096


class SolveStatus(IntEnum):
    VALID=0
    NOT_FOUND=1
    UNRESOLVED=2
    REJECTED=3


class DispersionBatch:
    def __init__(self, velocity, periods, nmodes, wave, flags, levels, counts, bounds, backend, reason, accuracy=None, close_root_rtol=.005):
        self.phase_velocity=velocity
        self.period=periods.copy()
        self.mode=np.arange(nmodes)
        self.wave=wave
        self.accuracy=accuracy
        self.close_root_rtol=float(close_root_rtol)
        self.approximate_search=(flags & TRACE_APPROXIMATE_SEARCH)!=0
        self.mode_index_uncertain=self.approximate_search.copy()
        self.close_root_mask=np.zeros(velocity.shape,dtype=bool)
        if nmodes>1:
            gaps=np.diff(velocity,axis=1)
            close=np.isfinite(gaps)&(gaps<=self.close_root_rtol*velocity[:,1:,:])
            self.close_root_mask[:,:-1,:]|=close
            self.close_root_mask[:,1:,:]|=close
        self.close_root_risk=((flags&TRACE_CLOSE_ROOT_RISK)!=0)|np.any(self.close_root_mask,axis=1)
        self.potentially_ambiguous=self.approximate_search|self.close_root_risk
        self.mode_mask=np.isfinite(velocity)
        self.trace_flags=flags
        # User cropping is a range choice, not a numerical failure. Keep it
        # separable so downstream masks do not discard every explicitly
        # bounded trace.
        self.numerical_flags=flags & np.uint32(0xFFFFFFFF ^ TRACE_RANGE_TRUNCATED)
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
        self.mode_convention=("ascending recovered roots; close-cluster omissions may shift indices" if np.any(self.approximate_search) else "ascending roots in the reported search interval")

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
    def __init__(self,nc=512,cmin=None,cmax=None,c_tol_rel=None,f_accept=1.,block=256,
                 search_strategy="qe_adaptive",workspace_bytes=256*1024**2,chunk_size=None,
                 *,backend="auto",device=0,threads=None,profile=False,accuracy=None,
                 adaptive_period_threshold=None,close_root_rtol=.005):
        if backend not in ("auto","cpu","cuda"):raise ValueError("backend must be auto, cpu or cuda")
        if search_strategy not in ("qe_adaptive","uniform"):raise ValueError("invalid search_strategy")
        if accuracy is None:
            accuracy="balanced" if search_strategy=="qe_adaptive" else "uniform"
        elif accuracy not in ("high","balanced","fast"):
            raise ValueError("accuracy must be high, balanced or fast")
        elif search_strategy=="uniform":
            raise ValueError("accuracy applies only to qe_adaptive; uniform is the legacy fixed path")
        if search_strategy=="uniform":accuracy="uniform"
        if adaptive_period_threshold is not None:
            if accuracy in ("high","uniform"):raise ValueError("adaptive_period_threshold applies only to balanced/fast")
            if not np.isfinite(adaptive_period_threshold) or adaptive_period_threshold<=0:raise ValueError("adaptive_period_threshold must be finite and positive")
        if not np.isscalar(close_root_rtol) or np.iscomplexobj(close_root_rtol) or not np.isfinite(close_root_rtol) or not 0<close_root_rtol<1:raise ValueError("close_root_rtol must be finite and between 0 and 1")
        if c_tol_rel is None:c_tol_rel={"high":1e-6,"balanced":1e-5,"fast":1e-4,"uniform":1e-6}[accuracy]
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
        if not isinstance(profile,(bool,np.bool_)):raise ValueError("profile must be boolean")
        self.nc=int(nc);self.cmin=cmin;self.cmax=cmax;self.c_tol_rel=float(c_tol_rel);self.f_accept=float(f_accept)
        self.block=int(block);self.search_strategy=search_strategy;self.workspace_bytes=int(workspace_bytes);self.chunk_size=chunk_size
        self.backend=backend;self.device=int(device);self.threads=int(threads)
        self.profile=bool(profile)
        self.accuracy=accuracy;self.adaptive_period_threshold=adaptive_period_threshold;self.close_root_rtol=float(close_root_rtol)
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
        started=time.perf_counter() if self.profile else 0.
        timings={}
        del names
        if wave not in ("rayleigh","love"):raise ValueError("wave must be rayleigh or love")
        if not isinstance(nmodes,(int,np.integer)) or not 1<=nmodes<=256:raise ValueError("nmodes must be 1..256")
        h,a,b,r=_models(thickness,vp,vs,rho);B,L=h.shape
        if np.iscomplexobj(periods):raise ValueError("periods must be real")
        per=np.ascontiguousarray(periods,dtype=np.float64)
        if per.ndim!=1 or not per.size or not np.isfinite(per).all() or np.any(per<=0):raise ValueError("periods must be a finite positive 1-D array")
        if self.profile:timings['input_validation']=time.perf_counter()-started
        selected=time.perf_counter() if self.profile else 0.
        backend,reason=self._select()
        if self.profile:timings['backend_selection']=time.perf_counter()-selected
        if self.search_strategy=="uniform":
            if backend!="cuda" or wave!="rayleigh" or np.any(b==0) or L<2:raise NotImplementedError("uniform is the legacy CUDA solid-Rayleigh option; use qe_adaptive for this combination")
            if self._legacy is None:
                from ._legacy import Solver as LegacySolver
                self._legacy=LegacySolver(self.nc,.2 if self.cmin is None else self.cmin,self.cmax,self.c_tol_rel,self.f_accept,self.block,search_strategy="uniform")
            old=self._legacy(h,a,b,r,per,nmodes,wave)
            bounds=_cpu.bounds(h,a,b,r,0,.2 if self.cmin is None else self.cmin,self.cmax)
            flags=old.trace_flags.copy()|TRACE_RANGE_TRUNCATED
            result=DispersionBatch(old.phase_velocity,per,nmodes,wave,flags,np.full((B,per.size),-2,dtype=np.int32),np.sum(old.mode_mask,axis=1,dtype=np.int32),bounds,backend,reason,accuracy="uniform",close_root_rtol=self.close_root_rtol)
            self._last_stats=dict(self._legacy.stats(),accuracy="uniform",close_root_rtol=self.close_root_rtol);self._calls+=1;self.last_result=result
            return result
        stage=time.perf_counter() if self.profile else 0.
        bounds=_cpu.bounds(h,a,b,r,int(wave=="love"),self.cmin,self.cmax)
        if self.profile:timings['bounds']=time.perf_counter()-stage
        if not np.isfinite(bounds).all() or np.any(bounds[:,0]<=0):raise ValueError("automatic search range could not be constructed")
        if backend=="cpu":
            fixed=self.threads*(4*2048*16+5*(nmodes+2)*8+3*L*8
                                +0)+4096
            cost=8*(6*L+2*(129+2*L))+per.size*(8*nmodes+12+16)
            limit=(self.workspace_bytes-fixed)//cost
        else:
            fixed=4096+8*per.size
            cache_points=256
            cost=8*(6*L+2*1024+4)+4+per.size*(64*(nmodes+2)+8*nmodes+48+2*cache_points*8+16)
            limit=(self.workspace_bytes-fixed)//cost
        capacity=min(B,limit,self.chunk_size or B)
        if capacity<1:raise MemoryError("workspace_bytes is too small for one model and the backend's fixed workspace; increase the budget or reduce threads")
        threshold=0. if self.adaptive_period_threshold is None else float(self.adaptive_period_threshold)
        resolved_threshold=(float(self.adaptive_period_threshold) if self.adaptive_period_threshold is not None
                            else (1.0 if self.accuracy=="balanced" else (0.63 if self.accuracy=="fast" else 0.0)))
        key=(backend,self.device,self.threads,capacity,L,per.size,int(nmodes),self.accuracy,threshold,self.close_root_rtol)
        # Preserve the high-water capacity for short tail batches. Never retain
        # an allocation that exceeds a subsequently reduced budget/chunk cap.
        compatible=(self._engine_key is not None and self._engine_key[:3]==key[:3]
                    and self._engine_key[4:]==key[4:] and capacity<=self._engine_key[3]
                    <=min(limit,self.chunk_size or limit))
        allocated=not compatible
        stage=time.perf_counter() if self.profile else 0.
        if allocated:
            self.close()
            code={"high":0,"balanced":1,"fast":2,"uniform":3}[self.accuracy]
            self._engine=(_cpu.CpuSolver(self.threads,code,threshold,self.close_root_rtol) if backend=="cpu"
                          else self._module.QeSolver(capacity,L,per.size,int(nmodes),code,threshold,self.close_root_rtol))
            self._engine_key=key
        else:capacity=self._engine_key[3]
        if self.profile:timings['engine_prepare']=time.perf_counter()-stage
        stage=time.perf_counter() if self.profile else 0.
        velocity=np.empty((B,nmodes,per.size));flags=np.empty((B,per.size),dtype=np.uint32)
        levels=np.empty((B,per.size),dtype=np.int32);counts=np.empty_like(levels);stats={};chunks=0
        if self.profile:timings['output_allocate']=time.perf_counter()-stage
        for start in range(0,B,capacity):
            end=min(B,start+capacity)
            args=(h[start:end],a[start:end],b[start:end],r[start:end],per,bounds[start:end])
            if backend=="cpu":out=self._engine.run(*args,int(nmodes),int(wave=="love"),self.cmin is None,self.c_tol_rel,self.f_accept,self.profile)
            else:out=self._engine.run(*args,self.c_tol_rel,self.f_accept,int(wave=="love"),self.cmin is None,self.profile)
            stage=time.perf_counter() if self.profile else 0.
            velocity[start:end],flags[start:end],part,levels[start:end],counts[start:end]=out
            if self.profile:timings['chunk_assembly']=timings.get('chunk_assembly',0.)+time.perf_counter()-stage
            for k,v in part.items():
                if k=='timings_s':
                    for name,seconds in v.items():timings[name]=timings.get(name,0.)+float(seconds)
                elif k=='active_traces_by_stage':
                    stats[k]=[x+int(y) for x,y in zip(stats.get(k,[0]*len(v)),v)]
                elif k in ("density_table_knots","cache_capacity","pinned_host_bytes","device_workspace_bytes","sampling_capacity","dense_floor4","dense_floor5","dense_tile"):stats[k]=int(v)
                elif isinstance(v,(int,np.integer)) and k not in ("threads","capacity"):stats[k]=stats.get(k,0)+int(v)
            chunks+=1
        if self.cmin is not None or self.cmax is not None:flags|=TRACE_RANGE_TRUNCATED
        counts[(flags&TRACE_SEARCH_UNCONVERGED)!=0]=-1
        stats.update(backend=backend,strategy=("qe_adaptive" if self.accuracy=="high" else "observation_log"),accuracy=self.accuracy,
                     period_threshold=resolved_threshold,early_acceptance_enabled=False,approximate_search_enabled=self.accuracy in ("balanced","fast"),velocity_step_rtol=(self.close_root_rtol*(.5 if self.accuracy=="balanced" else 1.) if self.accuracy!="high" else None),close_root_rtol=self.close_root_rtol,refinement_tolerance=self.c_tol_rel,chunks=chunks,threads=self.threads if backend=="cpu" else 0)
        stats.update(approximate_traces=int(np.count_nonzero(flags&TRACE_APPROXIMATE_SEARCH)),close_root_risk_traces=int(np.count_nonzero(flags&TRACE_CLOSE_ROOT_RISK)))
        stats.update(workspace_bytes=int(fixed+capacity*cost),workspace_budget_bytes=self.workspace_bytes)
        stats.update(capacity_models=int(capacity),engine_allocations=int(allocated))
        for k in ("rejected_sign","troughs","hidden_roots","relocated"):stats.setdefault(k,0)
        self._last_stats=stats;self._key=(B,L,per.size,self.nc,nmodes);self._calls+=1
        stage=time.perf_counter() if self.profile else 0.
        result=DispersionBatch(velocity,per,nmodes,wave,flags,levels,counts,bounds,backend,reason,accuracy=self.accuracy,close_root_rtol=self.close_root_rtol)
        if self.profile:
            timings['result_packaging']=time.perf_counter()-stage
            timings['solver_total']=time.perf_counter()-started
            stats['timings_s']=timings
        self.last_result=result
        return result

    @property
    def calls(self):return self._calls
    def stats(self):
        if self._last_stats is None:raise RuntimeError("call the solver first")
        result=dict(self._last_stats)
        if 'timings_s' in result:result['timings_s']=dict(result['timings_s'])
        if 'active_traces_by_stage' in result:result['active_traces_by_stage']=list(result['active_traces_by_stage'])
        return result
    def kernel_report(self):
        backend=self.last_result.backend_used if self.last_result is not None else self.backend
        if backend=="cuda" and self.search_strategy=="uniform":return {"backend":"cuda","precision":"float32 scan / float64 refinement","accuracy":"uniform","legacy":self._legacy.kernel_report()}
        resolved=(float(self.adaptive_period_threshold) if self.adaptive_period_threshold is not None
                  else (1.0 if self.accuracy=="balanced" else (0.63 if self.accuracy=="fast" else 0.0)))
        if backend=="cuda" and self._key is not None:return dict(self._module.qe_kernel_report(self._key[1],int(self.accuracy!="high")),backend=backend,precision="float64",accuracy=self.accuracy,period_threshold=resolved,sampling_capacity=2048,cache_points_per_trace=256,close_root_rtol=self.close_root_rtol,refinement_tolerance=self.c_tol_rel)
        return {"backend":backend,"precision":"float64","accuracy":self.accuracy,"period_threshold":resolved,"close_root_rtol":self.close_root_rtol,"refinement_tolerance":self.c_tol_rel,"search_kernel":"qe_adaptive" if self.accuracy=="high" else "observation_log","threads":self.threads,"module":_cpu.__file__}
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


def dispersion_batch(thickness,vp,vs,rho,periods,nmodes=5,wave="rayleigh",solver=None,names=None,search_strategy=None,accuracy=None,**options):
    global _DEFAULT
    if solver is not None:
        if (options or (search_strategy is not None and search_strategy!=solver.search_strategy)
                or (accuracy is not None and accuracy!=solver.accuracy)):
            raise ValueError("options conflict with the supplied Solver")
        return solver(thickness,vp,vs,rho,periods,nmodes,wave,names)
    if options or accuracy is not None:
        with Solver(search_strategy="qe_adaptive" if search_strategy is None else search_strategy,
                    accuracy=accuracy,**options) as s:return s(thickness,vp,vs,rho,periods,nmodes,wave,names)
    strategy="qe_adaptive" if search_strategy is None else search_strategy
    expected="balanced" if strategy=="qe_adaptive" else "uniform"
    if _DEFAULT is None or _DEFAULT.search_strategy!=strategy or _DEFAULT.accuracy!=expected:
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
