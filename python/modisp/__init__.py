"""Layered-earth multimodal phase dispersion. Units: km, km/s, g/cm³, s."""
from __future__ import annotations
from enum import IntEnum, IntFlag
import os
import threading
import time
import numpy as np
from . import _cpu, _continuity
from ._continuity import ContinuityFlag
from ._loader import ABI_VERSION as _ABI_VERSION, VERSION as __version__
from ._loader import BackendUnavailableError, CudaUnavailable, load as _load_cuda
from .models import LayeredModel, CANONICAL_MODELS, random_model_batch, pad_batch, to_batch

if _cpu.VERSION!=__version__ or _cpu.ABI_VERSION!=_ABI_VERSION:raise RuntimeError("CPU package and extension versions differ; reinstall a matching wheel")
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
TRACE_ADAPTIVE_FALLBACK=8192
TRACE_LOVE_PHASE_FALLBACK=16384
TRACE_FP64_REPAIR=32768
TRACE_CONTINUATION_APPROXIMATE=65536
TRACE_AUTO_RELAXED=1 << 17
DEFAULT_ACCURACY="auto"
_ACCURACY_CODE={"high":0,"balanced":1,"fast":2,"uniform":3,"auto":4}
_APPROXIMATE_ACCURACIES=("auto","balanced","fast")
from .reliability import ModeReliabilityFlag, ModeReliability, check_mode_reliability


class SolveStatus(IntEnum):
    VALID=0
    NOT_FOUND=1
    UNRESOLVED=2
    REJECTED=3


class GroupVelocityFlag(IntFlag):
    MISSING_ROOT=1
    POLISH_FAILED=2
    DERIVATIVE_UNSTABLE=4
    ILL_CONDITIONED=8
    CLOSE_ROOT=16
    NONFINITE=32


class PredictionFlag(IntFlag):
    ANCHOR=1
    ACCEPTED=2
    FALLBACK=4
    BYPASSED=8
    AUDIT=16
    AUDIT_FAILED=32
    SOURCE_SEARCH_UNCERTAIN=64
    ROOT_COUNT_UNSAFE=128
    ANCHOR_UNSAFE=256
    WINDOW_UNSAFE=512
    BRACKET_FAILED=1024
    CORRECTION_FAILED=2048
    REVIEW=4096


class DispersionBatch:
    def __init__(self, velocity, periods, nmodes, wave, flags, levels, counts, bounds, backend, reason, accuracy=None, close_root_rtol=.005,search_strategy=None,sampling_rtol=None,secular=None,refinement=None,love_strategy=None,group_velocity=None,group_velocity_flags=None,prediction_flags=None,group_rtol=None,frequency_strategy="independent",frequency_strategy_used="independent",frequency_strategy_reason="independent frequency search requested",anchor_stride=4,continuity_flags=None,continuity_unresolved=False):
        self.phase_velocity=velocity
        self.precision=velocity.dtype.name
        self.refinement=refinement
        self.love_strategy=love_strategy
        self.secular=secular
        self.period=periods.copy()
        self.mode=np.arange(nmodes)
        self.wave=wave
        self.accuracy=accuracy
        self.search_strategy=search_strategy
        self.sampling_rtol=sampling_rtol
        self.close_root_rtol=float(close_root_rtol)
        self.group_velocity=group_velocity
        self.group_velocity_flags=group_velocity_flags
        self.group_velocity_mask=(None if group_velocity is None else np.isfinite(group_velocity)&(group_velocity_flags==0))
        self.group_velocity_valid=self.group_velocity_mask
        self.prediction_flags=prediction_flags
        self.continuity_flags=continuity_flags
        self.group_rtol=group_rtol
        self.frequency_strategy=frequency_strategy
        self.frequency_strategy_used=frequency_strategy_used
        self.frequency_strategy_reason=frequency_strategy_reason
        self.anchor_stride=int(anchor_stride)
        self.approximate_search=(flags & (TRACE_APPROXIMATE_SEARCH|TRACE_CONTINUATION_APPROXIMATE))!=0
        self.mode_index_uncertain=self.approximate_search.copy()
        if continuity_unresolved:
            self.mode_index_uncertain |= (continuity_flags & ContinuityFlag.UNRESOLVED) != 0
        self.close_root_mask=np.zeros(velocity.shape,dtype=bool)
        if nmodes>1:
            gaps=np.diff(velocity,axis=1)
            close=np.isfinite(gaps)&(gaps<=self.close_root_rtol*velocity[:,1:,:])
            self.close_root_mask[:,:-1,:]|=close
            self.close_root_mask[:,1:,:]|=close
        self.close_root_risk=((flags&TRACE_CLOSE_ROOT_RISK)!=0)|np.any(self.close_root_mask,axis=1)
        self.potentially_ambiguous=self.mode_index_uncertain|self.close_root_risk
        self.mode_mask=np.isfinite(velocity)
        self.trace_flags=flags
        # User cropping is a range choice, not a numerical failure. Keep it
        # separable so downstream masks do not discard every explicitly
        # bounded trace.
        self.numerical_flags=flags & np.uint32(0xFFFFFFFF ^ (TRACE_RANGE_TRUNCATED | TRACE_AUTO_RELAXED))
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

    def check_modes(self,**options):
        return check_mode_reliability(self,**options)

    @property
    def auto_relaxed(self):
        """Computed traces whose accepted search used a wider automatic sampling tolerance."""
        return (self.trace_flags & TRACE_AUTO_RELAXED)!=0 if self.accuracy=="auto" else None

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
    """Reusable phase dispersion solver; use one instance per concurrent caller.

    backend='auto' selects an available CUDA plugin, otherwise CPU. threads
    controls the persistent CPU pool. secular=None selects fast on CUDA and
    dunkin on CPU; legacy uniform always defaults to dunkin. Explicit cmin/cmax define a search window
    and consequently the ascending root labels within that window.
    accuracy='auto' allows wider shape sampling on smooth, same-sign intervals.
    Its root refinement and independent continuity review use the balanced settings.
    precision='float32' requires an FP32-capable CUDA plugin and an adaptive
    strategy. It uses FP32 secular evaluation and returns float32 velocities.
    group_velocity=True adds FP64 groups without altering reported phase roots.
    frequency_strategy='anchor_continuation' is opt-in CPU/CUDA balanced/fast
    prediction on at least eight unique periods. Balanced and fast screen adjacent
    phase curves by default and independently recheck suspicious traces in FP64.
    """
    def __init__(self,nc=512,cmin=None,cmax=None,c_tol_rel=None,f_accept=1.,block=256,
                 search_strategy="shape_adaptive",workspace_bytes=256*1024**2,chunk_size=None,
                 *,backend="auto",device=0,threads=None,profile=False,accuracy=None,
                 adaptive_period_threshold=None,close_root_rtol=.005,sampling_rtol=None,secular=None,
                 refinement=None,love_strategy="phase",precision="float64",group_velocity=False,
                 group_rtol=.001,frequency_strategy="independent",anchor_stride=4,continuity_check=True):
        if backend not in ("auto","cpu","cuda"):raise ValueError("backend must be auto, cpu or cuda")
        if search_strategy not in ("shape_adaptive","qe_adaptive","uniform"):raise ValueError("invalid search_strategy")
        if not isinstance(continuity_check,(bool,np.bool_)):raise ValueError("continuity_check must be boolean")
        self.continuity_check=bool(continuity_check)
        if not isinstance(group_velocity,(bool,np.bool_)):raise ValueError("group_velocity must be boolean")
        if (not np.isscalar(group_rtol) or isinstance(group_rtol,(str,bytes,bool,np.bool_))
                or np.iscomplexobj(group_rtol) or not np.isfinite(group_rtol) or not 0<group_rtol<1):
            raise ValueError("group_rtol must be finite and between 0 and 1")
        if frequency_strategy not in ("independent","anchor_continuation"):
            raise ValueError("frequency_strategy must be independent or anchor_continuation")
        if not isinstance(anchor_stride,(int,np.integer)) or isinstance(anchor_stride,(bool,np.bool_)) or anchor_stride<2:
            raise ValueError("anchor_stride must be an integer of at least 2")
        if search_strategy=="uniform" and (group_velocity or frequency_strategy!="independent"):
            raise NotImplementedError("group velocity and frequency continuation require an adaptive solver")
        self.group_velocity=bool(group_velocity);self.group_rtol=float(group_rtol)
        self.frequency_strategy=frequency_strategy;self.anchor_stride=int(anchor_stride)
        if not isinstance(precision, str) or precision not in ("float64", "float32"):
            raise ValueError("precision must be float64 or float32")
        if precision == "float32" and (backend == "cpu" or search_strategy == "uniform"):
            raise ValueError("precision='float32' requires an adaptive CUDA solver")
        self.precision = precision
        if secular not in (None,"dunkin","fast"):raise ValueError("secular must be None, 'dunkin' or 'fast'")
        if refinement is None:refinement="legacy" if search_strategy=="uniform" else "brent"
        if refinement not in ("legacy","brent","chandrupatla"):raise ValueError("refinement must be legacy, brent or chandrupatla")
        if love_strategy not in ("phase","scan"):raise ValueError("love_strategy must be phase or scan")
        if search_strategy=="uniform" and refinement!="legacy":raise NotImplementedError("uniform supports only legacy refinement")
        self.refinement=refinement;self.love_strategy=love_strategy
        if secular=="fast" and search_strategy=="uniform":
            raise NotImplementedError("secular='fast' is not available on the legacy uniform CUDA path")
        if accuracy is None or (accuracy=="auto" and search_strategy=="uniform"):
            accuracy=DEFAULT_ACCURACY if search_strategy!="uniform" else "uniform"
        elif accuracy not in ("auto","high","balanced","fast"):
            raise ValueError("accuracy must be auto, high, balanced or fast")
        elif search_strategy=="uniform":
            raise ValueError("accuracy applies only to adaptive strategies; uniform is the legacy fixed path")
        if search_strategy=="uniform":accuracy="uniform"
        if adaptive_period_threshold is not None:
            if accuracy in ("high","uniform"):raise ValueError("adaptive_period_threshold applies only to auto/balanced/fast")
            if not np.isfinite(adaptive_period_threshold) or adaptive_period_threshold<=0:raise ValueError("adaptive_period_threshold must be finite and positive")
        if not np.isscalar(close_root_rtol) or np.iscomplexobj(close_root_rtol) or not np.isfinite(close_root_rtol) or not 0<close_root_rtol<1:raise ValueError("close_root_rtol must be finite and between 0 and 1")
        if sampling_rtol is not None and search_strategy!="shape_adaptive":raise ValueError("sampling_rtol applies only to shape_adaptive")
        self._auto_sampling=(accuracy=="auto" and search_strategy=="shape_adaptive" and sampling_rtol is None)
        if sampling_rtol is None:sampling_rtol={"auto":1e-2,"high":1e-3,"balanced":1e-2,"fast":5e-2,"uniform":1e-3}[accuracy]
        if not np.isscalar(sampling_rtol) or np.iscomplexobj(sampling_rtol) or not np.isfinite(sampling_rtol) or not 0<sampling_rtol<=1:raise ValueError("sampling_rtol must be finite and between 0 and 1")
        if c_tol_rel is None:c_tol_rel={"auto":1e-5,"high":1e-6,"balanced":1e-5,"fast":1e-4,"uniform":1e-6}[accuracy]
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
        self.profile=bool(profile);self.sampling_rtol=float(sampling_rtol)
        self.accuracy=accuracy;self.adaptive_period_threshold=adaptive_period_threshold;self.close_root_rtol=float(close_root_rtol);self.secular=secular
        self._engine=None;self._engine_key=None;self._key=None;self._module=None;self._legacy=None;self._calls=0;self._last_stats=None
        self.last_result=None
        self._lock=threading.RLock()

    def _resolved_secular(self,backend):
        if self.secular is not None:return self.secular
        if self.search_strategy=="uniform" or backend=="cpu":return "dunkin"
        if backend=="cuda":return "fast"
        return None

    def _select(self):
        if self.backend=="cpu":return "cpu","explicit CPU selection"
        try:
            module=_load_cuda()
            if self.precision == "float32" and not getattr(module, "SUPPORTS_FP32", False):
                raise BackendUnavailableError("precision='float32' requires an FP32-capable CUDA plugin")
            if self.accuracy=="auto" and not getattr(module,"SUPPORTS_AUTO_ACCURACY",False):
                raise BackendUnavailableError("accuracy='auto' requires a matching CUDA plugin with automatic sampling support")
            if not module.available(self.device):raise BackendUnavailableError(f"CUDA device {self.device} is unavailable")
            try:module.set_device(self.device)
            except RuntimeError as exc:raise BackendUnavailableError(f"CUDA device {self.device} initialization failed: {exc}") from exc
            self._module=module
            return "cuda","CUDA plugin and device available"
        except BackendUnavailableError as exc:
            if self.backend=="cuda" or self.precision=="float32":raise
            return "cpu",str(exc)

    def __call__(self,thickness,vp,vs,rho,periods,nmodes=5,wave="rayleigh",names=None):
        if not self._lock.acquire(blocking=False):raise RuntimeError("a Solver instance cannot be used concurrently; create one instance per caller")
        try:return self._solve(thickness,vp,vs,rho,periods,nmodes,wave,names)
        finally:self._lock.release()

    def _solve(self,thickness,vp,vs,rho,periods,nmodes=5,wave="rayleigh",names=None,*,_bounds=None,_reserved=0,_owner=None,_row_periods=None):
        started=time.perf_counter() if self.profile else 0.
        timings={}
        del names
        if wave not in ("rayleigh","love"):raise ValueError("wave must be rayleigh or love")
        if not isinstance(nmodes,(int,np.integer)) or not 1<=nmodes<=256:raise ValueError("nmodes must be 1..256")
        h,a,b,r=_models(thickness,vp,vs,rho);B,L=h.shape
        if np.iscomplexobj(periods):raise ValueError("periods must be real")
        per=np.ascontiguousarray(periods,dtype=np.float64)
        if per.ndim!=1 or not per.size or not np.isfinite(per).all() or np.any(per<=0):raise ValueError("periods must be a finite positive 1-D array")
        if self.precision == "float32":
            fp32 = np.finfo(np.float32)
            max_value, zero_cutoff = float(fp32.max), .5 * float(fp32.smallest_subnormal)
            for name, values in zip(("thickness", "vp", "vs", "rho"), (h, a, b, r)):
                if values.max() > max_value or np.any((values > 0.) & (values <= zero_cutoff)):
                    raise ValueError(f"{name} contains values outside the float32 range")
            if per.max() > max_value or per.min() < 2. * np.pi / max_value:
                raise ValueError("periods or their angular frequencies are outside the float32 range")
        if self.profile:timings['input_validation']=time.perf_counter()-started
        selected=time.perf_counter() if self.profile else 0.
        backend,reason=self._select()
        if backend=="cpu" and self.accuracy=="auto" and not getattr(_cpu,"SUPPORTS_AUTO_ACCURACY",False):
            raise RuntimeError("accuracy='auto' requires a matching CPU extension with automatic sampling support")
        secular=self._resolved_secular(backend)
        if self.profile:timings['backend_selection']=time.perf_counter()-selected
        if self.search_strategy=="uniform":
            if secular=="fast":raise NotImplementedError("secular='fast' is not available on the legacy uniform CUDA path")
            if backend!="cuda" or wave!="rayleigh" or np.any(b==0) or L<2:raise NotImplementedError("uniform is the legacy CUDA solid-Rayleigh option; use qe_adaptive for this combination")
            if self._legacy is None:
                from ._legacy import Solver as LegacySolver
                self._legacy=LegacySolver(self.nc,.2 if self.cmin is None else self.cmin,self.cmax,self.c_tol_rel,self.f_accept,self.block,search_strategy="uniform")
            old=self._legacy(h,a,b,r,per,nmodes,wave)
            bounds=_cpu.bounds(h,a,b,r,0,.2 if self.cmin is None else self.cmin,self.cmax)
            flags=old.trace_flags.copy()|TRACE_RANGE_TRUNCATED
            result=DispersionBatch(old.phase_velocity,per,nmodes,wave,flags,np.full((B,per.size),-2,dtype=np.int32),np.sum(old.mode_mask,axis=1,dtype=np.int32),bounds,backend,reason,accuracy="uniform",close_root_rtol=self.close_root_rtol,search_strategy="uniform",secular="dunkin",refinement="legacy")
            self._last_stats=dict(self._legacy.stats(),accuracy="uniform",precision=result.precision,close_root_rtol=self.close_root_rtol,secular="dunkin",secular_kernel="dunkin",refinement="legacy",love_strategy=None,group_velocity=False,group_rtol=self.group_rtol,frequency_strategy="independent",frequency_strategy_used="independent",frequency_strategy_reason=result.frequency_strategy_reason,anchor_stride=self.anchor_stride);self._calls+=1;self.last_result=result
            return result
        input_periods = per
        unique_periods = per
        inverse = None
        # CUDA high FP32 budgets its FP64 repairs by the native trace count.
        preserve_grid = backend == "cuda" and self.precision == "float32" and self.accuracy == "high"
        if not preserve_grid and per.size > 1 and not np.all(per[1:] > per[:-1]):
            needs_sorted = (self.frequency_strategy == "anchor_continuation" and
                            self.accuracy != "high" and per.size >= 8)
            if needs_sorted or not np.all(per[1:] < per[:-1]):
                unique_periods, inverse = np.unique(per, return_inverse=True)
        continuation=False
        frequency_reason="independent frequency search requested"
        if self.frequency_strategy=="anchor_continuation":
            if self.accuracy=="high":frequency_reason="high accuracy retains independent search"
            elif backend=="cpu" and not getattr(_cpu,"SUPPORTS_CONTINUATION",False):
                frequency_reason="CPU extension lacks continuation support"
            else:
                if unique_periods.size<8:frequency_reason="fewer than 8 unique periods; using independent search"
                elif not np.any(np.log(unique_periods[2:])-np.log(unique_periods[:-2])<=.25):
                    frequency_reason="period grid has no interior target within the anchor log-gap limit; using independent search"
                else:
                    continuation=True
                    frequency_reason=f"{backend.upper()} anchor continuation on sorted unique periods"
        if continuation or unique_periods.size < per.size:
            per = unique_periods
        else:
            inverse = None  # Keep the caller's order for independent, all-unique grids.
        frequency_used="anchor_continuation" if continuation else "independent"
        stage=time.perf_counter() if self.profile else 0.
        bounds=_cpu.bounds(h,a,b,r,int(wave=="love"),self.cmin,self.cmax) if _bounds is None else _bounds
        if self.profile:timings['bounds']=time.perf_counter()-stage
        if not np.isfinite(bounds).all() or np.any(bounds[:,0]<=0):raise ValueError("automatic search range could not be constructed")
        shape_queue_bytes=5632 if backend=="cuda" and self.search_strategy=="shape_adaptive" else 0
        if backend=="cpu":
            fixed=self.threads*(4*2048*16+5*(nmodes+2)*8+3*L*8
                                +(72*50 if self.search_strategy=="shape_adaptive" else 0))+4096
            cost=8*(6*L+2*(129+2*L)+(6*L+174 if self.search_strategy=="shape_adaptive" else 0))+2*L*8+per.size*(8*nmodes+12+16)
        else:
            fixed=4096+8*per.size
            cache_points=256
            # Conservatively reserve a compact shared queue for every trace,
            # even though only the resident persistent blocks use it at once.
            cost=8*(6*L+2*1024+4)+2*L*8+(8*2048+4 if self.search_strategy=="shape_adaptive" else 0)+4+per.size*(64*(nmodes+2)+8*nmodes+48+2*cache_points*8+16+shape_queue_bytes)
            if self.accuracy=="high" and self.precision=="float64" and per.size==1:
                # The first period is already in the shared-grid budget.
                fixed-=16
                cost+=16  # One pinned and one device double per verification row.
            if self.precision == "float32":
                cost += (6 * L + 1) * np.dtype(np.float32).itemsize
        if self.group_velocity:cost+=per.size*nmodes*(24 if backend=="cuda" else 28)
        if continuation:cost+=per.size*(16*(nmodes+2)+8+(24*L if backend=="cuda" else 0))
        budget=self.workspace_bytes
        if _owner is not None and _owner._engine is not None:
            if budget-_reserved >= fixed+cost:
                budget-=_reserved
            else:
                # Share the caller's native workspace budget with compact FP64 review.
                _owner.close()
        limit=(budget-fixed)//cost
        capacity=min(B,limit,self.chunk_size or B)
        if capacity<1:raise MemoryError("workspace_bytes is too small for one model and the backend's fixed workspace; increase the budget or reduce threads")
        threshold=0. if self.adaptive_period_threshold is None else float(self.adaptive_period_threshold)
        resolved_threshold=(float(self.adaptive_period_threshold) if self.adaptive_period_threshold is not None
                            else (1.0 if self.accuracy in ("balanced","auto") else (0.63 if self.accuracy=="fast" else 0.0)))
        native_sampling=(-1. if self._auto_sampling and self.sampling_rtol==.01 else self.sampling_rtol)
        key=(backend,self.device,self.threads,capacity,L,per.size,int(nmodes),self.accuracy,threshold,self.close_root_rtol,self.search_strategy,native_sampling,secular,self.refinement,self.love_strategy,self.precision,self.group_velocity,self.group_rtol,self.frequency_strategy,continuation,self.anchor_stride)
        # Preserve the high-water capacity for short tail batches. Never retain
        # an allocation that exceeds a subsequently reduced budget/chunk cap.
        compatible=(self._engine_key is not None and self._engine_key[:3]==key[:3]
                    and self._engine_key[4:]==key[4:] and capacity<=self._engine_key[3]
                    <=min(limit,self.chunk_size or limit))
        allocated=not compatible
        stage=time.perf_counter() if self.profile else 0.
        if allocated:
            self.close()
            code=_ACCURACY_CODE[self.accuracy]
            secular_code=0 if secular=="dunkin" else 1
            refinement_code={"legacy":0,"brent":1,"chandrupatla":2}[self.refinement]
            phase_code=int(self.love_strategy=="phase")
            config = (code, threshold, self.close_root_rtol, self.search_strategy == "shape_adaptive",
                      native_sampling, secular_code, refinement_code, phase_code)
            if backend == "cpu":
                self._engine = _cpu.CpuSolver(self.threads, *config)
            else:
                self._engine = self._module.QeSolver(capacity, L, per.size, int(nmodes), *config,
                                                    self.precision=="float32",self.group_velocity,self.group_rtol,
                                                    continuation,self.anchor_stride)
            self._engine_key=key
        else:capacity=self._engine_key[3]
        if self.profile:timings['engine_prepare']=time.perf_counter()-stage
        stage=time.perf_counter() if self.profile else 0.
        velocity=np.empty((B,nmodes,per.size),dtype=self.precision);flags=np.empty((B,per.size),dtype=np.uint32)
        levels=np.empty((B,per.size),dtype=np.int32);counts=np.empty_like(levels);stats={};chunks=0
        group_velocity=np.empty(velocity.shape,dtype=np.float64) if self.group_velocity else None
        group_flags=np.empty(velocity.shape,dtype=np.uint32) if self.group_velocity else None
        prediction_flags=(np.full((B,per.size),int(PredictionFlag.BYPASSED),dtype=np.uint32)
                          if self.frequency_strategy=="anchor_continuation" else None)
        canonical_lower = (True,) if (self.precision == "float32" and self.cmin is None and
                                      getattr(self._module, "SUPPORTS_CANONICAL_LOWER", False)) else ()
        if self.profile:timings['output_allocate']=time.perf_counter()-stage
        for start in range(0,B,capacity):
            end=min(B,start+capacity)
            native_periods=per if _row_periods is None else _row_periods[start:end,None]
            args=(h[start:end],a[start:end],b[start:end],r[start:end],native_periods,bounds[start:end])
            if backend=="cpu":
                lower_arg = (True,) if self.cmin is None and getattr(_cpu, "SUPPORTS_CANONICAL_LOWER", False) else ()
                if continuation:lower_arg=(self.cmin is None,True,self.anchor_stride)
                out=self._engine.run(*args,int(nmodes),int(wave=="love"),self.cmin is None,self.c_tol_rel,self.f_accept,self.profile,self.group_velocity,self.group_rtol,*lower_arg)
            else:out=self._engine.run(*args,self.c_tol_rel,self.f_accept,int(wave=="love"),self.cmin is None,self.profile,*canonical_lower)
            stage=time.perf_counter() if self.profile else 0.
            velocity[start:end],flags[start:end],part,levels[start:end],counts[start:end],extras=out
            if self.group_velocity:
                group_velocity[start:end]=extras["group_velocity"]
                group_flags[start:end]=extras["group_velocity_flags"]
            if continuation:
                chunk_prediction=extras["prediction_flags"]
                prediction_flags[start:end]=chunk_prediction
                for name,bit in (("root_count_unsafe",128),("anchor_unsafe",256),
                                 ("window_unsafe",512),("bracket_failed",1024),
                                 ("correction_failed",2048),("audit_failed",32)):
                    counter="continuation_fallback_"+name
                    stats[counter]=stats.get(counter,0)+int(np.count_nonzero(chunk_prediction&bit))
            if self.profile:timings['chunk_assembly']=timings.get('chunk_assembly',0.)+time.perf_counter()-stage
            for k,v in part.items():
                if k=='timings_s':
                    for name,seconds in v.items():timings[name]=timings.get(name,0.)+float(seconds)
                elif k=='active_traces_by_stage':
                    stats[k]=[x+int(y) for x,y in zip(stats.get(k,[0]*len(v)),v)]
                elif k in ("density_table_knots","cache_capacity","pinned_host_bytes","device_workspace_bytes","sampling_capacity","dense_floor4","dense_floor5","dense_tile"):stats[k]=int(v)
                elif isinstance(v,(int,np.integer)) and k not in ("threads","capacity"):stats[k]=stats.get(k,0)+int(v)
            chunks+=1
        continuity_flags = None
        if self.accuracy in _APPROXIMATE_ACCURACIES and (self.continuity_check or continuation or self.accuracy=="auto"):
            stage=time.perf_counter() if self.profile else 0.
            continuity_flags, suspect = _cpu.continuity_screen(per,velocity,prediction_flags,self.continuity_check)
            if self.accuracy=="auto":
                search_failure = (flags & (TRACE_UNRESOLVED_BRACKET | TRACE_SEARCH_UNCONVERGED | TRACE_ROOT_CONFLICT))!=0
                if np.any(search_failure):
                    continuity_flags[search_failure] |= int(ContinuityFlag.SEARCH)
                    suspect = True
            stats.update(continuity_suspected=0,continuity_local_searches=0,continuity_full_searches=0,
                         continuity_repaired=0,continuity_unresolved=0,continuity_evaluations=0)
            if suspect:
                _continuity.review(self,backend,(h,a,b,r),per,wave,bounds,velocity,flags,levels,counts,
                                   group_velocity,group_flags,continuity_flags,stats,fixed+capacity*cost)
            if self.profile:timings["continuity"]=time.perf_counter()-stage
        if self.accuracy=="auto":
            stats["auto_relaxed_traces"]=int(np.count_nonzero(flags & TRACE_AUTO_RELAXED))
            stats["sampling_rtol_max"]=None if self.search_strategy!="shape_adaptive" else (.05 if native_sampling<0 else native_sampling)
        computed_periods=per.size
        if inverse is not None:
            velocity = velocity.take(inverse, axis=-1)
            flags = flags.take(inverse, axis=-1)
            levels = levels.take(inverse, axis=-1)
            counts = counts.take(inverse, axis=-1)
            if group_velocity is not None:
                group_velocity = group_velocity.take(inverse, axis=-1)
                group_flags = group_flags.take(inverse, axis=-1)
            if prediction_flags is not None:
                prediction_flags = prediction_flags.take(inverse, axis=-1)
            if continuity_flags is not None:
                continuity_flags = continuity_flags.take(inverse, axis=-1)
        if self.cmin is not None or self.cmax is not None:flags|=TRACE_RANGE_TRUNCATED
        counts[(flags&TRACE_SEARCH_UNCONVERGED)!=0]=-1
        love_phase_tolerance = None
        if wave == "love" and self.love_strategy == "phase":
            love_phase_tolerance = min(self.c_tol_rel, 1e-8)
            if self.precision == "float32":
                love_phase_tolerance = max(8. * float(np.finfo(np.float32).eps), love_phase_tolerance)
        stats.update(backend=backend,precision=self.precision,strategy=("shape_adaptive" if self.search_strategy=="shape_adaptive" else ("qe_adaptive" if self.accuracy=="high" else "observation_log")),requested_strategy=self.search_strategy,sampling_rtol=self.sampling_rtol,accuracy=self.accuracy,secular=secular,secular_kernel=("sh" if wave=="love" else "dunkin_fast" if secular=="fast" else "dunkin"),
                     refinement=self.refinement,love_strategy=self.love_strategy if wave=="love" else None,love_phase_tolerance=love_phase_tolerance,
                     period_threshold=resolved_threshold,early_acceptance_enabled=False,approximate_search_enabled=self.accuracy in _APPROXIMATE_ACCURACIES,velocity_step_rtol=(None if self.search_strategy=="shape_adaptive" else self.close_root_rtol*(.5 if self.accuracy in ("balanced","auto") else 1.) if self.accuracy!="high" else None),close_root_rtol=self.close_root_rtol,refinement_tolerance=self.c_tol_rel,chunks=chunks,threads=self.threads if backend=="cpu" else 0)
        stats.update(approximate_traces=int(np.count_nonzero(flags&TRACE_APPROXIMATE_SEARCH)),close_root_risk_traces=int(np.count_nonzero(flags&TRACE_CLOSE_ROOT_RISK)))
        stats['close_cluster_leaf_rtol']=self.close_root_rtol*(.5 if self.accuracy in ("balanced","auto") else 1.) if self.search_strategy=="shape_adaptive" and self.accuracy!="high" else None
        stats.update(workspace_bytes=int(fixed+capacity*cost),workspace_budget_bytes=self.workspace_bytes)
        stats['shape_queue_reserved_bytes']=int(shape_queue_bytes*capacity*per.size)
        stats.update(capacity_models=int(capacity),engine_allocations=int(allocated))
        stats.update(group_velocity=self.group_velocity,group_rtol=self.group_rtol,
                     frequency_strategy=self.frequency_strategy,frequency_strategy_used=frequency_used,
                     frequency_strategy_reason=frequency_reason,anchor_stride=self.anchor_stride,
                     periods_requested=int(input_periods.size),periods_computed=int(computed_periods),
                     continuity_check=self.continuity_check and self.accuracy in _APPROXIMATE_ACCURACIES)
        attempts=stats.get("continuation_accepted",0)+stats.get("continuation_fallbacks",0)
        stats["continuation_local_success_rate"]=(stats.get("continuation_accepted",0)/attempts if attempts else None)
        for k in ("rejected_sign","troughs","hidden_roots","relocated"):stats.setdefault(k,0)
        self._last_stats=stats;self._key=(B,L,per.size,self.nc,nmodes);self._calls+=1
        stage=time.perf_counter() if self.profile else 0.
        result=DispersionBatch(velocity,input_periods,nmodes,wave,flags,levels,counts,bounds,backend,reason,accuracy=self.accuracy,close_root_rtol=self.close_root_rtol,search_strategy=self.search_strategy,sampling_rtol=self.sampling_rtol if self.search_strategy=="shape_adaptive" else None,secular=secular,refinement=self.refinement,love_strategy=self.love_strategy if wave=="love" else None,group_velocity=group_velocity,group_velocity_flags=group_flags,prediction_flags=prediction_flags,group_rtol=self.group_rtol if self.group_velocity else None,frequency_strategy=self.frequency_strategy,frequency_strategy_used=frequency_used,frequency_strategy_reason=frequency_reason,anchor_stride=self.anchor_stride,continuity_flags=continuity_flags,continuity_unresolved=stats.get('continuity_unresolved',0))
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
        secular=self.last_result.secular if self.last_result is not None else self._resolved_secular(backend)
        precision = self.last_result.precision if self.last_result is not None else self.precision
        precision_args = (True,) if precision == "float32" else ()
        if backend=="cuda" and self.search_strategy=="uniform":return {"backend":"cuda","precision":"float32 scan / float64 refinement","accuracy":"uniform","secular":"dunkin","secular_kernel":"dunkin","refinement":"legacy","love_strategy":None,"group_velocity":False,"frequency_strategy":"independent","frequency_strategy_used":"independent","legacy":self._legacy.kernel_report()}
        secular_kernel=("sh" if self.last_result is not None and self.last_result.wave=="love"
                        else "dunkin_fast" if secular=="fast" else "dunkin" if secular=="dunkin" else None)
        resolved=(float(self.adaptive_period_threshold) if self.adaptive_period_threshold is not None
                  else (1.0 if self.accuracy in ("balanced","auto") else (0.63 if self.accuracy=="fast" else 0.0)))
        def actual(report):
            if self.last_result is not None and self.last_result.wave=="love" and self.love_strategy=="phase":
                if backend=="cuda" and self._key is not None:
                    report["scan_kernel_report"]=dict(report)
                    report.update(self._module.love_phase_kernel_report(self._key[1], *precision_args))
                report["scan_kernel"]=report.get("search_kernel")
                report["search_kernel"]="love_phase_with_scan_fallback"
                report["love_phase_accepted"]=self._last_stats.get("love_phase_accepted",0)
                report["love_phase_fallbacks"]=self._last_stats.get("love_phase_fallbacks",0)
            return report
        report = dict(backend=backend, precision=precision, accuracy=self.accuracy, secular=secular,
                      secular_kernel=secular_kernel, period_threshold=resolved, close_root_rtol=self.close_root_rtol,
                      refinement_tolerance=self.c_tol_rel, refinement=self.refinement, love_strategy=self.love_strategy,
                      group_velocity=self.group_velocity,group_rtol=self.group_rtol,
                      frequency_strategy=self.frequency_strategy,anchor_stride=self.anchor_stride)
        if self.accuracy=="auto":
            report["sampling_rtol_max"]=(None if self.search_strategy!="shape_adaptive" else
                                         .05 if self._auto_sampling and self.sampling_rtol==.01 else self.sampling_rtol)
        if self.last_result is not None:
            report.update(frequency_strategy_used=self.last_result.frequency_strategy_used,
                          frequency_strategy_reason=self.last_result.frequency_strategy_reason)
        if backend == "cuda" and self._key is not None:
            report.update(self._module.qe_kernel_report(self._key[1], int(self.accuracy != "high"),
                                                        self.search_strategy == "shape_adaptive", *precision_args))
            report.update(sampling_capacity=2048, cache_points_per_trace=256)
        else:
            search_kernel = "qe_adaptive" if self.accuracy == "high" else "observation_log"
            report.update(search_kernel="shape_adaptive" if self.search_strategy == "shape_adaptive" else search_kernel,
                          threads=self.threads, module=_cpu.__file__)
        return actual(report)
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
        if accuracy=="auto" and solver.search_strategy=="uniform":
            accuracy="uniform"
        if (options or (search_strategy is not None and search_strategy!=solver.search_strategy)
                or (accuracy is not None and accuracy!=solver.accuracy)):
            raise ValueError("options conflict with the supplied Solver")
        return solver(thickness,vp,vs,rho,periods,nmodes,wave,names)
    if options or accuracy is not None:
        with Solver(search_strategy="shape_adaptive" if search_strategy is None else search_strategy,
                    accuracy=accuracy,**options) as s:return s(thickness,vp,vs,rho,periods,nmodes,wave,names)
    strategy="shape_adaptive" if search_strategy is None else search_strategy
    expected=DEFAULT_ACCURACY if strategy!="uniform" else "uniform"
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
