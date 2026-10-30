"""Independent, conservative cross-period checks; no physical branch identity claim."""
from dataclasses import dataclass
from enum import IntFlag
import numpy as np


def _log_gap(a,b):
    # Preserve distinct adjacent FP64 periods even when log(a)==log(b).
    return float(np.log1p((b-a)/a)) if b-a<a else float(np.log(b)-np.log(a))


class ModeReliabilityFlag(IntFlag):
    UNASSESSED = 1
    NUMERICAL_FAILURE = 2
    ROOT_COUNT_CHANGE = 4
    CLOSE_ROOT = 8
    RANK_AMBIGUOUS = 16
    STEP_JUMP = 32
    PREDICTION_MISMATCH = 64
    DUPLICATE_MISMATCH = 128
    NONFINITE = 256
    PERIOD_GAP = 512
    SEARCH_UNCERTAIN = 1024


@dataclass(frozen=True)
class ModeReliability:
    flags: np.ndarray              # [B,K,T], original period order
    assessed: np.ndarray           # a finite, sufficiently sampled incoming edge
    previous_index: np.ndarray     # [T], original index of predecessor
    sorted_order: np.ndarray       # [T], stable period sort

    @property
    def suspect(self):
        return self.flags != 0

    @property
    def passes_checks(self):
        """Passed observed checks, not certified modal identity/completeness."""
        return self.assessed & ~self.suspect

    def summary(self):
        return dict(cells=int(self.flags.size), assessed=int(self.assessed.sum()),
                    passes_checks=int(self.passes_checks.sum()),
                    flags={f.name: int(np.count_nonzero(self.flags & int(f)))
                           for f in ModeReliabilityFlag})


def check_mode_reliability(result, periods=None, *, trace_flags=None, root_count=None,
                           close_root_rtol=None, max_relative_step=.1,
                           prediction_rtol=.05, max_log_period_gap=.25):
    """Check recovered ranks across sorted periods without editing the solution.

    Pass a DispersionBatch, or velocity[B,K,T] plus periods. Thresholds are
    heuristics. A smooth wrong/missing branch can pass; eigenfunctions and an
    independent root-count reference are not available to this checker.
    Duplicates are compared separately; large gaps and the first period are
    explicitly unassessed. Output indexing always follows the input order.
    """
    if hasattr(result, 'phase_velocity'):
        if periods is not None or trace_flags is not None or root_count is not None:
            raise ValueError('periods/trace_flags/root_count are supplied by the result')
        periods=result.period;trace_flags=result.trace_flags;root_count=result.root_count
        if close_root_rtol is None:close_root_rtol=result.close_root_rtol
        velocity=result.phase_velocity
    else:
        velocity=result
    if close_root_rtol is None:close_root_rtol=.005
    if np.iscomplexobj(velocity) or np.iscomplexobj(periods):
        raise ValueError('velocities and periods must be real')
    v=np.asarray(velocity,dtype=float);p=np.asarray(periods,dtype=float)
    if v.ndim!=3 or min(v.shape)<1 or p.ndim!=1 or len(p)!=v.shape[2] or not np.isfinite(p).all() or np.any(p<=0):
        raise ValueError('expected velocity [B,K,T] and finite positive periods [T]')
    if np.isinf(v).any() or np.any(v[np.isfinite(v)]<=0):
        raise ValueError('velocities must be positive finite values or NaN')
    for name,x in [('close_root_rtol',close_root_rtol),('max_relative_step',max_relative_step),
                   ('prediction_rtol',prediction_rtol),('max_log_period_gap',max_log_period_gap)]:
        if not np.isscalar(x) or np.iscomplexobj(x) or not np.isfinite(x) or x<=0:
            raise ValueError(f'{name} must be finite and positive')
    B,K,T=v.shape;finite=np.isfinite(v)
    if trace_flags is None:tf=np.zeros((B,T),dtype=np.uint32)
    else:
        tf=np.asarray(trace_flags)
        if tf.shape!=(B,T) or tf.dtype.kind not in 'iu' or np.any(tf<0):
            raise ValueError('trace_flags must be nonnegative integer [B,T]')
    if root_count is None:counts=finite.sum(axis=1)
    else:
        counts=np.asarray(root_count)
        if counts.shape!=(B,T) or counts.dtype.kind not in 'iu':
            raise ValueError('root_count must be integer [B,T]')
    F=ModeReliabilityFlag;flags=np.zeros(v.shape,dtype=np.uint16)
    assessed=np.zeros(v.shape,dtype=bool);previous=np.full(T,-1,dtype=np.int64)
    flags[~finite]|=int(F.NONFINITE)
    failure=((tf & (1|4|8|32|64|128|512|1024))!=0) | (counts<0)
    flags|=np.where(failure[:,None,:],int(F.NUMERICAL_FAILURE),0).astype(np.uint16)
    flags|=np.where(((tf & (2048|4096))!=0)[:,None,:],int(F.SEARCH_UNCERTAIN),0).astype(np.uint16)
    close=np.zeros(v.shape,dtype=bool)
    if K>1:
        gap=np.diff(v,axis=1)
        c=np.isfinite(gap)&(gap<=close_root_rtol*v[:,1:,:])
        close[:,:-1,:]|=c;close[:,1:,:]|=c
        # Sorting violations are not silently repaired.
        bad=np.isfinite(gap)&(gap<=0)
        flags[:,:-1,:]|=np.where(bad,int(F.RANK_AMBIGUOUS),0).astype(np.uint16)
        flags[:,1:,:]|=np.where(bad,int(F.RANK_AMBIGUOUS),0).astype(np.uint16)
    flags[close]|=int(F.CLOSE_ROOT)
    order=np.argsort(p,kind='stable');unique=[]
    for idx in order:
        if not unique:
            flags[:,:,idx]|=int(F.UNASSESSED);unique.append(int(idx));continue
        prev=unique[-1];previous[idx]=prev
        a=v[:,:,prev];b=v[:,:,idx];both=finite[:,:,prev]&finite[:,:,idx]
        changed=(counts[:,prev]!=counts[:,idx]) | np.any(finite[:,:,prev]!=finite[:,:,idx],axis=1)
        flags[:,:,idx]|=np.where(changed[:,None],int(F.ROOT_COUNT_CHANGE),0).astype(np.uint16)
        if p[idx]==p[prev]:
            mismatch=(finite[:,:,prev]!=finite[:,:,idx]) | (both & (np.abs(a-b)>1e-10*np.maximum(a,b)))
            flags[:,:,idx][mismatch]|=int(F.DUPLICATE_MISMATCH)
            flags[:,:,prev][mismatch]|=int(F.DUPLICATE_MISMATCH)
            assessed[:,:,idx]=both & ~failure[:,prev,None] & ~failure[:,idx,None]
            continue
        log_gap=_log_gap(p[prev],p[idx])
        if log_gap>max_log_period_gap:
            flags[:,:,idx]|=int(F.PERIOD_GAP)
        else:
            assessed[:,:,idx]=both & ~failure[:,prev,None] & ~failure[:,idx,None]
        step=np.divide(np.abs(b-a),np.maximum(a,b),out=np.zeros_like(a),where=both)
        flags[:,:,idx]|=np.where(both & (step>max_relative_step),int(F.STEP_JUMP),0).astype(np.uint16)
        # Nearest-rank disagreement is evidence of ambiguity, never a remapping.
        for model in range(B):
            ia=np.flatnonzero(finite[model,:,prev]);ib=np.flatnonzero(finite[model,:,idx])
            if not len(ia) or not len(ib):continue
            cost=np.abs(a[model,ia][:,None]-b[model,ib][None,:])
            nearest=ia[np.argmin(cost,axis=0)];reverse=ib[np.argmin(cost,axis=1)]
            ambiguous=(nearest!=ib) | (reverse[np.argmin(cost,axis=0)]!=ib)
            flags[model,ib[ambiguous],idx]|=int(F.RANK_AMBIGUOUS)
        if len(unique)>=2:
            older=unique[-2];den=_log_gap(p[older],p[prev])
            if den<=max_log_period_gap and log_gap<=max_log_period_gap:
                valid=both&finite[:,:,older]
                with np.errstate(invalid='ignore'):
                    predicted=a+(a-v[:,:,older])*(log_gap/den)
                    mismatch=valid & (np.abs(b-predicted)>prediction_rtol*b)
                flags[:,:,idx][mismatch]|=int(F.PREDICTION_MISMATCH)
        unique.append(int(idx))
    return ModeReliability(flags,assessed,previous,order)
