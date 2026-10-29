#!/Users/yusong/miniforge3/bin/python
"""Compare both search strategies with the same frozen independent archives.

The retained high path supplies only optional prefix-boundary guards. It never
replaces an independent reference cell. Unstable rows stay explicitly unjudged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import modisp as md
from tolerant_metrics import compare
from validate_shape_paper import strict_compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--references',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--threads',type=int,default=8)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cases=[]
    for path in sorted(a.references.glob('*.npz')):
        with np.load(path,allow_pickle=False) as z:
            model={k:z[k] for k in ['thickness','vp','vs','rho']}
            periods=z['periods'];reference=z['reference'];stable=z['reference_stable'].astype(bool)
            count_hi=z['reference_count_hi']
        if stable.shape==(reference.shape[0],):stable=np.broadcast_to(stable[:,None],(len(stable),len(periods)))
        if stable.shape!=(reference.shape[0],reference.shape[2]):raise ValueError((path,stable.shape))
        low=float(min(.2,.75*np.min(model['vs']))) if path.stem.startswith('qed_') else .2
        options=dict(backend=a.backend,threads=a.threads,cmin=low,workspace_bytes=1024**3)
        with md.Solver(accuracy='high',search_strategy='qe_adaptive',**options) as s:
            guard=s(**model,periods=periods,nmodes=8)
        extended=np.full((len(reference),8,len(periods)),np.nan);extended[:,:reference.shape[1]]=reference
        agree=np.all(np.isclose(guard.phase_velocity[:,:reference.shape[1]],reference,rtol=2e-6,atol=3e-6,equal_nan=True),axis=1)
        for b,t in np.argwhere(agree):extended[b,reference.shape[1]:,t]=guard.phase_velocity[b,reference.shape[1]:,t]
        complete=np.asarray(count_hi)<=np.isfinite(extended).sum(axis=1)
        case=dict(suite=path.stem,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),models=len(reference),
                  total_traces=int(stable.size),stable_traces=int(stable.sum()),unjudged_traces=int((~stable).sum()),
                  guard_prefix_disagree=int((~agree).sum()),results={})
        # Flatten only the judged rows; retain their original mask in raw output.
        ref=extended.transpose(0,2,1)[stable].T[None,:,:]
        complete_j=complete[stable][None,:]
        for strategy in ['qe_adaptive','shape_adaptive']:
            case['results'][strategy]={}
            for accuracy,budget in [('high',2e-6),('balanced',2e-5),('fast',2e-4)]:
                with md.Solver(accuracy=accuracy,search_strategy=strategy,**options) as s:
                    result=s(**model,periods=periods,nmodes=reference.shape[1]);stats=s.stats()
                velocity=result.phase_velocity.transpose(0,2,1)[stable].T[None,:,:]
                metric=compare(velocity,ref,result.trace_flags[stable][None,:],0. if accuracy=='high' else .005,
                               budget,reference_complete=complete_j)
                strict_missing=strict_spurious=unknown=0
                for b,t in np.argwhere(stable):
                    r=reference[b,:,t];r=r[np.isfinite(r)]
                    x=result.phase_velocity[b,:,t];x=x[np.isfinite(x)]
                    known=x if count_hi[b,t]<=len(r) else x[x<=r[-1]*(1+budget)] if len(r) else np.array([])
                    strict=strict_compare(known,r,budget)
                    strict_missing+=strict['missing'];strict_spurious+=strict['spurious'];unknown+=len(x)-len(known)
                case['results'][strategy][accuracy]=dict(metrics=metric,strict_missing=strict_missing,
                    strict_spurious=strict_spurious,unknown_beyond_independent_prefix=unknown,stats=stats)
                np.savez_compressed(a.output/f'{path.stem}-{strategy}-{accuracy}.npz',
                    velocity=result.phase_velocity,flags=result.trace_flags,reference=reference,stable=stable)
        cases.append(case);print('reference',path.stem,'stable',int(stable.sum()),flush=True)
    if len(cases)!=11:raise ValueError(f'expected 11 archives, got {len(cases)}')
    gates={}
    for preset in ['high','balanced','fast']:
        before=[c['results']['qe_adaptive'][preset] for c in cases]
        after=[c['results']['shape_adaptive'][preset] for c in cases]
        if preset=='high':
            gates['high_no_new_missing']=sum(x['strict_missing'] for x in after)<=sum(x['strict_missing'] for x in before)
            gates['high_no_new_spurious']=sum(x['strict_spurious'] for x in after)<=sum(x['strict_spurious'] for x in before)
        else:
            gates[preset+'_ordinary']=all(x['metrics']['accepted'] for x in after)
            gates[preset+'_strict_clusters']=sum(x['metrics']['close_cluster_unmatched'] for x in after)<=sum(x['metrics']['close_cluster_unmatched'] for x in before)
    report=dict(version=md.__version__,backend=a.backend,cases=cases,gates=gates,passed=all(gates.values()))
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('gates',gates,flush=True)
    if not report['passed']:raise SystemExit(2)


if __name__=='__main__':main()
