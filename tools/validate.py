#!/Users/yusong/miniforge3/bin/python
"""Numerical validation: paper witnesses, frozen external archives and seeded stress.

PyGRT is an external executable, never imported or bundled. Raw SH matrices
and the separate unnormalised CPS probe cross-check its roots. All numerical
outputs and logs belong in the caller's validation directory, not the repo.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from scipy.io import netcdf_file
from scipy.optimize import brentq
import modisp as md
from tolerant_metrics import compare

MODEL=np.array([[.01,1.55,.18,1.78],[.01,1.8,.35,1.85],
                [.02,1.,.12,1.5],[0.,2.,.6,1.93]])
FREQUENCIES=np.arange(1,1001,dtype=float)/10.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_grt(path):
    with netcdf_file(path,'r',mmap=False) as f:
        frequency=f.variables['freq'].data.copy()
        counts=f.variables['cnum'].data.copy()
        values=f.variables['c'].data.copy()
    np.testing.assert_allclose(frequency,FREQUENCIES,rtol=0,atol=1e-12)
    rows=[];start=0
    for n in counts:
        rows.append(values[start:start+n]);start+=n
    return rows


def raw_sh(c,frequency):
    """Unnormalised displacement/traction 2x2 matrices, independent of modisp."""
    if np.ndim(c)==0:
        omega=2*math.pi*frequency;k=omega/float(c)
        traction=MODEL[-1,3]*MODEL[-1,2]**2*math.sqrt(max(0.,k*k-(omega/MODEL[-1,2])**2));displacement=1.
        for h,_,vs,rho in MODEL[-2::-1]:
            square=(omega/vs)**2-k*k;q=math.sqrt(abs(square));phase=q*h
            if square>0:C=math.cos(phase);S=math.sin(phase)/q;D=-q*math.sin(phase)
            elif square<0:C=math.cosh(phase);S=math.sinh(phase)/q;D=q*math.sinh(phase)
            else:C=1.;S=h;D=0.
            mu=rho*vs*vs;traction,displacement=traction*C+displacement*mu*D,traction*S/mu+displacement*C
        return traction
    c=np.asarray(c);omega=2*np.pi*frequency;k=omega/c
    traction=MODEL[-1,3]*MODEL[-1,2]**2*np.sqrt(np.maximum(0.,k*k-(omega/MODEL[-1,2])**2))
    displacement=np.ones_like(c)
    for h,_,vs,rho in MODEL[-2::-1]:
        square=(omega/vs)**2-k*k;q=np.sqrt(np.abs(square));phase=q*h
        C=np.where(square>0,np.cos(phase),np.cosh(phase))
        sine=np.where(square>0,np.sin(phase),np.sinh(phase))
        S=np.divide(sine,q,out=np.full_like(q,h),where=q!=0)
        D=np.where(square>0,-q*np.sin(phase),q*np.sinh(phase))
        mu=rho*vs*vs
        traction,displacement=traction*C+displacement*mu*D,traction*S/mu+displacement*C
    return traction


def strict_compare(actual,reference,rtol):
    """Ordered one-to-one matching; a candidate cannot certify two roots."""
    missing=extra=0;maximum=0.;i=j=0
    while i<len(reference) and j<len(actual):
        x,r=actual[j],reference[i];error=abs(x-r)/r
        if error<=rtol:maximum=max(maximum,float(error));i+=1;j+=1
        elif x<r:extra+=1;j+=1
        else:missing+=1;i+=1
    return dict(missing=missing+len(reference)-i,spurious=extra+len(actual)-j,
                max_matched_relative_error=maximum)


def merge_witnesses(primary,additional,tolerance=2e-6):
    """One-to-one witness union; never coalesce neighbors from one root set."""
    result=np.asarray(primary,dtype=float)
    for values in additional:
        values=np.asarray(values);i=j=0;extra=[]
        while i<len(result) and j<len(values):
            x,r=values[j],result[i]
            if abs(x-r)<=tolerance*r:i+=1;j+=1
            elif x<r:extra.append(x);j+=1
            else:i+=1
        extra.extend(values[j:]);result=np.sort(np.r_[result,extra])
    return result


def build_reference(output,grt,threads):
    model=output/'paper-model1.txt'
    np.savetxt(model,np.c_[MODEL,np.full((4,2),1e30)],fmt='%.17g')
    pin=dict(tool='pygrt-kit 0.18.0',binary_sha256=digest(grt),model_sha256=digest(model),
             source='Zhu et al. (2026), doi:10.1785/0120260071, Table 1 model 1',
             dense_grids=[65537,98305],dense_lower=.05,dense_upper=.6*(1-1e-10))
    receipt=output/'reference-pin.json'
    if receipt.exists():
        previous=json.loads(receipt.read_text())
        for key in ['binary_sha256','model_sha256','dense_grids','dense_lower','dense_upper']:
            if previous.get(key)!=pin[key]:raise ValueError(f'reference cache pin changed: {key}; use a new directory')
    pin['reference_method']='one-to-one witness union v2; two GRT controls, two raw grids, raw local brackets'
    references={};reports={}
    for wave,label in [('rayleigh','R'),('love','L')]:
        grt_rows=[]
        for tol,threshold in [(4,6),(5,8)]:
            path=output/f'paper-{label}-t{tol}.nc'
            if not path.exists():
                command=[str(grt),'eigenv',f'-M{model}',f'-C{path}',f'-S{label}',
                         '-F0.1/100/0.1','-N',f'-T+t{tol}+c9+v9+r{threshold}',f'-P{threads}','-s']
                with (output/f'{label}-t{tol}.log').open('w') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            grt_rows.append(read_grt(path))
        cache=output/f'independent-v2-{wave}.npz'
        if cache.exists():
            with np.load(cache,allow_pickle=False) as z:
                reference=z['reference'];stable=z['stable'];details=json.loads(str(z['details']))
            references[wave]=(reference,stable);reports[wave]=details;continue
        reference=np.full((1,64,len(FREQUENCIES)),np.nan);stable=np.ones(len(FREQUENCIES),dtype=bool)
        details=dict(grt_parameter_disagreements=0,dense_parameter_disagreements=0,
                     grt_omitted_raw_roots=0,raw_bracket_failures=0,nonfinite_grid_traces=0)
        for t,frequency in enumerate(FREQUENCIES):
            raw=lambda c:raw_sh(c,frequency) if wave=='love' else md._cpu.md_dunkin_point(*MODEL.T,4,1/frequency,float(c),0)
            lo=.12*(1+1e-12) if wave=='love' else .05;hi=.6*(1-1e-10)
            dense=[]
            for nc in pin['dense_grids']:
                cs=np.linspace(lo,hi,nc)
                if nc==98305:
                    cs=np.r_[lo,lo+(np.arange(nc-1)+.5)*(hi-lo)/(nc-1),hi]
                if wave=='love':values=raw_sh(cs,frequency)
                else:
                    # The second grid is independently shifted. The diagnostic
                    # point evaluator and grid use the separate CPS reference.
                    if nc==65537:values=md._cpu.probe_grid(*MODEL.T,np.array([1/frequency]),lo,hi,nc,False,False)[0]
                    else:
                        step=(hi-lo)/(nc-1)
                        middle=md._cpu.probe_grid(*MODEL.T,np.array([1/frequency]),lo+.5*step,hi-.5*step,nc-1,False,False)[0]
                        values=np.r_[raw(lo),middle,raw(hi)]
                finite=np.isfinite(values)
                if not finite.all():details['nonfinite_grid_traces']+=1;stable[t]=False
                index=np.flatnonzero(finite[:-1]&finite[1:]&(np.signbit(values[:-1])!=np.signbit(values[1:])))
                roots=[]
                for j in index:
                    try:roots.append(brentq(raw,cs[j],cs[j+1],xtol=5e-15,rtol=4*np.finfo(float).eps))
                    except (ValueError,RuntimeError):stable[t]=False;details['raw_bracket_failures']+=1
                dense.append(np.asarray(roots))
            g0,g1=grt_rows[0][t],grt_rows[1][t]
            if any(strict_compare(g0,g1,2e-6)[k] for k in ['missing','spurious']):details['grt_parameter_disagreements']+=1;stable[t]=False
            if any(strict_compare(dense[0],dense[1],2e-6)[k] for k in ['missing','spurious']):details['dense_parameter_disagreements']+=1
            candidates=merge_witnesses(g1,[g0,*dense]);verified=[]
            for j,c in enumerate(candidates):
                left=(candidates[j-1]+c)*.5 if j else lo
                right=(c+candidates[j+1])*.5 if j+1<len(candidates) else hi
                delta=min(c-left,right-c)*.4
                found=None
                for factor in [1.,.1,.01,.001]:
                    a=max(lo,c-delta*factor);b=min(hi,c+delta*factor)
                    if a>=b:continue
                    try:
                        if np.signbit(raw(a))!=np.signbit(raw(b)):
                            found=brentq(raw,a,b,xtol=5e-15,rtol=4*np.finfo(float).eps);break
                    except (ValueError,RuntimeError):pass
                if found is None:stable[t]=False;details['raw_bracket_failures']+=1
                else:
                    verified.append(found)
                    if not np.any(abs(g1-found)<=2e-6*found):details['grt_omitted_raw_roots']+=1
            roots=np.sort(verified);reference[0,:len(roots),t]=roots
            if t%100==0:print('reference',wave,t,'stable',int(stable[:t+1].sum()),flush=True)
        np.savez_compressed(cache,reference=reference,stable=stable,details=json.dumps(details))
        references[wave]=(reference,stable);reports[wave]=details
    files=[*output.glob('independent-v2-*.npz'),*output.glob('paper-?-t?.nc')]
    pin['reference_files_sha256']={p.name:digest(p) for p in sorted(files)}
    raw_source=Path(__file__).resolve().parents[1]/'cpp/cpu/dunkin_ref.cpp'
    if raw_source.exists():pin['raw_cps_source_sha256']=digest(raw_source)
    receipt.write_text(json.dumps(dict(pin,checks=reports),indent=2)+'\n')
    return references,reports


def paper_main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--secular',choices=['dunkin','fast'])
    p.add_argument('--grt',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--forward-only',action='store_true',help='save forward outputs for comparison where private references reside')
    p.add_argument('--read-results',type=Path,help='judge previously saved forward outputs without recomputing')
    p.add_argument('--backend',choices=['cpu','cuda'],default='cpu');p.add_argument('--threads',type=int,default=8)
    a=p.parse_args(argv);a.output.mkdir(parents=True,exist_ok=True)
    if not a.forward_only and a.grt is None:p.error('--grt is required unless --forward-only')
    if a.forward_only:
        refs={wave:(None,None) for wave in ['rayleigh','love']};checks={}
    else:refs,checks=build_reference(a.output,a.grt,a.threads)
    rows=[]
    for wave in ['rayleigh','love']:
        reference,stable=refs[wave]
        for strategy in ['qe_adaptive','shape_adaptive']:
            for accuracy,budget in [('high',2e-6),('balanced',2e-5),('fast',2e-4)]:
                if a.read_results:
                    path=a.read_results/f'{a.backend}-{wave}-{strategy}-{accuracy}.npz'
                    with np.load(path,allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['periods'],1/FREQUENCIES)
                        result=SimpleNamespace(phase_velocity=z['velocity'],trace_flags=z['trace_flags'])
                        stats=json.loads(str(z['stats']))
                else:
                    with md.Solver(backend=a.backend,secular=a.secular,accuracy=accuracy,search_strategy=strategy,threads=a.threads,workspace_bytes=1024**3) as solver:
                        result=solver(*MODEL.T,periods=1/FREQUENCIES,nmodes=64,wave=wave)
                        stats=solver.stats()
                if a.forward_only:
                    np.savez_compressed(a.output/f'{a.backend}-{wave}-{strategy}-{accuracy}.npz',
                        velocity=result.phase_velocity,trace_flags=result.trace_flags,periods=1/FREQUENCIES,stats=json.dumps(stats))
                    print('forward',wave,strategy,accuracy,flush=True);continue
                judged=result.phase_velocity[:,:,stable];ref=reference[:,:,stable]
                metric=compare(judged,ref,result.trace_flags[:,stable],0. if accuracy=='high' else .005,budget,
                               reference_complete=np.ones((1,int(stable.sum())),dtype=bool))
                strict=[strict_compare(x[np.isfinite(x)],r[np.isfinite(r)],budget) for x,r in zip(judged[0].T,ref[0].T)]
                row=dict(wave=wave,strategy=strategy,accuracy=accuracy,stable_reference_traces=int(stable.sum()),
                         unjudged_reference_traces=int((~stable).sum()),metrics=metric,stats=stats,
                         strict_missing=sum(x['missing'] for x in strict),strict_spurious=sum(x['spurious'] for x in strict))
                rows.append(row);print('result',wave,strategy,accuracy,row['strict_missing'],row['strict_spurious'],flush=True)
                np.savez_compressed(a.output/f'{a.backend}-{wave}-{strategy}-{accuracy}.npz',velocity=result.phase_velocity,
                                    reference=reference,reference_stable=stable,trace_flags=result.trace_flags,periods=1/FREQUENCIES)
    (a.output/f'report-{a.backend}.json').write_text(json.dumps(dict(reference_checks=checks,cases=rows),indent=2)+'\n')




def external_main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--secular',choices=['dunkin','fast'])
    p.add_argument('--references',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--threads',type=int,default=8)
    a=p.parse_args(argv);a.output.mkdir(parents=True,exist_ok=False);cases=[]
    for path in sorted(a.references.glob('*.npz')):
        with np.load(path,allow_pickle=False) as z:
            model={k:z[k] for k in ['thickness','vp','vs','rho']}
            periods=z['periods'];reference=z['reference'];stable=z['reference_stable'].astype(bool)
            count_hi=z['reference_count_hi']
        if stable.shape==(reference.shape[0],):stable=np.broadcast_to(stable[:,None],(len(stable),len(periods)))
        if stable.shape!=(reference.shape[0],reference.shape[2]):raise ValueError((path,stable.shape))
        low=float(min(.2,.75*np.min(model['vs']))) if path.stem.startswith('qed_') else .2
        options=dict(backend=a.backend,secular=a.secular,threads=a.threads,cmin=low,workspace_bytes=1024**3)
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
    if not cases:raise ValueError('no reference archives found')
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




def stress_main(argv=None):
    p=argparse.ArgumentParser(description="Seeded regression against high; not an independent physical reference")
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--secular',choices=['dunkin','fast'])
    p.add_argument('--strategy',choices=['shape_adaptive','qe_adaptive'],default='shape_adaptive')
    p.add_argument('--seeds',type=int,nargs='+',default=[3490,3491,3492,3493])
    p.add_argument('--models',type=int,default=160)
    p.add_argument('--layers',type=int,default=6)
    p.add_argument('--periods',type=int,default=24)
    a=p.parse_args(argv)
    if min(a.models,a.layers,a.periods,a.threads)<1:p.error('counts must be positive')
    a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    options=dict(backend=a.backend,threads=a.threads,secular=a.secular,
                 search_strategy=a.strategy,cmin=.1,workspace_bytes=1024**3)
    for seed in a.seeds:
        batch=md.random_model_batch(a.models,a.layers,seed=seed,lvz_prob=1.,h_min=.05,h_max=8.)
        periods=np.geomspace(.3,30.,a.periods)
        for wave in ['rayleigh','love']:
            with md.Solver(accuracy='high',**options) as solver:
                high=solver(**batch,periods=periods,nmodes=8,wave=wave)
            row=dict(seed=seed,wave=wave,models=a.models,traces=a.models*a.periods,
                     secular=high.secular,metrics={})
            for preset,budget in [('balanced',2e-5),('fast',2e-4)]:
                with md.Solver(accuracy=preset,**options) as solver:
                    result=solver(**batch,periods=periods,nmodes=5,wave=wave)
                metric=compare(result.phase_velocity,high.phase_velocity,result.trace_flags,
                               .005,budget,high.trace_flags)
                row['metrics'][preset]=metric
                if not metric['accepted']:
                    np.savez_compressed(a.output/f'{seed}-{wave}-{preset}-failure.npz',**batch,
                                        periods=periods,high=high.phase_velocity,actual=result.phase_velocity,
                                        flags=result.trace_flags,high_flags=high.trace_flags)
            rows.append(row)
            print(seed,wave,row['metrics'],flush=True)
    report=dict(version=md.__version__,backend=a.backend,strategy=a.strategy,
                reference_kind='same-physics high regression; unresolved reference traces are unjudged',
                cases=rows,passed=all(m['accepted'] for row in rows for m in row['metrics'].values()))
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['passed'] else 2


def main(argv=None):
    import sys
    argv=list(sys.argv[1:] if argv is None else argv)
    handlers={'paper':paper_main,'external':external_main,'stress':stress_main}
    if argv and argv[0] in handlers:return handlers[argv[0]](argv[1:])
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=handlers,help='paper: independent witnesses; external: NPZ archives; stress: seeded regression')
    p.parse_args(argv)


if __name__=='__main__':raise SystemExit(main())
