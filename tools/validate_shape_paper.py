#!/Users/yusong/miniforge3/bin/python
"""Validate Zhu et al. (2026) Table 1 model 1 without using new high as truth.

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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grt',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--forward-only',action='store_true',help='save forward outputs for comparison where private references reside')
    p.add_argument('--read-results',type=Path,help='judge previously saved forward outputs without recomputing')
    p.add_argument('--backend',choices=['cpu','cuda'],default='cpu');p.add_argument('--threads',type=int,default=8)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
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
                    with md.Solver(backend=a.backend,accuracy=accuracy,search_strategy=strategy,threads=a.threads,workspace_bytes=1024**3) as solver:
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


if __name__=='__main__':main()
