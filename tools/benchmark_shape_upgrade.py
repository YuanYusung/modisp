#!/Users/yusong/miniforge3/bin/python
"""Interleave real 0.3.0/0.3.1 forward calls in separate persistent workers.

First calls and warmup are separate. Every steady output must reproduce its
worker's first scientific arrays bitwise. Only model constants may be reused;
the native solver runs on every request. Validation data remain outside source.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import numpy as np
from benchmark_modephys import convert,RESULTS,GpuMonitor
from tolerant_metrics import compare

PRESETS=('high','balanced','fast')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def identical(a,b):
    for name in RESULTS:
        x,y=np.asarray(a[name]),np.asarray(b[name])
        if x.dtype.kind=='f':x,y=x.view(np.uint64),y.view(np.uint64)
        if not np.array_equal(x,y):raise AssertionError(f'{name}: scientific output changed')


def worker(a):
    import modisp as md
    if md.__version__!=a.expected_version:raise RuntimeError((md.__version__,a.expected_version,md.__file__))
    raw=np.load(a.input,mmap_mode='r')
    if not 0<a.models<=len(raw):raise ValueError('invalid model count')
    model=convert(raw[:a.models]);periods=np.load(a.periods)
    options=dict(backend=a.backend,threads=a.threads,workspace_bytes=1024**3)
    solvers={};first={}
    def emit(value):print(json.dumps(value),flush=True)
    metadata=dict(version=md.__version__,cpu_extension_sha256=digest(md._cpu.__file__),
                  python=sys.version,numpy=np.__version__,platform=platform.platform(),
                  models=a.models,layers=model['vs'].shape[1],periods=len(periods),
                  input_sha256=digest(a.input),periods_sha256=digest(a.periods))
    if a.backend=='cuda':
        module=md._load_cuda();metadata['cuda_extension_sha256']=digest(module.__file__)
    emit(metadata)
    try:
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('close'):break
            preset=request['preset'];strategy=request.get('strategy',a.strategy);modes=request.get('modes',5)
            key=(preset,strategy,modes)
            if key not in solvers:solvers[key]=md.Solver(accuracy=preset,search_strategy=strategy,**options)
            solver=solvers[key];start=time.perf_counter()
            result=solver(**model,periods=periods,nmodes=modes)
            elapsed=time.perf_counter()-start
            arrays={name:getattr(result,name) for name in RESULTS}
            if key in first:identical(first[key],arrays)
            else:first[key]={name:x.copy() for name,x in arrays.items()}
            if request.get('save'):np.savez_compressed(request['save'],**arrays)
            emit(dict(seconds=elapsed,stats=solver.stats(),kernel=solver.kernel_report()))
    finally:
        for s in solvers.values():s.close()


class Worker:
    def __init__(self,a,label,path,version,strategy):
        self.stderr=(a.output/f'{label}.stderr').open('w')
        env=dict(os.environ,PYTHONPATH=str(path))
        command=[str(a.python),str(Path(__file__).resolve()),'--worker','--expected-version',version,
                 '--strategy',strategy,'--input',str(a.input),'--periods',str(a.periods),
                 '--models',str(a.models),'--backend',a.backend,'--threads',str(a.threads)]
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                      stderr=self.stderr,text=True,bufsize=1,env=env,cwd=a.output)
        self.metadata=self.read()
    def read(self):
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('benchmark worker exited; inspect its stderr')
        return json.loads(line)
    def run(self,**request):
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush();return self.read()
    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"close":true}\n');self.process.stdin.flush()
            self.process.wait(timeout=30)
        self.stderr.close()


def benchmark(a):
    if a.repeats<5:raise ValueError('release benchmark requires at least five repeats')
    a.output.mkdir(parents=True,exist_ok=False);workers={};reports={}
    try:
        workers['baseline']=Worker(a,'baseline',a.baseline,'0.3.0','qe_adaptive')
        workers['candidate']=Worker(a,'candidate',a.candidate,'0.3.1','shape_adaptive')
        for k in ['input_sha256','periods_sha256','models','layers','periods']:
            if workers['baseline'].metadata[k]!=workers['candidate'].metadata[k]:raise AssertionError(k)
        monitor=GpuMonitor({w.process.pid for w in workers.values()}) if a.backend=='cuda' else nullcontext()
        with monitor:
            for name,w in workers.items():
                reports[name]={}
                for preset in PRESETS:
                    first=w.run(preset=preset,save=str(a.output/f'{name}-{preset}-first.npz'))
                    reports[name][preset]=dict(first=first['seconds'],samples=[],stats=first['stats'],kernel=first['kernel'])
                    w.run(preset=preset)
            workers['baseline'].run(preset='high',modes=8,save=str(a.output/'baseline-high-guard.npz'))
            workers['candidate'].run(preset='high',strategy='qe_adaptive',save=str(a.output/'candidate-legacy-high.npz'))
            with np.load(a.output/'baseline-high-first.npz') as x,np.load(a.output/'candidate-legacy-high.npz') as y:identical(x,y)
            for i in range(a.repeats):
                presets=PRESETS[i%3:]+PRESETS[:i%3]
                order=['baseline','candidate'] if i%2==0 else ['candidate','baseline']
                for preset in presets:
                    for name in order:
                        request=dict(preset=preset)
                        if i==a.repeats-1:request['save']=str(a.output/f'{name}-{preset}-last.npz')
                        reply=workers[name].run(**request)
                        reports[name][preset]['samples'].append(reply['seconds'])
                        reports[name][preset]['stats']=reply['stats']
                if a.backend=='cuda':monitor.check()
                print('steady',i+1,{n:{p:round(reports[n][p]['samples'][-1],4) for p in PRESETS} for n in order},flush=True)
                # Preserve full precision timings before statistical analysis.
                (a.output/'timing-checkpoint.json').write_text(json.dumps(dict(
                    metadata={k:w.metadata for k,w in workers.items()},timings=reports),indent=2)+'\n')
            if a.backend=='cuda':
                monitor.check();gpu_samples=monitor.samples;competitors=monitor.competitors
            else:gpu_samples=[];competitors=[]
        with np.load(a.output/'baseline-high-first.npz') as h,np.load(a.output/'baseline-high-guard.npz') as g:
            primary=h['phase_velocity'];guard=g['phase_velocity']
            reference=np.full_like(guard,np.nan);reference[:,:5]=primary
            agrees=np.all(np.isclose(guard[:,:5],primary,rtol=2e-6,atol=3e-6,equal_nan=True),axis=1)
            reference[:,5:,:]=np.where(agrees[:,None,:],guard[:,5:,:],np.nan)
            complete=agrees&(np.isfinite(guard).sum(axis=1)<8);flags=h['trace_flags'].copy()
        np.savez_compressed(a.output/'validation-reference.npz',reference=reference,flags=flags,complete=complete,guard_prefix_agrees=agrees)
        gates={};comparisons={}
        for preset,budget in [('high',2e-6),('balanced',2e-5),('fast',2e-4)]:
            comparison={}
            for name in workers:
                with np.load(a.output/f'{name}-{preset}-last.npz') as z:
                    comparison[name]=compare(z['phase_velocity'],reference,z['trace_flags'],.005 if preset!='high' else 0.,budget,flags,complete)
                reports[name][preset]['median']=statistics.median(reports[name][preset]['samples'])
            speedup=reports['baseline'][preset]['median']/reports['candidate'][preset]['median']
            comparisons[preset]=dict(speedup=speedup,validation=comparison,
                                    reference_kind='frozen 0.3.0 high regression witness; high changes require independent adjudication')
            gates[preset+'_speed']=speedup>=(1/1.05 if preset=='high' else 1.2)
            if preset!='high':
                gates[preset+'_ordinary']=comparison['candidate']['accepted']
                gates[preset+'_clusters']=comparison['candidate']['close_cluster_unmatched']<=comparison['baseline']['close_cluster_unmatched']
            gates[preset+'_unresolved']=comparison['candidate']['unresolved_traces']<=comparison['baseline']['unresolved_traces']
        report=dict(backend=a.backend,threads=a.threads,repeats=a.repeats,
                    metadata={k:w.metadata for k,w in workers.items()},timings=reports,
                    comparisons=comparisons,gates=gates,passed=all(gates.values()),
                    legacy_high_bitwise_equal=True,gpu_competitors=competitors)
        (a.output/'gpu-monitor.log').write_text('\n'.join(gpu_samples)+'\n')
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('gates',gates,flush=True)
        if not report['passed']:raise SystemExit(2)
    finally:
        for w in workers.values():w.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--periods',type=Path,required=True)
    p.add_argument('--models',type=int,required=True);p.add_argument('--backend',choices=['cpu','cuda'],required=True)
    p.add_argument('--threads',type=int,default=32);p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--python',type=Path,default=Path(sys.executable));p.add_argument('--baseline',type=Path)
    p.add_argument('--candidate',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--worker',action='store_true');p.add_argument('--expected-version');p.add_argument('--strategy')
    a=p.parse_args()
    if a.worker:worker(a)
    else:
        if not all([a.baseline,a.candidate,a.output]):p.error('--baseline, --candidate and --output are required')
        benchmark(a)


if __name__=='__main__':main()
