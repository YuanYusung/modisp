#!/Users/yusong/miniforge3/bin/python
"""Measure fitting-tolerance sensitivity against frozen independent roots.

Without --references, save forward results only (no reference upload needed).
Use --read-results with --references to adjudicate those files locally later.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import modisp as md
from validate_shape_paper import MODEL,FREQUENCIES,strict_compare
from tolerant_metrics import compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--references',type=Path)
    p.add_argument('--read-results',action='store_true')
    p.add_argument('--backend',choices=['cpu','cuda'],default='cpu');p.add_argument('--threads',type=int,default=8)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=[]
    for wave in ['rayleigh','love']:
        if a.references:
            path=a.references/f'independent-v2-{wave}.npz'
            with np.load(path,allow_pickle=False) as z:reference=z['reference'];stable=z['stable']
            reference_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        for accuracy,default,budget in [('high',.001,2e-6),('balanced',.01,2e-5),('fast',.05,2e-4)]:
            for scale in [.5,1.,2.]:
                tolerance=default*scale;path=a.output/f'{a.backend}-{wave}-{accuracy}-{scale:g}.npz'
                if a.read_results:
                    with np.load(path,allow_pickle=False) as z:
                        velocity=z['velocity'];flags=z['flags'];stats=json.loads(str(z['stats']))
                        if float(z['sampling_rtol'])!=tolerance:raise ValueError(path)
                else:
                    with md.Solver(backend=a.backend,threads=a.threads,accuracy=accuracy,sampling_rtol=tolerance,workspace_bytes=1024**3) as s:
                        result=s(*MODEL.T,periods=1/FREQUENCIES,nmodes=64,wave=wave)
                        velocity=result.phase_velocity;flags=result.trace_flags;stats=s.stats()
                    np.savez_compressed(path,velocity=velocity,flags=flags,stats=json.dumps(stats),sampling_rtol=tolerance,periods=1/FREQUENCIES)
                row=dict(wave=wave,accuracy=accuracy,sampling_rtol=tolerance,stats=stats)
                if a.references:
                    row.update(reference_sha256=reference_hash,stable_reference_traces=int(stable.sum()),
                               unjudged_reference_traces=int((~stable).sum()))
                    row['metrics']=compare(velocity[:,:,stable],reference[:,:,stable],flags[:,stable],
                        0. if accuracy=='high' else .005,budget,reference_complete=np.ones((1,int(stable.sum())),bool))
                    strict=[strict_compare(x[np.isfinite(x)],r[np.isfinite(r)],budget) for x,r in zip(velocity[0][:,stable].T,reference[0][:,stable].T)]
                    row.update(strict_missing=sum(s['missing'] for s in strict),strict_spurious=sum(s['spurious'] for s in strict))
                rows.append(row);print(wave,accuracy,tolerance,row.get('strict_missing'),row.get('strict_spurious'),flush=True)
    (a.output/'report.json').write_text(json.dumps(dict(version=md.__version__,backend=a.backend,cases=rows),indent=2)+'\n')


if __name__=='__main__':main()
