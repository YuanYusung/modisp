#!/Users/yusong/miniforge3/bin/python
"""Seeded stress against high, retaining close-cluster and unjudged counts."""
import argparse
import json
from pathlib import Path
import numpy as np
import modisp as md
from tolerant_metrics import compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--threads',type=int,default=8)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for seed in range(4):
        batch=md.random_model_batch(160,6,seed=3490+seed,lvz_prob=1.,h_min=.05,h_max=8.)
        periods=np.geomspace(.3,30.,24)
        for wave in ['rayleigh','love']:
            with md.Solver(backend=a.backend,accuracy='high',threads=a.threads,cmin=.1,workspace_bytes=1024**3) as s:
                high=s(**batch,periods=periods,nmodes=8,wave=wave)
            row=dict(seed=3490+seed,wave=wave,models=160,traces=3840,metrics={})
            for preset,budget in [('balanced',2e-5),('fast',2e-4)]:
                with md.Solver(backend=a.backend,accuracy=preset,threads=a.threads,cmin=.1,workspace_bytes=1024**3) as s:
                    result=s(**batch,periods=periods,nmodes=5,wave=wave)
                m=compare(result.phase_velocity,high.phase_velocity,result.trace_flags,.005,budget,high.trace_flags)
                row['metrics'][preset]=m
                if not m['accepted']:
                    np.savez_compressed(a.output/f'failure-{seed}-{wave}-{preset}.npz',**batch,periods=periods,
                                        high=high.phase_velocity,actual=result.phase_velocity,
                                        flags=result.trace_flags,high_flags=high.trace_flags)
                print(seed,wave,preset,m,flush=True)
            rows.append(row)
    report=dict(reference_kind='same-physics high regression, not independent physical truth',
                backend=a.backend,close_root_rtol=.005,cases=rows,
                passed=all(m['accepted'] for r in rows for m in r['metrics'].values()))
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise SystemExit(2)


if __name__=='__main__':main()
