#!/Users/yusong/miniforge3/bin/python
"""Compare shape presets with the retained 0.3.0 path on frozen seeded inputs.

The legacy high path is a regression witness, not independent physical truth.
Changes in new high are exported for external adjudication, never silently
called improvements. Legacy unresolved rows remain explicitly unjudged.
"""
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
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for seed in range(3490,3494):
        model=md.random_model_batch(160,6,seed=seed,lvz_prob=1.,h_min=.05,h_max=8.)
        periods=np.geomspace(.3,30.,24)
        for wave in ['rayleigh','love']:
            results={}
            for strategy in ['qe_adaptive','shape_adaptive']:
                for accuracy in ['high','balanced','fast']:
                    with md.Solver(backend=a.backend,accuracy=accuracy,search_strategy=strategy,threads=a.threads,cmin=.1,workspace_bytes=1024**3) as s:
                        result=s(**model,periods=periods,nmodes=8 if accuracy=='high' else 5,wave=wave)
                        stats=s.stats()
                    results[strategy,accuracy]=(result,stats)
            old=results['qe_adaptive','high'][0];new=results['shape_adaptive','high'][0]
            stable=(old.trace_flags&4)==0
            changed=np.any(np.isfinite(old.phase_velocity[:,:5])!=new.mode_mask[:,:5],axis=1)
            common=old.mode_mask[:,:5]&new.mode_mask[:,:5]
            changed|=np.any(common&(abs(old.phase_velocity[:,:5]-new.phase_velocity[:,:5])>2e-6*abs(old.phase_velocity[:,:5])),axis=1)
            row=dict(seed=seed,wave=wave,traces=len(periods)*len(model['thickness']),
                     legacy_unresolved_traces=int((~stable).sum()),high_changed_stable_traces=int((changed&stable).sum()),
                     high_newly_finite_traces=int(((~stable)&np.any(new.mode_mask,axis=1)).sum()),presets={})
            np.savez_compressed(a.output/f'{seed}-{wave}-high-changes.npz',**model,periods=periods,
                                reference=old.phase_velocity,reference_flags=old.trace_flags,
                                actual=new.phase_velocity,flags=new.trace_flags,changed=changed)
            for accuracy,budget in [('balanced',2e-5),('fast',2e-4)]:
                legacy=results['qe_adaptive',accuracy][0];candidate,stats=results['shape_adaptive',accuracy]
                before=compare(legacy.phase_velocity,old.phase_velocity,legacy.trace_flags,.005,budget,old.trace_flags)
                after=compare(candidate.phase_velocity,old.phase_velocity,candidate.trace_flags,.005,budget,old.trace_flags)
                row['presets'][accuracy]=dict(legacy=before,candidate=after,stats=stats,
                    close_clusters_not_worse=after['close_cluster_unmatched']<=before['close_cluster_unmatched'])
                if not after['accepted']:
                    np.savez_compressed(a.output/f'{seed}-{wave}-{accuracy}-failures.npz',**model,periods=periods,
                                        reference=old.phase_velocity,reference_flags=old.trace_flags,
                                        actual=candidate.phase_velocity,flags=candidate.trace_flags)
                print(seed,wave,accuracy,'ordinary',after['ordinary_failures'],'clusters',before['close_cluster_missing'],after['close_cluster_missing'],flush=True)
            rows.append(row)
    report=dict(version=md.__version__,backend=a.backend,reference_kind='retained 0.3.0 high regression witness; high changes require independent adjudication',cases=rows)
    report['ordinary_gate_passed']=all(p['candidate']['accepted'] for row in rows for p in row['presets'].values())
    report['cluster_gate_passed']=all(sum(row['presets'][preset]['candidate']['close_cluster_unmatched'] for row in rows)<=sum(row['presets'][preset]['legacy']['close_cluster_unmatched'] for row in rows) for preset in ['balanced','fast'])
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['ordinary_gate_passed'] or not report['cluster_gate_passed']:raise SystemExit(2)


if __name__=='__main__':main()
