#!/Users/yusong/miniforge3/bin/python
"""Adjudicate changed high prefixes with two pinned external GRT runs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.io import netcdf_file
from validate_shape_paper import strict_compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--grt',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for path in sorted(a.input.glob('*-high-changes.npz')):
        with np.load(path) as z:
            for b,t in np.argwhere(z['changed']&((z['reference_flags']&4)==0)):
                label=f'{path.stem}-{b}-{t}';wave='L' if '-love-' in path.stem else 'R'
                model=a.output/f'{label}.txt'
                np.savetxt(model,np.c_[*[z[k][b] for k in ['thickness','vp','vs','rho']],np.full((z['vs'].shape[1],2),1e30)],fmt='%.17g')
                frequency=1/float(z['periods'][t]);roots=[]
                for tolerance,threshold in [(5,8),(6,10)]:
                    output=a.output/f'{label}-t{tolerance}.nc'
                    command=[str(a.grt),'eigenv',f'-M{model}',f'-C{output}',f'-S{wave}',
                             f'-F{frequency:.17g}/{frequency:.17g}/{frequency/10:.17g}','-N',f'-T+t{tolerance}+c9+v9+r{threshold}','-P1','-s']
                    with (a.output/f'{label}-t{tolerance}.log').open('w') as log:
                        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                    with netcdf_file(output,'r',mmap=False) as f:
                        if len(f.variables['freq'].data)!=1:raise ValueError('expected one frequency')
                        np.testing.assert_allclose(f.variables['freq'].data,[frequency],rtol=1e-12,atol=0.)
                        roots.append(f.variables['c'].data.copy())
                old=z['reference'][b,:5,t];old=old[np.isfinite(old)]
                new=z['actual'][b,:5,t];new=new[np.isfinite(new)]
                stability=strict_compare(roots[0],roots[1],2e-6)
                prefix_stability=strict_compare(roots[0][:5],roots[1][:5],2e-6)
                comparisons=[strict_compare(new,r[:5],2e-6) for r in roots]
                rows.append(dict(case=label,wave=wave,period=float(z['periods'][t]),
                    old_prefix=old.tolist(),new_prefix=new.tolist(),grt_prefixes=[r[:8].tolist() for r in roots],
                    full_controls=stability,prefix_controls=prefix_stability,new_vs_grt=comparisons,
                    passed=all(c['missing']==c['spurious']==0 for c in [prefix_stability,*comparisons])))
                print(label,rows[-1]['passed'],flush=True)
    report=dict(grt_version='pygrt-kit 0.18.0',grt_sha256=hashlib.sha256(a.grt.read_bytes()).hexdigest(),
                cases=rows,passed=all(r['passed'] for r in rows))
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise SystemExit(2)


if __name__=='__main__':main()
