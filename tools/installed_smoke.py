#!/usr/bin/env python3
"""Installation check using only NumPy, including the generated console entry."""
import argparse,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
import modisp as md
p=argparse.ArgumentParser();p.add_argument('--backend',default='cpu');args=p.parse_args()
with md.Solver(backend=args.backend) as s:
    half=s([0.],[6.],[3.5],[2.7],[1.,3.],nmodes=2)
    ratio=(3.5/6.)**2
    x=next(x.real for x in np.roots([1.,-8.,24.-16*ratio,-16*(1-ratio)]) if abs(x.imag)<1e-10 and 0<x.real<1)
    np.testing.assert_allclose(half.phase_velocity[0,0],3.5*np.sqrt(x),atol=1e-6,rtol=0)
    assert np.isnan(half.phase_velocity[0,1]).all()
    dry=s([.3,0.],[2.,5.],[.8,2.8],[2.,2.6],[.1,1.,5.],nmodes=3,wave='love')
    wet=s([.5,.3,0.],[1.5,2.,5.],[0.,.8,2.8],[1.02,2.,2.6],[.1,1.,5.],nmodes=3,wave='love')
    np.testing.assert_allclose(dry.phase_velocity,wet.phase_velocity,atol=1e-6,rtol=1e-7)
    water=s([20.,0.],[1.5,6.],[0.,3.5],[1.02,2.7],[.1],nmodes=4)
    assert np.isfinite(water.phase_velocity).all()
    assert np.any(water.phase_velocity>1.5)
cli=Path(sys.executable).parent/'modisp'
subprocess.run([str(cli),'doctor'],check=True)
with tempfile.TemporaryDirectory() as td:
    td=Path(td);np.savez(td/'input.npz',thickness=[0.],vp=[6.],vs=[3.5],rho=[2.7],periods=[1.])
    subprocess.run([str(cli),'forward',str(td/'input.npz'),'--output',str(td/'out.npz'),'--backend',args.backend,'--nmodes','1'],check=True)
    assert np.load(td/'out.npz')['phase_velocity'].shape==(1,1,1)
print(json.dumps(dict(installed_smoke='passed',version=md.__version__,backend=args.backend,module=md.__file__,cpu=md._cpu.__file__,python=sys.executable)))
