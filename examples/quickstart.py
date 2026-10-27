#!/usr/bin/env python3
"""Run after pip installation, from any working directory."""
import numpy as np
from modisp import Solver, dispersion

periods=np.logspace(-1,1,24)
solid=np.array([[.3,2.,.8,2.],[0.,5.,2.8,2.6]])
water=np.vstack([[.025,1.5,0.,1.02],solid])
rayleigh=dispersion(solid,periods,nmodes=1,wave='rayleigh',backend='cpu')
love=dispersion(solid,periods,nmodes=1,wave='love',backend='cpu')
print('Solid Rayleigh:',rayleigh.phase_velocity[0,0,:3])
print('Solid Love:',love.phase_velocity[0,0,:3])
wet=dispersion(water,periods,nmodes=3,backend='auto')
print('Water-loaded P-SV:',wet.phase_velocity[0,0,:3])
print('Backend:',wet.backend_used,wet.backend_reason)
# Use a zero-thickness water layer for the dry member of a mixed batch.
batch=np.stack([np.vstack([[0.,1.5,0.,1.02],solid]),water])
with Solver(backend='cpu',threads=2) as solver:
    result=solver(*(batch[:,:,i] for i in range(4)),periods,nmodes=3)
    print('Batch shape:',result.phase_velocity.shape)
    print('Diagnostics:',solver.stats())
    print('Last result:',solver.last_result is result)
