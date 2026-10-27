#!/usr/bin/env python3
"""Create the NPZ accepted by modisp forward."""
import argparse
import numpy as np
p=argparse.ArgumentParser();p.add_argument('output');args=p.parse_args()
np.savez(args.output,thickness=[.025,.3,0.],vp=[1.5,2.,5.],vs=[0.,.8,2.8],rho=[1.02,2.,2.6],periods=np.logspace(-1,1,24))
