"""
Call signature: python testclimate.py $OBSDIR $PACKAGEDIR $NPROC
"""

import sys
import logging
import numpy as np
from pathlib import Path
import xarray as xr

OBSDIR = Path(sys.argv[1])
PACKAGEDIR = sys.argv[2] # Currently not in use for the SurfaceObservations class (hardcoded there)
NPROC = sys.argv[3]

sys.path.append(PACKAGEDIR)

from Weave.src.processing import ClimateComputer

if __name__ == '__main__':
    logging.basicConfig(filename='testclimate.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')
    cc = ClimateComputer(datapath = OBSDIR / 't2m_europe.nc', group = 'mean', shared = True)

    test = cc.compute(nprocs = int(NPROC))

