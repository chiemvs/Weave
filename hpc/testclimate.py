"""
Call signature: python testclimate.py $OBSDIR $PACKAGEDIR $NPROC
"""

import sys
import logging
import numpy as np
from pathlib import Path
import xarray as xr

TMPDIR = Path(sys.argv[1])
OBSDIR = Path(sys.argv[2])
PACKAGEDIR = sys.argv[3] # Currently not in use for the SurfaceObservations class (hardcoded there)
NPROC = sys.argv[4]

sys.path.append(PACKAGEDIR)

from Weave.src.processing import ClimateComputer

if __name__ == '__main__':
    logging.basicConfig(filename= OBSDIR / 'testclimate.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')
    cc = ClimateComputer(datapath = OBSDIR / 'siconc_nhmin.nc', group = 'mean', shared = False)

    test = cc.compute(nprocs = int(NPROC))

