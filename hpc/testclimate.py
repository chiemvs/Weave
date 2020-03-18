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
from Weave.src.processing import AnomComputer

if __name__ == '__main__':
    logging.basicConfig(filename= OBSDIR / 'testclimate.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')
    cc = ClimateComputer(datapath = OBSDIR / 't2m_europe.nc', group = 'mean', share_input = True)

    clim = cc.compute(nprocs = int(NPROC))
    
    ac = AnomComputer(datapath = OBSDIR / 't2m_europe.nc', group = 'mean', share_input = True, climate = clim)
    anom = ac.compute(nprocs = int(NPROC))

