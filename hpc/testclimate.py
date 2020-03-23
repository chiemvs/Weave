"""
Call signature: python testclimate.py $TEMPDIR $OBSDIR $PACKAGEDIR $NPROC
"""

import sys
import logging
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

TMPDIR = Path(sys.argv[1])
OBSDIR = Path(sys.argv[2])
PACKAGEDIR = sys.argv[3] # Currently not in use for the SurfaceObservations class (hardcoded there)
NPROC = sys.argv[4]

sys.path.append(PACKAGEDIR)

from Weave.src.processing import ClimateComputer, AnomComputer, TimeAggregator

if __name__ == '__main__':
    logging.basicConfig(filename= OBSDIR / 'testclimate.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')
    
    cc = ClimateComputer(datapath = OBSDIR / 't2m_europe.nc', group = 'mean', share_input = True)
    clim = cc.compute(nprocs = int(NPROC))
    clim.to_netcdf(OBSDIR / 't2m_europe_clim.nc')
    
    ac = AnomComputer(datapath = OBSDIR / 't2m_europe.nc', group = 'mean', share_input = True, climate = clim)
    anom = ac.compute(nprocs = int(NPROC))
    anom.to_netcdf(OBSDIR / 't2m_europe_anom.nc')

    ta = TimeAggregator(datapath = OBSDIR / 't2m_europe_anom.nc', share_input = True)
    mean = ta.compute(nprocs = int(NPROC), ndayagg = 4, method = 'mean', firstday = pd.Timestamp('1979-01-01'), rolling = False)
    mean.to_netcdf(OBSDIR / 't2m_europe_anom_4D.nc')
