"""
Call signature: python testclimate.py $TEMPDIR $OBSDIR $PACKAGEDIR $NPROC
"""

import sys
import logging
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
from pathlib import Path

TMPDIR = Path(sys.argv[1])
OBSDIR = Path(sys.argv[2])
PACKAGEDIR = sys.argv[3] # Currently not in use for the SurfaceObservations class (hardcoded there)
NPROC = sys.argv[4]
OUTDIR = Path(sys.argv[5])

sys.path.append(PACKAGEDIR)

from Weave.src.processing import ClimateComputer, AnomComputer, TimeAggregator

if __name__ == '__main__':
    logging.basicConfig(filename= TMPDIR / 'testclimate.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')

    # ERA5 variables present in the data directory
    files = [ f.parts[-1] for f in OBSDIR.glob('*.nc') if f.is_file()]
    
    for inputfile in ['siconc_nhmin.nc']:
        name = inputfile.split('.')[0]
        # Discover the group
        with nc.Dataset(OBSDIR / inputfile, mode = 'r') as dat:
            group = list(dat.groups.keys())[0]
        cc = ClimateComputer(datapath = OBSDIR / inputfile, group = group, share_input = True)
        clim = cc.compute(nprocs = int(NPROC))
        clim.to_netcdf(OUTDIR / '.'.join([name,'clim','nc']))
        
        ac = AnomComputer(datapath = OBSDIR / inputfile, group = group, share_input = True, climate = clim)
        anom = ac.compute(nprocs = int(NPROC))
        anom.to_netcdf(OUTDIR / '.'.join([name,'anom','nc']))

        ta = TimeAggregator(datapath = OUTDIR / '.'.join([name,'anom','nc']), share_input = True)
        mean = ta.compute(nprocs = int(NPROC), ndayagg = 4, method = 'mean', firstday = pd.Timestamp('1979-01-01'), rolling = False)
        mean.to_netcdf(OUTDIR / '.'.join([name,'anom',str(4),'nonroll','nc']))
