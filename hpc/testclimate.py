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
from Weave.src.inputoutput import Writer 

if __name__ == '__main__':
    logging.basicConfig(filename= TMPDIR / 'testclimate.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')

    # ERA5 variables present in the data directory
    files = [ f.parts[-1] for f in OBSDIR.glob('*.nc') if f.is_file()]
    
    for inputfile in ['snowc_nhmin.nc']:
        name = inputfile.split('.')[0]
        varname = name.split('_')[0]
        # Discover the group
        with nc.Dataset(OBSDIR / inputfile, mode = 'r') as dat:
            group = list(dat.groups.keys())[0]

        #dat = xr.open_dataarray(OBSDIR / inputfile, group = group).astype('float16')
        cc = ClimateComputer(datapath = OBSDIR / inputfile, group = group, share_input = True)
        clim = cc.compute(nprocs = int(NPROC))
        w = Writer(datapath = OUTDIR / '.'.join([name,'clim','nc']), varname = varname, ncvarname = clim.name) # Written without group structure
        w.create_dataset(example = clim)
        w.write(clim, units = clim.units)
        del cc # Too memory intensive to keep around
        
        ac = AnomComputer(datapath = OBSDIR / inputfile, group = group, share_input = True, climate = clim)
        anom = ac.compute(nprocs = int(NPROC))
        w = Writer(datapath = OUTDIR / '.'.join([name,'anom','nc']), varname = varname, ncvarname = anom.name)
        w.create_dataset(example = anom)
        w.write(anom, units = anom.units)
        del ac, anom

        #ta = TimeAggregator(datapath = OUTDIR / '.'.join([name,'anom','nc']), share_input = True)
        #mean = ta.compute(nprocs = int(NPROC), ndayagg = 4, method = 'mean', firstday = pd.Timestamp('1979-01-01'), rolling = False)
        #w = Writer(datapath = OUTDIR / '.'.join([name,'anom',str(4),'nonroll','nc']), varname = varname, ncvarname = mean.name)
        #w.create_dataset(example = mean)
        #w.write(mean, units = mean.units)
        #del ta, mean
