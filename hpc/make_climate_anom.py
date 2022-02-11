"""
Call signature: python make_climate_anom.py $TEMPDIR $PACKAGEDIR $NPROC $OBSDIR $OUTDIR
"""

import sys
import logging
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
from pathlib import Path

TMPDIR = Path(sys.argv[1])
PACKAGEDIR = sys.argv[2] 
NPROC = int(sys.argv[3])
OBSDIR = Path(sys.argv[4])
OUTDIR = Path(sys.argv[5])

sys.path.append(PACKAGEDIR)

from Weave.processing import ClimateComputer, AnomComputer, TimeAggregator
from Weave.inputoutput import Writer 

if __name__ == '__main__':
    logging.basicConfig(filename= TMPDIR / 'make_climate_anom.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')

    # ERA5 variables present in the data directory
    files = [ f.parts[-1] for f in OBSDIR.glob('*.pre1981.nc') if f.is_file()]
    
    for inputfile in files:
        name = inputfile.split('.')[0]
        varname = name.split('_')[0]
        # Discover the group, and ncvarname (can be a complex concatenation of {varname}-{operation})
        with nc.Dataset(OBSDIR / inputfile, mode = 'r') as dat:
            try:
                group = list(dat.groups.keys())[0]
                ncvarname = '-'.join([varname, group])
            except IndexError:
                group = None
                variables = list(dat.variables.keys())
                ncvarname = [s for s in variables if s.startswith(varname)][0]

        #cc = ClimateComputer(datapath = OBSDIR / inputfile, group = group, ncvarname = ncvarname, share_input = True, reduce_input = False)
        #clim = cc.compute(nprocs = NPROC)
        #w = Writer(datapath = OUTDIR / '.'.join([name,'clim','nc']), varname = varname, ncvarname = clim.name) # Written without group structure
        #w.create_dataset(example = clim)
        #w.write(clim, units = clim.units)
        #del cc # Too memory intensive to keep around
        clim = xr.open_dataarray(OUTDIR / '.'.join([name,'clim','nc']))
        
        ac = AnomComputer(datapath = OBSDIR / inputfile, group = group, share_input = True, ncvarname = ncvarname, reduce_input = False, climate = clim)
        anom = ac.compute(nprocs = NPROC)
        w = Writer(datapath = OUTDIR / '.'.join([name,'anom','pre1981','nc']), varname = varname, ncvarname = anom.name)
        #w = Writer(datapath = OUTDIR / '.'.join([name,'anom','nc']), varname = varname, ncvarname = anom.name)
        w.create_dataset(example = anom)
        w.write(anom, units = anom.units)
        del ac, anom

