"""
Call signature: python agg_anoms.py $TEMPDIR $PACKAGEDIR $NPROC $ANOMDIR $OUTDIR
"""

import sys
import logging
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from scipy.signal import detrend
from scipy.stats import spearmanr

TMPDIR = Path(sys.argv[1])
PACKAGEDIR = sys.argv[2] 
NPROC = int(sys.argv[3])
ANOMDIR = Path(sys.argv[4])
OUTDIR = Path(sys.argv[5])

sys.path.append(PACKAGEDIR)

from Weave.processing import TimeAggregator
from Weave.inputoutput import Writer

#inputfile = 'z300_nhnorm.anom.nc'
#inputfile = 'tcc_europe.anom.nc'
inputfile = 'sst_nhplus.anom.nc'

timeagg = 21

name = inputfile.split('.')[0]
varname = name.split('_')[0]
outpath = OUTDIR / '.'.join([name,str(timeagg),'anom','nc'])

ta = TimeAggregator(datapath = ANOMDIR / inputfile, share_input = True, reduce_input = False)
mean = ta.compute(nprocs = NPROC, ndayagg = timeagg, method = 'mean', rolling = True) # No first day

ta.encoding.pop('source')
ta.encoding.pop('original_shape')
mean.encoding = ta.encoding
del ta
mean.to_netcdf(outpath)

#w = Writer(outpath, varname = varname)
#w.create_dataset(example = mean)
#w.write(array = mean, blocksize = 500, attrs = mean.attrs, units = mean.units)
