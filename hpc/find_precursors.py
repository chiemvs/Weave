"""
Call signature: python find_precursors.py $TEMPDIR $PACKAGEDIR $NPROC $ANOMDIR $CLUSTERDIR $OUTDIR
"""

import sys
import logging
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from scipy.signal import detrend

TMPDIR = Path(sys.argv[1])
PACKAGEDIR = sys.argv[2] 
NPROC = int(sys.argv[3])
ANOMDIR = Path(sys.argv[4])
CLUSTERDIR = Path(sys.argv[5])
OUTDIR = Path(sys.argv[6])

sys.path.append(PACKAGEDIR)

from Weave.src.processing import TimeAggregator
from Weave.src.association import Associator
from Weave.src.inputoutput import Writer
from Weave.src.utils import agg_time

logging.basicConfig(filename= TMPDIR / 'testprecursor_snowc.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')
# Open a response timeseries. And extract a certain cluster with a cluster template
response = xr.open_dataarray(ANOMDIR / 't2m_europe.anom.nc')
clusterfield = xr.open_dataarray(CLUSTERDIR / 't2m-q095.nc').sel(nclusters = 14)
reduced = response.groupby(clusterfield).mean('stacked_latitude_longitude')
reduced = reduced.sel(clustid = 9) # In this case cluster 9 is western europe.
response.close()

# Scan for present anomaly files. 
files = [ f.parts[-1] for f in ANOMDIR.glob('*anom.nc') if f.is_file()]
# Don't do the response itself
files.remove('t2m_europe.anom.nc')
#files.remove('snowc_nhmin.anom.nc')

timeaggs = [1, 3, 5, 7, 9, 11, 15] # Block aggregations.
# Open a precursor array
for timeagg in timeaggs:
    # Determine the lags as a multiple of the timeagg
    laglist = list(timeagg * np.arange(1,11))
    # Aggregate the response, subset and detrend
    responseagg = agg_time(array = reduced, ndayagg = timeagg, method = 'mean', rolling = False, firstday = pd.Timestamp('1981-01-01'))
    summersubset = responseagg[responseagg.time.dt.season == 'JJA']
    summersubset.values = detrend(summersubset.values)
    for inputfile in files:
        # Investigate the precursors
        name = inputfile.split('.')[0]
        varname = name.split('_')[0]
        outpath = OUTDIR / '.'.join([name,str(timeagg),'corr','nc'])
        if not outpath.exists():
            ta = TimeAggregator(datapath = ANOMDIR / inputfile, share_input = True, reduce_input = (varname == 'snowc'))
            mean = ta.compute(nprocs = NPROC, ndayagg = timeagg, method = 'mean', firstday = pd.Timestamp(responseagg.time[0].values), rolling = False)
            del ta
            ac = Associator(responseseries = summersubset, data = mean, laglist = laglist)
            del mean
            corr = ac.compute(NPROC, alpha = 0.05)
            w = Writer(outpath, varname = corr.name)
            w.create_dataset(example = corr)
            w.write(array = corr, attrs = corr.attrs, units = '')
            del ac, corr, w
