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

from Weave.processing import TimeAggregator
from Weave.association import Associator
from Weave.inputoutput import Writer
from Weave.utils import agg_time, spearmanr_wrap, kendall_predictand #kendall_choice, chi

logging.basicConfig(filename= TMPDIR / 'find_precursors.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')
# Open a response timeseries. And extract a certain cluster with a cluster template
response = xr.open_dataarray(ANOMDIR / 't2m_europe.anom.nc')
clusterfield = xr.open_dataarray(CLUSTERDIR / 't2m-q095.nc').sel(nclusters = 14)
#reduced = response.groupby(clusterfield).mean('stacked_latitude_longitude')
#reduced = reduced.sel(clustid = 9) # In this case cluster 9 is western europe.
reduced = response.groupby(clusterfield).quantile(q = 0.8, dim = 'stacked_latitude_longitude')
reduced = reduced.sel(clustid = 9) # In this case cluster 9 is western europe.
response.close()
del response

# Scan for present anomaly files. 
files = [ f.parts[-1] for f in ANOMDIR.glob('*anom.nc') if f.is_file()]
# Don't do the response itself
files.remove('t2m_europe.anom.nc')
files.remove('swvl1_europe.anom.nc') # We only want to keep the merged one: swvl13
files.remove('swvl2_europe.anom.nc')
files.remove('swvl3_europe.anom.nc')
files.remove('z300_nhmin.anom.nc') # Only nhnorm retained
to_reduce = ['snowc','siconc'] # Variables that are reduced and stacked etc, such that they are not too large for parallel association
#files = ['sst_nhplus.anom.nc', 'z300_nhnorm.anom.nc', 'swvl13_europe.anom.nc']

timeaggs = [1, 3, 5, 7, 11, 15, 21, 31] # Block/rolling aggregations.
#timeaggs = [1,7,30] # Block/rolling aggregations.
# Open a precursor array
for timeagg in timeaggs:
    #laglist = [-1, -3, -5, -7, -9, -11, -15, -20, -25, -30, -35, -40, -45] #list(timeagg * np.arange(1,11))
    absolute_separation = np.array([0,1,3,5,7,11,15,21,31]) # Days inbetween end of precursor and beginning of response 
    laglist = [0,] + list(-timeagg - absolute_separation) # Dynamic lagging to avoid overlap, lag zero is the overlap
    # Aggregate the response, subset and detrend
    responseagg = agg_time(array = reduced, ndayagg = timeagg, method = 'mean', rolling = True, firstday = pd.Timestamp('1981-01-01'))
    summersubset = responseagg[responseagg.time.dt.season == 'JJA']
    summersubset.values = detrend(summersubset.values)
    for inputfile in files:
        # Investigate the precursors
        name = inputfile.split('.')[0]
        varname = name.split('_')[0]
        outpath = OUTDIR / '.'.join([name,str(timeagg),'corr','nc'])
        if not outpath.exists():
            ta = TimeAggregator(datapath = ANOMDIR / inputfile, share_input = True, reduce_input = (varname in to_reduce))
            mean = ta.compute(nprocs = NPROC, ndayagg = timeagg, method = 'mean', firstday = pd.Timestamp(responseagg.time[0].values), rolling = True)
            del ta
            ac = Associator(responseseries = summersubset, data = mean, laglist = laglist, association = spearmanr_wrap)
            del mean
            corr = ac.compute(NPROC, alpha = 5*10**(-6 - 0.2*(timeagg-1))) # Variable alpha, ranges from 5e-6 to 5e-12 for timeaggs 1 to 31
            if varname in to_reduce:
                example = xr.open_dataarray(ANOMDIR / inputfile)[0]
                corr = corr.unstack('stacked').reindex_like(example) # For correct ordering of the coords
                del example
            w = Writer(outpath, varname = corr.name)
            w.create_dataset(example = corr)
            w.write(array = corr, attrs = corr.attrs, units = '')
            del ac, corr, w
