"""
Call signature: python toytimescale.py $TEMPDIR $PACKAGEDIR $NPROC $ANOMDIR $CLUSTERDIR $PATTERNDIR $OUTDIR
"""

import sys
import logging
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
#from scipy.signal import detrend # Should spatial covariance and regression be done with detrended data?

TMPDIR = Path(sys.argv[1])
PACKAGEDIR = sys.argv[2] 
NPROC = int(sys.argv[3])
ANOMDIR = Path(sys.argv[4])
CLUSTERDIR = Path(sys.argv[5])
PATTERNDIR = Path(sys.argv[6])
OUTDIR = Path(sys.argv[7])

sys.path.append(PACKAGEDIR)

from Weave.src.processing import TimeAggregator
from Weave.src.inputoutput import Writer, Reader
from Weave.src.utils import agg_time, get_europe, get_natlantic
from Weave.src.dimreduction import spatcov_multilag

logging.basicConfig(filename= TMPDIR / 'toy.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')
# Open a response timeseries. And extract a certain cluster with a cluster template
response = xr.open_dataarray(ANOMDIR / 't2m_europe.anom.nc')
clusterfield = xr.open_dataarray(CLUSTERDIR / 't2m-q095.nc').sel(nclusters = 14)
reduced = response.groupby(clusterfield).mean('stacked_latitude_longitude')
reduced = reduced.sel(clustid = 9) # In this case cluster 9 is western europe.
response.close()
del response

# Define variable / region combinations for the dummy problem
combinations = {'sst_nhplus.anom.nc':get_natlantic(), 'z300_nhmin.anom.nc':get_europe()}
#combinations = {'sst_nhplus.anom.nc':get_natlantic()}
# Define the time scale of the response (to be regressed to)
responsetimeagg = 3
responseagg = agg_time(array = reduced, ndayagg = responsetimeagg, method = 'mean', rolling = True, firstday = pd.Timestamp('1981-01-01'))
summersubset = responseagg[responseagg.time.dt.season == 'JJA']
#summersubset.values = detrend(summersubset.values) # Detrend here?
#summersubset.to_netcdf(OUTDIR / '.'.join(['response',str(responsetimeagg),'nc'])) # Quick and dirty save

# Only rolling aggregation is possible for intercomparing timescales, as those are equally (daily) stamped
timeaggs = [1, 3, 5, 7, 9, 11, 15]
laglist = [-1, -3, -5, -7, -9, -11, -15, -20, -25, -30, -35, -40, -45] # Eventually will be negative values
#timeaggs = [3]
# first level loop is variable / block combinations
for inputfile, region in combinations.items():
    # Investigate the precursors
    name = inputfile.split('.')[0]
    varname = name.split('_')[0]
    outpath = OUTDIR / '.'.join([name,region[0],'spatcov','nc'])
    spatcovs = [] # One frame for each timeagg, to be merged later, and written per variable
    for timeagg in timeaggs:
        ta = TimeAggregator(datapath = ANOMDIR / inputfile, share_input = True, reduce_input = False, region = region)
        mean = ta.compute(nprocs = NPROC, ndayagg = timeagg, method = 'mean', firstday = pd.Timestamp(responseagg.time[0].values), rolling = True)
        # After aggregation we are going to get the correlation pattern
        patternpath = PATTERNDIR / '.'.join([name,str(timeagg),'corr','nc'])
        r = Reader(patternpath, region = region)
        corr = r.read(into_shared = False)
        corr = xr.DataArray(corr, dims = r.dims, coords = r.coords)
        full_spatcov = spatcov_multilag(pattern = corr, precursor = mean, laglist = laglist)
        spatcovs.append(full_spatcov.reindex_like(summersubset)) # Detrend here after the subsetting? or before dimreduction at each gridcell?

    # Collect the results in an array. Outside the timeaggregation loop the ridge regression should be done. (One to all, or one per lag but with all timeaggs, as Kiri said) 
    res = xr.concat(spatcovs, dim = pd.Index(timeaggs, name = 'timeagg')) # Quick and dirty, because of analysis in a notebook
    #res.stack({'stacked':['timeagg','lag']})
    res.to_netcdf(outpath)
