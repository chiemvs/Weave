"""
Call signature: python toytimescale.py $TEMPDIR $PACKAGEDIR $NPROC $ANOMDIR $CLUSTERDIR $PATTERNDIR $OUTDIR
"""

import sys
import logging
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
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
from Weave.src.dimreduction import spatcov 

logging.basicConfig(filename= TMPDIR / 'toy.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')
# Open a response timeseries. And extract a certain cluster with a cluster template
response = xr.open_dataarray(ANOMDIR / 't2m_europe.anom.nc')
clusterfield = xr.open_dataarray(CLUSTERDIR / 't2m-q095.nc').sel(nclusters = 14)
reduced = response.groupby(clusterfield).mean('stacked_latitude_longitude')
reduced = reduced.sel(clustid = 9) # In this case cluster 9 is western europe.
response.close()
del response

# Define variable / region combinations for the dummy problem
combinations = {'sst_nhplus.anom.nc':get_natlantic()} # 'z300_nhmin.anom.nc':get_europe()

# Only rolling aggregation is possible for intercomparing timescales, as those are equally (daily) stamped
#timeaggs = [1, 3, 5, 7, 9, 11, 15] # Block/rolling aggregations.
timeaggs = [3]
for timeagg in timeaggs:
    laglist = [1, 3, 5, 7, 9, 11, 15, 20, 25, 30, 35, 40, 45] #list(timeagg * np.arange(1,11))
    # Aggregate the response, subset and detrend
#    responseagg = agg_time(array = reduced, ndayagg = timeagg, method = 'mean', rolling = True, firstday = pd.Timestamp('1981-01-01'))
#    summersubset = responseagg[responseagg.time.dt.season == 'JJA']
#    summersubset.values = detrend(summersubset.values)
    # Second level loop is variable / block combinations
    for inputfile, region in combinations.items():
        # Investigate the precursors
        name = inputfile.split('.')[0]
        varname = name.split('_')[0]
#        outpath = OUTDIR / '.'.join([name,str(timeagg),'comp','nc'])
        ta = TimeAggregator(datapath = ANOMDIR / inputfile, share_input = True, reduce_input = False, region = region)
        mean = ta.compute(nprocs = NPROC, ndayagg = timeagg, method = 'mean', firstday = None, rolling = True) #pd.Timestamp(responseagg.time[0].values), rolling = True)
        # After aggregation we are going to get the correlation pattern
        patternpath = PATTERNDIR / '.'.join([name,str(timeagg),'corr','nc'])
        r = Reader(patternpath, region = region)
        corr = r.read(into_shared = False)
        test = spatcov(corr, mean.values)
        test = xr.DataArray(test, dims = [r.dims[0], mean.dims[0]], coords = {r.dims[0]:r.coords[r.dims[0]], mean.dims[0]:mean.coords[mean.dims[0]]})
        # Start with the spatial covariance. Should be coded in the src/dimreduction. vectorized over lags? or by loop? Perhaps not decode the lag coordinates.

        # Collect the results in a dataframe. Outside the timeaggregation loop the ridge regression should be done. (One to all, or one per lag but with all timeaggs, as Kiri said) 
#            ac = Associator(responseseries = summersubset, data = mean, laglist = laglist, association = composite1d)
#            #comp = composite(responseseries = summersubset, data = mean, quant = quants)
#            units = mean.units
#            ncvarname = mean.name
#            del mean
#            comp = ac.compute(NPROC, alpha = 0.05)
#            if varname in to_reduce:
#                example = xr.open_dataarray(ANOMDIR / inputfile)[0]
#                comp = comp.unstack('stacked').reindex_like(example) # For correct ordering of the coords
#                del example
#            w = Writer(outpath, varname = varname, ncvarname = ncvarname)
#            w.create_dataset(example = comp)
#            w.write(array = comp, attrs = {'q':str(quant)}, units = units)
#            del comp, w
