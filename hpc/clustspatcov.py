"""
Call signature: python clustsspatcov.py $TEMPDIR $PACKAGEDIR $NPROC $ANOMDIR $CLUSTERDIR $PATTERNDIR $OUTDIR
Goal is to compute a dataframe of timeseries, defined on a daily resolution
One series per variable and timeagg (distributed over files) and per lag and clustid (saved within patternfile)
"""

import sys
import logging
import numpy as np
import xarray as xr
import pandas as pd
import pyarrow as pa
from pathlib import Path

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
from Weave.src.utils import agg_time
from Weave.src.dimreduction import spatcov_multilag

logging.basicConfig(filename= TMPDIR / 'spatcov.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')
firstday = pd.Timestamp('1981-01-01')
responseclustid = 9
timeaggs = [1, 3, 5, 7, 9, 11, 15]
# Response timeseries is not linked to any of the processing of the response
# Only the starting date is important. 
# We will make a seperate response dataframe now first
# Only rolling aggregation is possible for intercomparing timescales, as those are equally (daily) stamped
response_output = OUTDIR / '.'.join(['response','multiagg','trended','parquet']) 
if not response_output.exists():
    response = xr.open_dataarray(ANOMDIR / 't2m_europe.anom.nc')
    clusterfield = xr.open_dataarray(CLUSTERDIR / 't2m-q095.nc').sel(nclusters = 14)
    reduced = response.groupby(clusterfield).mean('stacked_latitude_longitude')
    reduced = reduced.sel(clustid = responseclustid) # In this case cluster 9 is western europe.
    response.close()
    output = []
    for responsetimeagg in timeaggs:
        responseagg = agg_time(array = reduced, ndayagg = responsetimeagg, method = 'mean', rolling = True, firstday = firstday)
        summersubset = responseagg[responseagg.time.dt.season == 'JJA']
        summersubset = pd.DataFrame(summersubset.values, index = summersubset.coords['time'].to_index(), columns = pd.MultiIndex.from_tuples([(summersubset.name,responsetimeagg,responseclustid)], names = ['variable','timeagg','clustid']))
        # Smallest aggregation should be a reasonable starting point (it has the largest length)
        # But large chance that all have all summer timesteps (left stamping causes this only at the end of the anomaly series)
        output.append(summersubset)

    output = pd.concat(output, axis = 1, join = 'outer')
    pa.parquet.write_table(pa.Table.from_pandas(output), response_output)
    del response, reduced, output

# Only rolling aggregation is possible for intercomparing timescales, as those are equally (daily) stamped
laglist = [-1, -3, -5, -7, -9, -11, -15, -20, -25, -30, -35, -40, -45] # Eventually will be negative values
# first level loop is variable / block combinations
for inputfile, region in combinations.items():
    # Investigate the precursors
    name = inputfile.split('.')[0]
    varname = name.split('_')[0]
    outpath = OUTDIR / '.'.join([name,region[0],'spatcov','nc'])
    if not outpath.exists():
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
