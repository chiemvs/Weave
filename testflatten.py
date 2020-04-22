"""
Call signature: python testflatten.py $TEMPDIR $PACKAGEDIR $NPROC $ANOMDIR $CLUSTERDIR 
"""

import sys
import logging
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from scipy.signal import detrend
from scipy.stats import spearmanr, pearsonr

TMPDIR = Path(sys.argv[1])
PACKAGEDIR = sys.argv[2] 
NPROC = int(sys.argv[3])
ANOMDIR = Path(sys.argv[4])
CLUSTERDIR = Path(sys.argv[5])

sys.path.append(PACKAGEDIR)

from Weave.src.processing import TimeAggregator
from Weave.src.association import Associator
from Weave.src.inputoutput import Writer
from Weave.src.utils import agg_time

#logging.basicConfig(filename= TMPDIR / 'testprecursor_snowc_pearson.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')
logging.basicConfig(filename= TMPDIR / 'testflatten.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')
# Open a response timeseries. And extract a certain cluster with a cluster template
response = xr.open_dataarray(ANOMDIR / 't2m_europe.anom.nc')
clusterfield = xr.open_dataarray(CLUSTERDIR / 't2m-q095.nc').sel(nclusters = 14)
reduced = response.groupby(clusterfield).mean('stacked_latitude_longitude')
reduced = reduced.sel(clustid = 9) # In this case cluster 9 is western europe.
response.close()
del response

inputfile = Path('/scistor/ivm/jsn295/processed/swvl13_europe.anom.nc').parts[-1]
laglist = [1, 3]
timeagg = 3

responseagg = agg_time(array = reduced, ndayagg = timeagg, method = 'mean', rolling = True, firstday = pd.Timestamp('1981-01-01'))
summersubset = responseagg[responseagg.time.dt.season == 'JJA']
summersubset.values = detrend(summersubset.values)
# Investigate the precursors
name = inputfile.split('.')[0]
varname = name.split('_')[0]
outpath = Path('/scistor/ivm/jsn295/testflat.nc') 
ta = TimeAggregator(datapath = ANOMDIR / inputfile, share_input = True, reduce_input = True)
mean = ta.compute(nprocs = NPROC, ndayagg = timeagg, method = 'mean', firstday = pd.Timestamp(responseagg.time[0].values), rolling = True)
#del ta
#ac = Associator(responseseries = summersubset, data = mean, laglist = laglist, association = spearmanr)
#del mean
#corr = ac.compute(NPROC, alpha = 0.05)
#corr2 = corr.unstack('stacked')
#w = Writer(outpath, varname = corr2.name)
#w.create_dataset(example = corr2)
#w.write(array = corr2, attrs = corr2.attrs, units = '')
#del ac, corr, w
