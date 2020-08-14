"""
Call signature: python make_composite.py $TEMPDIR $PACKAGEDIR $NPROC $ANOMDIR $CLUSTERDIR $OUTDIR
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
from Weave.association import Associator, composite, composite1d
from Weave.inputoutput import Writer
from Weave.utils import agg_time

def set_quant(q, func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs, quant = q)
    return wrapper

quant = 0.95
composite1d = set_quant(q = quant, func = composite1d) # Decorate the composite1d to set the quantile

logging.basicConfig(filename= TMPDIR / 'make_composite.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')
#logging.basicConfig(filename= TMPDIR / 'testprecursor_spearman.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')
# Open a response timeseries. And extract a certain cluster with a cluster template
response = xr.open_dataarray(ANOMDIR / 't2m_europe.anom.nc')
clusterfield = xr.open_dataarray(CLUSTERDIR / 't2m-q095.nc').sel(nclusters = 14)
reduced = response.groupby(clusterfield).mean('stacked_latitude_longitude')
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
to_reduce = ['snowc','siconc'] # Variables that are reduced and stacked etc, such that they are not too large for parallel association
#files = ['transp_europe.anom.nc']

timeaggs = [1, 3, 5, 7, 9, 11, 15] # Block/rolling aggregations.
#quants = [0.66,0.8,0.9,0.95,0.98] # Only in use for non-lagging setup
# Open a precursor array
for timeagg in timeaggs:
    # Determine the lags as a multiple of the rolling aggregation
    laglist = [1, 3, 5, 7, 9, 11, 15, 20, 25, 30, 35, 40, 45] #list(timeagg * np.arange(1,11))
    # Aggregate the response, subset and detrend
    responseagg = agg_time(array = reduced, ndayagg = timeagg, method = 'mean', rolling = True, firstday = pd.Timestamp('1981-01-01'))
    summersubset = responseagg[responseagg.time.dt.season == 'JJA']
    summersubset.values = detrend(summersubset.values)
    for inputfile in files:
        # Investigate the precursors
        name = inputfile.split('.')[0]
        varname = name.split('_')[0]
        outpath = OUTDIR / '.'.join([name,str(timeagg),'comp','nc'])
        if not outpath.exists():
            ta = TimeAggregator(datapath = ANOMDIR / inputfile, share_input = True, reduce_input = (varname in to_reduce))
            mean = ta.compute(nprocs = NPROC, ndayagg = timeagg, method = 'mean', firstday = pd.Timestamp(responseagg.time[0].values), rolling = True)
            del ta
            ac = Associator(responseseries = summersubset, data = mean, laglist = laglist, association = composite1d)
            #comp = composite(responseseries = summersubset, data = mean, quant = quants)
            units = mean.units
            ncvarname = mean.name
            del mean
            comp = ac.compute(NPROC, alpha = 0.05)
            if varname in to_reduce:
                example = xr.open_dataarray(ANOMDIR / inputfile)[0]
                comp = comp.unstack('stacked').reindex_like(example) # For correct ordering of the coords
                del example
            w = Writer(outpath, varname = varname, ncvarname = ncvarname)
            w.create_dataset(example = comp)
            w.write(array = comp, attrs = {'q':str(quant)}, units = units)
            del comp, w
