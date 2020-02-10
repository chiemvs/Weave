#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extraction of clustered t2m, computation of correlation maps
"""
import sys
import logging
import time
import xarray as xr
import numpy as np
import multiprocessing as mp
import pandas as pd
#sys.path.append('/usr/people/straaten/Documents/RGCPD/clustering')
#from clustering_spatial import binary_occurences_quantile #, skclustering
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests 

# Some stuff with the desired precursor and spatial regions again?


# Loading of data
t2m = xr.open_dataarray('/nobackup_1/users/straaten/ERA5/t2m/t2m_europe.nc', group = 'mean')

clusters = xr.open_dataarray('/nobackup_1/users/straaten/Clustering/t2m-q095.nc') # Daily exceedence of q095
clusters.name = 'clustid'

precursorfield = xr.open_dataarray('/nobackup_1/users/straaten/ERA5/z500/z500_europe.nc', group = '12UTC')

# Time aggregation if applicable. Should be applied before subsetting
def agg_time(array: xr.DataArray, ndayagg: int = 1, method: str = 'mean', firstday: pd.Timestamp = None, rolling: bool = False) -> xr.DataArray:
    """
    Aggegates a daily time dimension, that should be continuous, otherwise non-neighbouring values are taken together. 
    It returns a left stamped aggregation of ndays
    For non-rolling aggregation it is possible to supply a firstday, to sync the blocks with another timeseries.
    Trailing Nan's are removed.
    """
    assert (np.diff(array.time) == np.timedelta64(1,'D')).all(), "time axis should be a continuous daily to be aggregated, though nan is allowed"
    if rolling:
        name = array.name
        attrs = array.attrs
        f = getattr(array.rolling({'time':ndayagg}, center = False), method) # Stamped right
        array = f()
        array = array.assign_coords(time = array.time - pd.Timedelta(str(ndayagg - 1) + 'D')).isel(time = slice(ndayagg - 1, None)) # Left stamping, trailing nans removed
        array.name = name
        array.attrs = attrs
    else:
        array = array.sel(time = slice(firstday, None))
        input_length = len(array.time)
        f = getattr(array.resample(time = str(ndayagg) + 'D', closed = 'left', label = 'left'), method)
        array = f(dim = 'time', keep_attrs = True, skipna = False)

        if (input_length % ndayagg) != 0:
            array = array.isel(time = slice(0,-1,None)) # Remove the last aggregation, if it has not been based on the full ndayagg

    return array

ndays = 4
rolling = True 
t2m = agg_time(t2m, ndayagg = ndays, method = 'mean', rolling = rolling)
precursorfield = agg_time(precursorfield, ndayagg = ndays, method = 'mean', firstday = pd.Timestamp(t2m.time[0].values), rolling = rolling)

# Extracting average response timeseries belonging to a certain set of clusters
t2m = t2m[t2m.time.dt.season == 'JJA', :,:]
nclusters = 11
series = t2m.groupby(clusters.sel(nclusters = nclusters)).mean('stacked_latitude_longitude')
series.coords['clustid'] = series.coords['clustid'].astype(np.int16)


# Shifting the precursor with a certain lag
lag = -4 
lagdays = lag if rolling else lag * ndays # Lag in days. Negative x means the values x days before the reponse variable timestamp are correlated to the response timestamp
print(f'lagdays: {lagdays}, ndayagg: {ndays}, rolling: {rolling}')
ori_timeaxis = precursorfield.coords['time']
lag_timeaxis = ori_timeaxis - pd.Timedelta(str(lagdays) + 'D')
precursorfield.coords['time'] = lag_timeaxis # Assign the shifted timeaxis
laggedfield = precursorfield.reindex_like(series)

# Correlations.
# The scipy.stats.pearsonr, can only handle two 1D timeseries (not sure about nans in the series) It returns a two side p-value, but this relies on the assumption of normality. Beta distribution based
# Do a vectorized implementation. Outputting one corr field and then one p-value field. Perhaps immediately with multiple testing?
# Do a looped implementation looping over NaN cells and calling scipy on the pairs. Then multiple testing from statsmodels

stacklag = laggedfield.stack({'latlon':['latitude','longitude']})

test = np.apply_along_axis(pearsonr, axis = 0, arr = stacklag, **{'y':series[:,0]}) # zeroth dimension is [cor, pvalue]. How does it handle nan's?
print(f'fraction p < 0.05 before correction: {(test[1,:] < 0.05).sum() / test.shape[-1]}')
pfield = multipletests(test[1,:], alpha = 0.05, method = 'fdr_bh', is_sorted = False) # Very small pvalues changes.
print(f'fraction p < 0.05 after correction: {pfield[0].sum() / test.shape[-1]}')
print(np.quantile(test[0,:], [0.2,0.5,0.8]))
# Afterwards haversine distances etc. on the correlated regions

#if __name__ == '__main__':
#    main()
