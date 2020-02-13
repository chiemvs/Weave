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
import matplotlib.pyplot as plt
#sys.path.append('/usr/people/straaten/Documents/RGCPD/clustering')
#from clustering_spatial import binary_occurences_quantile #, skclustering
from scipy.stats import pearsonr, linregress
from scipy.signal import detrend
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

# Function for making a climatology (used for anomalies)
def localclim(obsarray, daysbefore = 0, daysafter = 0):
    """
    Takes in a 3D/2D xarray of observations and computes the climatological mean belonging to each window around a doy.
    So the spatial dim can be clustid or pre-aggregated lat/lon.
    A continuous and highres daily time axis will give the best results as most observations will be present.    
    For non-daily aggregations the procedure is the same, the supplied obs are ideally roll-aggregated and thus stamped daily. Meaning that all needed aggregation, both in space and time, and all needed event classification shoulg already have taken place.
    """
    doygroups = obsarray.groupby('time.dayofyear')
    doygroups = {str(key):value for key,value in doygroups} # Change to string
    maxday = 366
    results = []
    for doy in doygroups.keys():        
        doy = int(doy)
        # for aggregated values the day of year is the first day of the period. 
        window = np.arange(doy - daysbefore, doy + daysafter + 1, dtype = 'int64')
        # small corrections for overshooting into previous or next year.
        window[ window < 1 ] += maxday
        window[ window > maxday ] -= maxday
        
        complete = xr.concat([doygroups[str(key)] for key in window if str(key) in doygroups.keys()], dim = 'time')
        # Prepare for removing the time dimension on this complete block, by computing the desired statistic
        spatialdims = tuple(complete.drop('time').coords._names)
        #spatialcoords = {key:complete.coords[key].values for key in spatialdims}
        reduced = complete.mean('time', keep_attrs = True)        
        # Setting a minimum on the amount of observations that went into the mean computation, and report the number of locations that went to NaN
        number_nan = reduced.isnull().sum(spatialdims).values
        reduced = reduced.where(complete.count('time') >= 10, np.nan)
        number_nan = reduced.isnull().sum(spatialdims).values - number_nan
        
        print('computed clim of', doy, ', removed', number_nan, 'locations with < 10 obs.')
        reduced.coords['doy'] = doy
        results.append(reduced)
    
    return xr.concat(results, dim = 'doy')

def make_anom(obsarray, climarray):
    """
    Takes in a 3D/2D observations xarray. And a similar dimensioned climatology array with doy as the zero-th axis
    """
    doygroups = obsarray.groupby('time.dayofyear')
    def subtraction(inputarray):
        doy = int(np.unique(inputarray['time.dayofyear']))
        climatology = climarray.sel(doy = doy, drop = True)
        return(inputarray - climatology)
    
    subtracted = doygroups.apply(subtraction)
    result = subtracted.drop('dayofyear')
    result.attrs = {'units':obsarray.units, 'new_units':obsarray.units}
    result.name = '-'.join([obsarray.name, 'anom'])
    return(result)

ndays = 4
rolling = False
lag = -4
de_trend = True
anom = True
clustid = 0

# Do the potential anomalie modification (full series, highest resolutions) and aggregate both in time
if anom:
    t2mhighresclim = localclim(t2m, daysbefore = 5, daysafter = 5)
    t2m = make_anom(t2m, climarray = t2mhighresclim)
    precursorhighresclim = localclim(precursorfield, daysbefore = 5, daysafter = 5)
    precursorfield = make_anom(precursorfield, climarray= precursorhighresclim)

t2m = agg_time(t2m, ndayagg = ndays, method = 'mean', rolling = rolling)
precursorfield = agg_time(precursorfield, ndayagg = ndays, method = 'mean', firstday = pd.Timestamp(t2m.time[0].values), rolling = rolling)

# Now start with the first subsets. Aggregate to the average response timeseries belonging to a certain set of clusters
t2m = t2m[t2m.time.dt.season == 'JJA', :,:]
nclusters = 11
series = t2m.groupby(clusters.sel(nclusters = nclusters)).mean('stacked_latitude_longitude')
series.coords['clustid'] = series.coords['clustid'].astype(np.int16)

if de_trend: # Detrending only the summer.
    series = xr.DataArray(detrend(series, axis = 0), dims = series.dims, coords = series.coords)
    
# Plot the seasonality left in the response:
gr = series.groupby(series.time.dt.dayofyear).mean('time')
gr[:,clustid].plot()
plt.show()
# Plot the trend still left in the response:
regresult = linregress(series[:,clustid], list(range(series.shape[0])))
plt.plot(series[:,clustid])
plt.title(f'slope:{np.round(regresult[0], 4)}, intercept:{np.round(regresult[1], 2)}, p:{np.round(regresult[3], 2)}')
plt.show()

ori_timeaxis = precursorfield.coords['time'].copy()

#============================ LAGGING, each time constructing a correlation series =======

lags = list(range(-6,0))
lagdays = [l if rolling else l * ndays for l in lags]  # Lag in days. Negative x means the values x days before the reponse variable timestamp are correlated to the response timestamp
d1corrcoefs = [None] * len(lags)
for lag in lags:
    # Shifting the precursor with a certain lag, and selecting a subset
    #lagdays = lag if rolling else lag * ndays # Lag in days. Negative x means the values x days before the reponse variable timestamp are correlated to the response timestamp
    print(f'lagdays: {lagdays}, ndayagg: {ndays}, rolling: {rolling}, anom: {anom}, detrend: {de_trend}')
    lag_timeaxis = ori_timeaxis - pd.Timedelta(str(lagdays) + 'D')
    precursorfield.coords['time'] = lag_timeaxis # Assign the shifted timeaxis
    laggedfield = precursorfield.reindex_like(series)
    
    # Correlations.
    # The scipy.stats.pearsonr, can only handle two 1D timeseries (not sure about nans in the series) It returns a two side p-value, but this relies on the assumption of normality. Beta distribution based
    # Do a vectorized implementation. Outputting one corr field and then one p-value field. Perhaps immediately with multiple testing?
    # Do a looped implementation looping over NaN cells and calling scipy on the pairs. Then multiple testing from statsmodels
    
    stacklag = laggedfield.stack({'latlon':['latitude','longitude']})
    if de_trend:
        stacklag = xr.DataArray(detrend(stacklag, axis = 0), dims = stacklag.dims, coords = stacklag.coords ) # linear detrending of only the summer trend.
    
    alpha = 0.05
    test = np.apply_along_axis(pearsonr, axis = 0, arr = stacklag, **{'y':series[:,clustid]}) # zeroth dimension is [cor, pvalue]. How does it handle nan's?
    print(f'fraction p < {alpha} before correction: {(test[1,:] < alpha).sum() / test.shape[-1]}')
    pfield = multipletests(test[1,:], alpha = alpha, method = 'fdr_bh', is_sorted = False) # Very small pvalues changes.
    print(f'fraction p < {alpha} after correction: {pfield[0].sum() / test.shape[-1]}')
    print(np.quantile(test[0,:], [0.2,0.5,0.8]))
    d1corrcoefs[lags.index(lag)] = np.ma.masked_array(data = test[0,:], mask = ~pfield[0])

# Plot which cells are significant.
d1fields = xr.DataArray(np.ma.stack(d1corrcoefs), dims = ('lagdays','latlon',), coords = {'lagdays':lagdays,'latlon':stacklag.coords['latlon']}, name = 'corcor')
fields = d1fields.unstack('latlon').reindex_like(laggedfield)



#===== For now just skip the clustering. Just group the positive and negative values.
np.apply_along_axis(lambda a: a[a>0].mean(), axis = 1, arr = d1fields) # Mean correlation of the positive field at each lag.
plt.plot(lagdays,np.apply_along_axis(lambda a: a[a>0].mean(), axis = 1, arr = d1fields))
plt.title('average strength of positively correlated regions')
plt.show()

plt.plot(lagdays,np.apply_along_axis(lambda a: a[a<0].mean(), axis = 1, arr = d1fields))
plt.title('average strength of negatively correlated regions')
plt.show()


# Afterwards haversine distances etc. on the correlated regions and clustering
from sklearn.metrics import pairwise_distances
from src.clustering import Clustering
from sklearn.cluster import DBSCAN
# First should be latitude, second should be longitude (in radians)
positives = d1field[d1field > 0]
coords = np.radians(np.stack(positives.coords['latlon'].values)).astype(np.float32)
dist = pairwise_distances(coords, metric = 'haversine').astype(np.float32) 

cl = Clustering() # Could not do the masking in here because not the obs but the lats and lons are neede
cl.prepare_for_distance_algorithm(array = coords.T)
cl.call_distance_algorithm(func = pairwise_distances, kwargs = {'metric':'haversine'})
cl.clustering(clusterclass = DBSCAN, kwargs = {'metric':'precomputed'}, nclusters = [2,3]) # DBSCAN does not work with n_clusters, n_clusters is the outcome.
est = DBSCAN(metric = 'precomputed', eps=700, min_samples=10) # Weighting? distance_eps=700, min_area_in_degrees2=2
est.fit(dist)
est.labels_
# Seperate the regions (of opposing sign?) For Z500 there does not seem to be an opposing sign. Making into anomalies does not help because correlation is already difference from its mean? So detrending, as that might inflate the correlation
# The story is of course different if you make anomalies with respect to the doy. This removes the seasonal cycle.
# Detrending indeed lowers the correlation


# Compare with a ridge regression: 
# https://stackoverflow.com/questions/46400262/scikitlearn-regression-design-matrix-x-too-big-for-regression-what-do-i-do


#if __name__ == '__main__':
#    main()
