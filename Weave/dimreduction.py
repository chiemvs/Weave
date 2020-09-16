import logging
import numpy as np
import xarray as xr
import pandas as pd
# spatial covariance of a loaded anomaly block with a pattern. Possibly flattened (when no domain subset region is supplied at reading) but no need for sharing as this cannot be paralelized. Although.. perhaps over time, or over lags.


def spatcov_multilag(pattern: xr.DataArray, precursor: xr.DataArray, laglist: list) -> xr.DataArray:
    """
    pattern: (nlags, ...)
    aggregated precursor (ntime, ...) Unlagged
    spatial dimensions should be equal
    laglist should have units days
    computes spatial covariance timeseries in a matrix (nlags,ntime) 
    then lags each of the timeseries by the appropriate lag. 
    """
    # Flatten the spatial dimensions
    flatpat = pattern.values.reshape((pattern.shape[0],-1)).astype(np.float32)
    flatprec = precursor.values.reshape((precursor.shape[0],-1)).astype(np.float32)

    patterndiff = flatpat - np.nanmean(flatpat, -1)[:,np.newaxis] # 2D, (nlags,nspace)
    precursordiff = flatprec - np.nanmean(flatprec, -1)[:,np.newaxis] # 2D (ntime,nspace)

    def onelag_cov(onepatterndiff, precursordiff):
        """
        covariance timeseries for one lag, unbiased estimate by n - 1   
        input: onepatterndiff 1D (nobs,), precursordiff 2D (ntime,nobs)
        output: 1D (ntime,)
        """
        return(np.nansum(precursordiff * onepatterndiff, axis = -1)/(len(onepatterndiff) - np.isnan(onepatterndiff).sum()))

    allseries = np.apply_along_axis(onelag_cov, axis = 1, arr = patterndiff, precursordiff = precursordiff)
    logging.debug(f'computed unlagged spatial covariances with the patterns of lags: {laglist}')
    # Now start with appropriate lagging, after putting into the frame.
    allseries = xr.DataArray(allseries, dims = [pattern.dims[0],precursor.dims[0]], coords = {pattern.dims[0]:laglist, precursor.dims[0]:precursor.coords[precursor.dims[0]]}, name = 'spatcov')
    for lag in laglist: # Units is days
        oneserie = allseries.sel({pattern.dims[0]:lag}).copy()
        oneserie['time'] = oneserie['time'] - pd.Timedelta(str(lag) + 'D') # Ascribe each value to another timestamp (e.g. lag of -10 means precursor value of originally 1979-01-01 is assigned to 1979-01-11
        allseries.loc[{pattern.dims[0]:lag}] = oneserie.reindex_like(allseries) # Write the lagged stamped values
        logging.debug(f'finished lagging ntimes: {len(oneserie)} with lag: {lag}')
    return(allseries)

def mean_singlelag(precursor: xr.DataArray, lag: int):
    """
    Computes the mean over all points in the flattened (spatial) last dimension.
    Should already be the subset for a single cluster / square / whatever
    (Time aggregated) precursor is unlagged and lagged here.
    Returned 1D with same time axis
    """
    meanprec = np.nanmean(precursor.values.reshape((precursor.shape[0],-1)).astype(np.float32), axis = -1)
    logging.debug('computed unlagged mean')
    meanprec = xr.DataArray(meanprec, dims = precursor.dims[:1], coords = {precursor.dims[0]:precursor.coords[precursor.dims[0]]}, name = 'mean') 
    meanprec['time'] = meanprec['time'] - pd.Timedelta(str(lag) + 'D') # Ascribe each value to another timestamp (e.g. lag of -10 means precursor value of originally 1979-01-01 is assigned to 1979-01-11
    logging.debug(f'lagged mean of ntimes: {len(meanprec)} with lag: {lag}')
    return(meanprec.reindex_like(precursor))
