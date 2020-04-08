#!/usr/bin/env python3

"""
Quantify the association between one variable field (at one time aggregation) and a single response variable.
Meaning that this single response variable is already spatial cluster mean and is subsetted and detrended. 
Basically does lagging, selection of the subset, detrending
Aimed at acting per spatial cell. 
"""

import numpy as np
import xarray as xr
import logging
from scipy.stats import pearsonr, linregress
from scipy.signal import detrend
from .utils import get_corresponding_ctype

def init_worker(inarray, share_input, dtype, shape, intimeaxis, responseseries, outarray = None, lagrange: list = None):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['inarray'] = inarray
    var_dict['share_input'] = share_input
    var_dict['dtype'] = dtype
    var_dict['shape'] = shape
    var_dict['intimeaxis'] = intimeaxis
    var_dict['responseseries'] = responseseries
    var_dict['outarray'] = outarray # Needed?
    var_dict['lagrange'] = lagrange
    logging.debug('this initializer has populated a global dictionary')

def lag_subset_detrend_associate(spatial_index: tuple):
    """
    Worker function. Initialized with acess to an array (first axis is time, others are the spatial_index) , passed at inheritance
    Also needs access to the original xr time axis, the xr response series, the lagrange (in days)
    The response series is used both for subsetting with its time axis
    and for its values to compute association
    """
    full_index = (slice(None),) + spatial_index # Assumes the unlimited time on the first axis
    if var_dict['share_input']:
        inarray = np.frombuffer(var_dict['inarray'], dtype = var_dict['dtype']).reshape(var_dict['shape'])[full_index] # For shared Ctype arrays
    else:
        inarray = var_dict['inarray'][full_index]
    
    # Now inarray is a one dimensional numpy array, we need the original and series time coordinates to do the lagging
    intimeaxis = var_dict['intimeaxis']
    oriset = xr.DataArray(inarray, dims = ('time',), coords = {'time':intimeaxis})
    subsettimeaxis = var_dict['responseseries'].time
    # Prepare the computation results. First axis: lags, second axis: [corrcoef, pvalue]
    lagrange = var_dict['lagrange']
    results = np.full((len(lagrange),2), np.nan, dtype = np.float32)
    for lag in lagrange: # Units is days
        oriset['time'] = intimeaxis - pd.Timedelta(str(lag) + 'D') # Each point in time is assigned to a lagged date
        subset = oriset.reindex_like(subsettimeaxis) # We only retain the points assigned to the dates of the response timeseries
        subset = detrend(subset) # Only a single axis
        results[lagrange.index(lag),:] = pearsonr(var_dict['responseseries'], subset) # Returns (corr,pvalue)
    # Return an array with strength of association/ p-value for lags? Or write to shared array? Then this has a non-standard shape
    #outarray = np.frombuffer(var_dict['outarray'], dtype = var_dict['dtype']).reshape(var_dict['shape']) # For shared Ctype arrays
    return results

def associate_cells(nprocs, responseseries: xr.DataArray, data: xr.DataArray, laglist: list):
    """
    Always shares the input array
    responseseries should already be a 1D detrended subset
    laglist is how far to shift the data in number of days
    """
    # Perhaps data should only be suplied at initialization. Such that it is an object in this class, and the object does not have to stick around and consume memory.
    # Fill the shared array
    inarray = mp.RawArray(get_corresponding_ctype(data.dtype), size_or_initializer=data.size)
    inarray_np = np.frombuffer(data, dtype=data.dtype)
    np.copyto(inarray_np, data.values.reshape((data.size,)))

    # Prepare all the associated spatial coordinates
    coords = None # Use itertools?
    with mp.Pool(processes = nprocs, initializer=init_worker, initargs=(self.inarray,self.share_input,self.dtype,self.shape,self.doyaxis,None,None,None,None)) as pool:
        results = pool.map(reduce_doy,doys)

def init_worker(inarray, share_input, dtype, shape, intimeaxis, responseseries, outarray = None, lagrange: list = None):
