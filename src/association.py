#!/usr/bin/env python3

"""
Quantify the association between one variable field (at one time aggregation) and a single response variable.
Meaning that this single response variable is already spatial cluster mean and is subsetted and detrended. 
Basically does lagging, selection of the subset, detrending
Aimed at acting per spatial cell. 
"""

import numpy as np
import logging

def init_worker(inarray, share_input, dtype, shape, intimeaxis, responseseries, outarray = None, lagrange = None):
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

def lag_subset_detrend(spatial_index: tuple):
    """
    Worker function. Initialized with acess to an array (first axis is time, others are the spatial_index) , passed at inheritance
    Also needs access to the original xr time axis, the xr series axis, the lagrange (in days)
    """
    if var_dict['share_input']:
        inarray = np.frombuffer(var_dict['inarray'], dtype = var_dict['dtype']).reshape(var_dict['shape'])[:,spatial_index] # For shared Ctype arrays
    else:
        inarray = var_dict['inarray'][:,spatial_index]
    
    # Now inarray is a one dimensional numpy array, we need the original and series time coordinates to do the lagging
    for lag in var_dict['lagrange']: # Units is days
        lag_timeaxis = var_dict['intimeaxis'] - pd.Timedelta(str(lag) + 'D')


    # Return an array with strength of association/ p-value for lags?
    #outarray = np.frombuffer(var_dict['outarray'], dtype = var_dict['dtype']).reshape(var_dict['shape']) # For shared Ctype arrays
