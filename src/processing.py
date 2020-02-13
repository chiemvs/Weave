#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functionality for parallel computation of climatologies, making anomalies and temporal aggregation.
By multiple processes on a shared C array. (All in memory, not mapped to disk)
"""
import numpy as np
import xarray as xr
import multiprocessing as mp
from utils import get_corresponding_ctype

# A global dictionary storing the variables that are filled with an initializer and inherited by each of the worker processes
var_dict = {}

def init_worker(array,shared,dtype,shape,doyaxis):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['array'] = array
    var_dict['shared'] = shared
    var_dict['dtype'] = dtype
    var_dict['shape'] = shape
    var_dict['doyaxis'] = doyaxis
    

def reduce_doy(doy):
    """
    Worker function. Initialized with acess to an array (first axis is time) and a doy axis array through the global var_dict, passed at inheritance
    Should not be part of the class.
    """
    maxday = 366
    daysbefore = 5
    daysafter = 5
    window = np.arange(doy - 5, doy + 5, dtype = 'int32')
    # small corrections for overshooting into previous or next year.
    window[ window < 1 ] += maxday
    window[ window > maxday ] -= maxday
    if var_dict['shared']:
        array = np.frombuffer(var_dict['array'], dtype = var_dict['dtype']).reshape(var_dict['shape']) # For shared Ctype arrays
    else:
        array = var_dict['array']
    ax0ind = np.isin(var_dict['doyaxis'], window, assume_unique= True)
    
    return array[ax0ind,...].mean(axis = 0)
    

class ClimateComputer(object):
    
    def __init__(self, shared = False):
        data = xr.open_dataarray('/nobackup_1/users/straaten/ERA5/t2m/t2m_europe.nc', group = 'mean')
        self.dtype = data.dtype
        self.shared = shared
        self.shape = data.shape
        if shared:
            self.array = mp.RawArray(get_corresponding_ctype(self.dtype), size_or_initializer=data.size)
            outarray_np = np.frombuffer(self.array, dtype=self.dtype)
            np.copyto(outarray_np, data.values.reshape((data.size,)))
        else:
            self.array = data.values
        self.doyaxis = data.time.dt.dayofyear.values
    
    def compute(self):
        with mp.Pool(processes = 10, initializer=init_worker, initargs=(self.array,self.shared,self.dtype,self.shape,self.doyaxis)) as pool:
            #pool = mp.Pool(processes = 5)
            results = pool.map(reduce_doy, range(1,367))
        return results

if __name__ == '__main__':
    cc = ClimateComputer(shared = False)
    test = cc.compute()

def localclim(obsarray, daysbefore = 0, daysafter = 0):
    """
    Takes in a 3D/2D xarray of observations and computes the climatological mean belonging to each window around a doy.
    So the spatial dim can be clustid or pre-aggregated lat/lon.
    A continuous and highres daily time axis will give the best results as most observations will be present.    
    For non-daily aggregations the procedure is the same, the supplied obs are ideally roll-aggregated and thus stamped daily. Meaning that all needed aggregation, both in space and time, and all needed event classification shoulg already have taken place.
    """
    doygroups = obsarray.groupby('time.dayofyear')
    doygroups = {str(key):value for key,value in doygroups} # Change to string
    
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
