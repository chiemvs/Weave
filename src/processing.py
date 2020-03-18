#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functionality for parallel computation of climatologies, making anomalies and temporal aggregation.
By multiple processes on a shared C array. (All in memory, not mapped to disk)
"""
import numpy as np
import xarray as xr
import multiprocessing as mp
import logging
from pathlib import Path
from .utils import get_corresponding_ctype

# A global dictionary storing the variables that are filled with an initializer and inherited by each of the worker processes
var_dict = {}

def init_worker(inarray, share_input, dtype, shape, doyaxis = None, climate = None, outarray = None):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    # Currently the np.frombuffer done in each worker(slight overhead in doing this here), but array will appear as if switching different adresses.
    var_dict['inarray'] = inarray
    var_dict['share_input'] = share_input
    var_dict['dtype'] = dtype
    var_dict['shape'] = shape
    var_dict['doyaxis'] = doyaxis
    var_dict['climate'] = climate
    var_dict['outarray'] = outarray
    logging.debug('this initializer has populated a global dictionary')
    

def reduce_doy(doy):
    """
    Worker function. Initialized with acess to an array (first axis is time) and a doy axis array through the global var_dict, passed at inheritance
    Should not be part of the class. Here it is determined that the 
    window is 5 days before and 5 days after. And that the minimum number of observations should be 10
    """
    maxdoy = 366
    daysbefore = 5
    daysafter = 5
    window = np.arange(doy - daysbefore, doy + daysafter, dtype = 'int32')
    # small corrections for overshooting into previous or next year.
    window[ window < 1 ] += maxdoy
    window[ window > maxdoy ] -= maxdoy
    if var_dict['share_input']:
        array = np.frombuffer(var_dict['inarray'], dtype = var_dict['dtype']).reshape(var_dict['shape']) # For shared Ctype arrays
    else:
        array = var_dict['inarray']
    logging.debug(f'accessing full array at {hex(id(var_dict["inarray"]))}')
    ax0ind = np.isin(var_dict['doyaxis'], window, assume_unique= True)
    reduced = array[ax0ind,...].mean(axis = 0)
    
    number_nan = np.isnan(reduced).sum()
    reduced = np.where((~np.isnan(array[ax0ind,...])).sum(axis = 0) < 10, np.nan, reduced)
    number_nan = np.isnan(reduced).sum() - number_nan
    
    logging.debug(f'computed clim of {doy}, removed {number_nan} locations with < 10 obs.')

    return reduced
    

class ClimateComputer(object):
    """
    Assumes that the time dimension is the zero-th dimension
    Output is never shared as it is only a small array. One spatial field per doy, with a maximum of 366
    """
    def __init__(self, datapath: Path, group = None, share_input: bool = False):
        data = xr.open_dataarray(datapath, group = group)
        assert data.dims[0] == 'time'
        self.spatdims = data.dims[1:]
        self.spatcoords = {dim:data.coords[dim] for dim in self.spatdims}
        self.maxdoy = 366
        self.dtype = data.dtype
        self.share_input = share_input
        self.shape = data.shape
        if self.share_input:
            self.inarray = mp.RawArray(get_corresponding_ctype(self.dtype), size_or_initializer=data.size)
            inarray_np = np.frombuffer(self.inarray, dtype=self.dtype)
            np.copyto(inarray_np, data.values.reshape((data.size,)))
        else:
            self.inarray = data.values
        self.doyaxis = data.time.dt.dayofyear.values
        logging.info(f'ClimateComputer placed inarray of dimension {self.shape} in memory, shared = {self.share_input}')
    
    def compute(self, nprocs = 1):
        doys = list(range(1, self.maxdoy + 1))
        with mp.Pool(processes = nprocs, initializer=init_worker, initargs=(self.inarray,self.share_input,self.dtype,self.shape,self.doyaxis,None,None)) as pool:
            results = pool.map(reduce_doy,doys)
        self.spatcoords.update({'doy':doys})
        results = xr.DataArray(data = np.stack(results, axis = 0), dims = ('doy',) + self.spatdims, coords = self.spatcoords) # delivers the array concatenated along the zeroth doy-dimension.
        logging.debug(f'stacked all doys along zeroth axis, returning array of shape {results.shape}')
        return results


def subtract_per_doy(doy):
    """
    Worker function. Initialized with acess to an array (first axis is time) and a doy axis array through the global var_dict, passed at inheritance
    Should not be part of the class. 
    Writes to the shared array according the same doy axis
    Will the order be correct?
    """
    if var_dict['share_input']:
        inarray = np.frombuffer(var_dict['inarray'], dtype = var_dict['dtype']).reshape(var_dict['shape']) # For shared Ctype arrays
    else:
        inarray = var_dict['inarray']
    logging.debug(f'accessing full inarray at {hex(id(var_dict["inarray"]))}')
    outarray = np.frombuffer(var_dict['outarray'], dtype = var_dict['dtype']).reshape(var_dict['shape']) # For shared Ctype arrays
    ax0ind = var_dict['doyaxis'] == doy
    outarray[ax0ind,...] = inarray[ax0ind,...] - var_dict['climate'].sel(doy = doy).values
    return doy
    
class AnomComputer(object):
    """
    Assumes that the time dimension is the zero-th dimension
    The data is read from a path. Put in a shared array. The climatology is supplied as a loaded xarray. 
    Will write to a shared outarray of same dimensions.
    """
    def __init__(self, datapath: Path, climate: xr.DataArray, group = None, share_input: bool = False):
        data = xr.open_dataarray(datapath, group = group)
        assert data.dims[0] == 'time'
        self.coords = data.coords
        self.dims = data.dims
        self.attrs = data.attrs
        self.name = data.name
        self.climate = climate
        assert climate.dims[0] == 'doy'
        # Checking compatibility of non-zeroth dimensions
        assert self.dims[1:] == climate.dims[1:]
        for dim in self.dims[1:]:
            assert (self.coords[dim].values == climate.coords[dim].values).all()
        self.dtype = data.dtype
        self.share_input = share_input
        self.shape = data.shape
        if self.share_input:
            self.inarray = mp.RawArray(get_corresponding_ctype(self.dtype), size_or_initializer=data.size)
            inarray_np = np.frombuffer(self.inarray, dtype=self.dtype)
            np.copyto(inarray_np, data.values.reshape((data.size,)))
        else:
            self.inarray = data.values
        logging.info(f'AnomComputer placed inarray of dimension {self.shape} in memory, shared = {self.share_input}')
        # Preparing and sharing the output
        self.outarray = mp.RawArray(get_corresponding_ctype(self.dtype), size_or_initializer=data.size)
        logging.info(f'AnomComputer placed outarray of dimension {self.shape} in shared memory')
        self.doyaxis = data.time.dt.dayofyear.values
        self.maxdoy = 366 

    def compute(self, nprocs):
        """
        Will call workers to compute on each doy subset of the timeseries
        They will write to the same array (but non-overlapping locations)
        """
        doys = list(range(1, self.maxdoy + 1))
        with mp.Pool(processes = nprocs, initializer=init_worker, initargs=(self.inarray,self.share_input,self.dtype,self.shape,self.doyaxis,self.climate,self.outarray)) as pool:
            results = pool.map(subtract_per_doy,doys)

        # Reconstruction from shared out array
        np_outarray = np.frombuffer(self.outarray, dtype = self.dtype).reshape(self.shape) # For shared Ctype arrays
        result = xr.DataArray(np_outarray, dims = self.dims, coords = self.coords, name = '-'.join([self.name, 'anom']))
        result.attrs = self.attrs
        return result

