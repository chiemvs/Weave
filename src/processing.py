#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functionality for parallel computation of climatologies, making anomalies and temporal aggregation.
By multiple processes on a shared C array. (All in memory, not mapped to disk)
"""
import numpy as np
import xarray as xr
import multiprocessing as mp
import pandas as pd
import logging
from pathlib import Path
from .utils import get_corresponding_ctype
from .inputoutput import Reader

# A global dictionary storing the variables that are filled with an initializer and inherited by each of the worker processes
var_dict = {}

def init_worker(inarray, share_input, dtype, shape, doyaxis = None, climate = None, outarray = None, ndayagg = None, method = None):
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
    var_dict['ndayagg'] = ndayagg
    var_dict['method'] = method
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
    logging.debug(f'Worker accessing full array at {hex(id(var_dict["inarray"]))}')
    ax0ind = np.isin(var_dict['doyaxis'], window, assume_unique= True)
    reduced = array[ax0ind,...].mean(axis = 0)
    
    number_nan = np.isnan(reduced).sum()
    reduced = np.where((~np.isnan(array[ax0ind,...])).sum(axis = 0) < 10, np.nan, reduced)
    number_nan = np.isnan(reduced).sum() - number_nan
    
    logging.debug(f'Worker computed clim of {doy}, removed {number_nan} locations with < 10 obs.')

    return reduced
    

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
    logging.debug(f'Worker accessing full inarray at {hex(id(var_dict["inarray"]))}')
    outarray = np.frombuffer(var_dict['outarray'], dtype = var_dict['dtype']).reshape(var_dict['shape']) # For shared Ctype arrays
    ax0ind = var_dict['doyaxis'] == doy
    outarray[ax0ind,...] = inarray[ax0ind,...] - var_dict['climate'].sel(doy = doy).values
    logging.debug(f'Worker subtracted clim of doy {doy} from {int(ax0ind.sum())} fields.')

def aggregate_at(index):
    """
    Worker function. Recieves an index value along the zeroth axis. Reads ndayagg fields from the accessible input array, starting at index. Then aggregates those fields along the zeroth axis with a chosen method, and then writes this field to the index location of the outarray. 
    """
    if var_dict['share_input']:
        inarray = np.frombuffer(var_dict['inarray'], dtype = var_dict['dtype']).reshape(var_dict['shape']) # For shared Ctype arrays
    else:
        inarray = var_dict['inarray']
    logging.debug(f'Worker accessing full inarray at {hex(id(var_dict["inarray"]))}')
    outarray = np.frombuffer(var_dict['outarray'], dtype = var_dict['dtype']).reshape(var_dict['shape']) # For shared Ctype arrays
    # Selection and choice of aggregation method
    func = getattr(inarray[index:(index + var_dict['ndayagg']),...], var_dict['method'])
    outarray[index,...] = func(axis = 0)
    logging.debug(f'Worker aggregated {var_dict["ndayagg"]} fields starting at index {index}.')

class Computer(object):
    """
    Super class that provides a common initialization for computers that need acces to an on disk netcdf array, extract coordinate information from them and prepare the array as numpy in (shared) memory for sub-processes to access it.
    Assumes that the time dimension is the zero-th dimension
    """
    def __init__(self, datapath: Path = None, group: str = None, ncvarname: str = None, share_input: bool = True, reduce_input: bool = False, data: xr.DataArray = None):
        """
        Opening data with xarray has not the ability to read with less precision
        Therefore we standard use the custom reader, to which a precision can be supplied.
        if reduce_input is chosen, then the writer class is instructed to read with precision float16, otherwise float32, and also to flatten the spatial dimension, discarding spatial points with a full nan first zeroth axis nan block
        There is a choice to share the input ot not share the input
        Also has the option to be supplied with an already loaded array (called data). Can again be shared or not.
        """
        if data is None:
            data = Reader(datapath = datapath, ncvarname = ncvarname, groupname = group)# This object has similar attributes as an xr.DataArray, only no values, these are returned upon reading, with desired precision
            self.inarray = data.read(into_shared = share_input, dtype = np.float16 if reduce_input else np.float32, flatten = reduce_input) 
        else:
            if share_input:
                self.inarray = mp.RawArray(get_corresponding_ctype(data.dtype), size_or_initializer=data.size)
                inarray_np = np.frombuffer(self.inarray, dtype=data.dtype)
                np.copyto(inarray_np, data.values.reshape((data.size,)))
            else:
                self.inarray = data.values
        assert data.dims[0] == 'time'
        self.coords = data.coords
        self.dims = data.dims
        self.size = data.size
        self.attrs = data.attrs
        self.name = data.name
        self.dtype = data.dtype
        self.encoding = data.encoding
        self.share_input = share_input
        self.shape = data.shape
        logging.info(f'Computer placed inarray of dimension {self.shape} in memory, shared = {self.share_input}, dtype = {self.dtype}')
        self.doyaxis = data.coords['time'].dt.dayofyear.values
        self.maxdoy = 366

    def cleanup(self):
        """
        Delete some potentially very large objects
        """
        for attrkey in ['inarray','outarray']:
            try:
                delattr(self, attrkey)
            except AttributeError:
                pass

class ClimateComputer(Computer):
    """
    Output is never shared as it is only a small array. One spatial field per doy, with a maximum of 366
    """
    def __init__(self, datapath: Path, group: str = None, ncvarname: str = None, share_input: bool = False, reduce_input: bool = False):
        Computer.__init__(self, datapath = datapath, group = group, ncvarname = ncvarname, share_input = share_input, reduce_input = reduce_input)
    
    def compute(self, nprocs = 1):
        doys = list(range(1, self.maxdoy + 1))
        with mp.Pool(processes = nprocs, initializer=init_worker, initargs=(self.inarray,self.share_input,self.dtype,self.shape,self.doyaxis,None,None,None,None)) as pool:
            results = pool.map(reduce_doy,doys)
        coords = dict(self.coords)
        coords.pop('time')
        coords.update({'doy':doys})
        results = xr.DataArray(data = np.stack(results, axis = 0), dims = ('doy',) + self.dims[1:], coords = coords, name = self.name) # delivers the array concatenated along the zeroth doy-dimension.
        results.attrs = self.attrs
        logging.info(f'ClimateComputer stacked all doys along zeroth axis, returning xr.DataArray of shape {results.shape} with dtype {results.dtype}')
        Computer.cleanup(self)
        return results

    
class AnomComputer(Computer):
    """
    The climatology is supplied as a loaded xarray. 
    Will write to a shared outarray of same dimensions.
    """
    def __init__(self, datapath: Path, climate: xr.DataArray, group: str = None, ncvarname: str = None, share_input: bool = False, reduce_input: bool = False):
        Computer.__init__(self, datapath = datapath, group = group, ncvarname = ncvarname, share_input = share_input, reduce_input = reduce_input)
        self.climate = climate
        assert climate.dims[0] == 'doy'
        # Checking compatibility of non-zeroth dimensions
        assert self.dims[1:] == climate.dims[1:]
        for dim in self.dims[1:]:
            assert (self.coords[dim].values == climate.coords[dim].values).all()
        # Preparing and sharing the output
        self.outarray = mp.RawArray(get_corresponding_ctype(self.dtype), size_or_initializer=self.size)
        logging.info(f'AnomComputer placed outarray of dimension {self.shape} in shared memory, dtype = {self.dtype}')

    def compute(self, nprocs):
        """
        Will call workers to compute on each doy subset of the timeseries
        They will write to the same array (but non-overlapping locations)
        Will not use multiprocessing when the desired nprocs is only one
        """
        doys = list(range(1, self.maxdoy + 1))
        if nprocs > 1:
            with mp.Pool(processes = nprocs, initializer=init_worker, initargs=(self.inarray,self.share_input,self.dtype,self.shape,self.doyaxis,self.climate,self.outarray,None,None)) as pool:
                pool.map(subtract_per_doy,doys)
        else: # Just a sequential loop, still use of a shared memory array
            init_worker(self.inarray,self.share_input,self.dtype,self.shape,self.doyaxis,self.climate,self.outarray,None,None)
            for doy in doys:
                subtract_per_doy(doy)

        # Reconstruction from shared out array
        np_outarray = np.frombuffer(self.outarray, dtype = self.dtype).reshape(self.shape) # For shared Ctype arrays
        result = xr.DataArray(np_outarray, dims = self.dims, coords = self.coords, name = '-'.join([self.name, 'anom']))
        result.attrs = self.attrs
        logging.info(f'AnomComputer added coordinates and attributes to anom outarray with shape {result.shape} and will return as xr.DataArray with dtype {result.dtype}')
        Computer.cleanup(self)
        return result

class TimeAggregator(Computer):
    """
    Aggregates a daily time dimension that should be continuous,
    Otherwise non-neighboring values are taken together
    It returns a left stamped aggregation of ndays, trailing windows without full coverage (filled by NaN) are removed
    For non-rolling aggregation it is possible to supply a first day
    This allows a full overlap with another block-aggregated time series
    """
    def __init__(self, datapath: Path, group: str = None, ncvarname: str = None, share_input: bool = False, reduce_input: bool = False):
        Computer.__init__(self, datapath = datapath, group = group, ncvarname = ncvarname, share_input = share_input, reduce_input = reduce_input)
        assert (np.diff(self.coords['time']) == np.timedelta64(1,'D')).all(), "Time axis should be continuous daily to be allegible for aggregation"
        # Preparing and sharing the output
        self.outarray = mp.RawArray(get_corresponding_ctype(self.dtype), size_or_initializer=self.size)
        logging.info(f'TimeAggregator placed outarray of dimension {self.shape} in shared memory, dtype = {self.dtype}')

    def compute(self, nprocs, ndayagg: int = 1, method: str = 'mean', firstday: pd.Timestamp = None, rolling: bool = False):
        """
        Will call workers to aggregate a number of days, starting at the first index, stamped left, and moving one step further in case of rolling. And in case of non-rolling, starting with a certain index, stamping left and jumping further
        Will not use multiprocessing when the desired nprocs is only one
        """
        if rolling:
            # Slicing off the end where not enough days are present to aggregate
            time_axis_indices = np.arange(0,self.shape[0] - ndayagg + 1, 1)
            logging.debug(f'TimeAggregator will start rolling aggregation')
        else:
            # Getting the first day 
            try:
                which_first = int(np.where(self.coords['time'] == firstday.to_datetime64())[0]) # Returns a tuple of indices (first row then column) but we only have the first dimension
                logging.debug(f'TimeAggregator found firstday {firstday} at location {which_first} to start non-rolling aggregation')
            except:
                which_first = 0
                logging.debug(f'TimeAggregator found no firstday {firstday}, non-rolling aggregation will start at location 0')
            time_axis_indices = np.arange(which_first,self.shape[0] - ndayagg + 1, ndayagg)

        if nprocs > 1:
            with mp.Pool(processes = nprocs, initializer=init_worker, initargs=(self.inarray,self.share_input,self.dtype,self.shape,self.doyaxis,None,self.outarray,ndayagg,method)) as pool:
                pool.map(aggregate_at,time_axis_indices)
        else: # Just a sequential loop, still use of a shared memory array
            init_worker(self.inarray,self.share_input,self.dtype,self.shape,self.doyaxis,None,self.outarray,ndayagg,method)
            for timeindex in time_axis_indices:
                aggregate_at(timeindex)

        # Reconstruction from shared out array
        np_outarray = np.frombuffer(self.outarray, dtype = self.dtype).reshape(self.shape)[time_axis_indices,...] 
        coords = dict(self.coords)
        coords['time'] = coords['time'][time_axis_indices]
        result = xr.DataArray(np_outarray, dims = self.dims, coords = coords, name = '-'.join([self.name, str(ndayagg), 'roll' if rolling else 'nonroll', method]))
        result.attrs = self.attrs
        logging.info(f'TimeAggregator added coordinates and attributes to aggregated outarray with shape {result.shape} and will return as xr.DataArray with dtype {result.dtype}')
        Computer.cleanup(self)
        return result

