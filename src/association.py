#!/usr/bin/env python3

"""
Quantify the association between one variable field (at one time aggregation) and a single response variable.
Meaning that this single response variable is already spatial cluster mean and is subsetted and detrended. 
Basically does lagging, selection of the subset, detrending
Aimed at acting per spatial cell. 
"""

import numpy as np
import xarray as xr
import multiprocessing as mp
import logging
import itertools
from scipy.stats import pearsonr
from scipy.signal import detrend
from .utils import get_corresponding_ctype
from .processing import Computer

var_dict = {}

def init_worker(inarray, dtype, shape, intimeaxis, responseseries, outarray, outdtype, outshape, laglist):
    var_dict['inarray'] = inarray
    var_dict['dtype'] = dtype
    var_dict['shape'] = shape
    var_dict['intimeaxis'] = intimeaxis
    var_dict['responseseries'] = responseseries
    var_dict['outarray'] = outarray 
    var_dict['outdtype'] = outdtype
    var_dict['outshape'] = outshape
    var_dict['laglist'] = laglist
    logging.debug('this initializer has populated a global dictionary')

def lag_subset_detrend_associate(spatial_index: tuple):
    """
    Worker function. Initialized with acess to an array (first axis is time, others are the spatial_index) , passed at inheritance
    Also needs access to the original xr time axis, the xr response series, the lagrange (in days)
    The response series is used both for subsetting with its time axis
    and for its values to compute association
    """
    full_index = (slice(None),) + spatial_index # Assumes the unlimited time on the first axis
    inarray = np.frombuffer(var_dict['inarray'], dtype = var_dict['dtype']).reshape(var_dict['shape'])[full_index]
    outarray = np.frombuffer(var_dict['outarray'], dtype = var_dict['outdtype']).reshape(var_dict['outshape'])
    # Now inarray is a one dimensional numpy array, we need the original and series time coordinates to do the lagging
    intimeaxis = var_dict['intimeaxis']
    oriset = xr.DataArray(inarray, dims = ('time',), coords = {'time':intimeaxis})
    subsettimeaxis = var_dict['responseseries'].time
    # Prepare the computation results. First axis: lags, second axis: [corrcoef, pvalue]
    laglist = var_dict['laglist']
    for lag in laglist: # Units is days
        oriset['time'] = intimeaxis - pd.Timedelta(str(lag) + 'D') # Each point in time is assigned to a lagged date
        subset = oriset.reindex_like(subsettimeaxis) # We only retain the points assigned to the dates of the response timeseries
        subset = detrend(subset) # Only a single axis
        out_index = (laglist.index(lag), slice(None)) + spatial_index # Remember the shape of (len(lagrange),2) + spatdims 
        outarray[out_index] = pearsonr(var_dict['responseseries'], subset) # Returns (corr,pvalue)

class Associator(Computer):

    def __init__(self, responseseries: xr.DataArray, data: xr.DataArray, laglist: list):
        """
        Is fed with an already loaded data array, this namely is an intermediate timeaggregated array
        Always shares the input array, such that it can be deleted after initialization
        responseseries should already be a 1D detrended subset
        laglist is how far to shift the data in number of days
        """
        Computer.__init__(self, share_input=True, data = data) # Fills a shared array at self.inarray

        # Prepare all spatial coordinates of the cells, to map our function over
        # Has to produce tuples of (coord_dim1, coord_dim2), skipping the zeroth time dim
        #spatcoords = tuple(self.coords[dim].values for dim in self.dims[1:])
        #self.coordtuples = itertools.product(*spatcoords)
        spatdimlengths = tuple(list(range(length)) for length in self.shape[1:])
        self.indextuples = itertools.product(*spatdimlengths)

        self.laglist = laglist
        self.responseseries = responseseries
        # For each cell we are going to produce 2 association values, a strength and a significance
        # for each lag. The dimensions of the outarray therefore become (nlags, 2 ) + spatshape
        # Which can be too large to handle by return values, therefore we share an outarray
        self.outshape = (len(self.laglist),2) + self.shape[1:]
        self.outsize = self.size // self.shape[0] * len(self.laglist) * 2
        self.outdtype = np.dtype(np.float32)
        self.outarray = mp.RawArray(get_corresponding_ctype(self.outdtype), size_or_initializer=self.outsize)
        logging.info(f'Associator placed outarray of dimension {self.outshape} in shared memory')

    def compute(nprocs):
        # Prepare all the associated spatial coordinates
        with mp.Pool(processes = nprocs, initializer=init_worker, initargs=(self.inarray, self.dtype, self.shape, self.coords['time'], self.responseseries, self.outarray, self.outdtype, self.outshape, self.laglist)) as pool:
            results = pool.map(lag_subset_detrend_associate,self.indextuples)

        # Debugged the workers. Continue with restructuring the output

