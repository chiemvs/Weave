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
import pandas as pd
import logging
import itertools
from scipy.signal import detrend
from statsmodels.stats.multitest import multipletests 
from typing import Callable
from .utils import get_corresponding_ctype
from .processing import Computer

var_dict = {}

def init_worker(inarray, dtype, shape, intimeaxis, responseseries, outarray, outdtype, outshape, laglist, asofunc):
    var_dict['inarray'] = inarray
    var_dict['dtype'] = dtype
    var_dict['shape'] = shape
    var_dict['intimeaxis'] = intimeaxis
    var_dict['responseseries'] = responseseries
    var_dict['outarray'] = outarray 
    var_dict['outdtype'] = outdtype
    var_dict['outshape'] = outshape
    var_dict['laglist'] = laglist
    var_dict['asofunc'] = asofunc # Association function defined in utils 
    logging.debug('this initializer has populated a global dictionary')

def lag_subset_detrend_associate(spatial_index: tuple):
    """
    Worker function. Initialized with acess to an array (first axis is time, others are the spatial_index) , passed at inheritance
    Also needs access to the original xr time axis, the xr response series, the lagrange (in days, and probably negative, as then data is treated as preceding the response)
    The response series is used both for subsetting with its time axis
    and for its values to compute association
    """
    full_index = (slice(None),) + spatial_index # Assumes the unlimited time on the first axis
    inarray = np.frombuffer(var_dict['inarray'], dtype = var_dict['dtype']).reshape(var_dict['shape'])[full_index]
    outarray = np.frombuffer(var_dict['outarray'], dtype = var_dict['outdtype']).reshape(var_dict['outshape'])
    if np.isnan(inarray).all(): # We are dealing with an empty cell
        logging.debug(f'Worker found empty cell at {spatial_index}, returning np.nan')
        out_index = (slice(None),slice(None)) + spatial_index
        outarray[out_index] = np.nan # Writes np.nan both for the association strength and the significance.
    else:
        # Now inarray is a one dimensional numpy array, we need the original and series time coordinates to do the lagging
        intimeaxis = var_dict['intimeaxis']
        oriset = xr.DataArray(inarray, dims = ('time',), coords = {'time':intimeaxis})
        subsettimeaxis = var_dict['responseseries'].time
        # Prepare the computation results. First axis: lags, second axis: [corrcoef, pvalue]
        laglist = var_dict['laglist']
        logging.debug(f'Worker starts lagging, detrending and correlating of cell {spatial_index} for lags {laglist}')
        for lag in laglist: # Units is days
            oriset['time'] = intimeaxis - pd.Timedelta(str(lag) + 'D') # Each point in time is assigned to a lagged date
            combined = np.column_stack((oriset.reindex_like(subsettimeaxis), var_dict['responseseries'])) # We only retain the points assigned to the dates of the response timeseries. Potentially this generates some nans. Namely when the response series extends further than the precursor (ERA5 vs ERA5-land)
            # Combined is an array (n_obs,2) with zeroth column the precursor (x) and first column the response (y) 
            combined = combined[~np.isnan(combined[:,0]),:] # Then we only retain non-nan. detrend cant handle them
            combined[:,0] = detrend(combined[:,0]) # Response was already detrended
            out_index = (laglist.index(lag), slice(None)) + spatial_index # Remember the shape of (len(lagrange),2) + spatdims 
            outarray[out_index] = var_dict['asofunc'](combined) # Asofunc should accept 2D data array and return (corr,pvalue)

class Associator(Computer):

    def __init__(self, responseseries: xr.DataArray, data: xr.DataArray, laglist: list, association: Callable):
        """
        Is fed with an already loaded data array, this namely is an intermediate timeaggregated array
        Always shares the input array, such that it can be deleted after initialization
        responseseries should already be a 1D detrended subset
        laglist is how far to shift the data in number of days. A lag of -10 means that data values of 1979-01-01 are associated to the response values at 1979-01-11
        Choice to supply the function that determines association between two timeseries
        Should return (corr_float, p_value_float)
        """
        Computer.__init__(self, share_input=True, data = data) # Fills a shared array at self.inarray


        self.laglist = laglist
        self.asofunc = association
        self.responseseries = responseseries
        # For each cell we are going to produce 2 association values, a strength and a significance
        # for each lag. The dimensions of the outarray therefore become (nlags, 2 ) + spatshape
        # Which can be too large to handle by return values, therefore we share an outarray
        self.outshape = (len(self.laglist),2) + self.shape[1:]
        self.outsize = self.size // self.shape[0] * len(self.laglist) * 2
        self.outdtype = np.dtype(np.float32)
        self.outarray = mp.RawArray(get_corresponding_ctype(self.outdtype), size_or_initializer=self.outsize)
        logging.info(f'Associator placed outarray of dimension {self.outshape} in shared memory')

    def compute(self, nprocs, alpha: float = 0.05):
        # Prepare all spatial coordinates of the cells, to map our function over
        # Has to produce tuples of (coord_dim1, coord_dim2), skipping the zeroth time dim
        #spatcoords = tuple(self.coords[dim].values for dim in self.dims[1:])
        #self.coordtuples = itertools.product(*spatcoords)
        spatdimlengths = tuple(list(range(length)) for length in self.shape[1:])
        indextuples = itertools.product(*spatdimlengths)
        with mp.Pool(processes = nprocs, initializer=init_worker, initargs=(self.inarray, self.dtype, self.shape, self.coords['time'], self.responseseries, self.outarray, self.outdtype, self.outshape, self.laglist, self.asofunc)) as pool:
            results = pool.map(lag_subset_detrend_associate,indextuples)

        # Reconstruction from shared out array
        np_outarray = np.frombuffer(self.outarray, dtype = self.outdtype).reshape(self.outshape) # For shared Ctype arrays
        # We are going to store with dimension (lags, spatdims), either masked or seperate as dataset
        returnshape = list(self.outshape)
        returnshape.pop(1)
        logging.info(f'Associator recieved all computed correlations and pvalues and will use those to produce a masked array of shape {returnshape}')
        # Mask the correlation array with the p-values? And record the fraction of cells that became unsignificant
        mask = np.full(returnshape, True, dtype = np.bool) # Everything starts with not-rejected null hyp and masked
        for lag in self.laglist:
            pfield = np_outarray[self.laglist.index(lag),1,...]
            nonan_indices = np.where(~np.isnan(pfield)) # Tuple with arrays of indices. one array if only 1D, but two in a 2D tuple if two spatial dimensions
            pfield_flat = pfield[nonan_indices].flatten() # As multipletest can only handle 1D p-value arrays, and subsetting to not-nan because we don't want nan to contribute to n_tests
            fracbefore = round((pfield_flat < alpha).sum() / pfield_flat.size , 5)
            reject, pfield_flat, garbage1, garbage2 = multipletests(pfield_flat, alpha = alpha, method = 'fdr_bh', is_sorted = False)
            fracafter = round((pfield_flat < alpha).sum() / pfield_flat.size , 5)
            mask[(self.laglist.index(lag),) + nonan_indices] = ~reject # First part of this joined tuple is only one number, the index of the current lag in the lagaxis
            self.attrs.update({f'lag{lag}':f'frac p < {alpha} before: {fracbefore}, after: {fracafter}'})
            
        # Preparing the to be returned dataarray
        coords = {'lag':self.laglist}
        dims = ('lag',) + self.dims[1:]
        for dim in self.dims[1:]:
            coords.update({dim:self.coords[dim]})
        corr = xr.DataArray(np.ma.masked_array(data = np_outarray[:,0,...], mask = mask), dims = dims, coords = coords, name = 'correlation')
        corr.attrs = self.attrs
        Computer.cleanup(self)
        return corr

def composite1d(responseseries: xr.DataArray, precursorseries: xr.DataArray, quant: float):
    indexer = responseseries > responseseries.quantile(quant)
    mean = float(precursorseries.where(indexer, drop = True).mean())
    return (mean, 1e-9) # Return a hardcoded significance. Only because that was how multiple testing worked.

def composite(responseseries: xr.DataArray, data: xr.DataArray, quant: float = 0.9):
    """
    Basic function that subsets the supplied already loaded data array, along the first axis, based on exceedence of the quantile in the response timeseries.
    Quantile can also be a list of quantiles
    And returns the mean of that. No detrending (of data). Response could already have been detrended.
    """
    indexer = responseseries > responseseries.quantile(quant)
    if isinstance(quant, list):
        returns = [None] * len(quant)
        for q in quant:
            returns[quant.index(q)] = data.where(indexer.sel(quantile = q), drop = True).mean(dim = data.dims[0])
        return(xr.concat(returns, dim = 'quantile'))
    else:
        return(data.where(indexer, drop = True).mean(dim = data.dims[0]))
