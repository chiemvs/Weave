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

def init_worker(inarray, dtype, shape, intimeaxis, responseseries, outarray, outdtype, outshape, laglist, asofunc, do_crossval):
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
    var_dict['do_crossval'] = do_crossval
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
    do_crossval = var_dict['do_crossval']

    if np.isnan(inarray).all(): # We are dealing with an empty cell
        logging.debug(f'Worker found empty cell at {spatial_index}, returning np.nan')
        out_index = (slice(None),slice(None),slice(None)) + spatial_index
        outarray[out_index] = np.nan # Writes np.nan both for the association strength and the significance.
    elif (inarray == inarray[0]).all():
        logging.debug(f'Worker found constant cell at {spatial_index}, correlation not defined, returning np.nan')
        out_index = (slice(None),slice(None),slice(None)) + spatial_index
        outarray[out_index] = np.nan # Writes np.nan both for the association strength and the significance.
    else:
        # Now inarray is a one dimensional numpy array, we need the original and series time coordinates to do the lagging
        intimeaxis = var_dict['intimeaxis']
        subsettimeaxis = var_dict['responseseries'].time
        laglist = var_dict['laglist']
        # extract some info 
        asofunc = var_dict['asofunc']
        if asofunc.is_partial: # attribute is set at initialization of the associator 
            timeagg = asofunc.timeagg # Needed when we want to compute partial correlation and provide also X_t-1 (the stepsize is exactly the size of the window)
            # First column of X data is normal. Second column is the value of (-windowsize) ago
            intimeaxis = intimeaxis[timeagg:]
            oriset = xr.DataArray(np.column_stack([inarray[timeagg:],inarray[:-timeagg]]), dims = ('time','what'), coords = {'time':intimeaxis,'what':['t0','t-1']})
            logging.debug(f'Worker detected is_partial. Creating t-1 with {timeagg} days reduced length from {inarray.shape[0]} to {oriset.shape[0]}')
        else:
            oriset = xr.DataArray(inarray, dims = ('time',), coords = {'time':intimeaxis})

        logging.debug(f'Worker starts lagging, detrending and correlating of cell {spatial_index} for lags {laglist}')
        for lag in laglist: # Units is days
            oriset['time'] = intimeaxis - pd.Timedelta(str(lag) + 'D') # Each point in time is assigned to a lagged date
            X_set = oriset.reindex_like(subsettimeaxis).dropna(dim = 'time') # We only retain the points assigned to the dates of the response timeseries. Potentially this generates some nans. Namely when the response series extends further than the precursor (ERA5 vs ERA5-land) only retain non_nan because detrend cant handle them
            X_set.values = detrend(X_set, axis = 0) # Response was already detrended
            y_set = var_dict['responseseries'].loc[X_set.time] # We want both as matching series
            # For non-cv asofuncs we feed the two dataarrays (n_obs,) of the precursor (x) and (y), for cv we to provide input as pandas and capture output of the crossvalidated function, all that should be already in asofunc
            out_index = (laglist.index(lag), slice(None), slice(None)) + spatial_index # Remember the shape of (len(lagrange),self.n_folds,2) + spatdims. So regardless of crossval at least one fold is mimiced
            if do_crossval:
                outarray[out_index] = asofunc(X_in = X_set.to_pandas().to_frame() if X_set.ndim == 1 else X_set.to_pandas(), y_in = y_set.to_pandas()) # Outarray returns a dataframe. We want X_set as a dataframe in there (even when 1D) because of time axis sorting and recording in potial grouped Kfold asofunc
            else:
                outarray[out_index] = asofunc(X_set, y_set) # Asofunc should accept two 1D data arrays and return (corr,pvalue)

class Associator(Computer):

    def __init__(self, responseseries: xr.DataArray, data: xr.DataArray, laglist: list, association: Callable, is_partial: bool = False, timeagg: int = None, n_folds: int = None) -> None:
        """
        Is fed with an already loaded data array, this namely is an intermediate timeaggregated array
        Always shares the input array, such that it can be deleted after initialization
        responseseries should already be a 1D detrended subset
        laglist is how far to shift the data in number of days. A lag of -10 means that data values of 1979-01-01 are associated to the response values at 1979-01-11
        Choice to supply the function that determines association between two timeseries
        Should return (corr_float, p_value_float)
        If the association should be computed in cross-val mode (computation on train only) then supply an n_folds parameter (has to match the return size for the decorated association func). If it is a partial correlation, some extra lagging needs to take place, and you also need to supply the timeagg
        """
        Computer.__init__(self, share_input=True, data = data) # Fills a shared array at self.inarray


        self.laglist = laglist
        self.asofunc = association
        self.asofunc.is_partial = is_partial
        self.asofunc.timeagg = timeagg
        self.responseseries = responseseries
        """
        For each cell we are going to produce 2 association values, a strength and a significance
        This is done per lag and possibly per fold. 
        If there are no folds we are going to mimic a single one for consistent amount dimensions in the outarray.
        The dimensions of the shared outarray therefore become (nlags, nfolds, 2) + spatshape
        """
        if n_folds is None:
            self.n_folds = 1
            self.do_crossval = False
            logging.debug('mimicing a fold of size 1. NO cross-validated association computing')
        else:
            self.n_folds = n_folds
            self.do_crossval = True
            logging.debug(f'found n_folds {n_folds}. Cross-validated association computing')
        self.outshape = (len(self.laglist),self.n_folds,2) + self.shape[1:]
        self.outsize = self.size // self.shape[0] * len(self.laglist) * self.n_folds * 2
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
        with mp.Pool(processes = nprocs, initializer=init_worker, initargs=(self.inarray, self.dtype, self.shape, self.coords['time'], self.responseseries, self.outarray, self.outdtype, self.outshape, self.laglist, self.asofunc, self.do_crossval)) as pool:
            results = pool.map(lag_subset_detrend_associate,indextuples)

        # Reconstruction from shared out array
        np_outarray = np.frombuffer(self.outarray, dtype = self.outdtype).reshape(self.outshape) # For shared Ctype arrays
        # We are going to store with dimension (lags, spatdims), either masked or seperate as dataset
        returnshape = list(self.outshape)
        returnshape.pop(2) # Popping out the fixed dimension of size 2 (corr, pval) which will collapse to a single masked corr
        logging.info(f'Associator recieved all computed correlations and pvalues and will use those to produce a masked array of shape {returnshape}')
        # Mask the correlation array with the p-values? And record the fraction of cells that became unsignificant
        mask = np.full(returnshape, True, dtype = np.bool) # Everything starts with not-rejected null hyp and masked
        for lagindex, foldindex in itertools.product(range(len(self.laglist)),range(self.n_folds)):
            pfield = np_outarray[lagindex,foldindex,1,...]
            nonan_indices = np.where(~np.isnan(pfield)) # Tuple with arrays of indices. one array if only 1D, but two in a 2D tuple if two spatial dimensions
            pfield_flat = pfield[nonan_indices].flatten() # As multipletest can only handle 1D p-value arrays, and subsetting to not-nan because we don't want nan to contribute to n_tests
            if pfield_flat.size == 0: # No valid correlations defined, nothing could be significant, no need for multiple testing
                fracbefore = .0
                fracafter = .0
            else:
                fracbefore = round((pfield_flat < alpha).sum() / pfield_flat.size , 5)
                reject, pfield_flat, garbage1, garbage2 = multipletests(pfield_flat, alpha = alpha, method = 'fdr_bh', is_sorted = False)
                fracafter = round((pfield_flat < alpha).sum() / pfield_flat.size , 5)
                mask[(lagindex,foldindex) + nonan_indices] = ~reject # First part of this joined tuple is only one number, the index of the current lag in the lagaxis
            if self.do_crossval:
                self.attrs.update({f'lag{self.laglist[lagindex]}_fold{foldindex}':f'frac p < {alpha} before: {fracbefore}, after: {fracafter}'})
            else:
                self.attrs.update({f'lag{self.laglist[lagindex]}':f'frac p < {alpha} before: {fracbefore}, after: {fracafter}'})
            
        # Preparing the to be returned dataarray
        if self.do_crossval:
            coords = {'lag':self.laglist, 'fold':list(range(self.n_folds))}
            dims = ('lag','fold') + self.dims[1:]
            corr = np.ma.masked_array(data = np_outarray[:,:,0,...], mask = mask)
        else:
            coords = {'lag':self.laglist}
            dims = ('lag',) + self.dims[1:]
            corr = np.ma.masked_array(data = np_outarray[:,:,0,...], mask = mask).squeeze(axis = 1) # Squeezing out the mimiced fold dimension
        for dim in self.dims[1:]:
            coords.update({dim:self.coords[dim]})
        corr = xr.DataArray(data = corr, dims = dims, coords = coords, name = 'correlation')
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
