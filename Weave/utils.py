#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:50:34 2020

@author: straaten
"""
import numpy as np
import xarray as xr
import pandas as pd
import ctypes as ct
import matplotlib.pyplot as plt
from collections import namedtuple
from typing import Union, Callable, Tuple
from scipy.stats import rankdata, spearmanr, pearsonr, weightedtau, t
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from sklearn.calibration import calibration_curve
from .models import crossvalidate

Region = namedtuple("Region", ["name", "latmax","lonmin", "latmin", "lonmax"])

def get_europe() -> Region:
    """
    This should be interpreted as the box with the midpoints of the bounding gridcells.
    """
    return(Region("europe", 75, -30, 30, 40))

def get_nhplus() -> Region:
    """
    This is the full northern hemisphere plus a (sub)tropical part of the sourthern hemisphere.
    """
    return(Region("nhplus", 90, -180, -40, 180))

def get_nhnorm() -> Region:
    """
    This is the full northern hemisphere plus an equatorial band
    """
    return(Region("nhnorm", 90, -180, -20, 180))

def get_nhmin() -> Region:
    """
    This is the part of the northern hemisphere experiencing snow cover and sea ice.
    """
    return(Region("nhmin", 90, -180, 30, 180))

def get_nhblock() -> Region:
    """
    Untill Equator, section that includes north america on the western and india on the eastern end
    """
    return(Region("nhblock", 90, -130, 0, 100))

def get_natlantic() -> Region:
    """
    North atlantic, from newfoundland in the west and up and including the british isles in the east
    """
    return(Region("natlantic", 60, -60, 30, 0))

def get_corresponding_ctype(npdtype: type) -> type:
    simple_types = [ct.c_byte, ct.c_short, ct.c_int, ct.c_long, ct.c_longlong,
    ct.c_ubyte, ct.c_ushort, ct.c_uint, ct.c_ulong, ct.c_ulonglong,
    ct.c_float, ct.c_double, ct.c_bool,]
    nptypes = {np.dtype(t):t for t in simple_types}
    # np.float16 does not exist as a ctype. Because the ctype is just to store in shared memory and first converted for all computation and reading/wriding we use a placeholder of similar number of bytes (2)
    nptypes.update({np.dtype(np.float16):ct.c_short})
    return nptypes[np.dtype(npdtype)]

def nanquantile(array: np.ndarray, q: float) -> np.ndarray:
    """
    Get quantile along the first axis of the array. Faster than numpy, because it has only a quantile function ignoring nan's along one dimension.
    Quality checked against numpy native method.
    Check here: https://krstn.eu/np.nanpercentile()-there-has-to-be-a-faster-way/
    Modified to take both 2d and 3d array. For 1d use normal np.nanquantile.
    """
    # amount of valid (non NaN) observations along the first axis. Plus repeated version
    valid_obs = np.sum(np.isfinite(array), axis=0)
    valid_obs_full = np.repeat(valid_obs[np.newaxis,...], array.shape[0], axis=0)
    # replace NaN with maximum, but only for slices with more than one valid observation along the first axis.
    max_val = np.nanmax(array)
    array[np.logical_and(np.isnan(array), valid_obs_full > 0 )] = max_val
    # sort - former NaNs will move to the end
    array = np.sort(array, axis=0)

    # desired position as well as floor and ceiling of it
    k_arr = (valid_obs - 1) * q
    f_arr = np.floor(k_arr).astype(np.int32)
    c_arr = np.ceil(k_arr).astype(np.int32)
    fc_equal_k_mask = f_arr == c_arr

    # linear interpolation (like numpy percentile) takes the fractional part of desired position
    floor_val = _zvalue_from_index(arr=array, ind=f_arr) * (c_arr - k_arr)
    ceil_val = _zvalue_from_index(arr=array, ind=c_arr) * (k_arr - f_arr)

    quant_arr = floor_val + ceil_val
    quant_arr[fc_equal_k_mask] = _zvalue_from_index(arr=array, ind=k_arr.astype(np.int32))[fc_equal_k_mask]  # if floor == ceiling take floor value

    return(quant_arr)

def _zvalue_from_index(arr: np.ndarray, ind: np.ndarray) -> np.ndarray:
    """private helper function to work around the limitation of np.choose() by employing np.take()
    arr has to be a 3D array or 2D (inferred by ndim)
    ind has to be an array without the first arr dimension containing values for z-indicies to take from arr
        self.encoding = data.encoding
        self.share_input = share_input
    See: http://stackoverflow.com/a/32091712/4169585
    This is faster and more memory efficient than using the ogrid based solution with fancy indexing.
    """
    ndim = arr.ndim
    # get number of columns and rows
    if ndim == 3:
        _,nC,nR = arr.shape
        # get linear indices and extract elements with np.take()
        idx = nC*nR*ind + np.arange(nC*nR).reshape((nC,nR))
    elif ndim == 2:
        _,nC = arr.shape
        idx = nC*ind + np.arange(nC)
        
    return(np.take(arr, idx))

def collapse_restore_multiindex(df: Union[pd.DataFrame,pd.Series], axis: int, names: list = None, ignore_level: int = None, separator: str = '.', inplace: bool = False) -> list:
    """
    Used to collapse a pandas multi_index, for instance for the use case where a plotting procedure requires single level understandable column names 
    Ignore_level only used when collapsing, by calling droplevel, so could also be a string according to the pandas api
    Returns a list with the old column names, if not inplace also the new frame
    """
    assert (axis == 0) or (axis == 1), "can collapse/restore either the index or the columns, choose axis 0 or 1"
    if axis == 0:
        what = 'index' 
    else:
        what = 'columns' 
    index = getattr(df, what)
    if isinstance(index, pd.MultiIndex): # In this case we are going to collapse into a string
        if not ignore_level is None:
            index = index.droplevel(ignore_level)               
        names = index.names.copy()
        index = pd.Index([separator.join([str(c) for c in col]) for col in index.values], dtype = object, name = 'collapsed')
    else: # In this case we are going to restore from string, new levels will all be string dtype
        index = pd.MultiIndex.from_tuples([tuple(string.split(separator)) for string in index.values], names = names)
    if inplace:
        setattr(df, what, index)
        return(None, names)
    else:
        df = df.copy()
        setattr(df, what, index)
        return(df, names)

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

def rankdirection(x,y):
    """
    Returns a rank array (0 most important) with emphasis on negative x when negative relation
    emphasis on positive x when positive overall relation
    """
    ranks = rankdata(x, method = 'ordinal')
    if pearsonr(x = x, y = y)[0] < 0:
        return ranks
    else:
        return ranks.max() - ranks

def kendall_choice(data: np.ndarray) -> float:
    """
    Takes in two timeseries in a 2D array (n_obs,[x,y]). computes weighted kendall tau. Weighting direction in terms of precursor ranks is chosen based on pearsons
    Significance is not implemented
    """
    corr, _ = weightedtau(x = data[:,0], y = data[:,1], rank=rankdirection(x = data[:,0], y = data[:,1]))
    return corr

def kendall_predictand(data: np.ndarray) -> float:
    """
    Takes in two timeseries in a 2D array (n_obs,[x,y]). computes weighted kendall tau.
    Weights are determined by the y (done by rank is None, meaning that weighting is determined by x)
    (rank = True, would compute twice, once with x and second with y)
    Significance is not implemented but might be obtained by bootstrapping
    """
    corr, _ = weightedtau(x = data[:,1], y = data[:,0], rank = None)
    return corr

def quick_kendall(data: np.ndarray) -> tuple:
    """
    no significance testing
    """
    corr, pval = weightedtau(x = data[:,1], y = data[:,0], rank = None)
    return (corr, 1e-9)

def pearsonr_wrap(data: np.ndarray) -> tuple:
    """
    wraps scipy pearsonr by decomposing a 2D dataarray (n_obs,[x,y]) into x and y
    """
    return pearsonr(x = data[:,0], y = data[:,1]) 

def spearmanr_wrap(data: np.ndarray) -> tuple:
    """
    wraps scipy spearmanr by decomposing a 2D dataarray (n_obs,[x,y]) into x and y
    """
    return spearmanr(a = data[:,0], b = data[:,1]) 

def spearmanr_cv(n_folds: int, split_on_year: bool = True, sort = False) -> Callable:
    """
    Constructor to supply function that computes correlation only on the training set
    Wrapping input after crossval into spearmanr, and wrapping output after crossval to array
    The returned function should be called with X_in and y_in arguments such that those can be filtered from kwargs and supplied as X_train and y_train to the wrapper
    """
    def wrapper(X_train, y_train, X_val, y_val = None) -> pd.Series: # Wrapping input format of pandas to spearmanr
        return pd.DataFrame([spearmanr(X_train, y_train)], columns = ['corr','pvalue'], index = X_val.index[[0]]) # Returning the start timestamp of the validation fold (later sorted by crossvalidate if split_on_year and sorted are True)
    interimfunc = crossvalidate(n_folds = n_folds, split_on_year = split_on_year, sort = sort)(wrapper) 
    def returnfunc(*args, **kwargs) -> np.ndarray: # Function to modify output to be written to shared array
        return interimfunc(*args, **kwargs) # Returns a dataframe
    return returnfunc
        

def chi(responseseries: xr.DataArray, precursorseries: xr.DataArray, nq: int = 100, qlim: tuple = None, alpha: float = 0.05, trunc: bool = True, full = False) -> tuple:
    """
    modified from https://github.com/cran/texmex/blob/master/R/chi.R
    Conversion to ECDF space. Computation of chi over a range of quantiles
    """
    assert len(responseseries) == len(precursorseries), '1D series should have equal length'
    n = len(responseseries)
    # To ecdf space [0-1]
    t_method = 'ordinal' # for scipy.stats.rankdata, similar to 'first' for rank in R
    ecdfx = rankdata(precursorseries, method = t_method)/(n+1) # Gumbel plotting position
    ecdfy = rankdata(responseseries, method = t_method)/(n+1) # Gumbel plotting position

    data = np.stack([ecdfx,ecdfy], axis = 1)
    
    rowmax = np.max(data, axis = 1) # Both X and Y are below this quantile
    rowmin = np.min(data, axis = 1) # Both X and Y are above this quantile

    # To check whether the desired quantile range can also be reliably estimated
    eps = np.finfo('float64').eps
    qlim_empirical = (min(rowmax) + eps, max(rowmin) - eps) # Eps gives necessary difference for two numbers to be numerically the same. This makes sure that always at least one data pair is simultaneously below the lower bount and simultaneously above the upper bound. (Avoids log(0)) 
    if not qlim is None:
        assert qlim_empirical[0] < qlim[0]
        assert qlim_empirical[1] > qlim[1]
        assert qlim[0] < qlim[1], 'upper limit should be higher than lower'
    else:
        qlim = qlim_empirical

    # Construct the quantile range, at each point we will calculate chi and chibar
    qs = np.linspace(qlim[0],qlim[1], num = nq)
    # Broadcasting to a boolean smaller than array of shape (nq,n) and then mean over axis 1 to get (nq,)
    prob_below = np.mean(rowmax[np.newaxis,:] < qs[:,np.newaxis], axis = 1)
    prob_above = np.mean(rowmin[np.newaxis,:] > qs[:,np.newaxis], axis = 1)

    chiq = 2 - np.log(prob_below)/np.log(qs)
    chibarq = (2 * np.log(1-qs))/np.log(prob_above) - 1 # Limiting value 1 means asymptotic dependence --> look at chi. Limiting value less then one means asymptotic independence. --> chi irrelevant

    # Estimate the standard error (variance), assumes independence of observations
    #chiqvar = ((1/np.log(qs)**2)/qs * (1-qs))/n
    #chibarqvar = (((4 * np.log(1-qs)**2)/(np.log(chibarq)**4 * chibarq**2)) * chibarq * (1-chibarq))/n

    if full:
        return (chiq, chibarq)
    else:
        # We basically want to get the dependence strengt for those pairs that are asymptotically dependent. 
        if chibarq[-1] > 0.6:
            return (chiq[-1], 1e-9)
        else:
            return (np.nan,np.nan) # Artificial creation of significance

def bootstrap(n_draws: int, return_numeric = True, blocksize: int = None, quantile: Union[int, list] = None) -> Callable:
    """
    Bootstrapping a function that takes 1D/2D data array as a first argument 
    By resampling this array along the zeroth axis.
    Resampling is done with non-overlapping blocks when a blocksize is given
    the function should return only one value. Collects n_draws return values in an array
    Either returns those, when no quantiles are given
    Or returns the requested quantiles of the collection 
    """
    def actual_decorator(func: Callable) -> Callable:
        if not return_numeric:
            assert quantile is None, 'If your function is going to return objects (return_numeric = False) you cannot extract quantiles'
            return_type = object
        else:
            return_type = np.nan
        def bootstrapper(*args, **kwargs) -> np.ndarray:
            collection = np.full((n_draws,), return_type)
            try: # Fish the data from any of the arguments, either called data or the first argument
                data = kwargs.pop('data')
            except KeyError:
                data = args[0]
            n_obs = len(data)
            if blocksize is not None: # Do some precomputation
                blocks = np.array_split(np.arange(n_obs), np.arange(blocksize,n_obs,blocksize))
            for i in range(n_draws):
                if blocksize is None: # Determine indices along the zeroth axis. np.random.choice(data,n_obs,replace = True) does not work with 2D data
                    indices = np.random.randint(low = 0, high = n_obs, size = n_obs)
                else:
                    indices = np.concatenate([blocks[i] for i in np.random.randint(low = 0, high = len(blocks), size = len(blocks))])
                collection[i] = func(data[indices,...], *args[1:], **kwargs)
            if quantile is None:
                return collection
            else:
                return np.array(np.nanquantile(collection, quantile))
        return bootstrapper
    return actual_decorator

def add_pvalue(func: Callable) -> Callable:
    """
    Converts an empirical distribrution (array) of a sample statistic
    as returned by the function to its mean estimate
    and a p-value for the probability that that value would be the result
    of an assumed True Null Hypothesis. This Null hypothesis is that 
    of a student t dist with similar variability (estimated by std_err) and true statistic of zero
    p-value is the parametric estimate of the two tailed probability of producing abs(estimate)
    (copying scipy)
    """
    def wrapper(*args, **kwargs) -> Tuple[float,float]:
        sample_dist = func(*args, **kwargs)
        estimate = sample_dist.mean()
        std_err_estimate = sample_dist.std() 
        n_samples = len(sample_dist)
        return estimate, 2*t.sf(x = abs(estimate), df = n_samples - 2, loc = 0, scale = std_err_estimate)
        
    return wrapper

def get_timeserie_properties(series: pd.Series, submonths: list = None, scale_trend_intercept = True, auto_corr_at: list = [1,5]) -> pd.Series:
    """ 
    Function to be called on a timeseries (does not need to be contiguous)
    Extracts some statistics that can be of interest and returns them as a Series
    If you want only certain months submonths of the series to be taken into account that provide those as a list of integers 
    Also it tries to lag the series with a given number of days, to compute autocorrelation
    """
    if not submonths is None:
        series = series.loc[series.index.month.map(lambda m: m in submonths)]
    std = series.std()
    mean = series.mean() 
    length = len(series)
    n_nan = series.isna().sum()
    series = series.dropna() # Remove nans
    lm = LinearRegression()
    if scale_trend_intercept:
        lm.fit(y = scale(series), X = series.index.year.values.reshape(-1,1))
    else:
        lm.fit(y = series, X = series.index.year.values.reshape(-1,1))
    trend = float(lm.coef_) # (standardized) coefficient / yr
    intercept = lm.intercept_
    # Smoothness is the autocorrelation at a lag of 1 day and 5 days.
    results = pd.Series({'std':std,'mean':mean,'length':length, 'n_nan':n_nan,'trend':trend, 'intercept':intercept})
    for lag in auto_corr_at:
        lagged = pd.Series(series.values, index = series.index - pd.Timedelta(f'{lag}D'), name = f'{lag}D')
        results.loc[f'auto{lag}'] = pd.merge(left = series, right = lagged, left_index=True, right_index=True, how = 'inner').corr().iloc[0,-1]
    return(results)

def brier_score_clim(p: float) -> float:
    """
    Determines the climatological reference score for an event with a probability p
    bs_ref = p*(p-1)**2 + (1-p)*(p-0)**2
    """
    return p*(p-1)**2 + (1-p)*p**2

def reliability_plot(y_true: pd.Series, y_probs: Union[pd.Series,pd.DataFrame], nbins: int = 10):
    """
    Computes the calibration curve for probabilistic predictions of a binary variable
    The true binary labels are supplied by y_true
    The matching probabilistic predictions (same row-index) are supplied in y_probs,
    different predictions can be supplied as columns 
    These predictions are binned and for each bin the corresponding frequency is computed
    returns the figure and two axes
    """
    assert np.all(y_true.index == y_probs.index), 'indices should match'

    fig = plt.figure(figsize=(7,7))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfect")
    ax2.set_yscale('log')
    if isinstance(y_probs, pd.Series):
        y_probs = y_probs.to_frame()

    for column_tuple in y_probs.columns:
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_probs[column_tuple], n_bins=nbins)
        ax1.plot(mean_predicted_value,fraction_of_positives, 's-',label = str(column_tuple))
        ax2.hist(y_probs[column_tuple], range=(0,1), bins=nbins, histtype="step", lw=2)

    ax1.legend(title = ','.join(y_probs.columns.names))
    ax1.set_ylabel('conditional observed frequency')
    ax2.set_ylabel('counts')
    ax2.set_xlabel('mean predicted probability')

    return fig, [ax1,ax2]
