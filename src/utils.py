#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:50:34 2020

@author: straaten
"""
import numpy as np
import ctypes as ct
from collections import namedtuple

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

def get_nhmin() -> Region:
    """
    This is the part of the northern hemisphere experiencing snow cover and sea ice.
    """
    return(Region("nhmin", 90, -180, 30, 180))

def get_corresponding_ctype(npdtype: type) -> type:
    simple_types = [ct.c_byte, ct.c_short, ct.c_int, ct.c_long, ct.c_longlong,
    ct.c_ubyte, ct.c_ushort, ct.c_uint, ct.c_ulong, ct.c_ulonglong,
    ct.c_float, ct.c_double, ct.c_bool,]
    nptypes = {np.dtype(t):t for t in simple_types}
    # np.float16 does not exist as a ctype. Because the ctype is just to store in shared memory and first converted for all computation and reading/wriding we use a placeholder of similar number of bytes (2)
    nptypes.update({np.dtype(np.float16):ct.c_short})
    return nptypes[np.dtype(npdtype)]

def nanquantile(array, q):
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

def _zvalue_from_index(arr, ind):
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
