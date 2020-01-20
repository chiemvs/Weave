#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for the T2m clustering of sem. 
"""
import sys
sys.path.append('/usr/people/straaten/Documents/RGCPD/clustering')
from clustering_spatial import *
import xarray as xr
import numpy as np
from scipy.spatial.distance import jaccard

# Land-sea mask derived from sea ice concentration
siconc = xr.open_dataarray('/nobackup_1/users/straaten/ERA5/siconc/siconc_nhmin.nc', group = 'mean')[0]
t2m = xr.open_dataarray('/nobackup_1/users/straaten/ERA5/t2m/t2m_europe.nc', group = 'mean')
t2m = t2m[t2m.time.dt.season == 'JJA', :,:]

mask = siconc.sel(latitude = t2m['latitude'], longitude = t2m['longitude']).isnull()
mask1d = mask.stack({'latlon':['latitude','longitude']})

# Make into a memmap? Or small enough to put to each of the subprocesses?
exceedence = binary_occurences_quantile(t2m, q=95).astype('bool') # Binary array. First reduces along one dimension to get the local q80 in time
exceedence = exceedence.stack({'latlon':['latitude','longitude']}) # Flatten it to 1D
exceedence = exceedence[:,mask1d.values] # Remove masked out 

# Jaccard distance between two 1D boolean arrays, loop to take on 1D and 2D matrix
def jaccard_1d_2d(array1d, array2d):
    """
    input 1d [n], 2d [n,m], returns[m]
    """
    ret = np.apply_along_axis(jaccard, 0, array2d, **{'v':array1d})
    return(ret)

# Initialize result gathering
ncells = exceedence.shape[-1]
n_triangular = int((ncells**2 - ncells)/2)
distmat = np.full((n_triangular), 1, dtype = 'float16') # Initialize at biggest possible distance. Pretty big in memory. Perhaps initialize as a shared Ctype multiprocessing array?
                
# triangular loop. Write to the 1D maxcormat matrix
# This is all sequential code. Both finding the maxima in non vectorized lags and the 
firstemptycell = 0
for i in range(ncells - 1):
    colindices = slice(i+1,ncells) # Not the correlation with itself.
    writelength = ncells - 1 - i
    cormatindices = slice(firstemptycell, firstemptycell + writelength) # write indices to the 1D triangular matrix.
    distmat[cormatindices] = jaccard_1d_2d(exceedence[:,i], exceedence[:,colindices])
    firstemptycell += writelength
    print('computed', str(writelength), 'links for cell', str(i + 1), 'of', str(ncells))

#returns = skclustering(indices, mask2d=mask, clustermethodkey='AgglomerativeClustering', kwrgs={'n_clusters':8})
# Go and test a precomputed jaccard distance?