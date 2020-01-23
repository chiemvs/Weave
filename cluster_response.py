#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for the T2m clustering of sem. 
"""
import sys
import logging
import time
import xarray as xr
import numpy as np
import multiprocessing as mp
from scipy.spatial.distance import jaccard
from ctypes import c_bool, c_float # Float is equivalent to numpy float32 I believe
sys.path.append('/usr/people/straaten/Documents/RGCPD/clustering')
from clustering_spatial import binary_occurences_quantile #, skclustering

def main(n_par_comps: int, quantile: int):
    """
    Quantile should be 00 to 100
    """
    # Land-sea mask derived from sea ice concentration
    siconc = xr.open_dataarray('/nobackup_1/users/straaten/ERA5/siconc/siconc_nhmin.nc', group = 'mean')[0]
    t2m = xr.open_dataarray('/nobackup_1/users/straaten/ERA5/t2m/t2m_europe.nc', group = 'mean')
    t2m = t2m[t2m.time.dt.season == 'JJA', :,:]
    
    mask = siconc.sel(latitude = t2m['latitude'], longitude = t2m['longitude']).isnull()
    mask1d = mask.stack({'latlon':['latitude','longitude']})
    siconc.close()
    
    # Compute the binary series array (n_time,n_cells). Remove masked cells.
    # Initialize it as an array with shared memory for subprocesses
    exceedence = binary_occurences_quantile(t2m, q=quantile).astype('bool') # Binary array. First reduces along one dimension to get the local q80 in time
    logging.info(f'computed the binary exceedence of quantile {quantile}')
    exceedence = exceedence.stack({'latlon':['latitude','longitude']}) # Flatten it to 1D
    exceedence = exceedence[:,mask1d.values] # Remove masked out
    spatcoords = exceedence.coords['latlon'] # Storing spatial coordinates
    exceedence = exceedence.values
    t2m.close()

    EX = mp.RawArray(c_bool, exceedence.size) # The shared array, initialized at zero
    #EX[:] = exceedence.reshape((exceedence.size,)) # Slower way of writing.
    EX_np = np.frombuffer(EX, dtype=np.bool) # Establish numpy access
    np.copyto(EX_np, exceedence.reshape((exceedence.size,))) # Fill array from numpy
    
    # Initialize the triangular part of the distance matrix (n_cells,n_cells) also as a shared array to which subprocesses will write.
    ncells = exceedence.shape[-1]
    n_triangular = int((ncells**2 - ncells)/2)
    COR = mp.RawArray(c_float, n_triangular)
    
    # Setup a queue. And fill it with Tuples(i_cell,compare_cells,cormat_indices) to be read by workers
    task_queue = mp.Queue()
    firstemptycell = 0
    for i in range(ncells - 1):
        compareindices = slice(i+1,ncells) # Not the correlation with itself.
        writelength = ncells - 1 - i
        cormatindices = slice(firstemptycell, firstemptycell + writelength) # write indices to the 1D triangular matrix.
        firstemptycell += writelength
        task_queue.put((i,compareindices,cormatindices))
    
    # Initialize workers with access to the shared array.
    for i in range(n_par_comps):
        p = mp.Process(target = worker, args=(task_queue, EX, exceedence.shape, COR))
        p.start()
        task_queue.put(('STOP','STOP','STOP'))
    
    while task_queue.qsize() > 0:
        time.sleep(5)
    
    task_queue.close()
    task_queue.join_thread()
    
    # Store results on disk
    storekwds = {'filename':'/nobackup_1/users/straaten/ERA5/dist_ex_q' + str(quantile) + '.dat', 'shape':(n_triangular,), 'dtype':np.float32}
    ondisk = np.memmap(mode = 'w+', **storekwds)
    ondisk[:] = np.frombuffer(COR, dtype = np.float32)
    return((spatcoords, storekwds))

def worker(inqueue: mp.Queue, readarray: mp.RawArray, readarrayshape: tuple, writearray: mp.RawArray) -> None:
    """
    Reshapes the shared reading array once. Will take queued messages to compute the links between cell i and a set of other cells. 
    Will write to non-overlapping parts of the shared triangular array. 
    """
    EX_np = np.frombuffer(readarray, dtype = np.bool).reshape(readarrayshape)
    COR_np = np.frombuffer(writearray, dtype = np.float32)
    logging.info('this process has started with access to the shared arrays')
    
    while True:
        i, compare_cells, cormat_indices = inqueue.get()
        if i == 'STOP':
            logging.info('this process is shutting down, after STOP')
            break
        else:
            dist = np.apply_along_axis(jaccard, 0, EX_np[:,compare_cells], **{'v':EX_np[:,i]})
            COR_np[cormat_indices] = dist
            logging.debug(f'computed {len(dist)} links for cell {i}.')

def cluster_distance(diskkwds, spatcoords, nclusters: int = 5):
    """
    Should receive kwds to read the 1d distance matrix
    Stores a netcdf to disk with the clusters
    """
    import scipy.cluster.hierarchy as sch
    opened = np.memmap(mode = 'r', **diskkwds)
    Z = sch.linkage(y = opened, method = 'average')
    labels = sch.cut_tree(Z, n_clusters=nclusters).squeeze()
    da = xr.DataArray(data = labels, coords = {'latlon':spatcoords}, dims=['latlon'])
    newfilename = diskkwds['filename'] + '.spatial.nc'
    da.unstack('latlon').sortby(['latitude','longitude']).to_netcdf(newfilename)
    logging.info(f'written out {newfilename}')
    
#returns = skclustering(indices, mask2d=mask, clustermethodkey='AgglomerativeClustering', kwrgs={'n_clusters':8})
# Go and test a precomputed jaccard distance?
if __name__ == '__main__':
    logging.basicConfig(filename='responsecluster.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(levelname)s-%(message)s')
    #spatcoord1, diskkwds1 = main(n_par_comps=6, quantile=95)
    #cluster_distance(diskkwds=diskkwds1, spatcoords=spatcoord1, nclusters=8)
    #spatcoord2, diskkwds2 = main(n_par_comps=6, quantile=85)
    #cluster_distance(diskkwds=diskkwds2, spatcoords=spatcoord2, nclusters=8)
