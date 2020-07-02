#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustering class to handle the clustering of an array dataset (usually with coordinates) based on standard or custom distance metrics.
For the computation of distances it can use the sklearn stack (because of some standard fast pairwise distance implementations)
But in some cases the data won't fit into memory and a custom parallelization is used.
For the clustering of the precomputed matrix, either the hierarchical scipy implementation is used (using condensed matrices) 
or a class from sklearn.cluster can be supplied (that takes a distance matrix)
Extraction of the labels 
"""

#import sys
import logging
import time
import xarray as xr
import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from scipy.spatial.distance import jaccard, squareform
#sys.path.append('/usr/people/straaten/Documents/RGCPD/clustering')
#from clustering_spatial import binary_occurences_quantile #, skclustering
from typing import Union, Callable, List
from .utils import nanquantile, get_corresponding_ctype
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
from haversine import haversine_vector

class Manipulator(object):
    """
    Standard Manipulator, will do no computation, only write the inarray to the desired outarray location
    Handles both numpy and xarray in the process.
    """
    def __init__(self) -> None:
        pass
    
    def __repr__(self) -> str:
        return(f'{self.__class__.__name__}()')

    def get_outshape(self, inarray: Union[np.ndarray, xr.DataArray]) -> tuple:
        return inarray.shape
    
    def get_outnpdtype(self, inarray: Union[np.ndarray, xr.DataArray]) -> type:
        return inarray.dtype
    
    def manipulate(self, inarray: Union[np.ndarray, xr.DataArray]) -> None:
        # Make sure that it are the numpy values only we retain
        if isinstance(inarray, xr.DataArray):
            self.array = inarray.values 
        else:
            self.array = inarray
        logging.debug(f'{self} manipulator created an internal identical copy of the array')
    
    def write(self, outarray: Union[np.ndarray, np.memmap, mp.RawArray]) -> None:
        if not isinstance(outarray, (np.ndarray, np.memmap)):
            outarray_np = np.frombuffer(outarray, dtype=self.array.dtype)
            np.copyto(outarray_np, self.array.reshape((self.array.size,)))
        else:
            outarray[:] = self.array
        logging.info(f'{self} manipulator has written to {type(outarray)}')

class Lagshift(Manipulator):
    """
    Creates multiple versions of a 2D xarray (n_features,n_samples) by lagging and shifting the n_features time axis 
    Deals with a non-continuous time axis (e.g. gaps between consecutive summers) by shifting and reindexing
    Writes the resulting 3D data to the supplied output array.
    """
    def __init__(self, lags: list) -> None:
        self.lags = lags
        assert self.lags[0] == -self.lags[-1], "List of lags should be symmetric around zero"
    
    def get_outshape(self, inarray: Union[np.ndarray, xr.DataArray]) -> tuple:
        # Lag axis will eventually be inserted as the zero-th axis
        return (len(self.lags),) + inarray.shape
        
    def manipulate(self, inarray: xr.DataArray) -> None:
        # Make sure that it are the numpy values only we retain
        ori_timeaxis = inarray.coords['time'].copy()
        self.array = [None] * len(self.lags)
        for lag in self.lags:
            lag_timeaxis = ori_timeaxis - pd.Timedelta(str(lag) + 'D')
            inarray.coords['time'] = lag_timeaxis # Assign the shifted timeaxis
            self.array[self.lags.index(lag)] = inarray.reindex_like(ori_timeaxis).values
            logging.debug(f'{self} manipulator created an internal lag {lag} shifted copy of the array')
        self.array = np.stack(self.array, axis = 0)

class Exceedence(Manipulator):
    """
    Takes a quantiles along the first axis of the supplied array 
    """
    def __init__(self, quantile: float) -> None:
        self.quantile = quantile
        
    def get_outnpdtype(self, inarray: Union[np.ndarray, xr.DataArray]) -> type:
        return np.bool
        
    def manipulate(self, inarray: Union[np.ndarray, xr.DataArray]) -> None:
        # Make sure that it are the numpy values only we retain
        if isinstance(inarray, xr.DataArray):
            self.array = inarray.values 
        else:
            self.array = inarray
        # nanquantile reduces the zero'th axis of the array
        qfield = nanquantile(self.array, self.quantile)
        self.array = self.array > qfield[np.newaxis,...]
        logging.debug(f'{self} manipulator created an internal binary q{self.quantile} exceedece of the array')

class Clustering(object):
    
    def __init__(self, varname: str = None, groupname: str = None, varpath: Path = None, 
                 storedir: Path = None) -> None:
        """
        Possible to supply a variable name and variable path when data has not 
        been in memory from other computations yet and needs to be loaded in this class.
        Should then be netcdf file readable by xarray.
        Here a default storage directory is defined for where intermediate results should be saved
        """
        self.varname = varname
        self.groupname = groupname
        self.varpath = varpath
        self.storedir = storedir

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        """
        Some cleanup of potentially huge attributes
        """
        for item in ['distmat','array']:
            if hasattr(self, item):
                delattr(self, item)
    
    def __repr__(self) -> str:
        return f'Clustering(varname = {self.varname}, groupname = {self.groupname}, varpath = {self.varpath})'
    
    def load_obs(self) -> None:
        try:
            self.array = xr.open_dataarray(self.varpath, group = self.groupname)
            logging.info(f'{self} loaded an xr.DataArray from disk in {self.varpath}')
        except AttributeError:
            raise AttributeError(f'{self} cannot load an xr.DataArray from disk in {self.varpath}')
    
    def reshape_and_drop_obs(self, array: xr.DataArray = None, mask: xr.DataArray = None, season: str = None) -> None:
        """
        The goal is a 2D array with (n_features, n_samples) with irrelevant observations dropped.
        When the array is 3D (n_features,l,m) this reshapes it into 2D (n_features,l*m)
        An (l,m) or (l*m) mask can be supplied (True means retained), which in coordinates should match the array of course. 
        Note that in the case of spatial clustering the first n_features dimension is the time axis
        Note that in the case of temporal clustering the n_features dimensions are climate indices and time is a samples axis
        A season can be supplied to take a subset of the time axis
        """
        if not array is None:
            self.array = array
            logging.info(f'You have supplied an array to {self} which will override any previous one')
        else:
            self.load_obs()
        
        assert self.array.ndim <= 3 and self.array.ndim >= 2 , "Only 3D and 2D arrays are suited for clustering. Are you sure the shape is (n_features, n_samples, ...)?"
        
        if season:
            self.array = self.array.sel(time = self.array.time[self.array.time.dt.season == season]) # axis order dependent: self.array = self.array[self.array.time.dt.season == 'JJA', ...]
        
        # Storing one original (not features) set of coordinates for later restructuring after potential unstacking
        # ordering of the dimensions is crucial
        sampledims = self.array.dims[1:]
        self.samplefield = self.array[0,...]
        
        # Flattening of data: Ordering of the dimensions is crucial
        if len(sampledims) == 2:
            self.stackdim = {'_'.join(sampledims): sampledims}
            self.array = self.array.stack(self.stackdim)
            logging.debug(f'{self} has flattened {sampledims} into one coordinate')
        
        if not mask is None:
            if mask.ndim == 2: # Flatten the mask too if applicable
                mask = mask.stack(self.stackdim)
            logging.debug(f'{self} will mask out {(~mask).sum().values} of {self.array.shape[-1]} samples')
            self.array = self.array[:,mask.values]
        
        # Capture the coords of data after masking.
        self.samplecoords = self.array.coords[self.array.dims[-1]]
    
    def prepare_for_distance_algorithm(self, where: str = None, array: np.array = None, manipulator: Manipulator = Manipulator, args: tuple = tuple(), kwargs: dict = dict()) -> None:
        """
        Accepts a 2D array with (n_features, n_samples) or works on the one created by self.reshape_and_drop_obs
        A manipulator class can be supplied that will be initialized with (*args,**kwargs) and its manipulator.manipulate is called. This can enlarge the array and potentially create memory poblems
        Optionally it therefore places the final array on disk (memmapped) or in shared memory (requires ctypes)
        where: 'memmap', 'shared'
        """
        if not array is None:
            self.array = array
            logging.info(f'You have directly supplied an array to {self}, potentially without reshaping and dropping samples. No coordinate restructuring will take place.')
        
        # Initialize the custom or standard manipulator and let it contain the manipulated array.
        manipulator = manipulator(*args, **kwargs)
        manipulator.manipulate(inarray = self.array)
        
        # Prepare the storage of the manipulated array
        self.mandtype = manipulator.get_outnpdtype(self.array)
        self.manshape = manipulator.get_outshape(self.array)
        if where is None:
            self.array = np.empty(shape = self.manshape, dtype = self.mandtype)
        elif where == 'memmap':
            tempfilepath = self.storedir / ('.'.join([self.varname,'dat']))
            self.array = np.memmap(tempfilepath, dtype=self.mandtype, mode='w+', shape=self.manshape)
        elif where == 'shared':
            # In this case we essentially create a flat array in memory
            self.manctype = get_corresponding_ctype(self.mandtype)
            self.array = mp.RawArray(self.manctype, size_or_initializer=manipulator.array.size)
        else:
            raise KeyError('Do not know what to do with the where argument. Choose: [None,"memmap","shared"]')
        logging.info(f'{self} has prepared a {where} array with dtype {self.mandtype} and shape {self.manshape} as requested by {manipulator}')

        # Let the manipulater write to it.
        manipulator.write(outarray = self.array)
        # Reopen the memmap with reading only
        if where == 'memmap':
            self.array = np.memmap(tempfilepath, dtype=self.mandtype, mode='r', shape=self.manshape)
            logging.info(f'the filled memmap array exists at {tempfilepath}')
    
    def call_distance_algorithm(self, func: Callable, kwargs: dict = dict(), n_par_processes: int = 1, distmatdtype: type = np.float32) -> None:
        """
        Calls the function on the previously created array. The function is either pairwise_distances from sklearn.metrics
        Or it is a custom function that accepts a queue of sample combinations to compute distances between and is tailored to
        This means that then a parallel processing queue is set up, and processes write to a shared matrix
        Both of them can be supplied with additional kwargs.
        The result of the first is a square matrix, the result of the latter is a condensed 1D distance matrix.
        """
        if func == pairwise_distances:
            # Pairwise distance uses the transformed format of the data array, namely (n_samples, n_features) 
            # It returns the square distance matrix (n_samples,n_samples)
            assert isinstance(self.array, (np.ndarray, np.memmap)), f"Shared ctypes arrays are not directly by func={func}. Prepare differently or supply a custom func."
            kwargs.update({'n_jobs':n_par_processes})
            logging.debug(f'{self} is about to call {func}(X = prepared array) with kwargs = {kwargs}')
            self.distmat = func(X = self.array.T, **kwargs).astype(distmatdtype)
        else:
            # These custom functions should accept the queue, a reading array, its shape, its dtype, and the writing array and its dtype, 
            # Initialize the triangular (condensed) distance matrix (n_cells,n_cells) as a shared array to which subprocesses will write.
            ncells = self.manshape[-1]
            n_triangular = int((ncells**2 - ncells)/2)
            DIST = mp.RawArray(get_corresponding_ctype(npdtype = distmatdtype), n_triangular)
            
            # Setup a queue. And fill it with Tuples(i_cell,compare_cells,cormat_indices) to be read by workers
            task_queue = mp.Queue()
            firstemptycell = 0
            for i in range(ncells - 1):
                compareindices = slice(i+1,ncells) # Not the correlation with itself.
                writelength = ncells - 1 - i
                cormatindices = slice(firstemptycell, firstemptycell + writelength) # write indices to the 1D triangular matrix.
                firstemptycell += writelength
                task_queue.put((i,compareindices,cormatindices))
            logging.debug(f'{self} has filled a queue of size {ncells - 1} with tasks and writing positions in distance matrix {DIST}')
            
            # Initialize the custom functions with access to the queue, a reading array, its shape, its dtype, and the writing array and its dtype, 
            self.procs = []
            logging.info(f'{self} is about to create {n_par_processes} processses running func = {func} with kwargs = {kwargs}. Func needs to accept the argument order and obey STOP commands')
            for i in range(n_par_processes):
                p = mp.Process(target = func, args=(task_queue, self.array, self.manshape, self.mandtype, DIST, distmatdtype), kwargs=kwargs)
                p.start()
                self.procs.append(p)
                task_queue.put(('STOP','STOP','STOP'))
            logging.info(f'{self} started processes {[p.pid for p in self.procs]}')
            
            while task_queue.qsize() > 0:
                time.sleep(10)
                logging.info(f'{task_queue.qsize() - n_par_processes} tasks remaining of {ncells - 1}')
            task_queue.close()
            task_queue.join_thread()
            
            # Get the shared distance matrix back to numpy
            self.distmat = np.frombuffer(DIST, dtype = distmatdtype)

    def store_dist_matrix(self, directory: Path = None) -> tuple:
        # Store results on disk
        if directory is not None:
            filepath = directory / ('.'.join([self.varname,'distmat','dat']))
        else:
            filepath = self.storedir / ('.'.join([self.varname,'distmat','dat']))
        storekwds = {'filename':filepath, 'shape':self.distmat.shape, 'dtype':self.distmat.dtype}
        ondisk = np.memmap(mode = 'w+', **storekwds)
        ondisk[:] = self.distmat
        logging.info(f'{self} wrote a np.memmap copy of the distance matrix at {storekwds}')
        return (self.samplecoords, storekwds)
    
    def clustering(self, nclusters: List[int] = None, dissimheights: List[float] = None, clusterclass: Callable = None, args: tuple = tuple(), kwargs: dict = dict()) -> Union[xr.DataArray,np.ndarray]:
        """
        Enables sklearn clustering algorithms like  DBSCAN to be called with precomputed matrices. Otherwise do the hierachal spatial clustering.
        This scipy implementation can be called with nclusters or with dissimilarity heights at which to cut the tree (resulting in unknown numbers of clusters)
        Have a possibility to weight the samples?
        The function returns an int16 array with two dimensions (nclusters or ndissimheigts,n_samples)
        This potentially is an xarray with coordinates if from a previous step samplecoords were extracted and even a 3D array if flattening has taken place. When unstacking introduces Nan, the returntype is float64
        """
        try:
            requestsize = len(nclusters)
            requestname = 'nclusters' 
            logging.debug(f'clusters {nclusters} will be asked from {clusterclass} or standard scipy hierarchy.')
        except TypeError:
            try:
                requestsize = len(dissimheights)
                requestname = 'dissimheight'
                logging.debug(f'levels {dissimheights} will be asked from standard scipy hierarchy. {clusterclass} should be None.')
            except TypeError:
                requestsize = 1
                requestname = 'nclusters'
                assert clusterclass == DBSCAN, 'You can only refrain from specifying nclusters and dissimheigts when using DBSCAN, which determines this by itself'
                logging.debug(f'The amount of clusters will be determined by {clusterclass}.')

        returnarray = np.zeros((requestsize,self.manshape[-1]), dtype = np.int16)
        if not clusterclass is None:
            # The sklearn classes want a square distance matrix, and can only be called once per ncluster value
            if not self.distmat.ndim == 2:
                self.distmat = squareform(self.distmat)
            if clusterclass == DBSCAN:
                kwargs.update({'metric':'precomputed'})
                cl = clusterclass(*args, **kwargs)
                logging.debug(f'Initialized the supplied sklearn clusterclass as {cl}')
                cl.fit(self.distmat) # weigths? This can be huge on memory even though there was precomputation when a lot of cells are connected through each others eps regions
                nclusters = [ sum(np.unique(cl.labels_) != -1) ]
                returnarray[:,:] = cl.labels_
            else:
                for ncluster in nclusters:
                    kwargs.update({'n_clusters':ncluster, 'affinity':'precomputed'})
                    cl = clusterclass(*args, **kwargs)
                    logging.debug(f'Initialized the supplied sklearn clusterclass as {cl}')
                    cl.fit(self.distmat) # weigths?
                    returnarray[nclusters.index(ncluster),:] = cl.labels_
        else:
            # We use the standard scipy hierarchal clustering, wants a condensed distance matix. Luckily squareform can also do the inverse
            if not self.distmat.ndim == 1:
                self.distmat = squareform(self.distmat)
            import scipy.cluster.hierarchy as sch
            logging.debug(f'computing clusters with {sch} average linkage')
            Z = sch.linkage(y = self.distmat, method = 'average')
            returnarray[:] = sch.cut_tree(Z, n_clusters=nclusters, height=dissimheights).T # Either use the nclusters or the dissim heights
        
        logging.info('the clusters in sample space have been computed')

        if hasattr(self, 'samplecoords'):
            returnarray = xr.DataArray(returnarray, dims = (requestname,self.samplecoords.name), coords = {requestname:nclusters if nclusters else dissimheights, self.samplecoords.name: self.samplecoords}, name = 'clustid')
            returnarray = returnarray.where(cond = returnarray >= 0, drop = False) # Put nan where clustid is -1, which is the noise cluster of dbscan, changes the np.dtype
            returnarray.encoding.update({'dtype': 'int16', '_FillValue': -32767})
            if hasattr(self, 'stackdim'):
                returnarray = returnarray.unstack(self.samplecoords.name).reindex_like(self.samplefield)
                logging.debug('The previously flattened sample dimensions have been restored')

        return returnarray
    
def jaccard_worker(inqueue: mp.Queue, readarray: mp.RawArray, readarrayshape: tuple, readarraydtype: type, writearray: mp.RawArray, writearraydtype: type) -> None:
    """
    Worker that takes messages from a queue to compute the jaccard distance between sample i and a set of other samples in the reading array
    Reshapes the reading array once if it is a shared ctype, otherwise it is on disk or copied into the memory of each worker 
    The computation reduces the reading array along the zero-th dimension
    it writes the resulting distances to non-overlapping parts of the shared triangular array. 
    """
    DIST_np = np.frombuffer(writearray, dtype =  writearraydtype)
    logging.info('this process has given itself access to a shared writing array')
    if not isinstance(readarray, (np.ndarray, np.memmap)): # Make the thing numpy accesible in this process
        readarray_np = np.frombuffer(readarray, dtype = readarraydtype).reshape(readarrayshape)
        logging.info('this process has given itself access to a shared reading array')
    else:
        readarray_np = readarray
    
    while True:
        i, compare_samples, distmat_indices = inqueue.get()
        if i == 'STOP':
            logging.info('this process is shutting down, after STOP')
            break
        else:
            DIST_np[distmat_indices] = np.apply_along_axis(jaccard, 0, readarray_np[:,compare_samples], **{'v':readarray_np[:,i]})

def haversine_worker(inqueue: mp.Queue, readarray: mp.RawArray, readarrayshape: tuple, readarraydtype: type, writearray: mp.RawArray, writearraydtype: type) -> None:
    """
    Worker that takes messages from a queue to compute the jaccard distance between sample i and a set of other samples in the reading array
    Reshapes the reading array once if it is a shared ctype, otherwise it is on disk or copied into the memory of each worker 
    The computation reduces the reading array along the zero-th dimension
    it writes the resulting distances to non-overlapping parts of the shared triangular array. 
    """
    DIST_np = np.frombuffer(writearray, dtype =  writearraydtype)
    logging.info('this process has given itself access to a shared writing array')
    if not isinstance(readarray, (np.ndarray, np.memmap)): # Make the thing numpy accesible in this process
        readarray_np = np.frombuffer(readarray, dtype = readarraydtype).reshape(readarrayshape)
        logging.info('this process has given itself access to a shared reading array')
    else:
        readarray_np = readarray
    
    while True:
        i, compare_samples, distmat_indices = inqueue.get()
        if i == 'STOP':
            logging.info('this process is shutting down, after STOP')
            break
        else:
            DIST_np[distmat_indices] = haversine_vector(array1 = readarray_np[:,i], array2 = readarray_np[:,compare_samples].T)

def dummy_worker(inqueue: mp.Queue, readarray: mp.RawArray, readarrayshape: tuple, readarraydtype: type, writearray: mp.RawArray, writearraydtype: type) -> None:
    DIST_np = np.frombuffer(writearray, dtype =  writearraydtype)
    logging.info('this process has given itself access to a shared writing array')
    if not isinstance(readarray, (np.ndarray, np.memmap)): # Make the thing numpy accesible in this process
        readarray_np = np.frombuffer(readarray, dtype = readarraydtype).reshape(readarrayshape)
        logging.info('this process has given itself access to a shared reading array')
    else:
        readarray_np = readarray
    
    while True:
        i, compare_samples, distmat_indices = inqueue.get()
        if i == 'STOP':
            logging.info('this process is shutting down, after STOP')
            break
        else:
            DIST_np[distmat_indices] = readarray_np[readarrayshape[0]//2 + 1,10,i]
            time.sleep(0.2)
            logging.debug(f'computed links for sample {i}.')

def maxcorrcoef_worker(inqueue: mp.Queue, readarray: mp.RawArray, readarrayshape: tuple, readarraydtype: type, writearray: mp.RawArray, writearraydtype: type) -> None:
    """
    Worker that takes messages from a queue to compute linear correlation coefficients between a single sample timeseries (n,) and a set of lagged timeseries of other samples (d,n,m)
    For each pairwise combination of samples it takes the maximum over lags d to output (m)
    adaptation of https://waterprogramming.wordpress.com/2014/06/13/numpy-vectorized-correlation-coefficient/
    Reshapes the reading array once if it is a shared ctype, otherwise it is on disk or copied into the memory of each worker 
    The computation reduces the reading array along the zero-th and first dimension
    it writes the resulting distances to non-overlapping parts of the shared triangular array. 
    """
    DIST_np = np.frombuffer(writearray, dtype =  writearraydtype)
    logging.info('this process has given itself access to a shared writing array')
    if not isinstance(readarray, (np.ndarray, np.memmap)): # Make the thing numpy accesible in this process
        readarray_np = np.frombuffer(readarray, dtype = readarraydtype).reshape(readarrayshape)
        logging.info('this process has given itself access to a shared reading array')
    else:
        readarray_np = readarray

    while True:
        i, compare_samples, distmat_indices = inqueue.get()
        if i == 'STOP':
            logging.info('this process is shutting down, after STOP')
            break
        else:
            y = readarray_np[readarrayshape[0]//2 + 1, :, i] # Counts on symmetric lagging and an unlagged version in the middle
            X = readarray_np[:,:,compare_samples] # (d,n,m)
            Xm = np.nanmean(X,axis=1) # result (d,m)
            ym = np.nanmean(y) # result (,)
            r_num = np.nansum((X-Xm[:,np.newaxis,:])*((y-ym)[np.newaxis,:,np.newaxis]),axis=1) # Summing covariances (yi -ymean)(xi -xmean) over n
            r_den = np.sqrt(np.nansum((X-Xm[:,np.newaxis,:])**2,axis=1)*np.nansum((y-ym)**2)) # Summing over n
            r = r_num/r_den # result (d,m)
            DIST_np[distmat_indices] = 1 - np.nanmax(r, axis = 0) # Maximum over d, 1 minus maxcor to adapt to a distance.

