"""
Call signature: python cluster_precursors.py $TEMPDIR $PACKAGEDIR $NPROC $PATTERNDIR
Reading of a correlation field. HDBSCAN clustering on spatial distance
Will write clustids inside the files of the patterndir
"""
import sys
import logging
import itertools
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
#from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN

TMPDIR = Path(sys.argv[1])
PACKAGEDIR = sys.argv[2] 
NPROC = int(sys.argv[3])
PATTERNDIR = Path(sys.argv[4])

sys.path.append(PACKAGEDIR)

from Weave.inputoutput import Writer, Reader
from Weave.clustering import Clustering, haversine_worker, Latlons, MaskingError

logging.basicConfig(filename= TMPDIR / 'cluster_precursors_eom.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')

# Clusterkwargs for hdbscan, coming from manual selection
hdbscan_kwargs = pd.DataFrame([[1000,2000,200,300,200,200,1000,200],[3000,6000,400,1000,450,600,3000,400],[0.17,0.17,0.09,0.20,0.08,0.08,0.11,0.1]], columns = ['t850','z300','tcc','sst','standard','transp','snowc','siconc'], index = ['min_samples','min_cluster_size','cluster_selection_epsilon'], dtype = object) # Attempt 18: full strict + epsilon clustering.
hdbscan_kwargs.loc['metric',:] = 'haversine'
hdbscan_kwargs.loc['core_dist_n_jobs',:] = NPROC
hdbscan_kwargs.loc['allow_single_cluster',:] = True
hdbscan_kwargs.loc['cluster_selection_method',:] = 'eom' #'leaf'
#corrfiles = [ f for f in PATTERNDIR.glob('*corr.nc') if f.is_file() ]
corrfiles = [ f for f in PATTERNDIR.glob('*corr.nc') if (f.is_file() and (f.name[:4] != 'z300'))]
#corrfiles = [ f for f in PATTERNDIR.glob('*corr.nc') if (f.is_file() and (f.name[:4] == 'z300'))]

# first level loop are association metric files, so variable / timeagg combinations
# The goal is to cluster these patterns based on distance on the globe
for corrpath in corrfiles:
    filename = corrpath.parts[-1]
    variable = filename.split('.')[0].split('_')[0] 
    invarname = 'correlation'
    outvarname = 'clustid'
    ds = xr.open_dataset(corrpath, decode_times = False)
    logging.debug(f'{filename} has been loaded.')
    already_clustered = hasattr(ds, outvarname)
    lags = ds.coords['lag'].values.tolist() # Lag was getting decoded. This is now solved with decode_lag passing to decode_times in xarray.
    try:
        folds = getattr(ds, 'fold').values.tolist() # Then we have crossvalidation present
    except AttributeError:
        folds = [None] # Now we mimic one
    if variable in hdbscan_kwargs:
        clusterkwargs = hdbscan_kwargs[variable].to_dict() # Non-standard if present
    else:
        clusterkwargs = hdbscan_kwargs['standard'].to_dict()
    
    if (not already_clustered): # and (filename != 'z300_nhnorm.31.corr.nc'): 
        combined = []
        attrs = {}
        for lag, fold in itertools.product(lags, folds):
            cl = Clustering()
            field = ds[invarname].sel(lag = lag) if fold is None else ds[invarname].sel(lag = lag, fold = fold)
            try:
                cl.reshape_and_drop_obs(array = field.expand_dims(dim = 'lag', axis = 0), mask = ~field.isnull(), min_samples = 700) # Lag dim is expanded to obey the [n_features, nsamples] format after spatial flattening, even though the lat and lon coordinates will take up the place of the features axis
                cl.prepare_for_distance_algorithm(manipulator = Latlons, kwargs = {'to_radians':True}) # Conversion to radians because HDBSCAN uses that.
                clusters = cl.clustering(clusterclass = HDBSCAN, kwargs = clusterkwargs)
#            with Clustering() as cl: # This is the memory heavy precomputed DBSCAN variety
#                cl.prepare_for_distance_algorithm(where = 'shared', manipulator = Latlons)
#                cl.call_distance_algorithm(func = haversine_worker, n_par_processes = NPROC, distmatdtype = np.float16) 
#                clusters = cl.clustering(clusterclass = DBSCAN, kwargs = {'eps':1300, 'min_samples':2000})
                nclusters = int(clusters.coords["nclusters"]) # nclusters returned as coordinate because this matches bahaviour of the non-DBSCAN algorithms, even though with DBSCAN it is only a dimension of length 1
                logging.debug(f'clustered {invarname} of {filename} by spatial haversine distance with HDBSCAN for lag: {lag}, fold: {fold}, resulting nclusters: {nclusters}')
            except MaskingError: # Happens when masking results in zero or less than the minimum samples 
                nclusters = 0
                clusters = xr.DataArray(np.nan, dims = cl.samplefield.dims, coords = cl.samplefield.drop_vars(['lag','fold'], errors = 'ignore').coords)
                logging.debug(f'No/too little samples were present after masking {invarname} of {filename} for lag: {lag}. fold: {fold}, HDBSCAN was not called. A field with zero clusters is returned.')
            if fold is None:    
                attrs.update({f'lag{lag}':f'nclusters: {nclusters}'}) 
            else:
                attrs.update({f'lag{lag}_fold{fold}':f'nclusters: {nclusters}'}) 
            combined.append(clusters.squeeze().drop_vars('nclusters', errors = 'ignore'))
        
        if fold is None:
            temp = xr.concat(combined, dim = pd.Index(lags, name = 'lag')) # Immediately at first position, and correct order
        else:
            temp = xr.concat(combined, dim = pd.MultiIndex.from_product([lags,folds], names = ['lag','fold'])).unstack('concat_dim').transpose(*ds[invarname].dims).reindex_like(ds[invarname]) # Unstack brings the lag/fold dimension to last place and scrambled order, so so transpose and reindex to get original ordering
        ds.close() # Need to close before writer can access
        attrs.update({key:str(item) for key,item in clusterkwargs.items()})
        w = Writer(corrpath,varname = outvarname) # Should be able to find the dataformat
        w.create_dataset(example = temp)
        w.write(array = temp, units = '', attrs = attrs) 
    else:
        logging.debug(f'{filename} was already clustered')
        ds.close()
