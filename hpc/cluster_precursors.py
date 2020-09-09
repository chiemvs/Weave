"""
Call signature: python cluster_precursors.py $TEMPDIR $PACKAGEDIR $NPROC $PATTERNDIR
Reading of a correlation field. HDBSCAN clustering on spatial distance
Will write clustids inside the files of the patterndir
"""
import sys
import logging
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

logging.basicConfig(filename= TMPDIR / 'cluster_precursors.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')

# Clusterkwargs for hdbscan, coming from manual selection
#hdbscan_kwargs = pd.DataFrame([[2000,600],[5000,2000],[True,False]], columns = ['z300','standard'], index = ['min_samples','min_cluster_size','allow_single_cluster'])
#hdbscan_kwargs = pd.DataFrame([[2000,2000,600,1500,600,2000],[5000,7000,2000,2500,2000,5000],[True,True,True,False,False,False]], columns = ['t850','z300','tcc','sst','standard','snowc'], index = ['min_samples','min_cluster_size','allow_single_cluster']) # attempt4
hdbscan_kwargs = pd.DataFrame([[2000,2000,600,2000,600,2000],[5000,7000,2000,2500,2000,5000],[True,True,True,False,False,False]], columns = ['t850','z300','tcc','sst','standard','snowc'], index = ['min_samples','min_cluster_size','allow_single_cluster']) # attempt3
#hdbscan_kwargs = pd.DataFrame([[2000,2000,600,1000,600,1500],[5000,7000,2000,3000,2000,4000],[True,True,True,False,False,False]], columns = ['t850','z300','tcc','sst','standard','snowc'], index = ['min_samples','min_cluster_size','allow_single_cluster']) # attempt2
#hdbscan_kwargs = pd.DataFrame([[2000,2000,600,1000,600,1500],[6000,6000,2000,2000,2000,2000],[True,True,True,False,False,False]], columns = ['t850','z300','tcc','sst','standard','snowc'], index = ['min_samples','min_cluster_size','allow_single_cluster']) # attempt1
hdbscan_kwargs.loc['metric',:] = 'haversine'
hdbscan_kwargs.loc['core_dist_n_jobs',:] = NPROC
corrfiles = [ f for f in PATTERNDIR.glob('*corr.nc') if f.is_file() ]

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
    if variable in hdbscan_kwargs:
        clusterkwargs = hdbscan_kwargs[variable].to_dict() # Non-standard if present
    else:
        clusterkwargs = hdbscan_kwargs['standard'].to_dict()
    
    if (not already_clustered): # and (filename != 'z300_nhnorm.31.corr.nc'): 
        combined = []
        attrs = {}
        for lag in lags:
            cl = Clustering()
            try:
                cl.reshape_and_drop_obs(array = ds[invarname], mask = ~ds[invarname].sel(lag = lag).isnull())
                cl.prepare_for_distance_algorithm(manipulator = Latlons, kwargs = {'to_radians':True}) # Conversion to radians because HDBSCAN uses that.
                clusters = cl.clustering(clusterclass = HDBSCAN, kwargs = clusterkwargs)
#            with Clustering() as cl: # This is the memory heavy precomputed DBSCAN variety
#                cl.prepare_for_distance_algorithm(where = 'shared', manipulator = Latlons)
#                cl.call_distance_algorithm(func = haversine_worker, n_par_processes = NPROC, distmatdtype = np.float16) 
#                clusters = cl.clustering(clusterclass = DBSCAN, kwargs = {'eps':1300, 'min_samples':2000})
                nclusters = int(clusters.coords["nclusters"]) # nclusters returned as coordinate because this matches bahaviour of the non-DBSCAN algorithms, even though with DBSCAN it is only a dimension of length 1
                logging.debug(f'clustered {invarname} of {filename} by spatial haversine distance with HDBSCAN for lag: {lag}, resulting nclusters: {nclusters}')
            except MaskingError: # Happens when masking results in zero samples 
                nclusters = 0
                clusters = xr.DataArray(np.nan, dims = cl.samplefield.dims, coords = cl.samplefield.drop_vars('lag').coords)
                logging.debug(f'No samples were present after masking {invarname} of {filename} for lag: {lag}. HDBSCAN was not called. A field with zero clusters is returned.')
                
            attrs.update({f'lag{lag}':f'nclusters: {nclusters}'}) 
            combined.append(clusters.squeeze().drop_vars('nclusters', errors = 'ignore'))
        
        temp = xr.concat(combined, dim = pd.Index(lags, name = 'lag'))
        ds.close() # Need to close before writer can access
        attrs.update({key:str(item) for key,item in clusterkwargs.items()})
        w = Writer(corrpath,varname = outvarname) # Should be able to find the dataformat
        w.create_dataset(example = temp)
        w.write(array = temp, units = '', attrs = attrs) 
    else:
        logging.debug(f'{filename} was already clustered')
        ds.close()
