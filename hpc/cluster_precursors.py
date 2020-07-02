"""
Call signature: python cluster_precursors.py $TEMPDIR $PACKAGEDIR $NPROC $PATTERNDIR
Reading of a correlation field. DBSCAN clustering on spatial distance
Will write clustids inside the files of the patterndir
"""
import sys
import logging
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN

TMPDIR = Path(sys.argv[1])
PACKAGEDIR = sys.argv[2] 
NPROC = int(sys.argv[3])
PATTERNDIR = Path(sys.argv[4])

sys.path.append(PACKAGEDIR)

from Weave.src.inputoutput import Writer, Reader
from Weave.src.clustering import Clustering, haversine_worker

logging.basicConfig(filename= TMPDIR / 'cluster_precursors.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')

#corrfiles = [ f in PATTERNDIR.glob('*corr.nc') if f.is_file()]
corrfiles = [PATTERNDIR / 'z300_nhmin.1.corr.alpha00005.nc']

# first level loop are association metric files, so variable / timeagg combinations
# The goal is to cluster these patterns based on distance on the globe
for corrpath in corrfiles:
    filename = corrpath.parts[-1]
    invarname = 'correlation'
    outvarname = 'clustid'
    ds = xr.open_dataset(corrpath, decode_times = False)
    already_clustered = hasattr(ds, outvarname)
    lags = ds.coords['lag'].values.tolist() # Lag was getting decoded. This is now solved with decode_lag passing to decode_times in xarray.
    #lags = [-25]
    
    latlons = np.stack(np.meshgrid(ds.coords['latitude'].astype(np.float16), ds.coords['longitude'].astype(np.float16), indexing = 'ij'), axis = 0) # these are the features on which we compute the haversine distances (input are decimal degree coordinates)
    latlons = xr.DataArray(latlons, dims = ('coordinates','latitude','longitude'), coords = {'coordinates':['lat','lon'],'latitude':ds.coords['latitude'], 'longitude': ds.coords['longitude']}) # Not strictly neccesary but nice for automated coordinate restructuring
    if not already_clustered: 
        combined = []
        attrs = {}
        for lag in lags:
            with Clustering() as cl:
                cl.reshape_and_drop_obs(array = latlons, mask = ~ds[invarname].sel(lag = lag).isnull())
                cl.prepare_for_distance_algorithm(where = 'shared')
                # arguments to haversine are (lat,lon)
                cl.call_distance_algorithm(func = haversine_worker, n_par_processes = NPROC, distmatdtype = np.float16) 
                clusters = cl.clustering(clusterclass = DBSCAN, kwargs = {'eps':800, 'min_samples':1000})
                nclusters = int(clusters.coords["nclusters"]) # nclusters returned as coordinate because this matches bahaviour of the non-DBSCAN algorithms, even though with DBSCAN it is only a dimension of length 1
                attrs.update({f'lag{lag}':f'nclusters: {nclusters}'}) 
                combined.append(clusters.squeeze().drop('nclusters'))
                logging.debug(f'clustered {invarname} of {filename} by spatial haversine distance with DBSCAN for lag: {lag}, resulting nclusters: {nclusters}')
        
        temp = xr.concat(combined, dim = pd.Index(lags, name = 'lag'))
        # Temporary fix
        #ds[outvarname] = temp
        #ds.to_netcdf(str(corrpath) + '.new')
        ds.close() # Need to close before writer can access
        w = Writer(corrpath,varname = outvarname) # Should be able to find the dataformat
        w.create_dataset(example = temp)
        w.write(array = temp, units = '', attrs = attrs) 
    else:
        ds.close()
