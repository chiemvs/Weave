import sys
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.cluster import DBSCAN
from pathlib import Path

sys.path.append('..')
from src.clustering import Clustering, haversine_worker
from src.inputoutput import Writer, Reader

# Testing of DBSCAN
corpath = Path('/scistor/ivm/jsn295/correlation_roll_spearman/tcc_europe.15.corr.nc')
with xr.open_dataarray(corpath, decode_times = False) as arr:
    lags = arr.coords['lag'].values.tolist() # Lag was getting decoded. This is now solved with decode_lag passing to decode_times in xarray.
    
    latlons = np.stack(np.meshgrid(arr.coords['latitude'],arr.coords['longitude'], indexing = 'ij'), axis = 0) # Ordering correct
    latlons = xr.DataArray(latlons, dims = ('coordinates','latitude','longitude'), coords = {'coordinates':['lat','lon'],'latitude':arr.coords['latitude'], 'longitude': arr.coords['longitude']}) # Nice for automated coordinate restructuring
    
combined = []
attrs = {}
for lag in lags:
    with Clustering(varname = 'distance') as cl:
        cl.reshape_and_drop_obs(array = latlons, mask = ~arr.sel(lag = lag).isnull())
        cl.prepare_for_distance_algorithm(where = 'shared')
        # arguments to haversine are (lat,lon)
        #self.call_distance_algorithm(func = pairwise_distances, kwargs = {'metric':haversine}, n_par_processes = 15) # extremely slow
        cl.call_distance_algorithm(func = haversine_worker, n_par_processes = 15) 
        clusters = cl.clustering(clusterclass = DBSCAN, kwargs = {'eps':200})
        attrs.update({f'lag{lag}':f'nclusters: {int(clusters.coords["nclusters"])}'})
        combined.append(clusters.squeeze().drop('nclusters'))

test = xr.concat(combined, dim = pd.Index(lags, name = 'lag'))
w = Writer(corpath,varname = 'clustid') # Should be able to find the dataformat
w.create_dataset(example = test)
w.write(array = test, units = '', attrs = attrs) 
