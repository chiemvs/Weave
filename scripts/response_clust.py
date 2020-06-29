import sys
import logging
import xarray as xr
import numpy as np
from pathlib import Path
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN

sys.path.append('..')
from src.clustering import Clustering, haversine_worker, Exceedence, jaccard_worker, Lagshift, maxcorrcoef_worker
    
#logging.basicConfig(filename='responsecluster.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(levelname)s-%(message)s')
#siconc = xr.open_dataarray('/nobackup_1/users/straaten/ERA5/siconc/siconc_nhmin.nc', group = 'mean')[0]
#t2m = xr.open_dataarray('/nobackup_1/users/straaten/ERA5/t2m/t2m_europe.nc', group = 'mean')
#mask = siconc.sel(latitude = t2m['latitude'], longitude = t2m['longitude']).isnull()
#mask[mask['latitude'] < 60,:] = False
#siconc.close()
#t2m.close()
#self = Clustering(varname = 't2m', groupname = 'mean', varpath = Path('/nobackup_1/users/straaten/ERA5/t2m/t2m_europe.nc'))
#self.reshape_and_drop_obs(season='JJA', mask=mask)
#self.prepare_for_distance_algorithm(where='memmap', manipulator=Lagshift, kwargs={'lags':list(range(-20,21))})
#self.prepare_for_distance_algorithm(where=None, manipulator=Exceedence, kwargs={'quantile':0.85})
#self.call_distance_algorithm(func = pairwise_distances, kwargs= {'metric':'jaccard'}, n_par_processes = 7)
#self.call_distance_algorithm(func = jaccard_worker, n_par_processes = 7)
#ret2 = self.clustering(nclusters = [2,3,4])

