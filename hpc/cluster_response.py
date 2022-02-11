"""
Call signature: python parcluster.py $TEMPDIR $PACKAGEDIR $NPROC $OBSDIR $CLUSTERDIR
"""

import sys
import logging
import numpy as np
from pathlib import Path

TMPDIR = Path(sys.argv[1])
PACKAGEDIR = sys.argv[2]
NPROC = int(sys.argv[3])
OBSDIR = Path(sys.argv[4]) 
OUTDIR = Path(sys.argv[5]) 

sys.path.append(PACKAGEDIR)

from Weave.clustering import Clustering, Exceedence
from Weave.inputoutput import Writer

# E-OBS part, has a very large memory footprint and thus seems to work best with 5 workers and shared reading memory
#logging.basicConfig(filename= OBSDIR / 'eobs_cluster3D_DJF.log', filemode='w', level=logging.DEBUG, format='%(asctime)s-%(process)d-%(levelname)s-%(message)s', datefmt='%m-%d %H:%M:%S')
#from SubSeas import observations as obs
#o = obs.SurfaceObservations('tg')
#o.load(tmin = '1989-01-01', tmax = '2018-12-31', llcrnr = (36,-24), rucrnr = (None,40))
#o.minfilter(season = 'DJF', n_min_per_seas = 80)
#o.aggregatetime(freq = '3D', method = 'mean', rolling = True)
#
#mask = ~ o.array[0].isnull()
##mask[mask['latitude'] < 60,:] = False
#
#c = Clustering(varname = 'tg3DDJF', storedir = Path(TMPDIR))
#c.reshape_and_drop_obs(array = o.array, season='DJF', mask=mask) # Avoid the load method of the class.
#del o
#c.prepare_for_distance_algorithm(where='shared', manipulator=cl.Lagshift, kwargs={'lags':list(range(-20,21))})
#c.call_distance_algorithm(func = cl.maxcorrcoef_worker, n_par_processes = NPROC)
#storekwargs = c.store_dist_matrix(directory = OBSDIR / '..')
##c.distmat = np.memmap(OBSDIR / 'tg.distmat.dat', shape = (118249131,), dtype = np.float32)
#returnarray = c.clustering(dissimheights = [0,0.005,0.01,0.025,0.05,0.1,0.15,0.2,0.3,0.4,0.5,1])
#returnarray.to_netcdf(OBSDIR / '../paper1/tg-DJF-clustered_3D.nc')

# Quantile exceedence ERA5 part, with jaccard distance, Does not need the 
logging.basicConfig(filename= TMPDIR / 'cluster_response.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(levelname)s-%(message)s')
from sklearn.metrics import pairwise_distances
import xarray as xr
siconc = xr.open_dataarray(OBSDIR / 'siconc_nhmin.nc', group = 'mean')[0]
t2m = xr.open_dataarray(OBSDIR / 't2m_europe.nc', group = 'mean')
mask = siconc.sel(latitude = t2m['latitude'], longitude = t2m['longitude']).isnull()
#mask[mask['latitude'] < 60,:] = False
siconc.close()
t2m.close()
c = Clustering(varname = 't2m', groupname = 'mean', storedir = Path(TMPDIR), varpath = OBSDIR / 't2m_europe.nc')
c.reshape_and_drop_obs(season='JJA', mask=mask)

c.prepare_for_distance_algorithm(where=None, manipulator=Exceedence, kwargs={'quantile':0.75})
c.call_distance_algorithm(func = pairwise_distances, kwargs= {'metric':'jaccard'}, n_par_processes = NPROC)
storekwargs = c.store_dist_matrix(directory = TMPDIR)
returnarray = c.clustering(dissimheights = np.linspace(0.01,1,100).tolist()) # Makes sure that it has a name: clustid
w = Writer(datapath = OUTDIR / 't2m-q075.nc', varname = returnarray.name)
w.create_dataset(example = returnarray)
w.write(array = returnarray)
