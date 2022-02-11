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
