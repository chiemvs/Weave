"""
Call signature: python find_precursors.py $TEMPDIR $ANOMDIR $CLUSTERDIR $PACKAGEDIR $NPROC
"""

import sys
import logging
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

TMPDIR = Path(sys.argv[1])
ANOMDIR = Path(sys.argv[2])
CLUSTERDIR = Path(sys.argv[3])
PACKAGEDIR = sys.argv[4] # Currently not in use for the SurfaceObservations class (hardcoded there)
NPROC = sys.argv[5]

sys.path.append(PACKAGEDIR)

from Weave.src.processing import TimeAggregator
from Weave.src.association import Associator

logging.basicConfig(filename= TMPDIR / 'testprecursor.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')
# Open a response timeseries. And extract a certain cluster with a cluster template
response = xr.open_dataarray(ANOMDIR / 't2m_europe.anom.nc')
clusterfield = xr.open_dataarray(CLUSTERDIR / 't2m-q085.nc').sel(nclusters = 14)
reduced = response.groupby(clusterfield).mean('stacked_latitude_longitude')
summersubset = reduced.sel(clustid = 8)[reduced.time.dt.season == 'JJA'] # In this case cluster 8 is western europe.
response.close()

# Open a precursor array
precursor = xr.open_dataarray(ANOMDIR / 'z500_europe.anom.nc')

self = Associator(responseseries = summersubset, data = precursor, laglist = [-6, -4, -2])
del precursor

