import sys
import logging
import xarray as xr
import numpy as np
import pandas as pd
import netCDF4 as nc
from pathlib import Path

sys.path.append('..')
from Weave.utils import spearmanr_cv, spearmanr_wrap
from Weave.association import Associator #, init_worker, var_dict
from Weave.inputoutput import Writer

logging.basicConfig(level = logging.DEBUG)
t2m = xr.open_dataarray('/nobackup_1/users/straaten/ERA5/t2m/t2m_europe.nc', group = 'mean')[:,50,50]
tcc = xr.open_dataarray('/nobackup_1/users/straaten/ERA5/tcc/tcc_europe.nc', group = 'mean')[:,:10,:10]

#y_in = t2m.loc[t2m.time.dt.season == 'JJA'].to_pandas()
y_in = t2m.loc[t2m.time.dt.season == 'JJA']
#X_in = tcc.to_pandas().reindex_like(y_in) 

asofunc = spearmanr_cv(5, True)
#asofunc = spearmanr_wrap
self = Associator(responseseries = y_in, data = tcc, laglist = [-1,-2], association = asofunc, n_folds = 5)
#ret = asofunc(X_in = X_in, y_in = y_in)
corr = self.compute(nprocs = 7)

w = Writer(Path('test.nc'), varname = corr.name)
w.create_dataset(example = corr)
w.write(corr, attrs = corr.attrs, units = '')

# Test if the crossvalidation linking is similar for the 
#startvals = nc.num2date(corr[0,:,0,0].values, 'days since 1979-01-01') 
#
#y_in = y_in.to_pandas()
#X_in = pd.read_parquet('/nobackup_1/users/straaten/spatcov/precursor.multiagg.parquet')['sst_nhplus'].iloc[:,0].reindex_like(y_in)
#X_in.loc[X_in.isnull()] = 280 # Resolve some nan-stuff
#from sklearn.linear_model import LinearRegression
#from Weave.models import fit_predict
#
#m = LinearRegression()
#ret = fit_predict(m, X_in = pd.DataFrame(X_in), y_in = y_in, n_folds = 5)
#order = ret.groupby('fold').apply(lambda df: df.index.get_level_values('time').min())
