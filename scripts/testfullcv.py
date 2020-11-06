import sys
import logging
import xarray as xr
import numpy as np
import pandas as pd
import netCDF4 as nc
from pathlib import Path
from scipy.stats import spearmanr
from scipy.signal import detrend

sys.path.append('..')
from Weave.utils import agg_time, spearmanr_par, bootstrap, prepare_scipy_stats_for_crossval, prepare_scipy_stats_for_array
from Weave.association import Associator #, init_worker, var_dict
from Weave.inputoutput import Writer
from Weave.models import crossvalidate

logging.basicConfig(level = logging.DEBUG)
#t2m = xr.open_dataarray('/nobackup_1/users/straaten/ERA5/t2m/t2m_europe.nc', group = 'mean')[:,50,50]
#t2m = xr.open_dataarray('/scistor/ivm/jsn295/processed/t2m_europe.anom.nc').sel(latitude = 52, longitude = 4)
t2m = xr.open_dataarray('/scistor/ivm/jsn295/processed/t2m_europe.anom.nc')[:,10,10]
#tcc = xr.open_dataarray('/nobackup_1/users/straaten/ERA5/tcc/tcc_europe.nc', group = 'mean')[:,:10,:10]
#z300 = xr.open_dataarray('/scistor/ivm/jsn295/processed/z300_nhmin.anom.nc').sel(latitude = 52, longitude = 4)
tcc = xr.open_dataarray('/scistor/ivm/jsn295/processed/tcc_europe.anom.nc')[:,:10,:10]

#y_in = t2m.loc[t2m.time.dt.season == 'JJA'].to_pandas()
#y_in = t2m.loc[t2m.time.dt.season == 'JJA']
#X_in = tcc.to_pandas().reindex_like(y_in) 
#data = np.column_stack([t2m,z300])

timeagg = 1
a = np.column_stack([t2m[timeagg:],t2m[:-timeagg]]) # First column is concurrent. 2nd is value from one step back
#b = np.column_stack([z300[1:],z300[:-1]]) # First column is concurrent. 2nd is value from one step back

y_in_par = xr.DataArray(a, dims = ('time','what'), coords = {'time':t2m.coords['time'][timeagg:], 'what':['t0','t-1']})
y_in_par = y_in_par.loc[y_in_par.coords['time'].dt.season == 'JJA']
y_in = t2m.loc[t2m.coords['time'].dt.season == 'JJA']

"""
How to construct the asofunc for regulare spearman cv
"""
asofunc = crossvalidate(5,True,True)(prepare_scipy_stats_for_crossval(spearmanr)) 
#asofunc(X_in = z300.to_pandas(), y_in = t2m.to_pandas())
"""
How to contruct for bootstrappint only spearman correlation (no p-vals)
"""
#asofunc = bootstrap(n_draws = 100, return_numeric = True)(prepare_scipy_stats_for_array(spearmanr))
#asofunc(data)
"""
How to construct the partial correlation spearman (not-splitonyear)
"""
#asofunc = crossvalidate(5,False,True)(prepare_scipy_stats_for_crossval(spearmanr_par)) 
#asofunc(X_in = b, y_in = a)
"""
How to construct the partial correlation spearman (with-splitonyear)
"""
asofunc_par = crossvalidate(5,True,True)(prepare_scipy_stats_for_crossval(spearmanr_par)) 
#asofunc(X_in = pd.DataFrame(b), y_in = pd.DataFrame(a)) # Not working because of course time axis is needed for indices

self = Associator(responseseries = y_in, data = tcc, laglist = [-1,-2], association = asofunc, is_partial = False, timeagg = 1, n_folds = 5)
##ret = asofunc(X_in = X_in, y_in = y_in)
corr = self.compute(nprocs = 15)
self2 = Associator(responseseries = y_in_par, data = tcc, laglist = [-1,-2], association = asofunc_par, is_partial = True, timeagg = 1, n_folds = 5)
corr_par = self2.compute(nprocs = 15)
#
#w = Writer(Path('test.nc'), varname = corr.name)
#w.create_dataset(example = corr)
#w.write(corr, attrs = corr.attrs, units = '')

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
