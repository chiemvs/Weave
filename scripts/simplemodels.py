import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging
import scipy as sc
import xarray as xr
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from pathlib import Path

TMPDIR = Path(sys.argv[1])
PACKAGEDIR = sys.argv[2] 
NPROC = int(sys.argv[3])
basepath = Path(sys.argv[4])

sys.path.append(PACKAGEDIR)
from Weave.inputoutput import Reader
from Weave.processing import TimeAggregator
from Weave.models import fit_predict, fit_predict_evaluate, permute_importance, map_foldindex_to_groupedorder, hyperparam_evaluation, BaseExceedenceModel, HybridExceedenceModel, compute_shaps
from Weave.utils import get_timeserie_properties, brier_score_clim, get_euratl, get_europe, agg_time

logging.basicConfig(filename= TMPDIR / 'simplemodels.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')

"""
Creation of simple input datasets
- Cassou 2005 regime approach 20–80N, 90W–30E, but not k-means, just mapping onto the first two EOFS
- Only climate change (logistic base model)
- Average of all variables in the cluster region 
"""
#basepath = Path('/scistor/ivm/jsn295/')
outpath = basepath / 'simple_models/input/' # Initial outpath for constructed input
anomdir = basepath / 'processed/'
clusterdir = basepath / 'clusters/'
"""
Regimes, also defined at multiple timescales
As an experiment we will extract also the EOFS within anomalies at different timescales 
"""
#timeaggs = [1, 3, 5, 7, 11, 15, 21, 31] 
#for timeagg in timeaggs:
#    ta = TimeAggregator(datapath = anomdir / 'z300_nhnorm.anom.nc', share_input = True, region = get_euratl())
#    z300 = ta.compute(nprocs = NPROC, ndayagg = timeagg, method = 'mean', rolling = True) 
#    del ta
#    coords = z300.coords # Storing coords for reference
#    shape = z300.shape 
#    z300 = z300.values.reshape((shape[0],-1)) # Raveling the spatial dimension, keeping only the values
#    z300summer = z300[coords['time'].dt.season == 'JJA',...]
#
#    U, s, Vt = np.linalg.svd(z300summer, full_matrices=False) # export OPENBLAS_NUM_THREADS=25 put upfront.
#
#    # Contributions in terms of normalized eigenvalues (s**2) / biggest one drop below 0.01 beyond 20
#    ncomps = 20
#    eigvals = (s**2)[:ncomps]
#    eigvectors = Vt[:ncomps,:].reshape((ncomps,) + shape[1:])
#    joined = coords.to_dataset().drop_dims('time')
#    joined.coords['component'] = pd.Index(list(range(ncomps)), dtype = np.int64)
#    joined['eigvectors'] = (('component','latitude','longitude'),eigvectors)
#    joined['eigvalues'] = (('component',),eigvals)
#    joined.to_netcdf(outpath / f'Z300_{timeagg}D_components_JJA_1979_2019.nc')
#    
#    #Project anom fields into the ncomps, this is already at the timescale
#    projection = z300 @ Vt.T[:,:ncomps] # Not summer only because we need lagged values too
#    timeseries = xr.DataArray(projection, name = 'projection', dims = ('time','component'), coords = {'component':list(range(ncomps)),'time': coords['time']})
#    timeseries.attrs['units'] = ''
#    timeseries.to_netcdf(outpath / f'Z300_{timeagg}D_projections.nc')

"""
Average of others within response area
Variables defined only at sea will fall out of course.
"""
#response = xr.open_dataarray(anomdir / 't2m_europe.anom.nc')
#clusterfield = xr.open_dataarray(clusterdir / 't2m-q095.nc').sel(nclusters = 15) # Used to be 14, now england is not a part
#clustermask = clusterfield.where(clusterfield == 9, other = np.nan) # In this case cluster 9 is western europe.
#
## Scan for present anomaly files. 
#files = [ f.parts[-1] for f in anomdir.glob('*anom.nc') if f.is_file()]
## Don't do the response itself
#files.remove('t2m_europe.anom.nc')
#files.remove('swvl1_europe.anom.nc') # We only want to keep the merged one: swvl13
#files.remove('swvl2_europe.anom.nc')
#files.remove('swvl3_europe.anom.nc')
#files.remove('z300_nhmin.anom.nc') # Only nhnorm retained
#files.remove('z500_europe.anom.nc') # too similar to z300
#
#timeseries = []
#for f in files:
#    r = Reader( anomdir / f, region = get_europe(), blocksize = 1000)
#    values = r.read(into_shared = False)  # Daily anomaly
#    array = xr.DataArray(values, dims = r.dims, coords = r.coords)
#    res = array.groupby(clustermask.reindex_like(array, method = 'nearest')).mean('stacked_latitude_longitude') # Resolutions might not match if ERA5 land, therefore reindex
#    res = res.squeeze().drop(['nclusters','clustid'], errors = 'ignore')
#    res.name = f.split('.')[0] + '_mean'
#    timeseries.append(res.to_dataframe())
#total = pd.concat(timeseries, axis = 0)
#total.columns = pd.Index(total.columns, name = 'variable')
#total.to_parquet(outpath / 'clustermean_daily.parquet')

"""
Lagging and aggregating to be quickly usable in models
Does not follow the full model in the sense of having separate timeseries per fold
"""

def lag_aggregate(array, timeaggs: list, aggregated: bool = False) -> pd.DataFrame: # First dimension of array should be time
    """
    Possibility to be already left stamp aggregated, boolean should be true 
    And then give a single value for timeaggs (because the different arrays need to be supplied with different function calls)
    """
    collection = []
    for timeagg in timeaggs:
        laglist = list(-timeagg - absolute_separation) # Dynamic lagging to avoid overlap, lag zero is the overlap
        if aggregated:
            aggarray = array.sel(time = slice(pd.Timestamp('1981-01-01'),None))
        else:
            aggarray = agg_time(array = array, ndayagg = timeagg, method = 'mean', rolling = True, firstday = pd.Timestamp('1981-01-01'))
        for separation, lag in zip(absolute_separation,laglist): # Lagging needed
            lagarray = aggarray.copy()
            lagarray['time'] = lagarray['time'] - pd.Timedelta(str(lag) + 'D') # Ascribe each value to another timestamp (e.g. lag of -10 means precursor value of originally 1979-01-01 is assigned to 1979-01-11
            lagframe = lagarray.reindex({'time':goalset.index}).to_pandas() # Write the lagged stamped values for summer only
            # To match the extended set the column levels have to be:
            # [fold, variable, timeagg, lag, separation] clustid and metric will surely be absent, variable and fold should be present already (if desired)
            newcols = lagframe.columns.to_frame()
            newcols['timeagg'] = np.full((len(newcols),),timeagg, dtype = int)
            newcols['lag'] = np.full((len(newcols),),lag, dtype = int)
            newcols['separation'] = np.full((len(newcols),),-separation, dtype = int)
            lagframe.columns = pd.MultiIndex.from_frame(newcols)
            collection.append(lagframe)
    return pd.concat(collection, axis = 1)

#goalset = pd.read_parquet(basepath / 'clusters_cv_spearmanpar_varalpha_strict' / 'response.multiagg.trended.parquet') 
#aggs = goalset.columns.get_level_values('timeagg').unique()
#absolute_separation = np.array([0,1,3,5,7,11,15,21,31]) # Days inbetween end of precursor and beginning of response 

#means = pd.read_parquet(outpath / 'clustermean_daily.parquet') 
#means = means.stack('variable').to_xarray() # Xarrays work with the agg_time function
#mean_frame = lag_aggregate(array = means, timeaggs = aggs, aggregated = False)
#pq.write_table(pa.Table.from_pandas(mean_frame, preserve_index = True), where = outpath / 'clustermean.parquet')

#eof_frames = []
#for agg in aggs:
#    eof = xr.open_dataarray(outpath / f'Z300_{agg}D_projections.nc').rename({'component':'variable'})
#    eof_frames.append(lag_aggregate(array = eof, timeaggs = [agg], aggregated = True)) # Already left stamped
#eof_frame = pd.concat(eof_frames, axis = 1)
#pq.write_table(pa.Table.from_pandas(eof_frame, preserve_index = True), where = outpath / 'Z300_projections_multiD_patterns.parquet')

"""
Model test for a regime based input
"""
#eofs = pd.read_parquet(outpath / 'Z300_projections_1D_patterns.parquet')
eofs = pd.read_parquet(outpath / 'Z300_projections_multiD_patterns.parquet')
mean = pd.read_parquet(outpath / 'clustermean.parquet')
response = pd.read_parquet(basepath / 'simple_models'/ 'input' / 'response.multiagg.trended.parquet') 
threshold = 0.666
separation = -15 
respagg = 5 
inputtimeagg = 5 
y = response.loc[:,(slice(None),respagg,slice(None))].iloc[:,0] # Only summer, starting 1981
X = eofs.loc[:, (slice(None),inputtimeagg,slice(None),separation)].dropna(axis = 0, how = 'any') 
X_soil = mean.loc[:,('swvl13_europe_mean',inputtimeagg,slice(None),separation)]
#test = X.apply(get_timeserie_properties, axis = 0, **{'scale_trend_intercept':True}) # Scaled trends are not insane (relative to the variance)
def to_simple(X, ncomponents = 2, soilm: pd.DataFrame = None):
    """
    Create a simple set, by selecting time and 2 EOFS 
    Possible to also add a soilm as a second to last column (time always last)
    All columns are scaled, if folds are detected in the EOF
    then time and soilm get fakefolds to work with crossvalidation
    """
    time = X.index.to_julian_date()
    if 'fold' in X.columns.names: # This will be the zeroth level
        folds = X.columns.get_level_values('fold').unique().tolist()
        eof = X.loc[:,(slice(None),list(range(ncomponents)),)].copy() # components is at the level of 'variable'
        dummy_levels = [list(dummy) for dummy in eof.columns[-1][2:]] # skipping fold and variable 
        time = pd.DataFrame(time.values[:,np.newaxis], columns = pd.MultiIndex.from_product([folds,['time']] + dummy_levels, names = eof.columns.names), index = eof.index)
        if not soilm is None:
            soilm = pd.concat([soilm]*len(folds), keys = folds, names = ['fold'])
            eof = eof.join(soilm)
        eof = eof.join(time)
    else:
        eof = X.loc[:,(list(range(ncomponents)),)].copy()
        if not soilm is None:
            eof = eof.join(soilm)
        eof['time'] = time
    scaler = StandardScaler(copy = True, with_mean = True, with_std = True)
    scaled = scaler.fit_transform(eof) # leads to attributes .mean_ and .var_
    return pd.DataFrame(scaled, index = eof.index, columns = eof.columns) 

y = y > y.quantile(threshold)

#model = HybridExceedenceModel(fit_base_to_all_cv = True, **dict(max_depth = 3, n_estimators = 1000, min_samples_split = 30, max_features = 0.8, n_jobs = 20))
base = BaseExceedenceModel(greedyfit = True)
log = LogisticRegression(penalty = 'l2', C = 1.0, fit_intercept = True)
log2 = LogisticRegression(penalty = 'l2', C = 1.0, fit_intercept = True)

""" 
Compare a simple logistic model with climate change only 
Should do a bit better? With especially positive weight on the second EOF (
"""
ncomp = 2
x = to_simple(X, ncomponents = ncomp)
x_time = x[['time']] # For future reference, we can actually just use the base which fits on time in index X 
x_soil = to_simple(X, ncomponents = ncomp, soilm = X_soil)
print(f'y {respagg} > {threshold}, x {inputtimeagg} at {separation}')
print(f'base model')
base.fit(X, y, fullX = X, fully = y)
bsbase = brier_score_loss(y, base.predict(X))
print(f'scaled coefs, time: {base.coef_[0][-1]}')
print(f'bs: {bsbase}')
print(f'simple EOF, ncomponents {ncomp}')
log.fit(X = x, y = y) # Automatically a greedyfit
bslog = brier_score_loss(y, log.predict_proba(x)[:,log.classes_.tolist().index(True)])
print(f'scaled coefs, time: {log.coef_[0][-1]}, eofs {log.coef_[0][:-1]}')
print(f'bs: {bslog}')
print(f'bss 1 - (eof/base): {1 - bslog/bsbase}')
predlog = fit_predict(log, x, y, n_folds = 5) # Cross validation
bslogcv = brier_score_loss(y, predlog)
print(f'bss cv 1 - (eof/base): {1 - bslogcv/bsbase}')
print(f'simple EOF, ncomponents {ncomp} plus soil')
log2.fit(X = x_soil, y = y)
bslog2 = brier_score_loss(y, log2.predict_proba(x_soil)[:,log2.classes_.tolist().index(True)])
print(f'scaled coefs, time: {log2.coef_[0][-1]}, eofs {log2.coef_[0][:-2]}, soil {log2.coef_[0][-2]}')
print(f'bs: {bslog2}')
print(f'bss 1 - (eof_soil/base): {1 - bslog2/bsbase}')
predlog2 = fit_predict(log2, x_soil, y, n_folds = 5)
bslogcv2 = brier_score_loss(y, predlog2)
print(f'bss cv 1 - (eof_soil/base): {1 - bslogcv2/bsbase}')

"""
SHAP inspection
 Expected value plays a role
 Whether you want to get the shap deviation from the global 0.333 chance or from the local climate change signal makes a difference for the background data
 In the former the bg should be from training, or even from all data (TODO: code that into shap).
 In the latter case the bg should be from validation only
"""
test = compute_shaps(log2, x_soil, y, on_validation = True, bg_from_training = False, sample_background = 'standard', n_folds = 5, split_on_year = True, explainer_kwargs = dict(), shap_kwargs = dict())

temp = test.drop('expected_value',level = 'variable', axis = 0).groupby('fold', axis = 0).apply(lambda df: df.dropna(how = 'all', axis = 1).sum(axis = 0))
exp_vals = test.loc[(slice(None),'expected_value'),:].max(axis = 0)
exp_probs = 1/(1+np.exp(-exp_vals))

# Predictions as given by the additive logic:
shapsum = 1/(1+np.exp(-temp.sort_index(level = 'time')))
shapcorr = 1/(1+np.exp(-temp.sort_index(level = 'time') - exp_vals))
