import sys
import os
import numpy as np
import pandas as pd
import shap
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess 

sys.path.append(os.path.expanduser('~/Documents/Weave/'))
from Weave.models import compute_shaps, fit_predict, fit_predict_evaluate, permute_importance, map_foldindex_to_groupedorder, hyperparam_evaluation, BaseExceedenceModel, HybridExceedenceModel
from Weave.utils import get_timeserie_properties, brier_score_clim, collapse_restore_multiindex
from Weave.inspection import ImportanceData, MapInterface

logging.basicConfig(level = logging.DEBUG)

quantile = 0.666
separation = 0 
respagg = 7 
#basepath = Path('/nobackup_1/users/straaten/shaptest/')
basepath = Path('/scistor/ivm/jsn295/clusters_cv_spearmanpar_varalpha_strict/')
Y_path = basepath / 'response.multiagg.trended.parquet'
Y_val_path = basepath / 'response.multiagg.trended.pre1981.parquet'
X_path = basepath / 'precursor.multiagg.parquet'
X_val_path = basepath / 'precursor.multiagg.pre1981.parquet'
y = pd.read_parquet(Y_path).loc[:,(slice(None),respagg,slice(None))].iloc[:,0] # Only summer, starting 1981
y_val = pd.read_parquet(Y_val_path).loc[:,(slice(None),respagg,slice(None))].iloc[:,0] # Only summer, 1950 - 1981
X = pd.read_parquet(X_path).loc[y.index, (slice(None),slice(None),slice(None),slice(None),separation,slice(None),slice(None))].dropna(axis = 0, how = 'any') # A single separation, extra level because of fold, reading both metrics
X_val = pd.read_parquet(X_val_path).loc[y_val.index, (slice(None),slice(None),slice(None),slice(None),separation,slice(None),slice(None))].dropna(axis = 0, how = 'any') # A single separation, extra level because of fold, reading both metrics
y = y.reindex(X.index)
y_val = y_val.reindex(X_val.index)
threshold = y.quantile(quantile) # Threshold comes from the crossval part
y = y > threshold
y_val = y_val > threshold

#
## Testing the relabeling according to new grouped order
map_foldindex_to_groupedorder(X = X, n_folds = 5, return_foldorder = False)
map_foldindex_to_groupedorder(X = X_val, n_folds = 5, return_foldorder = False)
##props = X.apply(get_timeserie_properties, axis = 0, **{'scale_trend_intercept':False})

#model = HybridExceedenceModel(max_depth = 5, n_estimators = 2500, min_samples_split = 30, max_features = 35, n_jobs = 20)
#test = permute_importance(model, X_in = X, y_in = y, evaluation_fn = brier_score_loss, scoring_strategy = 'argmax_of_mean', perm_imp_kwargs = dict(njobs = 10, nbootstrap = 1, nimportant_vars = 2), single_only = False, n_folds = 5, split_on_year = True)
#shappies = compute_forest_shaps(r2, X, y, on_validation = False, bg_from_training = True, sample = 'standard', n_folds = 5, split_on_year = True)

#from Weave.models import crossvalidate, get_validation_fold_time
#f = crossvalidate(5,True,True)(get_validation_fold_time)
#testi = f(X_in = y, y_in = y, end_too = True)

evaluate_kwds = dict(scores = [brier_score_loss], score_names = ['bs'])
hyperparams = dict(max_depth = [4,5,7,100],min_samples_split = [2,30,100],max_features = [25,35,100])
other_kwds = dict(n_jobs = 20, n_estimators = 2500, max_features = 20, fit_base_to_all_cv = True) # Whether it will fit to X_train plus X_val
#other_kwds = dict(n_jobs = 10, n_estimators = 1000, min_samples_split = 30, max_features = 35, fit_base_to_all_cv = True) # Whether it will fit to X_train plus X_val

"""
Old approach hyperparam evaluation in the traintest setup
TODO: Feature names error appears probably at calls of .fit and .predict of the RF's
"""
#ret = hyperparam_evaluation(HybridExceedenceModel, X, y, hyperparams, other_kwds, fit_predict_evaluate_kwds = dict(properties_too = False, n_folds = 5, evaluate_kwds = evaluate_kwds))
#mean = ret.groupby('score', axis = 0).mean()

"""
New approach with backward extension for evaluating hyperparameters.
Strategy for inside cv training, outside cv evaluation.
- Feed X_val and y_val to fit_predict evaluate, through kwds, same set always (dependent on fold).
- X_in needs to be subsetted with crossvalidation here (outside loop)

TODO: check extrapolation behaviour of the base model (is it ok to fit to the backward extension too?)
"""

n_folds = 5
kf = GroupKFold(n_splits = n_folds)
groupsize = int(np.ceil(len(X.index.year.unique()) / n_folds)) # Maximum even groupsize, except for the last fold, if unevenly divisible then last fold gets only the remainder.
groups = (X.index.year - X.index.year.min()).map(lambda year: year // groupsize)  
k = 0
#preds = []
#basemax = np.repeat(np.nan, n_folds)
returns = []
for train_index, val_index in kf.split(X = X, groups = groups): # This generates integer indices, ultimately pure numpy and slices would give views. No copy of data, better for distributed
    X_train = X.iloc[train_index, X.columns.get_loc(k)]
    y_train = y.iloc[train_index]
    X_test = X_val.iloc[:,X_val.columns.get_loc(k)]
    y_test = y_val
    ret = hyperparam_evaluation(HybridExceedenceModel, X_in = X_train, y_in = y_train, hyperparams = hyperparams, other_kwds = other_kwds, fit_predict_evaluate_kwds = dict(compare_val_train = False, properties_too = False, X_val = X_val.iloc[:,X_val.columns.get_loc(k)], y_val = y_val, evaluate_kwds = evaluate_kwds))
    returns.append(ret)
    #he = HybridExceedenceModel(max_depth = 4,**other_kwds)
    #preds.append(fit_predict(he, X_in = X_train, y_in = y_train, X_val = X_test, y_val = y_test))
    #basemax[k] = he.base.predict(X_test).max()
    k += 1

#preds = pd.concat(preds, axis = 1)
returns = pd.concat(returns, keys = pd.RangeIndex(n_folds, name = 'fold'))
mean = returns.mean(axis = 0)
returns.to_hdf(Path('/scistor/ivm/jsn295/hyperparams') / f'bs_{respagg}_{separation}_{quantile}.pre1981.h5', key = 'bs')

#baseline = BaseExceedenceModel() # Most strict baseline (non-cv) that we can imagine, just for the idea of what the bs values mean
#bs = fit_predict_evaluate(baseline, X_in = X, y_in = y, X_val = X, y_val = y, evaluate_kwds = evaluate_kwds)
#bs2 = brier_score_loss(y, fit_predict(baseline, X_in = X, y_in = y, n_folds = 5)) # Less strict (cv) baseline

