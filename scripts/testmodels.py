import sys
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import brier_score_loss
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess 

#sys.path.append('/usr/people/straaten/Documents/Weave/')
sys.path.append('..')
from Weave.models import compute_forest_shaps, fit_predict, fit_predict_evaluate, permute_importance, map_foldindex_to_groupedorder, hyperparam_evaluation, BaseExceedenceModel, HybridExceedenceModel
from Weave.utils import get_timeserie_properties, brier_score_clim, collapse_restore_multiindex
from Weave.inspection import ImportanceData, MapInterface

logging.basicConfig(level = logging.DEBUG)

threshold = 0.666
separation = -15 
respagg = 31 
basepath = Path('/nobackup_1/users/straaten/shaptest/')
#basepath = Path('/scistor/ivm/jsn295/clusters_cv_spearmanpar_varalpha_strict/')
Y_path = basepath / 'response.multiagg.trended.parquet'
X_path = basepath / 'precursor.multiagg.parquet'
y = pd.read_parquet(Y_path).loc[:,(slice(None),respagg,slice(None))].iloc[:,0] # Only summer, starting 1981
#y = y.loc[y.index.month.map(lambda m: m in [7,8])] # THis is an optional subsetting to only July/ August. should not change the ordering
X = pd.read_parquet(X_path).loc[y.index, (slice(None),slice(None),slice(None),slice(None),separation,slice(None),slice(None))].dropna(axis = 0, how = 'any') # A single separation, extra level because of fold, reading both metrics
y = y.reindex(X.index)
y = y > y.quantile(threshold)
#
## Testing the relabeling according to new grouped order
map_foldindex_to_groupedorder(X = X, n_folds = 5, return_foldorder = False)
##props = X.apply(get_timeserie_properties, axis = 0, **{'scale_trend_intercept':False})

model = HybridExceedenceModel(max_depth = 5, n_estimators = 2500, min_samples_split = 30, max_features = 35, n_jobs = 20)
#test = permute_importance(model, X_in = X, y_in = y, evaluation_fn = brier_score_loss, scoring_strategy = 'argmax_of_mean', perm_imp_kwargs = dict(njobs = 10, nbootstrap = 1, nimportant_vars = 2), single_only = False, n_folds = 5, split_on_year = True)
#shappies = compute_forest_shaps(r2, X, y, on_validation = False, bg_from_training = True, sample = 'standard', n_folds = 5, split_on_year = True)

#from Weave.models import crossvalidate, get_validation_fold_time
#f = crossvalidate(5,True,True)(get_validation_fold_time)
#testi = f(X_in = y, y_in = y, end_too = True)

#evaluate_kwds = dict(scores = [brier_score_loss], score_names = ['bs'])
#hyperparams = dict(n_estimators = [1500,2500,3500])
#other_kwds = dict(n_jobs = 25, min_samples_split = 30, max_depth = 5, max_features = 35) 
#ret = hyperparam_evaluation(HybridExceedenceModel, X, y, hyperparams, other_kwds, fit_predict_evaluate_kwds = dict(properties_too = False, n_folds = 5, evaluate_kwds = evaluate_kwds))
#mean = ret.groupby('score', axis = 0).mean()
#
#baseline = BaseExceedenceModel() # Most strict baseline (non-cv) that we can imagine, just for the idea of what the bs values mean
#bs = fit_predict_evaluate(baseline, X_in = X, y_in = y, X_val = X, y_val = y, evaluate_kwds = evaluate_kwds)
#bs2 = brier_score_loss(y, fit_predict(baseline, X_in = X, y_in = y, n_folds = 5)) # Less strict (cv) baseline

