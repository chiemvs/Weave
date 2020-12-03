import sys
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import brier_score_loss
from pathlib import Path

#sys.path.append('/usr/people/straaten/Documents/Weave/')
sys.path.append('..')
from Weave.models import compute_forest_shaps, fit_predict, fit_predict_evaluate, permute_importance, map_foldindex_to_groupedorder, hyperparam_evaluation
from Weave.utils import get_timeserie_properties, brier_score_clim, collapse_restore_multiindex
from Weave.inspection import ImportanceData, MapInterface

logging.basicConfig(level = logging.DEBUG)

Y_path = '/scistor/ivm/jsn295/clusters_cv_spearmanpar_varalpha_strict/response.multiagg.detrended.parquet'
X_path = Path('/scistor/ivm/jsn295/clusters_cv_spearmanpar_varalpha_strict/precursor.multiagg.parquet')
y = pd.read_parquet(Y_path).loc[:,(slice(None),11,slice(None))].iloc[:,0] # Only summer, starting 1981
X = pd.read_parquet(X_path).loc[y.index, (slice(None),slice(None),slice(None),slice(None),-15,slice(None),'spatcov')].dropna(axis = 0, how = 'any') # A single separation, extra level because of fold
y = y.reindex(X.index)
threshold = 0.75
y = y > y.quantile(threshold)

# Testing the relabeling according to new grouped order
map_foldindex_to_groupedorder(X = X, n_folds = 5)
#props = X.apply(get_timeserie_properties, axis = 0, **{'scale_trend_intercept':False})

# Testing a classifier
#r2 = RandomForestClassifier(max_depth = 5, n_estimators = 1500, min_samples_split = 20, max_features = 0.15, n_jobs = 10) # Balanced class weight helps a lot.
evaluate_kwds = dict(scores = [brier_score_loss], score_names = ['bs'])
#test = fit_predict_evaluate(r2, X, y, n_folds = 5, evaluate_kwds = evaluate_kwds) 
#preds = fit_predict(r2, X, y, n_folds = 5)
#bs = brier_score_loss(y,preds)
bsc = brier_score_clim(threshold)

#test = permute_importance(r2, X, y, evaluation_fn = brier_score_loss, scoring_strategy = 'argmax_of_mean', perm_imp_kwargs = dict(njobs = 10, nbootstrap = 1, nimportant_vars = 2), single_only = False, n_folds = 5, split_on_year = True)
#shappies = compute_forest_shaps(r2, X, y, on_validation = False, bg_from_training = True, sample = 'standard', n_folds = 5, split_on_year = True)

#df = ImportanceData(Path('/scistor/ivm/jsn295/testshap'), respagg = 5, separation = -7)
#df.load_data(inputpath = X_path.parent)
#
#sample = df.df.loc[(5,[3,4],['siconc_nhmin'],[31]),:].iloc[:,1500] # shap siconc importance of seriesat some day 1500. Defined for fold 3 and 4 because it was computed on training data
#m = MapInterface(X_path.parent)
#result = m.map_to_fields(sample)




#data = np.stack([y.values,test.values], axis = -1)
#from utils import bootstrap, brier_score_clim 
#f = bootstrap(5000, return_numeric = True, quantile = [0.05,0.5,0.95])(evaluate)
#f2 = bootstrap(5000, blocksize = 15, return_numeric = True, quantile = [0.05,0.5,0.95])(evaluate) # object dtype array
#evaluate_kwds = dict(scores = [brier_score_loss], score_names = ['bs'])
#ret = f(data, **evaluate_kwds)
#ret2 = f2(data, **evaluate_kwds)

hyperparams = dict(min_samples_split = [20,30], max_depth = [4,5,8])
#hyperparams = dict(min_impurity_decrease = [0.001,0.002,0.003,0.005,0.01])
#hyperparams = dict(max_features = [0.05,0.1,0.15,0.2,0.25])
other_kwds = dict(n_jobs = 20, n_estimators = 1500, max_features = 0.15) 
ret = hyperparam_evaluation(RandomForestClassifier, X, y, hyperparams, other_kwds,  fit_predict_evaluate_kwds = dict(properties_too = True, n_folds = 5, evaluate_kwds = evaluate_kwds))

#def wrapper(self, *args, **kwargs):
#    return self.predict_proba(*args,**kwargs)[:,-1] # Last class is True
#RandomForestClassifier.predict = wrapper # To avoid things inside permutation importance package  
#m = RandomForestClassifier(max_depth = 5, min_samples_split = 20, n_jobs = 20, max_features = 0.15, n_estimators = 1500)
#ret = permute_importance(m, X_in = X, y_in = y, n_folds = 5, evaluation_fn = brier_score_loss, perm_imp_kwargs = dict(nimportant_vars = None, njobs = 20, nbootstrap = 1500))


"""
Own function testing
"""
#df = compute_forest_shaps(model, X_train, y_train, X_val, y_val, sample = 'standard')
#def compute_forest_shaps(model: Callable, X_in, y_in, X_val = None, y_val = None, on_validation = True, bg_from_training = True, sample = 'standard', n_folds = 10, split_on_year = True, explainer_kwargs = dict()) -> pd.DataFrame:

# Maximum prediction: 
#maxind = np.argmax(model.predict_proba(X_val)[:,-1])

# Negative case background data (all from training. because there is more, max +- 570 positive in training)
#bg_neg = shap.maskers.Independent(X_train.loc[~y_train,:], max_samples=50)
#bg_standard = shap.maskers.Independent(X_train, max_samples=50)
#bg_pos = shap.maskers.Independent(X_train.loc[y_train,:], max_samples=50)

#expl_neg = shap.TreeExplainer(model = model, data = bg_neg) # I have the feeling that this is the same as when model_output = 'raw' (needed for interaction shapley) 
#expl_standard = shap.TreeExplainer(model = model, data = bg_standard, model_output = 'raw')
#expl_pos = shap.TreeExplainer(model = model, data = bg_pos, model_output = 'probability')
#
#shap_standard = expl_standard.shap_values(X_val)
#shap_neg = expl_neg.shap_values(X_val)
#shap_pos = expl_pos.shap_values(X_val)
#"""
#Output of the shapvalues is for predict_proba so in terms of probability.
#Values of shap_values[0] are the opposite of shap_values[1], which is for which depends on model.classes_ though up to now [-1] has always been positive.
#"""
## Better plotting with better names
#X_val.columns = [s[:5] for s in X_val.columns.get_level_values(0)]
#shap.force_plot(expl_neg.expected_value[-1], shap_neg[-1][maxind], X_val.iloc[maxind,:].round(5), matplotlib =True, show = False)
#
## These are equal
#shap_pos[-1][maxind].sum() + expl_pos.expected_value[-1]
#shap_neg[-1][maxind].sum() + expl_neg.expected_value[-1]
#
## Beeswarm plot
#shap.summary_plot(shap_neg[-1], X_val.round(5))
## The difference with a beeswarm plot of Shap_pos is that the baseline is very much shifted. The bulk of the data has to be corrected downwards from the expected value.
## Also spatcov series seem quite well-behaved. Usually a large density negative spatcov values (not looking like the correlation pattern)
#
## Interaction
#shap_inter = shap.TreeExplainer(model = model).shap_interaction_values(X_val) # raw output [-1] still seems true
#shap.dependence_plot(('snowc','swvl1'),shap_inter_neg[-1],X_val)
