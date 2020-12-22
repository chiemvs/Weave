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
Y_path = '/scistor/ivm/jsn295/clusters_cv_spearmanpar_varalpha_strict/response.multiagg.trended.parquet'
#Y_path = '/scistor/ivm/jsn295/clusters_cv_spearmanpar_varalpha_strict/response.multiagg.q0.8.detrended.parquet'
X_path = Path('/scistor/ivm/jsn295/clusters_cv_spearmanpar_varalpha_strict/precursor.multiagg.parquet')
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
test = permute_importance(model, X_in = X, y_in = y, evaluation_fn = brier_score_loss, scoring_strategy = 'argmax_of_mean', perm_imp_kwargs = dict(njobs = 10, nbootstrap = 1, nimportant_vars = 2), single_only = False, n_folds = 5, split_on_year = True)
#shappies = compute_forest_shaps(r2, X, y, on_validation = False, bg_from_training = True, sample = 'standard', n_folds = 5, split_on_year = True)

#df = ImportanceData(Path('/scistor/ivm/jsn295/shap_standard_val_q08'), respagg = 3, separation = -1)
#df.load_data(inputpath = X_path.parent)
#
#sample = df.df.loc[(5,[3,4],['siconc_nhmin'],[31]),:].iloc[:,1500] # shap siconc importance of seriesat some day 1500. Defined for fold 3 and 4 because it was computed on training data
#m = MapInterface(X_path.parent)
#result = m.map_to_fields(sample)

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
#
#score_only = ret.loc[(slice(None),'bs'),:].round(3).T
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
