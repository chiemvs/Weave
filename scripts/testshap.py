import sys
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from scipy.signal import detrend

sys.path.append('/usr/people/straaten/Documents/Weave/')
from Weave.models import evaluate
Y_path = '/nobackup_1/users/straaten/spatcov/response.multiagg.trended.parquet'
X_path = '/nobackup_1/users/straaten/spatcov/precursor.multiagg.parquet'
y = pd.read_parquet(Y_path).loc[:,(slice(None),7,slice(None))].iloc[:,0] # Only summer
X = pd.read_parquet(X_path).loc[y.index, (slice(None),slice(None),slice(None),-21,slice(None),'spatcov')].dropna(axis = 0, how = 'any')
y = y.reindex(X.index)
y = pd.Series(detrend(y), index = y.index, name = y.name) # Also here you see that detrending improves Random forest performance a bit
y = y > y.quantile(0.8)

# Wrapping a probabilistic predict function to only return the positve values)
# This cannot be done. results in an unrecognized ouput method

hyperparams = dict(min_samples_split = 20, max_depth = 5, n_estimators = 1500, max_features = 0.15, n_jobs = 7)
model = RandomForestClassifier(**hyperparams)

X_train = X.iloc[:int(0.8 * len(X)),:]
y_train = y.iloc[:int(0.8 * len(X))]
X_val = X.iloc[int(0.8 * len(X)):,:]
y_val = y.iloc[int(0.8 * len(X)):]

model.fit(X = X_train, y = y_train)

# Maximum prediction: 
maxind = np.argmax(model.predict_proba(X_val)[:,-1])

# Negative case background data (all from training. because there is more, max +- 570 positive in training)
bg_neg = shap.maskers.Independent(X_train.loc[~y_train,:], max_samples=500)
bg_standard = shap.maskers.Independent(X_train, max_samples=500)
bg_pos = shap.maskers.Independent(X_train.loc[y_train,:], max_samples=500)

expl_neg = shap.TreeExplainer(model = model, data = bg_neg, model_output = 'probability') # I have the feeling that this is the same as when model_output = 'raw' (needed for interaction shapley) 
expl_standard = shap.TreeExplainer(model = model, data = bg_standard, model_output = 'probability')
expl_pos = shap.TreeExplainer(model = model, data = bg_pos, model_output = 'probability')

shap_neg = expl_neg.shap_values(X_val)
shap_pos = expl_pos.shap_values(X_val)
"""
Output of the shapvalues is for predict_proba so in terms of probability.
Values of shap_values[0] are the opposite of shap_values[1], which is for which depends on model.classes_ though up to now [-1] has always been positive.
"""
# Better plotting with better names
X_val.columns = [s[:5] for s in X_val.columns.get_level_values(0)]
shap.force_plot(expl_neg.expected_value[-1], shap_neg[-1][maxind], X_val.iloc[maxind,:].round(5), matplotlib =True, show = False)

# These are equal
shap_pos[-1][maxind].sum() + expl_pos.expected_value[-1]
shap_neg[-1][maxind].sum() + expl_neg.expected_value[-1]

# Beeswarm plot
shap.summary_plot(shap_neg[-1], X_val.round(5))
# The difference with a beeswarm plot of Shap_pos is that the baseline is very much shifted. The bulk of the data has to be corrected downwards from the expected value.
# Also spatcov series seem quite well-behaved. Usually a large density negative spatcov values (not looking like the correlation pattern)

# Interaction
shap_inter = shap.TreeExplainer(model = model).shap_interaction_values(X_val) # raw output [-1] still seems true
shap.dependence_plot(('snowc','swvl1'),shap_inter_neg[-1],X_val)
