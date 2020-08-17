"""
Statistical modelling tools
tabular data
Fitting, optimization, interpretation
"""
import numpy as np
import pandas as pd
import itertools

from typing import Callable
from collections import OrderedDict
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def evaluate(y_true, y_pred, scores = [r2_score, mean_squared_error, mean_absolute_error], score_names = ['r2','mse','mae']) -> pd.Series:
    """
    Calls scores one by one a set of predictions and true values
    scores should be functions that accept (y_true,y_pred) and return a float
    default are determinisic regression scores
    supply others for classification and/or probabilistic prediction 
    """
    assert y_true.size == y_pred.size, "True values and predicted values should be array-like and have the same size"
    assert (len(y_true.shape) == 1) and (len(y_pred.shape) == 1), "True values and predicted values should be array-like and 1D"
    returns = pd.Series(np.full((len(scores),), np.nan, dtype = np.float64), index = pd.Index(score_names, name = 'score'))
    for score, name in zip(scores, score_names):
        returns.loc[name] = score(y_true, y_pred)
    return returns

def crossval(model: Callable, X_in, y_in, n_folds = 10) -> pd.Series:
    """
    Calls the fitting and predict method of the already initialized model 
    on k_fold subsets of the data
    Input data should be the train/validation set (keep test apart)
    """
    kf = KFold(n_splits=n_folds)
    results = []
    for train_index, val_index in kf.split(X): # This generates integer indices, ultimately pure numpy and slices would give views. No copy of data, better for distributed
        if isinstance(X, pd.DataFrame):
            X_t_fold, X_v_fold = X_in.iloc[train_index,:], X_in.iloc[val_index]
            y_t_fold, y_v_fold = y_in.iloc[train_index], y_in.iloc[val_index]
        else:
            X_t_fold, X_v_fold = X_in[train_index,:], X_in[val_index,:]
            y_t_fold, y_v_fold = y_in[train_index], y_in[val_index]
        model.fit(X = X_t_fold, y=y_t_fold)
        preds = model.predict(X = X_v_fold)
        results.append(evaluate(y_true = y_v_fold, y_pred = preds))
    return(pd.concat(results, axis = 0, keys = pd.RangeIndex(n_folds, name = 'fold')))

def hyperparam_evaluation(model: Callable, X_in, y_in, hyperparams: dict, other_kwds: dict = dict()) -> pd.DataFrame:
    """
    Model agnostic function for exploring sets of hyperparameters
    Initializes and trains the supplied model class k-times for each combination 
    Variable params should be supplied as lists in the varydict
    all combinations are tested 
    Input data should be the train/validation set (keep test apart)
    TODO: parallel cross-fitting plus evaluation?
    """
    # Ordered to make sure we preserve the naming when returning the parameter combinations
    hyperparams = OrderedDict(hyperparams)
    keynames = hyperparams.keys()
    keys = []
    outcomes = []
    for paramcomb in itertools.product(*hyperparams.values()): # Create value combinations
        keys.append(paramcomb)
        paramcomb = dict(zip(hyperparams.keys(),paramcomb)) # Give back keys
        paramcomb.update(other_kwds) # Complement with non-varying parameters
        mod = model(**paramcomb)
        outcomes.append(crossval(mod, X_in = X_in, y_in = y_in, n_folds = 10))

    full = pd.concat(outcomes, axis = 1, keys = pd.MultiIndex.from_tuples(keys))
    full.columns.names = list(keynames)
    return full

Y_path = '/nobackup_1/users/straaten/spatcov/response.multiagg.trended.parquet'
X_path = '/nobackup_1/users/straaten/spatcov/precursor.multiagg.parquet'
y = pd.read_parquet(Y_path).loc[:,(slice(None),5,slice(None))].iloc[:,0]
X = pd.read_parquet(X_path).loc[y.index, (slice(None),slice(None),slice(None),-6)].dropna(axis = 0, how = 'any')
y = y.reindex(X.index)

hyperparams = dict(max_features = ['sqrt','log2'], min_samples_leaf = [200,300])
other_kwds = dict(n_jobs = 7)
