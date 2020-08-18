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
from scipy.signal import detrend
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from PermutationImportance import sklearn_permutation_importance


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

def crossvalidate(n_folds: int = 10) -> Callable:
    def actual_decorator(func: Callable) -> Callable:
        """
        Manipulates the input arguments of func
        X_in and y_in are distributed over X_train, y_train, X_val, y_val
        func should be a function that returns pandas objects
        TODO: remove debugging print statements
        """
        def wrapper(*args, **kwargs) -> pd.Series:
            kf = KFold(n_splits=n_folds)
            results = []
            try:
                X_in = kwargs.pop('X_in')
                y_in = kwargs.pop('y_in')
                print('manipulating kwargs', kwargs)
            except KeyError:
                X_in, y_in = args[1:3] # Model is often the first argument
                args = args[:1] + args[3:]
                print('manipulating args', args)
            k = 0
            for train_index, val_index in kf.split(X_in): # This generates integer indices, ultimately pure numpy and slices would give views. No copy of data, better for distributed
                if isinstance(X_in, pd.DataFrame):
                    X_train, X_val = X_in.iloc[train_index,:], X_in.iloc[val_index]
                    y_train, y_val = y_in.iloc[train_index], y_in.iloc[val_index]
                else:
                    X_train, X_val = X_in[train_index,:], X_in[val_index,:]
                    y_train, y_val = y_in[train_index], y_in[val_index]
                kwargs.update({'X_train':X_train, 'y_train':y_train, 'X_val':X_val, 'y_val':y_val})
                print(f'fold {k}, kwargs: {kwargs.keys()}, args: {args}')
                k += 1
                results.append(func(*args, **kwargs))
            return(pd.concat(results, axis = 0, keys = pd.RangeIndex(n_folds, name = 'fold')))
        return wrapper
    return actual_decorator

def fit_predict_evaluate(model: Callable, X_in, y_in, X_val = None, y_val = None, n_folds = 10) -> pd.Series:
    """
    Calls the fitting and predict method of the already initialized model 
    If X_val and y_val are not supplied then it is assumed that X_in and y_in comprise both training and validation data
    We will then call the method on k_fold subsets of the data
    Input data should be the train/validation set (keep test apart)
    """
    def inner_func(model, X_train, y_train, X_val, y_val) -> pd.Series:
        model.fit(X = X_train, y=y_train)
        preds = model.predict(X = X_val)
        return evaluate(y_true = y_val, y_pred = preds)

    if (X_val is None) or (y_val is None):
        f = crossvalidate(n_folds = n_folds)(inner_func)
        return f(model, X_in, y_in) # Conversion of arguments from X_in/y_in to X_train/y_train and X_val/y_val happens inside the decorating wrapper function
    else:
        return inner_func(model = model, X_train = X_in, y_train = y_in, X_val = X_val, y_val = y_val)

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
        outcomes.append(fit_predict_evaluate(mod, X_in = X_in, y_in = y_in, n_folds = 10)) # Always cross validation

    full = pd.concat(outcomes, axis = 1, keys = pd.MultiIndex.from_tuples(keys))
    full.columns.names = list(keynames)
    return full

def permute_importance(model: Callable, single_only: bool = False):
    """
    Calls permutation importance functionality 
    This functionality does single-multi-pass permutation on the validation part of the data
    If model is already fitted then data is intepreted as validation data only
    If model is not yet fitted then data is interpreted as validation/training set
    And it proceeds in a cross validation setting
    """
    # Use similar setup as fit_predict_evaluate, with an inner_func
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
Y_path = '/nobackup_1/users/straaten/spatcov/response.multiagg.trended.parquet'
X_path = '/nobackup_1/users/straaten/spatcov/precursor.multiagg.parquet'
y = pd.read_parquet(Y_path).loc[:,(slice(None),5,slice(None))].iloc[:,0] # Only summer
X = pd.read_parquet(X_path).loc[y.index, (slice(None),slice(None),slice(None),-6)].dropna(axis = 0, how = 'any')
y = y.reindex(X.index)
y = pd.Series(detrend(y), index = y.index, name = y.name) # Also here you see that detrending improves Random forest performance a bit

hyperparams = dict(max_depth = [None,10,20,30], min_samples_split = [5,10,20,50,100,300])
#hyperparams = dict(min_samples_split = [10,20,50,100,300])
other_kwds = dict(n_jobs = 7, max_depth = None)
#ret = hyperparam_evaluation(RandomForestRegressor, X, y, hyperparams, other_kwds)
# Small min_samples_split? But difficult to see real trends in the hyperparams as generally not a very skillful situation in this small case study. Deeper than this max-depth seems better

model = RandomForestRegressor(max_depth = 40, min_samples_split = 50, n_jobs = 7)
ret = fit_predict_evaluate(model, X, y)
#model.fit(X = X.iloc[:int(0.8 * len(y)),:], y = y.iloc[:int(0.8 * len(y))]) 
#valx = X.iloc[int(0.8 * len(y)):,:]
#valx.columns = ['.'.join([str(c) for c in col]) for col in valx.columns.values] # Collapse of the index is required unfortunately
#valy = y.iloc[int(0.8 * len(y)):].to_frame() # Required form by permimp
#
#result = sklearn_permutation_importance(model, (valx,valy), mean_squared_error, 'argmax_of_mean', variable_names = valx.columns.values, nbootstrap = 500, subsample = 1, nimportant_vars = 8, njobs = 7)
#singlepass = result.retrieve_singlepass() # ordered Dictionary with tuples of (rank, n_boots_trap_scores) for each variable name
#ranksingle = pd.Series([tup[0] for tup in singlepass.values()], index = singlepass.keys())
## beyond the ranks, should I perhaps combine the bootstrapped score in the cross validation setup?u
#multipass = result.retrieve_multipass()
#rankmulti = pd.Series([tup[0] for tup in multipass.values()], index = multipass.keys())
