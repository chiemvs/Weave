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

def permute_importance(model: Callable, X_in, y_in, X_val = None, y_val = None, evaluation_fn = mean_absolute_error, scoring_strategy = 'argmax_of_mean', perm_imp_kwargs: dict = dict(nimportant_vars = 8, njobs = -1, nbootstrap = 500), single_only: bool = False, n_folds = 10):
    """
    Calls permutation importance functionality 
    This functionality does single-multi-pass permutation on the validation part of the data
    The model should be initialized but not fitted
    If only X_in and y_in are supplied, then data is intepreted as the complete validation/training set
    on which cross validation is called.
    perm_imp_kwargs are mainly computational arguments
    """
    if single_only:
        perm_imp_kwargs.update(dict(n_important_vars = 1)) # Only one pass neccessary

    # Use similar setup as fit_predict_evaluate, with an inner_func that is potentially called multiple times
    def inner_func(model, X_train, y_train, X_val: pd.DataFrame, y_val: pd.Series) -> pd.DataFrame:
        model.fit(X = X_train, y = y_train)
        y_val = y_val.to_frame() # Required form for perm imp
        X_val.columns = ['.'.join([str(c) for c in col]) for col in X_val.columns.values] # Collapse of the index is required unfortunately
        result = sklearn_permutation_importance(model = model, scoring_data = (X_val, y_val), evaluation_fn = evaluation_fn, scoring_strategy = scoring_strategy, variable_names = X_val.columns.values, **perm_imp_kwargs)
        singlepass = result.retrieve_singlepass()
        singlepass_rank_scores = pd.DataFrame([{'rank':tup[0], 'score':np.mean(tup[1])} for tup in singlepass.values()]) # We want to export both rank and mean score. (It is allowed to average here over all bootstraps even when this happens in one fold of the cross validation, as the grand mean will be equal as group sizes are equal over all cv-folds)
        singlepass_rank_scores.index = pd.MultiIndex.from_tuples([tuple(string.split('.')) for string in singlepass.keys()], names = X_train.columns.names)
        if single_only:
            return singlepass_rank_scores
        else: # Multipass dataframe probably contains the scores and ranks of only a subset of nimportant_vars variables, nrows is smaller than singlepass
            multipass = result.retrieve_multipass()
            multipass_rank_scores = pd.DataFrame([{'rank':tup[0], 'score':np.mean(tup[1])} for tup in multipass.values()])
            multipass_rank_scores.index = pd.MultiIndex.from_tuples([tuple(string.split('.')) for string in multipass.keys()], names = X_train.columns.names)
            multipass_rank_scores.columns = pd.MultiIndex.from_product([['multipass'], multipass_rank_scores.columns])# Add levels for the merging
            singlepass_rank_scores.columns = pd.MultiIndex.from_product([['singlepass'], singlepass_rank_scores.columns])# Add levels for the merging
            return singlepass_rank_scores.join(multipass_rank_scores, how = 'left') # Index based merge

    if (X_val is None) or (y_val is None):
        f = crossvalidate(n_folds = n_folds)(inner_func)
        return f(model = model, X_in = X_in, y_in = y_in) # Conversion of kwargs from X_in/y_in to X_train/y_train and X_val/y_val happens inside the decorating wrapper function
    else:
        return inner_func(model = model, X_train = X_in, y_train = y_in, X_val = X_val, y_val = y_val) 

if __name__ == '__main__':
    from scipy.signal import detrend
    Y_path = '/nobackup_1/users/straaten/spatcov/response.multiagg.trended.parquet'
    X_path = '/nobackup_1/users/straaten/spatcov/precursor.multiagg.parquet'
    y = pd.read_parquet(Y_path).loc[:,(slice(None),5,slice(None))].iloc[:,0] # Only summer
    X = pd.read_parquet(X_path).loc[y.index, (slice(None),slice(None),slice(None),-6)].dropna(axis = 0, how = 'any')
    y = y.reindex(X.index)
    y = pd.Series(detrend(y), index = y.index, name = y.name) # Also here you see that detrending improves Random forest performance a bit
    
    #hyperparams = dict(max_depth = [None,10,20,30], min_samples_split = [5,10,20,50,100,300])
    #hyperparams = dict(min_samples_split = [10,20,50,100,300])
    #other_kwds = dict(n_jobs = 7, max_depth = None)
    #ret = hyperparam_evaluation(RandomForestRegressor, X, y, hyperparams, other_kwds)
    # Small min_samples_split? But difficult to see real trends in the hyperparams as generally not a very skillful situation in this small case study. Deeper than this max-depth seems better
    
    #m = RandomForestRegressor(max_depth = 40, min_samples_split = 50, n_jobs = 7 )
    #ret = permute_importance(m, X_in = X, y_in = y, perm_imp_kwargs = dict(nimportant_vars = 8, njobs = 7, nbootstrap = 200))
    ## Select most important variables has become easy, but quite variabe over all folds
    #ret.loc[ret.loc[:,('multipass','rank')] == 0,:]
    ## Mean ranks / scores over all folds:
    #meanimp = ret.groupby(['variable','timeagg','lag','separation','clustid','metric']).mean()
