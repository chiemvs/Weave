"""
Statistical modelling tools
tabular data
Fitting, optimization, interpretation
"""
import numpy as np
import pandas as pd
import itertools
import logging
import shap

from typing import Callable
from collections import OrderedDict
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, brier_score_loss, log_loss
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from PermutationImportance import sklearn_permutation_importance

def evaluate(data = None, y_true = None, y_pred = None, scores = [r2_score, mean_squared_error, mean_absolute_error], score_names = ['r2','mse','mae']) -> pd.Series:
    """
    Calls scores one by one a set of predictions and true values
    these can be provided separately (both 1D) or as a joined 2D data array [:,0] true, [:,1] pred 
    this is to enable integration with bootstrapping util (repeated call of evaluate on resamples of data)
    scores should be functions that accept (y_true,y_pred) and return a float
    default are determinisic regression scores
    supply others for classification and/or probabilistic prediction like [brier_score,log_likelyhood,reliability] 
    """
    if (y_true is None) or (y_pred is None):
        y_true = data[:,0]
        y_pred = data[:,1]
    assert y_true.size == y_pred.size, "True values and predicted values should be array-like and have the same size"
    assert (len(y_true.shape) == 1) and (len(y_pred.shape) == 1), "True values and predicted values should be array-like and 1D"
    returns = pd.Series(np.full((len(scores),), np.nan, dtype = np.float64), index = pd.Index(score_names, name = 'score'))
    for score, name in zip(scores, score_names):
        returns.loc[name] = score(y_true, y_pred)
    return returns

def crossvalidate(n_folds: int = 10, split_on_year: bool = False) -> Callable:
    def actual_decorator(func: Callable) -> Callable:
        """
        Manipulates the input arguments of func
        X_in and y_in are distributed over X_train, y_train, X_val, y_val
        func should be a function that returns pandas objects
        TODO: remove debugging print statements
        possibility to split cleanly on years. Years are grouped into distinct consecutive groups, such that the amount of groups equals the disered amount of folds.
        With split on year there is no guarantee of the folds [0 to nfolds-1] to be chronological on the time axis. Therefore a sorting of time index is needed.
        """
        def wrapper(*args, **kwargs) -> pd.Series:
            try:
                X_in = kwargs.pop('X_in')
                y_in = kwargs.pop('y_in')
            except KeyError:
                X_in, y_in = args[1:3] # Model is often the first argument
                args = args[:1] + args[3:]
            if split_on_year:
                assert isinstance(X_in,(pd.Series,pd.DataFrame)), 'Pandas object with time index is needed'
                assert n_folds <= len(X_in.index.year.unique()), 'More folds than unique years requested. Unable to split_on_year.'
                kf = GroupKFold(n_splits = n_folds)
                groupsize = int(np.ceil(len(X_in.index.year.unique()) / n_folds)) # Maximum even groupsize, except for the last fold, if unevenly divisible then last fold gets only the remainder.
                groups = (X_in.index.year - X_in.index.year.min()).map(lambda year: year // groupsize)  
                assert len(groups.unique()) == n_folds
                kf_kwargs = dict(X = X_in, groups = groups)
            else:
                kf = KFold(n_splits = n_folds)
                kf_kwargs = dict(X = X_in)
            results = []
            k = 0
            for train_index, val_index in kf.split(**kf_kwargs): # This generates integer indices, ultimately pure numpy and slices would give views. No copy of data, better for distributed
                if isinstance(X_in, pd.DataFrame):
                    X_train, X_val = X_in.iloc[train_index,:], X_in.iloc[val_index]
                    y_train, y_val = y_in.iloc[train_index], y_in.iloc[val_index]
                else:
                    X_train, X_val = X_in[train_index,:], X_in[val_index,:]
                    y_train, y_val = y_in[train_index], y_in[val_index]
                kwargs.update({'X_train':X_train, 'y_train':y_train, 'X_val':X_val, 'y_val':y_val})
                logging.debug(f'fold {k}, kwargs: {kwargs.keys()}, args: {args}')
                k += 1
                results.append(func(*args, **kwargs))
            results = pd.concat(results, axis = 0, keys = pd.RangeIndex(n_folds, name = 'fold')) 
            if split_on_year:
                return(results.sort_index(axis = 0, level = -1)) # Lowest level perhaps called time, highest level in the hierarchy has just become fold,
            else:
                return(results)
        return wrapper
    return actual_decorator

def balance_training_data(how: str, X_train, y_train):
    """
    Balancing of a two-class (binarity) clasification problem
    The true class is assumed to be the minority. A 50/50 ratio can be obtained by
    i) oversampling the Trues, ii) undersampling the Falses
    (Only training data should be balanced, otherwise biased scoring)
    """
    assert y_train.dtype == np.bool, "two type boolean class is needed for balancing training data, n_True < n_False"
    if how == 'oversample': # Assumes that number of true is in the minority. Oversampling the Trues
        true_picks = np.random.choice(a = np.where(y_train == True)[0], size = np.sum(y_train == False), replace = True)
        false_true_picks = np.concatenate([np.where(y_train == False)[0], true_picks])
    elif how == 'undersample': # Assumers that the number of true is the minority. Undersampling the Falses
        assert y_train.dtype == np.bool, "two type boolean class is needed for balancing training data, n_True < n_False"
        false_picks = np.random.choice(a = np.where(y_train == False)[0], size = np.sum(y_train == True), replace = False)
        false_true_picks = np.concatenate([false_picks, np.where(y_train == True)[0]])
    else:
        raise ValueError("balance_training how argument should be one of ['oversample','undersample']")
    logging.debug(f'training set has been balanced with {how} from {len(y_train)} to {len(false_true_picks)}')
    try:
        return X_train.iloc[false_true_picks,:], y_train.iloc[false_true_picks]
    except AttributeError: # Not pandas objects but numpy arrays
        return X_train[false_true_picks,:], y_train[false_true_picks]

def fit_predict_evaluate(model: Callable, X_in, y_in, X_val = None, y_val = None, n_folds = 10, split_on_year = True, balance_training: str = None, compare_val_train: bool = True, properties_too: bool = False, evaluate_kwds: dict = dict()) -> pd.Series:
    """
    Calls the fitting and predict method of the already initialized model 
    If X_val and y_val are not supplied then it is assumed that X_in and y_in comprise both training and validation data
    We will then call the method on k_fold subsets of the data
    Input data should be the train/validation set (keep test apart)
    Has the option to not only evaluate the predictions, but to also extract forest properties if the model is a forest
    """
    if properties_too:
        assert isinstance(model, (RandomForestRegressor,RandomForestClassifier)), 'extracting properties works only with forest models'
    if isinstance(model, RandomForestClassifier): # Renaming of the methods, such that the preferred one for the classifier is a probabilistic prediction
        def wrapper(*args, **kwargs):
            return model.predict_proba(*args,**kwargs)[:,-1] # Last class is True
        model.predfunc = wrapper
    else:
        model.predfunc = model.predict

    def inner_func(model, X_train, y_train, X_val, y_val) -> pd.Series:
        if not balance_training is None:
            X_train, y_train = balance_training_data(how = balance_training, X_train = X_train, y_train = y_train)
        model.fit(X = X_train, y=y_train)
        preds = model.predfunc(X = X_val)
        scores = evaluate(y_true = y_val, y_pred = preds, **evaluate_kwds) # Unseen data
        if compare_val_train:
            preds_train = model.predfunc(X = X_train)
            scores_train = evaluate(y_true = y_train, y_pred = preds_train, **evaluate_kwds) # Seen data
            scores_valtrain = scores / scores_train # The difference between the two is informative
            scores_valtrain.index = pd.Index([f'{score}_val/train' for score in scores_valtrain.index], name = scores_valtrain.index.name)
            scores = pd.concat([scores,scores_valtrain])
        if properties_too:
            properties = get_forest_properties(model, average = True)
            properties.index.name = scores.index.name # Not completely accurate that a property is a score, but makes grouping over cross-validation folds later on easier
            scores = pd.concat([scores, properties])
        return scores

    if (X_val is None) or (y_val is None):
        f = crossvalidate(n_folds = n_folds, split_on_year = split_on_year)(inner_func)
        return f(model, X_in, y_in) # Conversion of arguments from X_in/y_in to X_train/y_train and X_val/y_val happens inside the decorating wrapper function
    else:
        return inner_func(model = model, X_train = X_in, y_train = y_in, X_val = X_val, y_val = y_val)

def fit_predict(model: Callable, X_in, y_in, X_val = None, y_val = None, n_folds = 10, split_on_year = True, balance_training: str = None):
    """
    Similar to fit_predict_evaluate, but only designed to make predictions for validation
    Within cv-mode this can give you a full set of predictions on which (in total) you can call an evaluation function,
    for that you can discard the fold index. And check (when used with split_on_year) that the index matches you y_val variable index 
    """
    if isinstance(model, RandomForestClassifier): # Renaming of the methods, such that the preferred one for the classifier is a probabilistic prediction
        def wrapper(*args, **kwargs):
            return model.predict_proba(*args,**kwargs)[:,-1] # Last class is True
        model.predfunc = wrapper
    else:
        model.predfunc = model.predict

    def inner_func(model, X_train, y_train, X_val, y_val) -> pd.Series:
        if not balance_training is None:
            X_train, y_train = balance_training_data(how = balance_training, X_train = X_train, y_train = y_train)
        model.fit(X = X_train, y=y_train)
        preds = model.predfunc(X = X_val)
        return pd.Series(preds, index = y_val.index)

    if (X_val is None) or (y_val is None):
        f = crossvalidate(n_folds = n_folds, split_on_year = split_on_year)(inner_func)
        return f(model, X_in, y_in) # Conversion of arguments from X_in/y_in to X_train/y_train and X_val/y_val happens inside the decorating wrapper function
    else:
        return inner_func(model = model, X_train = X_in, y_train = y_in, X_val = X_val, y_val = y_val)

def hyperparam_evaluation(model: Callable, X_in, y_in, hyperparams: dict, other_kwds: dict = dict(), fit_predict_kwds: dict = dict(), fit_predict_evaluate_kwds: dict = dict()) -> pd.DataFrame:
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
        if fit_predict_evaluate_kwds:
            outcomes.append(fit_predict_evaluate(mod, X_in = X_in, y_in = y_in, **fit_predict_evaluate_kwds)) # Always cross validation
        else:
            outcomes.append(fit_predict(mod, X_in = X_in, y_in = y_in, **fit_predict_kwds)) # Always cross validation

    full = pd.concat(outcomes, axis = 1, keys = pd.MultiIndex.from_tuples(keys))
    full.columns.names = list(keynames)
    return full

def permute_importance(model: Callable, X_in, y_in, X_val = None, y_val = None, evaluation_fn = mean_absolute_error, scoring_strategy = 'argmax_of_mean', perm_imp_kwargs: dict = dict(nimportant_vars = 8, njobs = -1, nbootstrap = 500), single_only: bool = False, n_folds = 10, split_on_year = True):
    """
    Calls permutation importance functionality 
    This functionality does single-multi-pass permutation on the validation part of the data
    The model should be initialized but not fitted
    If only X_in and y_in are supplied, then data is intepreted as the complete validation/training set
    on which cross validation is called.
    perm_imp_kwargs are mainly computational arguments
    """
    if single_only:
        perm_imp_kwargs['nimportant_vars'] = 1 # Only one pass neccessary

    # Use similar setup as fit_predict_evaluate, with an inner_func that is potentially called multiple times
    def inner_func(model, X_train, y_train, X_val: pd.DataFrame, y_val: pd.Series) -> pd.DataFrame:
        model.fit(X = X_train, y = y_train)
        y_val = y_val.to_frame() # Required form for perm imp
        X_val.columns = ['.'.join([str(c) for c in col]) for col in X_val.columns.values] # Collapse of the index is required unfortunately
        result = sklearn_permutation_importance(model = model, scoring_data = (X_val.values, y_val.values), evaluation_fn = evaluation_fn, scoring_strategy = scoring_strategy, variable_names = X_val.columns, **perm_imp_kwargs) # Pass the data as numpy arrays. Avoid bug in PermutationImportance, see scripts/minimum_example.py
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
        f = crossvalidate(n_folds = n_folds, split_on_year = split_on_year)(inner_func)
        return f(model = model, X_in = X_in, y_in = y_in) # Conversion of kwargs from X_in/y_in to X_train/y_train and X_val/y_val happens inside the decorating wrapper function
    else:
        return inner_func(model = model, X_train = X_in, y_train = y_in, X_val = X_val, y_val = y_val) 

def get_forest_properties(forest: RandomForestRegressor, average: bool = True):
    """
    needs a fitted forest, extracts properties of the decision tree estimators
    the amount of split nodes is always n_leaves - 1
    flatness is derived by the ratio of the actual amount of n_leaves over n_leaves_for_a_flat_tree_of_that_max_depth (2**maxdepth) (actual leaves usually less when branching of mostly in one direction).
    """
    properties = ['max_depth','node_count','n_leaves']
    #derived_property = ['flatness']
    derived_property = []
    counts = np.zeros((forest.n_estimators, len(properties + derived_property)), dtype = np.int64)
    for i, tree in enumerate(forest.estimators_):
        counts[i,:len(properties)] = [getattr(tree.tree_, name) for name in properties]
    
    # Derive the other
    #counts[:,-1] = counts[:,properties.index('n_leaves')] / 2**counts[:,properties.index('max_depth')]

    if not average:
        return pd.DataFrame(counts, index = pd.RangeIndex(forest.n_estimators, name = 'tree'), columns = properties + derived_property)
    else:
        return pd.Series(counts.mean(axis = 0), index = pd.Index(properties + derived_property, name = 'properties'))

def compute_forest_shaps(model: Callable, X_in, y_in, X_val = None, y_val = None, on_validation = True, bg_from_training = True, sample = 'standard', n_folds = 10, split_on_year = True, explainer_kwargs = dict()) -> pd.DataFrame:
    """
    Computation of (non-interaction) SHAP values through shap.TreeExplainer. Outputs a frame of shap values with same dimensions as X
    A non-fitted forest (classifier or regressor), options to get the background data from the training or the validation
    the sampling of the background can for instance be without balancing, but also in the case of classification
    with only positives or negatives
    other explainer kwargs are for instance a possible link function, or model_output
    Cross-validation if X_val and y_val are not supplied
    """
    assert sample in ['standard','negative','positive']
    max_samples = 500
    logging.debug(f'TreeShap will be started for {"validation" if on_validation else "training"}, with background data from {"validation" if not bg_from_training else "training"}, event sampling is {sample}')
    # Use similar setup as fit_predict_evaluate, with an inner_func that is potentially called multiple times
    def inner_func(model, X_train, y_train, X_val: pd.DataFrame, y_val: pd.Series) -> pd.DataFrame:
        """
        Will return a dataframe with the dimensions of X_train or X_val (depending on 'on_validation' argument
        """
        model.fit(X = X_train, y = y_train)
        if bg_from_training:
            X_bg_set, y_bg_set =  X_train, y_train
        else:
            X_bg_set, y_bg_set = X_val, y_val
        if sample == 'standard':
            background = shap.maskers.Independent(X_bg_set, max_samples = max_samples)
        elif sample == 'negative':
            background = shap.maskers.Independent(X_bg_set.loc[~y_bg_set,:], max_samples = max_samples)
        else:
            background = shap.maskers.Independent(X_bg_set.loc[y_bg_set,:], max_samples = max_samples)
        
        explainer = shap.TreeExplainer(model = model, data = background, feature_perturbation = 'interventional', **explainer_kwargs)

        shap_values = explainer.shap_values(X_val if on_validation else X_train) # slow. Outputs a numpy ndarray or a list of them when classifying. We need to add columns and indices
        if isinstance(model, RandomForestClassifier):
            shap_values = shap_values[model.classes_.tolist().index(True)] # Only the probabilities for the positive case
        return pd.DataFrame(shap_values, columns = X_val.columns if on_validation else X_train.columns, index = X_val.index if on_validation else X_train.index)

    if (X_val is None) or (y_val is None):
        f = crossvalidate(n_folds = n_folds, split_on_year = split_on_year)(inner_func)
        return f(model = model, X_in = X_in, y_in = y_in) # Conversion of kwargs from X_in/y_in to X_train/y_train and X_val/y_val happens inside the decorating wrapper function
    else:
        return inner_func(model = model, X_train = X_in, y_train = y_in, X_val = X_val, y_val = y_val) 

if __name__ == '__main__':
    from scipy.signal import detrend
    Y_path = '/nobackup_1/users/straaten/spatcov/response.multiagg.trended.parquet'
    X_path = '/nobackup_1/users/straaten/spatcov/precursor.multiagg.parquet'
    #Y_path = '/scistor/ivm/jsn295/clustertest_roll_spearman_varalpha/response.multiagg.trended.parquet'
    #X_path = '/scistor/ivm/jsn295/clustertest_roll_spearman_varalpha/precursor.multiagg.parquet'
    #Y_path = '/scistor/ivm/jsn295/clusterpar3_roll_spearman_varalpha/response.multiagg.trended.parquet'
    #X_path = '/scistor/ivm/jsn295/clusterpar3_roll_spearman_varalpha/precursor.multiagg.parquet'
    y = pd.read_parquet(Y_path).loc[:,(slice(None),7,slice(None))].iloc[:,0] # Only summer
    X = pd.read_parquet(X_path).loc[y.index, (slice(None),slice(None),slice(None),-21,slice(None),'spatcov')].dropna(axis = 0, how = 'any')
    #X = X.sort_index(axis = 1).loc[:,(slice(None),slice(21,22))] # Small subset fit test
    y = y.reindex(X.index)
    y = pd.Series(detrend(y), index = y.index, name = y.name) # Also here you see that detrending improves Random forest performance a bit
    y = y > y.quantile(0.8)

    # Testing cross validation split on_year vs not on_year
    # Validation folds should be distinc years
    #def test_func(model, X_train, y_train, X_val, y_val):
    #    print('trainslice:', y_train.index.min(), y_train.index.max())
    #    print('valslice:', y_val.index.min(), y_val.index.max())
    #    return pd.Series(dtype = np.int32)
    #
    #f1 = crossvalidate(n_folds = 5, split_on_year = False)(test_func)
    #f1(model = None, X_in = X, y_in = y)
    #f2 = crossvalidate(n_folds = 5, split_on_year = True)(test_func)
    #f2(model = None, X_in = X, y_in = y)

    # Testing a classifier
    r2 = RandomForestClassifier(max_depth = 5, n_estimators = 1500, min_samples_split = 20, max_features = 0.15, n_jobs = 7) # Balanced class weight helps a lot.
    shappies = compute_forest_shaps(r2, X, y, on_validation = True, bg_from_training = True, sample = 'standard', n_folds = 3, split_on_year = True)
    #test = fit_predict(r2, X, y, n_folds = 5) # evaluate_kwds = dict(scores = [brier_score_loss,log_loss], score_names = ['bs','ll'])
    #test.index = test.index.droplevel(0)
    #data = np.stack([y.values,test.values], axis = -1)
    #from utils import bootstrap, brier_score_clim 
    #f = bootstrap(5000, return_numeric = True, quantile = [0.05,0.5,0.95])(evaluate)
    #f2 = bootstrap(5000, blocksize = 15, return_numeric = True, quantile = [0.05,0.5,0.95])(evaluate) # object dtype array
    #evaluate_kwds = dict(scores = [brier_score_loss], score_names = ['bs'])
    #ret = f(data, **evaluate_kwds)
    #ret2 = f2(data, **evaluate_kwds)
    
    #hyperparams = dict(min_samples_split = [30,35], max_depth = [15,17,20,23])
    #hyperparams = dict(min_impurity_decrease = [0.001,0.002,0.003,0.005,0.01])
    #hyperparams = dict(max_features = [0.05,0.1,0.15,0.2,0.25])
    #other_kwds = dict(n_jobs = 20, n_estimators = 1000, max_features = 0.2) 
    #ret = hyperparam_evaluation(RandomForestRegressor, X, y, hyperparams, other_kwds,  fit_predict_evaluate_kwds = dict(properties_too = True))
    
    #def wrapper(self, *args, **kwargs):
    #    return self.predict_proba(*args,**kwargs)[:,-1] # Last class is True
    #RandomForestClassifier.predict = wrapper # To avoid things inside permutation importance package  
    #m = RandomForestClassifier(max_depth = 5, min_samples_split = 20, n_jobs = 20, max_features = 0.15, n_estimators = 1500)
    #ret = permute_importance(m, X_in = X, y_in = y, n_folds = 5, evaluation_fn = brier_score_loss, perm_imp_kwargs = dict(nimportant_vars = None, njobs = 20, nbootstrap = 1500))
