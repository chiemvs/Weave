import logging
import sys
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss

TMPDIR = Path(sys.argv[1])
PACKAGEDIR = sys.argv[2] 
NPROC = int(sys.argv[3])
timeseriespath = Path(sys.argv[4]) 
OUTPUTDIR = Path(sys.argv[5])
sys.path.append(PACKAGEDIR)
from Weave.utils import brier_score_clim, bootstrap
from Weave.models import fit_predict, evaluate, map_foldindex_to_groupedorder, BaseExceedenceModel, HybridExceedenceModel

def read_prepare_data(responseagg = 3, separation = -7, quantile: float = 0.9):
    """
    Returns the selcted X and y data
    A dataframe and a Series
    also does the classification step with exceeding a quantile threshold
    """
    path_y = timeseriespath / 'response.multiagg.trended.parquet'
    path_X = timeseriespath / 'precursor.multiagg.parquet'
    y = pd.read_parquet(path_y).loc[:,(slice(None),responseagg,slice(None))].iloc[:,0] # Only summer
    X = pd.read_parquet(path_X).loc[y.index,(slice(None),slice(None),slice(None),slice(None),separation,slice(None),slice(None))].dropna(axis = 0, how = 'any')
    y = y.reindex(X.index) # Shape matching because of possible Nan's 
    y = y > y.quantile(quantile)
    logging.debug(f'read y from {path_y} at resptimeagg {responseagg} trended, exceeding quantile {quantile}, and read dimreduced X from {path_X} at separation {separation}')
    map_foldindex_to_groupedorder(X = X, n_folds = 5)
    logging.debug('restored fold oreder on dimreduced X')
    return X, y

def get_classif_bs(X, y, hyperparams: dict, blocksizes: list = [None]):
    #r2 = RandomForestClassifier(**hyperparams) 
    base = BaseExceedenceModel()
    hybrid = HybridExceedenceModel(**hyperparams)
    #outcomes_base = fit_predict(base, X_in = X, y_in = y, n_folds = 5)
    outcomes_base = fit_predict(base, X_in = X, y_in = y, X_val = X, y_val = y) # Most strict non-cv basemodel
    outcomes_hybrid = fit_predict(hybrid, X_in = X, y_in = y, n_folds = 5) 
    """
    procedure to drop the fourth fold
    """
    #foldsubset = outcomes.index.get_loc_level(4,'fold')[0]
    #outcomes = outcomes.iloc[~foldsubset]
    #y = y.iloc[~foldsubset]
    """
    Till here
    """
    #data = np.stack([y.values,outcomes.values], axis = -1) # Preparing for bootstrap format
    data = np.stack([y.values,outcomes_base.values,outcomes_hybrid.values], axis = -1) # Preparing for bootstrap format, 3 columns: 0 and 1 used for base(reference) score and 0 and 2 for hybrid score
    evaluate_kwds = dict(scores = [brier_score_loss], score_names = ['bs'])
    def to_skillscore(dataarray, **evaluate_kwds):
        """
        Accepting a bootstrapped dataarray. Computes reference score from columns zero and one
        and model score from columns zero and two. 
        Returns the skillscore = 1 - model/reference for case where perfect scores equal 0.
        """
        referencescore = evaluate(dataarray[:,[0,1]], **evaluate_kwds)
        modelscore = evaluate(dataarray[:,[0,2]], **evaluate_kwds)
        return 1 - modelscore/referencescore
    bootstrap_quantiles = [0.05,0.5,0.95] 
    scores = np.full((len(blocksizes),len(bootstrap_quantiles)),np.nan)
    for i, blocksize in enumerate(blocksizes): # No recomputation of the fit is neccesary
        #evaluate_decor = bootstrap(5000, return_numeric = True, blocksize = blocksize, quantile = bootstrap_quantiles)(evaluate)
        evaluate_decor = bootstrap(5000, return_numeric = True, blocksize = blocksize, quantile = bootstrap_quantiles)(to_skillscore)
        scores[i,:] = evaluate_decor(data, **evaluate_kwds)
    return pd.DataFrame(scores, index = pd.Index(blocksizes, name = 'blocksize'), columns = pd.Index(bootstrap_quantiles, name = 'bss_quantile'))


params = dict(max_depth = 5, n_estimators = 2500, min_samples_split = 30, max_features = 35, n_jobs = NPROC)

# First without any bootstrap types (more auto-correlated, more skillful with increasing timeagg)
fullset = read_prepare_data(slice(None),slice(None))
timeaggs = fullset[0].columns.get_level_values('timeagg').unique()
separations = fullset[0].columns.get_level_values('separation').unique()
del fullset

outcomes = []
keys = []
for separation in separations: 
    for timeagg in timeaggs: 
        for quantile in [0.5,0.666,0.8,0.9]:
            test = get_classif_bs(*read_prepare_data(timeagg,separation,quantile), hyperparams = params, blocksizes = [None,5,15,30,60])
            outcomes.append(test)
            keys.append((timeagg,separation,quantile))

outcomes = pd.concat(outcomes, axis = 0, keys = keys)
outcomes.index = outcomes.index.set_names(['timeagg','separation','threshold'] + outcomes.index.names[-1:])
outpath = OUTPUTDIR /'.'.join([f'{key}={item}' for key,item in params.items()] + ['bs','parquet']) 
pq.write_table(pa.Table.from_pandas(outcomes), outpath)
