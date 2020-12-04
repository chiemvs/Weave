"""
called as optimal_model_search.py $TMPDIR $PACKAGEDIR $NPROC $PATTERNDIR $OUTDIR
"""
import sys
import itertools
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import brier_score_loss
from scipy.signal import detrend
from multiprocessing import Pool

TMPDIR = Path(sys.argv[1])
PACKAGEDIR = sys.argv[2] 
NPROC = int(sys.argv[3])
PATTERNDIR = Path(sys.argv[4])
OUTDIR = Path(sys.argv[5])

sys.path.append(PACKAGEDIR)
from Weave.models import hyperparam_evaluation, permute_importance, compute_forest_shaps

logging.basicConfig(filename= TMPDIR / 'shapley_negative_train.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')

path_complete = PATTERNDIR / 'precursor.multiagg.parquet'
path_y = PATTERNDIR / 'response.multiagg.detrended.parquet'

def merge_data():
    # Merging the snow and other dimreduced timeseries to the complete timeseries
    if not path_complete.exists(): 
        other = pq.read_table(path_other).to_pandas() # Not working yet
        snow = pq.read_table(path_snow).to_pandas() # Not working yet
        total = pa.Table.from_pandas(other.join(snow))
        pq.write_table(total, where = path_complete)

def read_data(responseagg = 3, separation = -7, detrend_y = True):
    """
    Returns the selcted X and y data
    A dataframe and a Series
    """
    y = pd.read_parquet(path_y).loc[:,(slice(None),responseagg,slice(None))].iloc[:,0] # Only summer
    X = pd.read_parquet(path_complete).loc[y.index,(slice(None),slice(None),slice(None),separation,slice(None),'spatcov')].dropna(axis = 0, how = 'any')
    y = y.reindex(X.index)
    if detrend_y:
        y = pd.Series(detrend(y), index = y.index, name = y.name) # Also here you see that detrending improves Random forest performance a bit
    logging.debug(f'read y from {path_y} at resptimeagg {responseagg} and detrend is {detrend_y} and read dimreduced X from {path_complete} at separation {separation}')
    return X, y

def execute_shap(respseptup):
    """
    FUnction to fit model and call shapley values computation with certain arguments
    Can be paralellized
    """
    responseagg, separation = respseptup
    retpath = OUTDIR / str(responseagg) / str(separation)
    if not retpath.exists():
        X,y = read_data(responseagg = responseagg, separation = separation)
        y = y > y.quantile(0.8)

        m = RandomForestClassifier(max_depth = 5, n_estimators = 1500, min_samples_split = 20, max_features = 0.15, n_jobs = njobs_per_imp)
        shappies = compute_forest_shaps(m, X, y, on_validation = False, bg_from_training = True, sample = 'negative', n_folds = 5)
        retpath.mkdir(parents = True)
        pq.write_table(pa.Table.from_pandas(shappies), retpath / 'responsagg_separation.parquet')
        logging.debug(f'subprocess has written out SHAP frame at {retpath}')
    else:
        logging.debug(f'SHAP frame at {retpath} already exists')


def execute_perm_imp(respseptup):
    """
    Standalone function writes the importance of this combination 
    of responseagg and separation to a subdirectory
    no returns
    """
    responseagg, separation = respseptup
    retpath = OUTDIR / str(responseagg) / str(separation)
    if not retpath.exists():
        X,y = read_data(responseagg = responseagg, separation = separation)
        y = y > y.quantile(0.8)
        #m = RandomForestRegressor(max_depth = 20, n_estimators = 1500, min_samples_split = 35, max_features = 0.2, n_jobs = njobs_per_imp)
        def wrapper(self, *args, **kwargs):
            return self.predict_proba(*args,**kwargs)[:,-1] # Last class is True
        RandomForestClassifier.predict = wrapper # To avoid things inside permutation importance package  where it is only possible to invoke probabilistic prediction with twoclass y.
        m = RandomForestClassifier(max_depth = 5, n_estimators = 1500, min_samples_split = 20, max_features = 0.15, n_jobs = njobs_per_imp)
        ret = permute_importance(m, X_in = X, y_in = y, evaluation_fn = brier_score_loss, n_folds = 5, perm_imp_kwargs = dict(nimportant_vars = 20, njobs = njobs_per_imp, nbootstrap = 1500))
        retpath.mkdir(parents = True)
        pq.write_table(pa.Table.from_pandas(ret), retpath / 'responsagg_separation.parquet')
        logging.debug(f'subprocess has written out importance frame at {retpath}')
    else:
        logging.debug(f'importance frame at {retpath} already exists')

if __name__ == "__main__":
    """
    Parallelized with multiprocessing over repsagg / separation models
    """
    njobs_per_imp = 1
    nprocs = NPROC // njobs_per_imp
    logging.debug(f'Spinning up {nprocs} processes with each {njobs_per_imp} for shapley')
    responseaggs = np.unique(pd.read_parquet(path_y).columns.get_level_values('timeagg'))
    separations = np.unique(pd.read_parquet(path_complete).columns.get_level_values('separation'))
    with Pool(nprocs) as p:
        p.map(execute_shap, itertools.product(responseaggs, separations))
    """
    Parallelized with threading for forest fitting and permutation importance per respagg / separation model
    """
    #responseaggs = np.unique(pd.read_parquet(path_y).columns.get_level_values('timeagg'))
    #separations = np.unique(pd.read_parquet(path_complete).columns.get_level_values('separation'))
    #njobs_per_imp = NPROC
    #for respagg_sep in itertools.product(responseaggs, separations):
    #    execute_perm_imp(respagg_sep)
