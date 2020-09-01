"""
called as optimal_model_search.py $TMPDIR $PACKAGEDIR $NPROC $OUTDIR
"""
import sys
import itertools
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from scipy.signal import detrend
from multiprocessing import Pool

TMPDIR = Path(sys.argv[1])
PACKAGEDIR = sys.argv[2] 
NPROC = int(sys.argv[3])
OUTDIR = Path(sys.argv[4])

sys.path.append(PACKAGEDIR)
from Weave.models import hyperparam_evaluation, permute_importance

logging.basicConfig(filename= TMPDIR / 'importance.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')

# Merging the snow and other dimreduced timeseries
path_other = Path('/scistor/ivm/jsn295/clustertest_roll_spearman_varalpha/precursor.other.multiagg.parquet')
path_snow = Path('/scistor/ivm/jsn295/clustertest_roll_spearman_varalpha/precursor.snowsea.multiagg.parquet')
path_complete = Path('/scistor/ivm/jsn295/clustertest_roll_spearman_varalpha/precursor.multiagg.parquet')
path_y = Path('/scistor/ivm/jsn295/clustertest_roll_spearman_varalpha/response.multiagg.trended.parquet')

if not path_complete.exists(): 
    other = pq.read_table(path_other).to_pandas() # Not working yet
    snow = pq.read_table(path_snow).to_pandas() # Not working yet
    total = pa.Table.from_pandas(other.join(snow))
    pq.write_table(total, where = path_complete)

# Hyperparameter stuff
def read_data(responseagg = 3, separation = -7, detrend_y = True):
    """
    Returns the selcted X and y data
    A dataframe and a Series
    """
    y = pd.read_parquet(path_y).loc[:,(slice(None),responseagg,slice(None))].iloc[:,0] # Only summer
    X = pd.read_parquet(path_complete).loc[y.index,(slice(None),slice(None),slice(None),separation)].dropna(axis = 0, how = 'any')
    y = y.reindex(X.index)
    if detrend_y:
        y = pd.Series(detrend(y), index = y.index, name = y.name) # Also here you see that detrending improves Random forest performance a bit
    logging.debug(f'read y from {path_y} at resptimeagg {responseagg} and detrend is {detrend_y} and read dimreduced X from {path_complete} at separation {separation}')
    return X, y

"""
Some loop for exploring scores over multiple hyperparams? Models? 
Optimality different for resptimeagg, separation? Yes probably slightly so. First of all the amount of predictors can vary a bit.
"""
#X,y = read_data(responseagg = 15, separation = -15)

#hyperparams = dict(max_depth = [None,10,20,30], min_samples_split = [5,10,20,50,100,300])
#hyperparams = dict(min_samples_split = [5,10,50,100], max_depth = [350,400])
#other_kwds = dict(n_jobs = n_procs, max_features = 'auto')

#hyperparams = dict(n_estimators = [100,150,200], max_depth = [400,500], max_features = [0.1,0.3,0.5,0.7])
#other_kwds = dict(n_jobs = n_procs, min_samples_split = 50)

#hyperparams = dict(min_samples_split = [50,70,90], max_depth =  [400,500,600])
#other_kwds = dict(n_jobs = n_procs, n_estimators = 200, max_features = 0.3)
#ret = hyperparam_evaluation(RandomForestRegressor, X, y, hyperparams, other_kwds)

"""
Importance check loop
We kindof want 3 processors per permutation importance job
"""

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
        m = RandomForestRegressor(max_depth = 500, n_estimators = 200, min_samples_split = 70, max_features = 0.3, n_jobs = njobs_per_imp)
        ret = permute_importance(m, X_in = X, y_in = y, perm_imp_kwargs = dict(nimportant_vars = 8, njobs = njobs_per_imp, nbootstrap = 500))
        retpath.mkdir(parents = True)
        pq.write_table(pa.Table.from_pandas(ret), retpath / 'responsagg_separation.parquet')
        logging.debug(f'subprocess has written out importance frame at {retpath}')
    else:
        logging.debug(f'importance frame at {retpath} already exists')

if __name__ == "__main__":
    njobs_per_imp = 1
    nprocs = NPROC // njobs_per_imp
    logging.debug(f'Spinning up {nprocs} processes with each {njobs_per_imp} for permutation importance')
    responseaggs = np.unique(pd.read_parquet(path_y).columns.get_level_values('timeagg'))
    separations = np.unique(pd.read_parquet(path_complete).columns.get_level_values('separation'))
    with Pool(nprocs) as p:
        p.map(execute_perm_imp, itertools.product(responseaggs, separations))
