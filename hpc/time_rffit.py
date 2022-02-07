"""
Call signature: python parcluster.py $TEMPDIR $PACKAGEDIR $NPROC $NTOTAL
"""
import sys
import os
import time
import itertools
import uuid
import pandas as pd
import numpy as np

from pathlib import Path
from pathlib import Path
from scipy.signal import detrend
from sklearn.metrics import brier_score_loss, log_loss

TMPDIR = Path(sys.argv[1])
PACKAGEDIR = sys.argv[2]
NPROC = int(sys.argv[3])
NTOTAL = int(sys.argv[4])

sys.path.append(PACKAGEDIR)

from Weave.models import fit_predict, fit_predict_evaluate, evaluate, hyperparam_evaluation, map_foldindex_to_groupedorder, BaseExceedenceModel, HybridExceedenceModel

basedir = Path('/scistor/ivm/jsn295/')

inputpath = basedir / 'clusters_cv_spearmanpar_varalpha_strict' # Latest dimreduced X and y data 

def read_data(responseagg = 3, separation = -7, trended = False, quantile = 0.8, pre1981 = False):
    """
    Returns the selcted X and y data
    A dataframe and a (de)trended Series
    """
    if trended:
        Y_path = (inputpath / 'response.multiagg.trended.pre1981.parquet') if pre1981 else (inputpath / 'response.multiagg.trended.parquet')
    else:
        Y_path = (inputpath / 'response.multiagg.detrended.pre1981.parquet') if pre1981 else (inputpath / 'response.multiagg.detrended.parquet')
    X_path = (inputpath / 'precursor.multiagg.pre1981.parquet') if pre1981 else (inputpath / 'precursor.multiagg.parquet')
    y = pd.read_parquet(Y_path).loc[:,(slice(None),responseagg,slice(None))].iloc[:,0] # Only summer
    X = pd.read_parquet(X_path).loc[y.index,(slice(None),slice(None),slice(None),slice(None),separation,slice(None),slice(None))].dropna(axis = 0, how = 'any') # both metrics
    y = y.reindex(X.index)
    if not quantile is None:
        y = y > y.quantile(quantile)
    map_foldindex_to_groupedorder(X = X, n_folds = 5)
    return X, y

model = HybridExceedenceModel(fit_base_to_all_cv = True, base_only = False, n_jobs = NPROC, max_depth = 5, min_samples_split = 30, n_estimators = 2500, max_features = None)


X, y = read_data(31, -15, trended = True, quantile = 0.666)
# Adding extra data
n_total = NTOTAL
for fold in X.columns.get_level_values('fold').unique():
    n_current = X.columns.get_loc_level(fold,'fold')[0].sum()
    n_missing = n_total - n_current
    if n_missing > 0:
        print(f'adding {n_missing} noise columns')
        extra = pd.DataFrame(np.random.normal(size = (X.shape[0],n_missing)), index = X.index, 
                                     columns =pd.MultiIndex.from_product([[fold],['a'],[31],[-46],[-15],list(range(n_missing)),['dummy']], names = X.columns.names))
        X = pd.merge(X,extra,left_index = True, right_index = True)

print('start timing')
st = time.perf_counter()

preds = fit_predict(model, X, y, n_folds = 5, split_on_year = True)

et = time.perf_counter()
elapsed = et - st
print('ended timing')
print(f'preds: {NTOTAL}')
print(f'ncores: {NPROC}')
print(f'seconds: {elapsed:0.4f}')

res = pd.Series([NTOTAL,NPROC,elapsed], index = ['npreds','ncores','time[s]'])
res.to_csv(TMPDIR / f'{uuid.uuid4().hex}.csv')
