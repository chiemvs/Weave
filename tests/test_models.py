import numpy as np
import pandas as pd
import sys

from Weave.models import crossvalidate

def test_crossvalidate():
    """
    Tests whether the data is splitted cleanly in non-overlapping and unique parts
    """
    Xdata = np.arange(100).reshape((100,1))
    ydata = np.arange(100)
    grandsum = np.sum(ydata)
    n_folds = 10
    def sumfunc(X_train, y_train, X_val, y_val):
        assert X_train.sum() + X_val.sum() == grandsum, 'All X data should be devided over train and validation and its total should match the grandsum of the dataset'
        assert y_train.sum() + y_val.sum() == grandsum, 'All y data should be devided over train and validation and its total should match the grandsum of the dataset' 
        return pd.Series(X_val.sum())
    f = crossvalidate(n_folds = n_folds)(sumfunc)
    allsums = f(X_in = Xdata, y_in = ydata)
    assert len(np.unique(allsums.values)) == n_folds, 'The data has been split into n non-overlapping folds, (the sum of) each resulting validation set should be unique'
