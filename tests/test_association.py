import numpy as np
import xarray as xr
import pandas as pd
from scipy.signal import detrend

from Weave.association import Associator, init_worker, lag_subset_detrend_associate 
from Weave.utils import pearsonr_wrap


def speed_lagging_subset():
    responseseries = xr.DataArray(np.arange(100), dims = ['dim_0'], coords = {'dim_0':np.arange(100)})
    oriset = xr.DataArray(np.random.choice([np.nan,1.0,2.0],(1000,)), dims = ['dim_0'], coords = {'dim_0':np.arange(1000)})
    subset = oriset.reindex_like(responseseries) 
    
    subset = subset[~subset.isnull()] # Then we only retain non-nan. detrend and pearsonr cant handle them
    subset.values = detrend(subset) # Only a single axis, replace values. We need the non-nan timeaxis to also get a potentially reduced subset of the response
    return pearsonr(responseseries.reindex_like(subset), subset) # Asofunc should return (corr,pvalue)
            
def speed_lagging_subset2():
    responseseries = xr.DataArray(np.arange(100), dims = ['dim_0'], coords = {'dim_0':np.arange(100)})
    oriset = xr.DataArray(np.random.choice([np.nan,1.0,2.0],(1000,)), dims = ['dim_0'], coords = {'dim_0':np.arange(1000)})
    combined = np.column_stack((oriset.reindex_like(responseseries),responseseries)) 
    
    combined = combined[~np.isnan(combined[:,0]),:] # Then we only retain non-nan. detrend and pearsonr cant handle them
    combined[:,0] = detrend(combined[:,0]) # Only a single axis, replace values. We need the non-nan timeaxis to also get a potentially reduced subset of the response
    return pearsonr(combined[:,0], combined[:,1]) # Asofunc should return (corr,pvalue)

def test_detrend_in_lagdetrend():
    """ Precursor that is only linear trend Should result in zero correlation 
    Precursor that is linear trend + signal should result in correlation close to one
    """
    resp = xr.DataArray(np.random.random(1000), dims = ('time',), coords = {'time':pd.date_range('1990-01-01','2000-01-01')[:1000]})
    nspace = 2
    precursor = xr.DataArray(np.arange(2000).reshape((1000,2)), dims = ('time','space'), coords = {'time':resp.time, 'space':np.arange(nspace)})
    precursor[:,-1] = precursor[:,-1] + resp * 1000
    laglist = [0]
    a = Associator(responseseries = resp, data = precursor, laglist = laglist, association = pearsonr_wrap) # generates the shared object
    init_worker(a.inarray, a.dtype, a.shape, a.coords['time'], resp, a.outarray, a.outdtype, a.outshape, a.laglist, a.asofunc) # Pretty ugly way of populating the global dict for lagsubset detr etc.
    for ind in [(i,) for i in range(nspace)]:
        lag_subset_detrend_associate(spatial_index= ind)
    np_outarray = np.frombuffer(a.outarray, dtype = a.outdtype).reshape(a.outshape) 
    corrs = np_outarray[:,0,...]
    pvals = np_outarray[:,1,...]
    assert a.outshape == (len(laglist),2,nspace), "Lagsubset_detrend should will collect 2 values per lag and spatial cell, shape of repeated calling should thus be (nlags,2,nspace)"
    assert np.allclose(corrs[:,:-1], np.zeros_like(corrs[:,:-1]), atol = 0.1), "only trend should after removal of the trend result in a correlation of 0"
    assert np.allclose(corrs[:,-1], np.full_like(corrs[:,-1], 1), atol = 0.1), "trend + signal should after removal of the trend result in a correlation of 1"
    assert (pvals[:,:-1] > 0.3).all(), "Only trend should after removal result in insignificant p-values for correlation" 

