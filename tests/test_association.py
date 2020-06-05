import numpy as np
import xarray as xr
from scipy.stats import pearsonr
from scipy.signal import detrend

#from src.utils import get_corresponding_ctype
#from src.association import init_worker, lag_subset_detrend_associate 

responseseries = xr.DataArray(np.arange(100), dims = ['dim_0'], coords = {'dim_0':np.arange(100)})
oriset = xr.DataArray(np.random.choice([np.nan,1.0,2.0],(1000,)), dims = ['dim_0'], coords = {'dim_0':np.arange(1000)})

def test_lagging_subset():
    subset = oriset.reindex_like(responseseries) 
    
    subset = subset[~subset.isnull()] # Then we only retain non-nan. detrend and pearsonr cant handle them
    subset.values = detrend(subset) # Only a single axis, replace values. We need the non-nan timeaxis to also get a potentially reduced subset of the response
    return pearsonr(responseseries.reindex_like(subset), subset) # Asofunc should return (corr,pvalue)
            
def test_lagging_subset2():
    combined = np.column_stack((oriset.reindex_like(responseseries),responseseries)) 
    
    combined = combined[~np.isnan(combined[:,0]),:] # Then we only retain non-nan. detrend and pearsonr cant handle them
    combined[:,0] = detrend(combined[:,0]) # Only a single axis, replace values. We need the non-nan timeaxis to also get a potentially reduced subset of the response
    return pearsonr(combined[:,0], combined[:,1]) # Asofunc should return (corr,pvalue)

def reindex1():
    resp_time = responseseries.dim_0
    for i in range(10):
        oriset.reindex_like(resp_time)

def reindex2():
    for i in range(10):
        oriset.reindex_like(responseseries)
