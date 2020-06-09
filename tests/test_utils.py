import numpy as np
import xarray as xr
import pandas as pd
import pytest
from scipy.stats import pearsonr

from src.utils import nanquantile, bootstrap, add_pvalue, kendall_predictand, spearmanr_wrap, agg_time

@pytest.fixture()
def supply_testfunc():
    def testfunc(data: np.ndarray, what = 'mean'):
        f = getattr(data, what)
        return float(f())
    return testfunc

def test_nanquantile_empty():
    data = np.full((5,5,5), np.nan)
    quantiles = nanquantile(array = data, q = 0.9)
    assert np.isnan(quantiles).all(), "supplying fully nan data should result in fully nan quantile array"
    assert quantiles.shape == (5,5), "a 3D data array should result in a 2D array of quantiles" 

def test_bootstrap(supply_testfunc):
    data = np.random.sample((5,2))
    n_draws = 10
    newfunc = bootstrap(n_draws = n_draws, quantile = None)(supply_testfunc)
    collection = newfunc(data = data, what = 'mean')
    assert len(collection) == n_draws, "Without quantile argument, bootstrap should return as many samples as requested"

def test_blockbootstrap_quantiles(supply_testfunc):
    data = [0]
    for i in range(80): # Generate some autocorrelated data (mu = 0, sigma =+- 1)
        data.append(0.7 * data[i] + np.random.normal(0, 0.7))
    data = np.array(data) 
    n_draws = 1000
    newfunc = bootstrap(n_draws = n_draws, quantile = [0.1,0.9])(supply_testfunc)
    newfunc_block = bootstrap(n_draws = n_draws, blocksize = 5, quantile = [0.1, 0.9])(supply_testfunc)
    qs = newfunc(data, 'mean')
    qs_block = newfunc_block(data, 'mean')
    assert len(qs) == 2 and len(qs_block) == 2, "Bootstrap should return the amount of requested quantiles, regardless of blocksize"
    assert qs_block[0] < qs[0], "The low quantile sample statistic of block-bootstrapped data should be lower than the low quantile of non-block-bootstrapped data, for AR1 data (more variability)"
    assert qs_block[1] > qs[1], "The high quantile sample statistic of block-bootstrapped data should be higher than the high quantile of non-block-bootstrapped data, for AR1 data (more variability)"

def test_bootstrap_p_values():
    """
    Estimates of bootstrapping + t-test pvalues for pearson correlation of independent normal data should give similar values as scipy.stats.pearsonr (its parametric p-values are based on normally distributed data).
    """
    independent_data = np.random.normal(loc = 0, scale = 1, size = (500,2))
    @add_pvalue
    @bootstrap(n_draws = 10000, quantile = None)
    def custom_corr(data):
        return pearsonr(x = data[:,0], y = data[:,1])[0]
    
    resample_stats = custom_corr(independent_data) # Returns (corr,p-value)
    param_stats = pearsonr(x = independent_data[:,0], y = independent_data[:,1])
    assert np.allclose(resample_stats[0], param_stats[0], atol = 0.001),"Two approaches should give similarly insignificant correlation strength with suffiecient bootstrap samples of independent normal data"
    assert np.allclose(resample_stats[1], param_stats[1], atol = 0.05),"Two approaches should give similarl p-values with sufficient bootstrap samples of independent normal data"  

def test_agg_time():
    """
    Tests if the correct number of days are the result
    """
    serie = xr.DataArray(np.arange(11), dims = ('time',), coords = {'time':pd.date_range('2000-01-01', '2000-01-11')})
    blocks = agg_time(serie, ndayagg = 3, rolling = False)
    assert len(blocks) == 11//3, "block aggregation with size 3 should result in the exact amount of full blocks that fit within the timeseries"
    rolling = agg_time(serie, ndayagg = 3, rolling = True)
    assert len(rolling) == 11 - 2, "rolling aggregation with window size 3 should result in two less left-stamped observation"
    blocks_start = agg_time(serie, ndayagg = 3, rolling = False, firstday = pd.Timestamp('2000-01-04'))
    assert len(blocks_start) == (11 - 3) // 3, "block aggregation with size 3 should result in the exact amount of full blocks that fit within the timeseries after the given first day"

# No testing, only profiling, either with timeit or cProfile. 
dummydat = np.random.random((1000,2))
@add_pvalue
@bootstrap(n_draws = 1000, quantile = None)
def corr(data):
    return kendall_predictand(data)

def speed_kendall():
    return corr(dummydat)

def speed_spearman():
    return spearmanr_wrap(dummydat)

if __name__ == "__main__":
    speed_kendall()
