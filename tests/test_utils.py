import numpy as np
import pytest
from scipy.stats import pearsonr

from src.utils import nanquantile, bootstrap, add_pvalue

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
    
