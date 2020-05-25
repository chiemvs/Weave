import numpy as np
# spatial covariance of a loaded anomaly block with a pattern. Possibly flattened (when no domain subset region is supplied at reading) but no need for sharing as this cannot be paralelized. Although.. perhaps over time, or over lags.


# At each time i 

def spatcov(pattern: np.ndarray, precursor: np.ndarray):
    """
    pattern: (nlags, ...)
    aggregated precursor (ntime, ...)
    spatial dimensions should be equal
    returns (nlags,ntime) 
    """
    # Flatten the spatial dimensions
    pattern = pattern.reshape((pattern.shape[0],-1))
    precursor = precursor.reshape((precursor.shape[0],-1))

    patterndiff = pattern - np.nanmean(pattern, -1)[:,np.newaxis] # 2D, (nlags,nspace)
    precursordiff = precursor - np.nanmean(precursor, -1)[:,np.newaxis] # 2D (ntime,nspace)

    def onelag_cov(onepatterndiff, precursordiff):
        """
        covariance timeseries for one lag, unbiased estimate by n - 1   
        input: onepatterndiff 1D (nobs,), precursordiff 2D (ntime,nobs)
        output: 1D (ntime,)
        """
        return(np.nansum(precursordiff * onepatterndiff, axis = -1)/(len(onepatterndiff) - 1))

    allseries = np.apply_along_axis(onelag_cov, axis = 1, arr = patterndiff, precursordiff = precursordiff)
    return(allseries)
