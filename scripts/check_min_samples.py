import sys
import numpy as np
import xarray as xr
from pathlib import Path

PATTERNDIR = Path(sys.argv[1])
PACKAGEDIR = sys.argv[2]

sys.path.append(PACKAGEDIR)

from Weave.clustering import Clustering, MaskingError

corrfiles = [ f for f in PATTERNDIR.glob('*corr.nc') if f.is_file() ]
min_samples = 700

for corrpath in corrfiles:
    invarname = 'correlation'
    ds = xr.open_dataset(corrpath, decode_times = False)
    lags = ds.coords['lag'].values.tolist()
    for lag in lags:
        cl = Clustering()
        try: 
            cl.reshape_and_drop_obs(array = ds[invarname], mask = ~ds[invarname].sel(lag = lag).isnull(), min_samples = min_samples)
        except MaskingError:
            print(f'{corrpath}, lag {lag} less than {min_samples} samples')
