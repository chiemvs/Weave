import sys
import numpy as np
import xarray as xr
from pathlib import Path

CORRDIR = Path(sys.argv[1])
PACKAGEDIR = sys.argv[2]

sys.path.append(PACKAGEDIR)

from Weave.src.inputoutput import Writer, Reader

example = xr.open_dataarray('/scistor/ivm/jsn295/processed/snowc_nhmin.anom.nc')[0]
lags = [1, 3, 5, 7, 9, 11, 15, 20, 25, 30, 35, 40, 45] # Currently only applicable to the rolling
files = [ f for f in CORRDIR.glob('snowc_nhmin*.corr.nc') if f.is_file() ]

for filepath in files:
    r = Reader(filepath)
    arr = r.read(into_shared = False)
    arr = xr.DataArray(arr, dims = r.dims, coords = r.coords, name = r.name)
    arr.coords['lag'] = lags
    arr = arr.reindex_like(example)
    filepath.unlink()
    w = Writer(filepath, arr.name) 
    w.create_dataset(example = arr)
    w.write(array = arr, attrs = r.attrs)
