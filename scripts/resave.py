import sys
import xarray as xr
from pathlib import Path

CLUSTERDIR = Path(sys.argv[1])
PACKAGEDIR = sys.argv[2] 

sys.path.append(PACKAGEDIR)

from Weave.inputoutput import Writer

filepaths = list(CLUSTERDIR.glob('*.nc'))
for path in filepaths:
    dat = xr.open_dataarray(path)
    dat.name = 'clustid'
    new_path = (path.parent / path.name).with_suffix(path.suffix + '2')
    w = Writer(new_path, varname = dat.name)
    w.create_dataset(example = dat)
    w.write(array = dat, units = '')

