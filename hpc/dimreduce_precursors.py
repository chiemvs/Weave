"""
Call signature: python dimreduce_precursors.py $TEMPDIR $PACKAGEDIR $NPROC $ANOMDIR $CLUSTERDIR $PATTERNDIR $OUTDIR
Goal is to compute a dataframe of timeseries, defined on a daily resolution
One series per variable and timeagg (distributed over files) and per lag and clustid (saved within patternfile)
"""

import sys
import logging
import numpy as np
import xarray as xr
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

TMPDIR = Path(sys.argv[1])
PACKAGEDIR = sys.argv[2] 
NPROC = int(sys.argv[3])
ANOMDIR = Path(sys.argv[4])
CLUSTERDIR = Path(sys.argv[5])
PATTERNDIR = Path(sys.argv[6])
OUTDIR = Path(sys.argv[7])

sys.path.append(PACKAGEDIR)

from Weave.processing import TimeAggregator
from Weave.inputoutput import Writer, Reader
from Weave.utils import agg_time, Region
from Weave.dimreduction import spatcov_multilag, mean_singlelag

logging.basicConfig(filename= TMPDIR / 'dimreduce_precursors_snowsea.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(relativeCreated)d-%(message)s')
firstday = pd.Timestamp('1981-01-01')
responseclustid = 9
timeaggs = [1, 3, 5, 7, 11, 15, 21, 31] 
# Response timeseries is not linked to any of the processing of the response
# Only the starting date is important. 
# We will make a seperate response dataframe now first
# Only rolling aggregation is possible for intercomparing timescales, as those are equally (daily) stamped
response_output = OUTDIR / '.'.join(['response','multiagg','trended','parquet']) 
if not response_output.exists():
    logging.debug(f'no previously existing file found at {response_output}')
    response = xr.open_dataarray(ANOMDIR / 't2m_europe.anom.nc')
    clusterfield = xr.open_dataarray(CLUSTERDIR / 't2m-q095.nc').sel(nclusters = 14)
    reduced = response.groupby(clusterfield).mean('stacked_latitude_longitude')
    reduced = reduced.sel(clustid = responseclustid) # In this case cluster 9 is western europe.
    response.close()
    output = []
    for responsetimeagg in timeaggs:
        responseagg = agg_time(array = reduced, ndayagg = responsetimeagg, method = 'mean', rolling = True, firstday = firstday)
        logging.debug(f'aggregated response to {responsetimeagg} day timeseries')
        summersubset = responseagg[responseagg.time.dt.season == 'JJA']
        summersubset = pd.DataFrame(summersubset.values, index = summersubset.coords['time'].to_index(), columns = pd.MultiIndex.from_tuples([(summersubset.name,responsetimeagg,responseclustid)], names = ['variable','timeagg','clustid']))
        # Smallest aggregation should be a reasonable starting point (it has the largest length)
        # But large chance that all have all summer timesteps (left stamping causes this only at the end of the anomaly series)
        output.append(summersubset)

    output = pd.concat(output, axis = 1, join = 'outer')
    pq.write_table(pa.Table.from_pandas(output), response_output)
    del response, reduced, output
else:
    logging.debug(f'previously existing file found at {response_output}, do nothing')

# Only rolling aggregation is possible for intercomparing timescales, as those are equally (daily) stamped
do = 'snowsea' # 'other'  
if do == 'other':
    files = [ f for f in PATTERNDIR.glob('*corr.nc') if (f.is_file() and not (f.name[:4] in ['sico','snow']))]
else:
    files = [ f for f in PATTERNDIR.glob('*corr.nc') if (f.is_file() and (f.name[:4] in ['sico','snow']))]
#files = [ f for f in PATTERNDIR.glob('*nhmin*corr.nc') if f.is_file()]
#files = [Path('/scistor/ivm/jsn295/clustertest_roll_spearman_varalpha/snowc_nhmin.31.corr.nc')]
to_reduce = ['snowc_nhmin','siconc_nhmin'] # Variables with huge files and remote clusters. Are handled differently (not stacked, but aggregated per cluster per lag)
# first level loop is variable / timeagg combinations over files
# Each file has unique clustid shapes per lag, so the
# Second level of the loop is over the lags contained within the file
# The third level is then the clustids. 
# On that subset we call the timeaggregator and compute spatial covarianceand mean, this increases read access
# Internal to spatcov multlilag we have the multiple lags, contained within the file, but this is not used because domains do not overlap
outcomes = [] # Collects both spatcov and mean
for inputpath in files:
    filename = inputpath.parts[-1]
    variable, timeagg = filename.split('.')[:2]
    anompath = list(ANOMDIR.glob(f'{variable}.anom*'))[0]
    ds = xr.open_dataset(inputpath, decode_times = False).drop_sel(lag = 0) # Remove the simultaneous lag = 0. Information cannot be used in the models
    def find_nunique(arr: xr.DataArray) -> xr.DataArray:
        uniques = np.unique(arr)
        nunique = len(uniques[~np.isnan(uniques)])
        return xr.DataArray(nunique, dims = ('count',), coords = {'count':[arr.name]})
    nclusters = ds['clustid'].groupby('lag').apply(find_nunique)
    lags = nclusters.coords["lag"].values
    if np.any(nclusters.values > 0):
        if not variable in to_reduce:
            logging.info(f'{nclusters.values.squeeze()} clusters found for lags {lags} in patternfile {inputpath}, proceed to aggregation once')
            # We are going to do a full aggregation once, to cover all lags and clustids
            # But slightly shrink the domain
            in_a_cluster = (~ds['clustid'].isnull()).any('lag') # 2D, False if not in a cluster
            in_a_cluster = in_a_cluster.stack({'latlon':['latitude','longitude']}) # 1D for boolean indexing
            in_a_cluster = in_a_cluster[in_a_cluster]
            subdomain = Region('subdomain', float(in_a_cluster.latitude.max()), float(in_a_cluster.longitude.min()), float(in_a_cluster.latitude.min()), float(in_a_cluster.longitude.max())) 
            ta = TimeAggregator(datapath = anompath, share_input = True, reduce_input = False, region = subdomain) # We are going to do a full aggregation once, to cover all lags and clustids
            mean = ta.compute(nprocs = NPROC, ndayagg = int(timeagg), method = 'mean', firstday = firstday, rolling = True)
            del ta
            mean = mean[np.logical_or(mean.time.dt.season == 'MAM',mean.time.dt.season == 'JJA'),...] # Throw away some values to reduce memoty cost of grouping but still keeping the ability to lag into previous season.
            ds = ds.reindex_like(mean) # Since the time aggregated version has the potential to be a spatial subset
            ds[mean.name] = mean # Should not copy the data

            for lag in lags:
                ncluster = int(nclusters.sel(lag = lag))
                if ncluster > 0:
                    logging.debug(f'starting clustersubset, n = {ncluster} for lag {lag}')
                    gr = ds.sel(lag = lag).groupby('clustid') # Nan clustids are discarded in grouping
                    for clustid, subset in gr:
                        pattern = subset['correlation'].expand_dims({'lag':1}, axis = 0) # Otherwise it does not fit into the spatcov function
                        spatcov = spatcov_multilag(pattern, subset[mean.name], laglist = [lag]) # Not really mutlilag. But then the pattern is also unique to the lag
                        spatcov = spatcov.squeeze().drop('lag').to_dataframe()
                        spatcov.columns = pd.MultiIndex.from_tuples([(variable,int(timeagg),int(lag),int(lag)+int(timeagg),int(clustid),'spatcov')], names = ['variable','timeagg','lag','separation','clustid','metric'])
                        outcomes.append(spatcov)
                        clustmean = mean_singlelag(precursor = subset[mean.name], lag = int(lag)).to_dataframe()
                        clustmean.columns = pd.MultiIndex.from_tuples([(variable,int(timeagg),int(lag),int(lag)+int(timeagg),int(clustid),'mean')], names = ['variable','timeagg','lag','separation','clustid','metric'])
                        outcomes.append(clustmean)
                else:
                    logging.debug(f'no clusters at lag {lag}')
            del mean, ds, gr
        else: # In this case the loop is organized differently. Sub-domain/aggregation per lag per clustid
            logging.info(f'{nclusters.values.squeeze()} clusters found for lags {lags} in big patternfile {inputpath}, proceed to aggregation per lag and clustid')
            for lag in lags:
                ncluster = int(nclusters.sel(lag = lag))
                if ncluster > 0:
                    for clustid in range(ncluster):
                        in_this_cluster = ds['clustid'].sel(lag = lag) == clustid
                        in_this_cluster = in_this_cluster.stack({'latlon':['latitude','longitude']})
                        in_this_cluster = in_this_cluster[in_this_cluster] 
                        subdomain = Region('subdomain', float(in_this_cluster.latitude.max()), float(in_this_cluster.longitude.min()), float(in_this_cluster.latitude.min()), float(in_this_cluster.longitude.max())) 
                        logging.debug(f'established subdomain for cluster {clustid} for lag {lag}')
                        ta = TimeAggregator(datapath = anompath, share_input = True, reduce_input = False, reduce_dtype = True, region = subdomain) # We are going to do a full aggregation once, to cover all lags and clustids
                        mean = ta.compute(nprocs = NPROC, ndayagg = int(timeagg), method = 'mean', firstday = firstday, rolling = True)
                        del ta
                        mean = mean[np.logical_or(mean.time.dt.season == 'MAM',mean.time.dt.season == 'JJA'),...] # Throw away some values to reduce memoty cost of grouping but still keeping the ability to lag into previous season.
                        tempset = ds.sel(lag = lag).reindex_like(mean) # Since the time aggregated version has the potential to be a spatial subset
                        tempset[mean.name] = mean
                        tempset = tempset.stack({'latlon':['latitude','longitude']})
                        indexer = tempset['clustid'] == clustid
                        spatcov = spatcov_multilag(tempset['correlation'][indexer].expand_dims({'lag':1}, axis = 0), tempset[mean.name][...,indexer], laglist = [lag]) # Not really mutlilag. But then the pattern is also unique to the lag
                        spatcov = spatcov.squeeze().drop('lag').to_dataframe()
                        spatcov.columns = pd.MultiIndex.from_tuples([(variable,int(timeagg),int(lag),int(lag)+int(timeagg),int(clustid),'spatcov')], names = ['variable','timeagg','lag','separation','clustid','metric'])
                        outcomes.append(spatcov)
                        clustmean = mean_singlelag(precursor = tempset[mean.name][...,indexer], lag = int(lag)).to_dataframe()
                        clustmean.columns = pd.MultiIndex.from_tuples([(variable,int(timeagg),int(lag),int(lag)+int(timeagg),int(clustid),'mean')], names = ['variable','timeagg','lag','separation','clustid','metric'])
                        outcomes.append(clustmean)
    else:
        logging.info(f'no clusters found in for all lags: {lags} in patternfile {inputpath}')
    logging.info('on to next variable/timeagg')

final = pd.concat(outcomes, axis = 1, join = 'outer')
outpath = OUTDIR / '.'.join(['precursor',do,'multiagg','parquet']) 
pq.write_table(pa.Table.from_pandas(final), outpath)
