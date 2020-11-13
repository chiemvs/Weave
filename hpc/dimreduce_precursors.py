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
if False: #not response_output.exists():
    logging.debug(f'no previously existing file found at {response_output}')
    response = xr.open_dataarray(ANOMDIR / 't2m_europe.anom.nc')
    clusterfield = xr.open_dataarray(CLUSTERDIR / 't2m-q095.nc').sel(nclusters = 15)
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
files = [ f for f in PATTERNDIR.glob('tcc_europe.1.corr.nc') if f.is_file()]
to_reduce = ['snowc_nhmin','siconc_nhmin'] # Variables with huge files and remote clusters. Are handled differently (not stacked, but aggregated per cluster per lag)
# first level loop is variable / timeagg combinations over files
# Each file has unique clustid shapes per lag, so the
# Second level of the loop is over the lags contained within the file
# The third level is then the clustids. 
# On that subset we call the timeaggregator and compute spatial covarianceand mean, this increases read access
# Internal to spatcov multlilag we have the multiple lags, contained within the file, but this is not used because domains do not overlap
outpath = OUTDIR / '.'.join(['precursor',do,'multiagg','parquet']) 
class disk_interface(object):
    """ Disk interface to write and to read if unique combination already present """
    def __init__(self, path: Path):
        self.path = path

    def check_if_present(self, combination: tuple, trim_fake_fold = False):
        """ combination is a column tuple. trim_fake_fold removes the zeroth level"""
        if not self.path.exists():
            return False
        else:
            present_cols = pq.ParquetFile(outpath).schema.names
            if trim_fake_fold:
                combination = combination[1:]
            as_string = str(tuple([str(item) for item in combination]))
            return as_string in present_cols

    def write_to_file(self, column: pd.DataFrame, trim_fake_fold = False):
        """ 
        You should check presence before writing, otherwise double entries I guess
        The column can actually consist of multiple columns for writing at once
        """
        if trim_fake_fold:
            column.columns = column.columns.droplevel('fold')
        if not self.path.exists(): # Write including the index
            pq.write_table(pa.Table.from_pandas(column, preserve_index = True), where = self.path)
        else:
            table = pq.read_table(outpath)
            to_append = pa.Table.from_pandas(column, preserve_index = False)
            for colindex in range(len(to_append.columns)):
                table = table.append_column(to_append.schema.field(colindex), to_append.column(colindex))
            pq.write_table(table, where = self.path)

output = disk_interface(outpath)

for inputpath in files:
    filename = inputpath.parts[-1]
    variable, timeagg = filename.split('.')[:2]
    anompath = list(ANOMDIR.glob(f'{variable}.anom*'))[0]
    ds = xr.open_dataset(inputpath, decode_times = False).drop_sel(lag = 0) # Remove the simultaneous lag = 0. Information cannot be used in the models
    # Perhaps just insert a fold dimension is there is None? Then loops can always be over folds and lags. Only upon writing the frame fold column is removed if there were no original folds
    if not 'fold' in ds.dims:
        ds = ds.expand_dims(dim = {'fold':[0.0]}, axis = 1) # (lag, fold, latitude, longitude)
        fakefold = True
    else:
        fakefold = False
    ds.coords['fold'] = ds.coords['fold'].astype(int)
    ds.coords['lag'] = ds.coords['lag'].astype(int)
    def find_nunique(arr: xr.DataArray) -> xr.DataArray:
        uniques = np.unique(arr)
        nunique = len(uniques[~np.isnan(uniques)])
        return xr.DataArray(nunique, dims = ('count',), coords = {'count':[arr.name]})
    stacked = ds['clustid'].stack({'st':['lag','fold']})
    nclusters = stacked.groupby('st').apply(find_nunique).unstack('st') # 2D grouping only possible by stacking https://github.com/pydata/xarray/issues/324 
    nclusters = nclusters.rename({'st_level_0':'lag','st_level_1':'fold'}) # groupby did not track the names
    # Construct the unique cluster ids that have to be present per fold and lag, such that indices can be generated and we know what is missing from the outfile
    temp = nclusters.to_dataframe(name = 'nclusters').iloc[:,0] # Need to have it as a series and drop nclusters = 0
    temp = temp[temp > 0]
    logging.info(f'{len(temp)} non-empty clustid fields found for {inputpath}')
    if not temp.empty:
        index_of_combinations = temp.groupby(['lag','fold']).apply(lambda x: pd.DataFrame(variable, columns = ['variable'], index = pd.MultiIndex.from_product([list(range(x.iloc[0])),['mean','spatcov']], names = ['clustid','metric'])))
        index_of_combinations['timeagg'] = int(timeagg) 
        index_of_combinations['separation'] = index_of_combinations.index.get_level_values('lag') + index_of_combinations['timeagg'] 
        index_of_combinations = index_of_combinations.set_index(['variable','timeagg','separation'], append = True).index
        index_of_combinations = index_of_combinations.reorder_levels(['fold','variable','timeagg','lag','separation','clustid', 'metric'])
        # Determine here whether the parquet outfile already contains all the combinations
        not_yet_present = [i for i, combination in enumerate(index_of_combinations) if not output.check_if_present(combination, trim_fake_fold = fakefold)] 
        todo = index_of_combinations[not_yet_present]
        logging.debug(f'extraction needs to happen for {todo.to_list()} as these clusterids were not yet found in {outpath}, fakefold = {fakefold}')

        if not variable in to_reduce:
            logging.info(f'variable not in to_reduce, proceeding to aggregation once, for all combinations')
            # We are going to do a full aggregation once, to cover all lags folds and clustids
            # But slightly shrink the domain
            in_a_cluster = (~ds['clustid'].isnull()).any(['lag','fold']) # 2D, False if not in a cluster
            in_a_cluster = in_a_cluster.stack({'latlon':['latitude','longitude']}) # 1D for boolean indexing
            in_a_cluster = in_a_cluster[in_a_cluster]
            subdomain = Region('subdomain', float(in_a_cluster.latitude.max()), float(in_a_cluster.longitude.min()), float(in_a_cluster.latitude.min()), float(in_a_cluster.longitude.max())) 
            ta = TimeAggregator(datapath = anompath, share_input = True, reduce_input = False, region = subdomain) # We are going to do a full aggregation once, to cover all lags and clustids
            mean = ta.compute(nprocs = NPROC, ndayagg = int(timeagg), method = 'mean', firstday = firstday, rolling = True)
            del ta
            mean = mean[np.logical_or(mean.time.dt.season == 'MAM',mean.time.dt.season == 'JJA'),...] # Throw away some values to reduce memoty cost of grouping but still keeping the ability to lag into previous season.
            ds = ds.reindex_like(mean) # Since the time aggregated version has the potential to be a spatial subset
            ds[mean.name] = mean # Should not copy the data

            for fullkey in todo: 
                fold,_,_,lag,_,clustid,metric = fullkey
                subset = ds.sel(lag = lag, fold = fold).stack({'latlon':['latitude','longitude']}) # 1D otherwise boolean indexing is not supported, unfortunately this stacking currently happens for each clustid and metric, but should be quite quick and is also what happens internally with groupby
                clustid_mask = subset['clustid'] == clustid # To discard other clustids and nan-field
                if metric == 'spatcov':
                    pattern = subset['correlation'][clustid_mask].expand_dims('lag', axis = 0) # Otherwise it does not fit into the spatcov function
                    spatcov = spatcov_multilag(pattern, subset[mean.name][:,clustid_mask], laglist = [lag]) # Not really mutlilag. But then the pattern is also unique to the lag
                    result = spatcov.squeeze().drop('lag').to_dataframe()
                elif metric == 'mean':
                    result = mean_singlelag(precursor = subset[mean.name][:,clustid_mask], lag = lag).to_dataframe()
                else:
                    raise ValueError('invalid value for metric, dimension reduction method unknown')
                result.columns = pd.MultiIndex.from_tuples([fullkey], names = todo.names)
                output.write_to_file(result, trim_fake_fold = fakefold)
            del mean, ds
        else: # In this case the loop is organized differently. Sub-domain/aggregation per lag per clustid
            pass
            # Loop on a reduced set of (lag, clustid) in todo. And then later search the remaining keys belonging to that pair, with something like loc
            todo.get_loc_level(lag,'lag') # Twice, perhaps np.logical_and?
            
#            logging.info('variable in to_reduce, proceeding to aggregation per lag and clustid')
#            for lag in lags:
#                ncluster = int(nclusters.sel(lag = lag))
#                if ncluster > 0:
#                    for clustid in range(ncluster):
#                        in_this_cluster = ds['clustid'].sel(lag = lag) == clustid
#                        in_this_cluster = in_this_cluster.stack({'latlon':['latitude','longitude']})
#                        in_this_cluster = in_this_cluster[in_this_cluster] 
#                        subdomain = Region('subdomain', float(in_this_cluster.latitude.max()), float(in_this_cluster.longitude.min()), float(in_this_cluster.latitude.min()), float(in_this_cluster.longitude.max())) 
#                        logging.debug(f'established subdomain for cluster {clustid} for lag {lag}')
#                        ta = TimeAggregator(datapath = anompath, share_input = True, reduce_input = False, reduce_dtype = True, region = subdomain) # We are going to do a full aggregation once, to cover all lags and clustids
#                        mean = ta.compute(nprocs = NPROC, ndayagg = int(timeagg), method = 'mean', firstday = firstday, rolling = True)
#                        del ta
#                        mean = mean[np.logical_or(mean.time.dt.season == 'MAM',mean.time.dt.season == 'JJA'),...] # Throw away some values to reduce memoty cost of grouping but still keeping the ability to lag into previous season.
#                        tempset = ds.sel(lag = lag).reindex_like(mean) # Since the time aggregated version has the potential to be a spatial subset
#                        tempset[mean.name] = mean
#                        tempset = tempset.stack({'latlon':['latitude','longitude']})
#                        indexer = tempset['clustid'] == clustid
#                        spatcov = spatcov_multilag(tempset['correlation'][indexer].expand_dims({'lag':1}, axis = 0), tempset[mean.name][...,indexer], laglist = [lag]) # Not really mutlilag. But then the pattern is also unique to the lag
#                        spatcov = spatcov.squeeze().drop('lag').to_dataframe()
#                        spatcov.columns = pd.MultiIndex.from_tuples([(variable,int(timeagg),int(lag),int(lag)+int(timeagg),int(clustid),'spatcov')], names = ['variable','timeagg','lag','separation','clustid','metric'])
#                        outcomes.append(spatcov)
#                        clustmean = mean_singlelag(precursor = tempset[mean.name][...,indexer], lag = int(lag)).to_dataframe()
#                        clustmean.columns = pd.MultiIndex.from_tuples([(variable,int(timeagg),int(lag),int(lag)+int(timeagg),int(clustid),'mean')], names = ['variable','timeagg','lag','separation','clustid','metric'])
#                        outcomes.append(clustmean)
#    else:
#        logging.info(f'no clusters found in for all lags: {lags} in patternfile {inputpath}')
#    logging.info('on to next variable/timeagg')
#
