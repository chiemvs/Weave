"""
Structure for opening and wrangling the output
of permutation importance and shapley model interpretation
And to prepare for visualization
"""
import os
import logging
import warnings
import itertools
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Union, List, Tuple

try:
    import shap
except ImportError:
    pass

try:
    import cartopy.crs as ccrs
except ImportError:
    pass

from .processing import TimeAggregator
from .models import map_foldindex_to_groupedorder, get_validation_fold_time, crossvalidate, HybridExceedenceModel, fit_predict
from .utils import collapse_restore_multiindex, Region, get_nhplus

def scale_high_to_high(series: pd.Series, fill_na: bool = False):
    """
    Min max scaling with the highest becoming one, and the lowest zero
    Na's can be filled with zero
    """
    series = (series - series.min())/(series.max() - series.min())
    if fill_na:
        series = series.fillna(0.0)
    return series

def scale_high_to_low(series: pd.Series, fill_na: bool = False):
    """
    Min max scaling with the lowest becoming one, and the highest zero
    Na's can be filled with zero
    """
    series = (series - series.max())/(series.min() - series.max())
    if fill_na:
        series = series.fillna(0.0)
    return series

class ImportanceData(object):

    def __init__(self, basepath: Path, respagg: Union[int, list], separation:Union[int, list], quantile: float = None, model = None) -> None:
        """
        Reading from a directory with importance dataframes. Which ones is determined by the respagg separation combinations
        Also possible to supply the threshold for which things were computed.
        Plus the initialized model type used (optional, but useful for the base rate of 
        """
        self.integer_indices = ['respagg','lag','separation','fold','clustid','timeagg']
        self.basepath = basepath
        try:
            self.respagg = tuple(respagg)
        except TypeError:
            self.respagg = (respagg,)
        try:
            self.separation = tuple(separation)
        except TypeError:
            self.separation = (separation,)

        self.quantile = quantile
        self.model = model
        if not self.model is None:
            assert not isinstance(self.model, type), 'When supplying a model it needs to be initialized already'

        if isinstance(self.model, HybridExceedenceModel):
            logging.debug('ImportanceData registered that data comes from hybrid model. Assuming that trended y is applicable')
            self.yname = 'response.multiagg.trended.parquet' 
        else:
            self.yname = 'response.multiagg.detrended.parquet' 

    def load_data(self, X_too = False, y_too = False, inputpath: Path = None):
        """
        will automatically load the X values if shap 
        Path discovery according: basepath/respagg/separation/*.parquet
        Path discovery for X and y is inputpath / precursor.multiagg.parquet and inputpath / response.multiagg.(de)trended.parquet
        If you wish a custom y either supply .y attribute directly or operate on it
        conversion of index datatypes (string to integer) but not needed for newest way of storing
        """
        def read_singlefile(fullpath):
            data = pd.read_parquet(fullpath) 
            if not hasattr(self, 'is_shap'):
                self.is_shap = 'expected_value' in data.index.get_level_values('variable')
            if self.is_shap:
                expected = data.loc[data.index.get_loc_level('expected_value','variable')[0],:].copy() # Copied as a Dataframe with folds as the index, time still in columns
                expected.index = expected.index.get_level_values('fold')
                data = data.drop('expected_value',level = 'variable', axis = 0)
                data.index = data.index.remove_unused_levels() # Necessary because otherwise the potential '' when expected value did not have dummy values for other levels will mess up string conversion to int, but not even neccesary when shaps were saved correctly
            for index_name in self.integer_indices:
                try:
                    level = data.index.names.index(index_name)
                    if not data.index.levels[level].dtype == int:
                        data.index.set_levels(data.index.levels[level].astype(int), level = level, inplace = True)
                except ValueError: # Name is not in the index
                    pass
            if self.is_shap:
                return data, expected
            else:
                return data, None
            
        combs = itertools.product(self.respagg,self.separation)
        keys = list(combs)
        results = []
        expected_values = [] # Only filled if self.is_shap
        for respagg, separation in keys:
            subdir = self.basepath / str(respagg) / str(separation)
            if subdir.exists():
                dircontent = list(self.basepath.glob(f'{respagg}/{separation}/*.parquet'))
                assert len(dircontent) == 1, f'None or multiple parquetfiles were found in {dircontent}'
                frame, expected = read_singlefile(dircontent[0]) 
                results.append(frame)
                expected_values.append(expected)
            else:
                logging.debug(f'{subdir} does not exist, skipping respagg {respagg} and separation {separation}')
                results.append(None)

        self.df = pd.concat(results, axis = 0, keys = keys, join = 'inner') # Adds two levels to the multiindex. Inner join could mess up the fold - timeslice mapping for shap, if this is variable.
        self.df.index.names = ['respagg','separation'] + self.df.index.names[2:]
        if self.df.index.names.count('separation') > 1: # Remove first separation level if it is in there twice
            self.df.index = self.df.index.droplevel(level = self.df.index.names.index('separation'))
        if X_too or self.is_shap:
            assert not inputpath is None, 'For X_too (automatic with shap) please supply an inputpath'
            X_path = list(inputpath.glob('precursor.multiagg.parquet'))[0]
            X = pd.read_parquet(X_path) 
            X = X.iloc[X.index.get_level_values('time').month.map(lambda m: m in [6,7,8]),:] # Seasonal subset starting at 1981 if correctly saved, for a correct grouped CV split when applied
            if 'fold' in X.columns.names:
                self.n_folds = len(X.columns.get_level_values('fold').unique())  
                self.lookup = map_foldindex_to_groupedorder(X, n_folds = self.n_folds, return_foldorder = True) # For later use in loading maps
            self.X = X.T # Transposing to match the storage of perm imp and shap
        if self.is_shap:
            # Prepare the expected values and load the X data
            self.expvals = pd.concat(expected_values, keys = keys, axis = 0) # concatenation as columns, not present because they were series
            self.expvals.index.names = ['respagg','separation'] + self.expvals.index.names[2:] 
            self.expvals.index = self.expvals.index.reorder_levels(['respagg'] + self.expvals.index.names[2:] + ['separation']) # Make sure that separation comes last, to match the possible removel of the first separation level when double
        if y_too:
            y_path = list(inputpath.glob(self.yname))[0]
            self.y = pd.read_parquet(y_path).T # summer only 
            # We drop the variable (t2m-anom) and clustid labels, as these are meaningless for the matching to importances. And we need to change timeagg to respagg 
            self.y.index = pd.Index(self.y.index.get_level_values('timeagg'), name = 'respagg') 

    def get_matching(self, toreindex: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
        """
        Uses values from toreindex frame according the index of the target
        When for instance the selection index or columns still have 'fold' in them, this won't work, because it is not present in X. This is also the mechanism with which you slice the X's to summer and spatcov values only
        absent levels are therefore dropped
        """
        not_present_in_index = [target.index.names.index(name) for name in target.index.names if not name in toreindex.index.names] # Level numbers
        not_present_in_columns = [target.columns.names.index(name) for name in target.columns.names if not name in toreindex.columns.names] # Level numbers
        try:
            target_index = target.index.droplevel(not_present_in_index) 
        except ValueError: # Cannot drop all levels, basically there is no index to compare against
            target_index = None
        try:
            target_columns = target.columns.droplevel(not_present_in_columns)
        except ValueError:
            target_columns = None

        return toreindex.reindex(index = target_index, columns = target_columns)
    
    def get_matching_X(self, selection: pd.DataFrame) -> pd.DataFrame:
        return self.get_matching(getattr(self,'X'), selection)

    def get_matching_y(self, selection: pd.DataFrame) -> pd.DataFrame:
        return self.get_matching(getattr(self,'y'), selection)

    def scale_within(self, what: Union[str,list] = ['respagg','separation','fold'], axis: int = 0, fill_na: bool = False) -> None:
        """
        Used to equalize the values (usually permutation importances, or a globalized shapley value: avg(|shap|)) to a single scale [0-1] within predictive situations that should be comparable over multiple_fits  
        The function makes groups and within each applies a scaling function to each series on the other axis 
        default function is high_to_high but if 'rank' is in the name then high_to_low
        """
        if self.is_shap:
            warnings.warn('Are you sure? Shapley is on the probability scale. So probably no need to scale those')
        def scale_func(series):
            if 'rank' in series.name:
                return(scale_high_to_low(series, fill_na = fill_na))
            else:
                return(scale_high_to_high(series, fill_na = fill_na))

        self.df = self.df.groupby(what, axis = axis).apply(lambda df: df.apply(scale_func, axis = axis))

    def reduce_over(self, what: Union[str, list] = 'fold', axis: int = 0, how: str = 'mean') -> None:
        """
        Take the mean/min/max over all values of a certain index/column label
        Values should be comparable over that label because it will disappear (so permimp scaled first)
        This means basically a grouping by unique combination except the label
        """
        if axis == 0:
            names = self.df.index.names.copy()
        elif axis == 1:
            names = self.df.columns.names.copy()
        else:
            raise ValueError(f'axis {axis} can only be 0 or 1')
        if not isinstance(what, list):
            what = [what]
        for do_not_keep in what:
            names.remove(do_not_keep)
        grouped = self.df.groupby(names, axis = axis)
        method = getattr(grouped, how)
        self.df = method()

    def global_shap(self) -> None:
        """
        Will aggregate the samples axis (time/fold) into a global shap importance, similar to global permutation importance
        """
        assert self.is_shap, 'dataframe needs to contain shap values'
        self.df = self.df.abs().mean(axis = 1) # Reduces it into a series.
        self.df.name = 'avgabsshap' 
        self.df = self.df.to_frame()

    def get_baserate(self, when: pd.Index, respagg: Union[pd.Index,int] = None, n_folds: int = 5) -> Union[float,pd.Series,pd.DataFrame]:
        """
        For the case of binary classification the base probability can be asked
        if when has length one a float is returned, otherwise a series with the same index
        The baserate is useful for e.g. shap plots to visualize probabilistic deviation from the base expectation
        This base model (strength of climatic trend) depends on the respagg (and quantile, but quantile is given at initialization)
        but of respagg we can have multiple so therefore a pd.DataFrame is returned
        """
        assert isinstance(when, pd.core.indexes.datetimes.DatetimeIndex), 'Datetimeindex needed, Multiindex with a fold level is taken care of when fitting'
        if not isinstance(self.model,HybridExceedenceModel): # In this case the y data is not trended
            if len(when) == 1:
                return self.quantile
            else:
                return pd.Series(data = np.full(shape = (len(when),), fill_value = self.quantile), index = when, name = 'baserate')
        else:
            if not self.model.base_only:
                self.model.base_only = True
                warnings.warn(f'model {self.model} set baseonly to return the baserates, in a way that is compatible with non-cv base model fitting')
            # Should it be connected to the varying folds? In a sense yes... that is what the model saw. 
            # Not separation dependent
            tempX = self.X.T # Temporary to row-indexed time. X values themselves do not matter. Only the index is used for the base rate model
            tempy = self.get_exceedences(when =  tempX.index) # Temporary y with row indexed time, full index for fitting
            returns = []
            for respagg_iterator in self.respagg:
                returns.append(fit_predict(self.model, X_in = tempX, y_in = tempy.loc[:,respagg_iterator], n_folds = n_folds))
            full = pd.concat(returns, axis = 1, keys = pd.Index(self.respagg, name = 'respagg')) # Full now has a multiindex as row, respagg ints in the column
            full.index = full.index.droplevel('fold')
            if respagg is None:
                return full.loc[when,self.respagg]
            elif isinstance(respagg, int):
                return full.loc[when,[respagg]] # List(respagg) is here to make sure we are returning a DataFrame
            else:
                return full.loc[when,respagg]

    def get_exceedences(self, when: pd.Index, respagg: int = None) -> Union[pd.Series,pd.DataFrame]:
        """
        To get the y in terms of the binary exceedences. Trended or not, 
        which depends on the loaded y path and the model supplied upon initialization
        Series if a single respagg is requested, otherwise pd.DataFrame
        """
        assert isinstance(when, pd.core.indexes.datetimes.DatetimeIndex), 'Datetimeindex needed, Multiindex with a fold level is taken care of when fitting'
        thresholds = self.y.quantile(self.quantile, axis = 1) # over axis 1 means over the whole timeseries
        binaries = pd.DataFrame(self.y.values > thresholds.values[:,np.newaxis], index = thresholds.index, columns = self.y.columns)
        return binaries.loc[self.respagg if respagg is None else respagg, when].T

    def fit_model(self, respagg: int, separation: int, n_folds: int = 5) -> pd.Series:
        """
        Fits the supplied model for one forecasting situation (defined by respagg and separation)
        with x and y that are attached to this object
        """
        if self.model.base_only:
            self.model.base_only = False
            warnings.warn(f'model {self.model} set False baseonly')
        tempX = self.get_matching_X(self.df.loc[np.logical_and(self.df.index.get_loc_level(respagg, 'respagg')[0],self.df.index.get_loc_level(separation,'separation')[0]),:]).T
        tempy = self.get_exceedences(when =  tempX.index, respagg = respagg).T # Temporary y with row indexed time, full index for fitting
        preds = fit_predict(self.model, X_in = tempX, y_in = tempy, n_folds = n_folds)
        preds.index = preds.index.droplevel('fold')
        return preds
    
    def get_predictions(self, when: pd.Index, respagg: int, separation: int):
        """
        Gets the predictions of the model for a certain timeslice
        Caches them for future use
        """
        if not hasattr(self, 'predictions'):
            self.predictions = pd.DataFrame(index = pd.MultiIndex.from_product([self.respagg,self.separation], names = ['respagg','separation']) ,columns = self.y.columns, dtype = 'float64') # Initialize empty

        if self.predictions.loc[(respagg,separation)].isnull().all():
            self.predictions.loc[(respagg,separation)] = self.fit_model(respagg = respagg, separation = separation) 

        return self.predictions.loc[(respagg,separation),when].T


class FacetMapResult(object):
    """
    The facetmapresults just is a consistent wrapping that is accepted by the mapplot function below
    The result is returned from a call to some mapinterface methods.
    columnkeys could be absent, in that case the listofarrays is not nested (for instance, just many shapley combinations mapped to clusters)
    However it can also be present: e.g. the different types of data with get_anom, in that case the listofarrays is nested row-major
    minimums and maximums are optional. Could be useful for plotting. These are not nested lists
    """
    def __init__(self, rowkeys, listofarrays, columnkeys = None, minimums: np.ndarray = None, maximums: np.ndarray = None):
        self.rowkeys = rowkeys
        self.columnkeys = columnkeys
        self.listofarrays = listofarrays
        self.minimums = minimums
        self.maximums = maximums

class MapInterface(object):
    """
    Class to couple (i.e. map) importance values to their geographical regions
    cluster arrays are searchable for variable/timeagg/lag combinations
    perhaps I want to cache these things in some internal datastructure?
    It provides a way to calculated and display the anomaly and correlation patterns
    related to important variables at a certain timestep
    """

    def __init__(self, corclustpath: Path, anompath: Path = Path(os.path.expanduser('~/processed/')), impdata: ImportanceData = None) -> None:
        """
        Default for anompath. Usually does not vary. corclustpath does (depending on clustering parameters and correlation measure)
        The original impdata object is optional but needs to be supplied when wanting to load validation fold specific patterns
        The translation table of impdata provides fieldfold numbers belonging to regular fold 
        """
        self.basepath = corclustpath
        self.anompath = anompath
        self.presentvars = []
        if not impdata is None:
            self.lookuptable = impdata.lookup

    def lookup(self, regularfold: int) -> int:
        """
        Called to lookup the fieldfold belonging to regular fold
        """
        if hasattr(self, 'lookuptable'):
            fieldfold = self.lookuptable.loc[regularfold, 'fieldfold']
            logging.debug(f'Fold {regularfold} leads to fieldfold {fieldfold}')
            return fieldfold 
        else:
            raise AttributeError('lookuptable is not present. Please initialize the MapInrerface with an ImportanceData object')

    def get_field(self, fold: int, variable: str, timeagg: int, separation: int, what: str = 'clustid') -> xr.DataArray:
        """
        Extracts and returns one (lat/lon) array from the dataset
        'what' denotes the array to get: possibilities are clustid and correlation
        """
        if not hasattr(self, variable):
            self.load_one_dataset(variable = variable, timeagg = timeagg)
        else:
            da = getattr(self, variable)
            if not timeagg in da.coords['timeagg']:
                self.load_one_dataset(variable = variable, timeagg = timeagg) # da needs to be reloaded
        # a problem arises with da[{'timeagg':timeagg,'separation':separation}] because separation (negative) is interpreted as positional arguments not a label value
        da = getattr(self, variable)[what].sel(fold = fold, timeagg = timeagg, separation = separation)
        return da

    def cache_everything(self) -> None:
        """
        Browses the basepath for correlation files. Loads them all as attributes
        """
        files = self.basepath.glob('*.corr.nc') 
        for filepath in files:
            variable, timeagg = filepath.parts[-1].split('.')[:2]
            self.load_one_dataset(variable = variable, timeagg = int(timeagg))

    def load_one_dataset(self, variable: str, timeagg: int) -> None:
        """
        Load one clustid file (multi-lag) into the internal structure 
        Replaces lag with the separation axis (this matches across timescales)
        """
        path = self.basepath / '.'.join([variable,str(timeagg),'corr','nc']) # e.g. clustertest_roll_spearman_varalpha/sst_nhplus.31.corr.nc
        ds = xr.open_dataset(path, decode_times = False) # Not using my own Reader (only handles one var)
        ds.coords.update({'separation': ds.coords['lag'].astype(int) + timeagg, 'timeagg':timeagg}) # create an alternative set of labels for the lag axis, these should match over the timescales
        ds = ds.swap_dims({'lag':'separation'}).drop('lag')[{'separation':slice(None,0,-1)}] # We want to drop non-matching lag. And deselect the simulataneous field (positive separation, zero lag)
        ds = ds.expand_dims('timeagg')
        if not hasattr(self, variable): # Then we put the array in place
            setattr(self, variable, ds)         
            self.presentvars.append(variable)
            logging.debug(f'{variable} was not yet present, placed as a new attribute for timeagg {timeagg}')
        else: # This might become quite inefficient when in the end every timeagg is needed. But it can be quite efficient when only one thing is needed
            setattr(self, variable, xr.concat([getattr(self,variable), ds], dim = 'timeagg')) 
            logging.debug(f'{variable} was present, concatenated timeagg {timeagg} to existing')

    def map_to_fields(self, imp: pd.Series, remove_unused: bool = True, unit: str = '') -> FacetMapResult:
        """
        Mapping a selection of properly indexed importance data to the clustids
        associated to the folds/variables/timeaggs/separations 
        it finds the unique groups. attemps mapping
        respagg does not play any role (not a property of input)
        returns a multiindex for the groups and a list with the filled clustid fields
        Because imp does not carry its own units (unlike xarray) possible to supply the new unit
        """
        assert 'clustid' in imp.index.names, 'mapping to fields by means of clustid, should be present in index, not be reduced over'
        assert isinstance(imp, pd.Series), 'Needs to be a series (preferably with meaningful name) otherwise indexing will fail'
        def map_to_field(impvals: pd.Series, fold: int, variable: str, timeagg: int, separation: int, remove_unused: bool) -> xr.DataArray:
            """ 
            does a single field. for all unique non-nan clustids in the field
            it calls a boolean comparison. Values not belonging are kept 
            the ones belonging to the clustid are set to the importance is found in impvals
            if not found (and remove_unused) then it is set to nan. So visually clusters can disappear
            The only way this repeated call can go wrong is if 
            the assigned importance has the exact value of a clustid integer called later
            This corner case can happen with shapvals of 0 and e.g. with the maximum scaled perimps (0 and 1)
            Therefore the Id's in the repeated call are taken outside the range of any impval
            by adding 10000
            """
            logging.debug(f'attempt field read and importance mapping for {variable}, fold {fold}, timeagg {timeagg}, separation {separation}')
            fieldfold = self.lookup(fold) 
            clustidmap = self.get_field(fold = fieldfold, variable = variable, timeagg = timeagg, separation = separation, what = 'clustid').copy() # Copy because replacing values. Don't want that to happen to our cached dataset
            clustidmap = clustidmap + 10000.0 # Dtype was already float32 because of the nans present
            ids_in_map = np.unique(clustidmap) # still nans possible
            ids_in_map = ids_in_map[~np.isnan(ids_in_map)]  
            ids_in_imp = impvals.index.get_level_values('clustid') + 10000.0 # conversion to float
            assert len(ids_in_imp) <= len(ids_in_map), 'importance series should not have more clustids than are contained in the corresponding map field'
            for clustid in ids_in_map:
                logging.debug(f'attempting mask for map clustid {clustid}')
                if clustid in ids_in_imp:
                    clustidmap = clustidmap.where(clustidmap != clustid, other = impvals.iloc[ids_in_imp == clustid].iloc[0]) # Boolean statement is where the values are maintaned. Other states where (if true) the value is taken from
                elif remove_unused:
                    clustidmap = clustidmap.where(clustidmap != clustid, other = np.nan)
                else:
                    pass
            clustidmap.name = impvals.name
            clustidmap.attrs.update({'units':unit})
            clustidmap.coords.update({'fold':fold}) # Read the fieldfold, now give it the fold numer of the fold it belongs to.
            return clustidmap 

        grouped = imp.groupby(['fold','variable','timeagg','separation']) # Discover what is in the imp series
        results = [] 
        keys = [] 
        minimums = [] # Just some statistics useful for plotting later
        maximums = []
        for key, series in grouped: # key tuple is composed of (variable, timeagg, separation)
            results.append(map_to_field(series, *key, remove_unused = remove_unused))
            keys.append(key) 
            minimums.append(float(series.min()))
            maximums.append(float(series.max()))
        keys = pd.MultiIndex.from_tuples(keys, names = ['fold','variable','timeagg','separation'])
        return FacetMapResult(rowkeys = keys, listofarrays = results, minimums = np.array(minimums), maximums = np.array(maximums))

    def get_anoms(self, imp: Union[pd.Series, pd.DataFrame], timestamp: pd.Timestamp = None, mask_with_clustid: bool = True, mask_strict: bool = False, correlation_too: bool = True) -> FacetMapResult:
        """
        Similar to map_to_fields this method accepts properly indexed importance values 
        [variable, timeagg, separation] should be in the index.
        For all present combinations it creates the anomaly field by aggregating the correct slice of daily anoms
        The slice is determined by the single! timestamp in the importance dataframe column
        Or by the pandas timestamp given manually. The timestamp is in terms of response time
        Has the possibility to also return the corresponding correlation pattern as a differnt array, both potentially masked with the clustids. The method will return those too.
        Masking can also be strict, then only the clustids in the imp index will be shown
        Returns a row major list of xarray objects [[var1_anom,var1_corr,var1_clust],[var2,...]]
        Indexed by row with the pd.MultiIndex, and by column with the pd.Index (length depends on whether correlation_too and clustid_too)
        """
        if timestamp is None:
            assert 'time' in imp.columns.names and len(imp.columns) == 1, 'determination of response timestamp (for which anoms are sought) requires time to be present as a single column, if timestamp is not given as an argument'
            timestamp = imp.columns.get_level_values('time')[0] 

        def get_anom(variable: str, timeagg: int, separation: int) -> xr.DataArray:
            """
            Searches the processed data directory. These data are still daily (not time aggregated at all) 
            I would need to aggregate only a slice. 
            The timestamp is at response time (left stamped regardless of respagg)
            So unfortunately we need some (hard)coded time axis math here.
            the timeaggregator does left-stamping, we need to load {timeagg} days, ending before / not including {separation} days before the timestamp  
            option to mask the anomalie field by the clustids
            """
            end = timestamp - pd.Timedelta(-separation + 1, unit = 'days') # +1 because for separation (gapsize) = 0 we want the adjacent, and -separation because negatively oriented
            start = end - pd.Timedelta(timeagg - 1, unit = 'days') # we need -1 extra days to obtain the disired timeagg because the end is inclusive
            input_range = pd.date_range(start, end, freq = 'D')
            datapath = list(self.anompath.glob(f'{variable}.anom.nc'))[0]
            data = xr.open_dataarray(datapath, decode_times = True).sel(time = input_range)
            logging.debug(f'Precursor anomaly field requested for {timestamp}. At separation {separation} and timeagg {timeagg} this amounts to daily data from {input_range}. Proceeding to aggregation')
            t = TimeAggregator(datapath, data = data, share_input = True) # datapath is actually ignored
            anom_field = t.compute(nprocs = 1, ndayagg = timeagg, rolling = False) # Only one process is needed (as we aggregate only one slice). Because the slice is the exact right length, rolling could be both True or False, both result in a time dim of length 1
            return anom_field.squeeze() # Get rid of the dimension of length 1 (timestamp becomes dimensionless coord

        grouped = imp.groupby(['fold','variable','timeagg','separation']) # Discover what is in the imp series A single fold is not neccessarily needed, the anomaly is independent of that
        results = [] 
        rowkeys = [] 
        columnkeys = ['anom', 'clustid'] 
        if correlation_too:
            columnkeys.insert(1,'correlation') # placed at first index, because like the anom (0) it will potentially be masked, and because like clustid (-1) it can be read with self.get_field
        for key, df in grouped: # key tuple is composed of (fold, variable, timeagg, separation)
            within_group_results = [None] * len(columnkeys)  # Will contain the anoms potentially more (forms one row in the return list) 
            within_group_results[0] = get_anom(*key[1:]) # Special function. For correlation and clustid we already have self.get_field. Fold is not neccesary here
            for extra in columnkeys[1:]:
                fieldfold = self.lookup(key[0]) # To load the correct cluster of correlation field we need to lookup the fieldfold and generate a new key
                newkey = (fieldfold,) + key[1:] 
                extramap = self.get_field(*newkey, what = extra)
                extramap.coords.update({'fold':key[0]}) # Read the fieldfold, now give it the fold numer of the fold it belongs to.
                within_group_results[columnkeys.index(extra)] = extramap 
            if mask_with_clustid:
                if mask_strict:
                    present_clustids = df.index.get_level_values('clustid').unique()
                    mask = xr.concat([within_group_results[-1] == i for i in present_clustids], dim = 'bools').any(dim = 'bools') # Where true values will be retained
                else:
                    mask = ~within_group_results[-1].isnull() # Where true values will be retained
                for index in range(len(within_group_results[:-1])): # Clustid is excluded from masking, cannot be masked with itself
                    within_group_results[index] = within_group_results[index].where(mask, other = np.nan) # Preserve where a cluster is found. np.nan otherwise

            results.append(within_group_results)
            rowkeys.append(key) 
        rowkeys = pd.MultiIndex.from_tuples(rowkeys, names = ['fold','variable','timeagg','separation'])
        columnkeys = pd.Index(columnkeys)
        return FacetMapResult(rowkeys = rowkeys, columnkeys = columnkeys, listofarrays = results)

    def fraction_significant(self, timeaggs: list = None, plot: bool = True, fold: int = None):
        """
        For the present variables, compute the amount of non-nan gridcells in the domain
        of each of variable and express it as a fraction.
        When correlations are loaded this fraction corresponds to the fraction of significant cells
        if fold is not given we compute an average over them
        if plot then plot fraction against separation with a panel per desired timeagg
        else the compted amounts and fractions are returned as a frame
        """
        results = []
        for variable in self.presentvars:
            array = getattr(self, variable)['correlation']
            n_signif = array.count(['latitude','longitude']).to_dataframe()
            n_signif['n'] = len(array.coords['latitude']) * len(array.coords['longitude'])
            n_signif['fraction'] = n_signif['correlation'] / n_signif['n']
            results.append(n_signif)
        results = pd.concat(results, keys = pd.Index(self.presentvars, name = 'variable'), axis = 0)
        if (fold is None) and ('fold' in results.index.names):
            groupers = list(results.index.names)
            groupers.remove('fold')
            results = results.groupby(groupers).mean()
        if not plot:
            return results
        else:
            timeaggs = results.index.get_level_values('timeagg').unique() if timeaggs is None else timeaggs
            fig, axes = plt.subplots(nrows = 1, ncols = len(timeaggs), sharex = True, sharey = True, figsize = (15,3.5), squeeze = False)
            for i, timeagg in enumerate(timeaggs):
                ax = axes[0,i]
                if not fold is None:
                    frame = results.loc[(slice(None),timeagg,slice(None),fold),'fraction'].unstack('variable')
                else:
                    frame = results.loc[(slice(None),timeagg,slice(None)),'fraction'].unstack('variable')
                ax.plot(frame.index.get_level_values('separation'), frame.values)
                ax.set_title(f'timeagg: {timeagg}, fold: {fold}')
                ax.set_xlabel('separation [days]')
                if i == 0:
                    ax.set_ylabel('fraction significant cells')
                if i == (len(timeaggs) - 1):
                    ax.legend(frame.columns.values)
            return fig, axes


def dotplot(df: pd.Series, fig = None, axes = None, custom_order: list = None, sizescaler = 50, alphascaler = 1, fix_alpha: float = None, nlegend_items = 4, color: str = 'tab:red'):
    """
    Takes a (scaled) importance df series (single variable)
    creates one panel per variable. (Custom order is possible, but variable names have to match exactly)
    the frame should have either a single separation (x axis becomes the respagg)
    or it has a single respagg (x axis becomes separation) 
    Both the size and alpha of the dots are scaled to importance (extra argument for alpha is scaler)
    if nlegend_items == 0 then no legend will be made
    """
    assert not 'clustid' in df.index.names, 'Clustid index should be reduced before plotting. Otherwise this results in stacked dots'
    y_var = 'timeagg'
    x_vars = ['separation','respagg']
    def get_val_len(name):
        vals = df.index.get_level_values(name).unique().values.tolist()
        return pd.Series([name, vals, len(vals)], index = ['name','values','len'])
    x_vars = pd.DataFrame([get_val_len(s) for s in x_vars], index = x_vars)
    assert (1 in x_vars['len'].values),'this type of dotplot requires an importance df with single respagg or single separation'
    unique_var = x_vars.loc[x_vars['len'].values == 1, 'name'][0]
    x_var = x_vars.loc[x_vars['len'].values != 1, 'name'][0] 
    imp_var = df.name # The name of the importance variable, like multipass rank or globalshap
    title = f"{unique_var} : {x_vars.loc[unique_var,'values'][0]}, {imp_var}"
    logging.debug(f'dotplot called, with {x_var} on the x-axis, now to determine the variables per panel') 
    if custom_order is None:
        custom_order = df.index.get_level_values('variable').unique().sort_values()
    
    max_per_row = 3 # Maximum amount of panels per row
    nrows = int(np.ceil(len(custom_order)/max_per_row))
    ncols = min(len(custom_order),max_per_row)
    if fig is None:
        fig, axes = plt.subplots(ncols = ncols, nrows = nrows, squeeze = False, figsize = (4*ncols,3.5 * nrows), sharex = True, sharey = True)
        presupplied = False
    else:
        presupplied = True
    plotdf = df.reset_index([y_var,x_var], name = imp_var) # We need the x and y values easily accesible
    global_min = plotdf[imp_var].max() # Needs updating to the selected variables
    global_max = plotdf[imp_var].min() 
    for i, variable in enumerate(custom_order):
        paneldf = plotdf.iloc[df.index.get_loc_level(key = variable, level = 'variable')[0],:] # Nice, now you don't have to know where in the levels 'variable' is to match the amount of required slice(None) in the slicing tuple
        global_min = min(global_min, paneldf[imp_var].min())
        global_max = max(global_max, paneldf[imp_var].max())
        rgba_colors = np.repeat(np.array(mcolors.to_rgba(color))[np.newaxis,:],len(paneldf), axis = 0) 
        if fix_alpha is None:
            rgba_colors[:,-1] = paneldf.loc[:,imp_var] * alphascaler # first three columns: rgb color values, 4th one: alpha
        else:
            rgba_colors[:,-1] = fix_alpha # first three columns: rgb color values, 4th one: alpha
        ax = axes[int(np.ceil((i+1)/max_per_row)) - 1,(i % max_per_row)]
        ax.scatter(x = paneldf[x_var], y = paneldf[y_var], s = paneldf[imp_var] * sizescaler, color = rgba_colors)
        if (i % max_per_row) == 0 and not presupplied:
            ax.set_ylabel('input timescale [days]')
        if i >= (len(custom_order) - max_per_row) and not presupplied:
            ax.set_xlabel(f'{x_var} [days]')
        if not presupplied:
            ax.set_title(f'var: {variable[:8]}')
    if not presupplied:
        fig.suptitle(title)
    # Setting up a custom legend (need to hand make the alpha/size of the labels) based on min/max of the selected variables
    if nlegend_items >= 1:
        imprange = np.round(np.linspace(global_min, global_max, num = nlegend_items), 3)
        items = [None] * len(imprange)
        for j, impval in enumerate(np.linspace(global_min, global_max, num = nlegend_items)):
            if fix_alpha is None:
                items[j] = plt.scatter([],[], s = impval * sizescaler, color = mcolors.to_rgb(color) + (impval*alphascaler,))
            else:
                items[j] = plt.scatter([],[], s = impval * sizescaler, color = mcolors.to_rgb(color) + (fix_alpha,))
        if presupplied:
            prev_legend = axes[-1,-1].get_legend()
            prev_imps = [text.get_text() for text in prev_legend.texts]
            axes[-1,-1].legend(prev_legend.legendHandles + items, prev_imps + [str(imp) for imp in imprange], loc = (1.05,0))
        else:
            axes[-1,-1].legend(items,imprange, loc = (1.05,0))
    return fig, axes


def mapplot(mapresult: FacetMapResult, wrap_per_row: int = 1, over_columns: str = None, match_scales: bool = False, fancyplot: bool = True, region: Region = get_nhplus(), projection = None):
    """
    No functionality to subset-select from the mapresult. This can be achieved by inputting a smaller dataframe/series to MapInterface methods
    For the case of absent columnkeys there is the option to plot vs a level in the rowindex
    otherwise the panels are distributed according wrap_per_row
    tuples are immutable which is the reason that FacetMapresult is not a named-tuple
    match_scales will put all maps on the same colorscale
    Fancyplot uses cartopy for the map, usually global, but can also be limited to a region
    Uses the Millweide projection as the default, but another initialized one can be supplied
    """
    if mapresult.columnkeys is None: # Then the option to check over_columns. Nevertheless we need to create a nesting ourselves
        assert not isinstance(mapresult.listofarrays[0], list), 'We should not be dealing with a nested list when columnkeys are absent'
        if over_columns is None:
            if wrap_per_row > 1:
                warnings.warn('Rowkeys are thinned, so you could be losing some information here if not also present in the xarray name or attributes')
            listofarrays = []
            rowkeys = mapresult.rowkeys[list(range(0,len(mapresult.listofarrays),wrap_per_row))]
            columnkeys = None
            for index in range(0,len(mapresult.listofarrays),wrap_per_row):
                listofarrays.append(mapresult.listofarrays[index:(index+wrap_per_row)]) # Slice is allowed to overshoot which is good
        else: # makes nested, but also set the columnkeys
            dummyframe = pd.Series(data = np.nan, index = mapresult.rowkeys).unstack(over_columns) # Unstack might generate combinations that are not there.
            rowkeys = dummyframe.index # Not overwritng we still need them to search, but also immutable because it is a tuple
            columnkeys = dummyframe.columns # Not overwritng, we still need them to search
            listofarrays = []
            for rowkey in rowkeys: # Nested loop (Building the new nested list row-major
                rowlist = [] 
                for columnkey in columnkeys:
                    originalkey = list(rowkey)
                    originalkey.insert(mapresult.rowkeys.names.index(over_columns),columnkey) # insert the column value at the right level
                    try:
                        originalindex = mapresult.rowkeys.get_loc(tuple(originalkey))  
                        rowlist.append(mapresult.listofarrays[originalindex])
                    except KeyError: # in this case it is an unstack combination that is not present
                        rowlist.append(None) # Leave the entry empty 

                listofarrays.append(rowlist)
        # Now we can reset
        mapresult.listofarrays = listofarrays
        mapresult.columnkeys = columnkeys
        mapresult.rowkeys = rowkeys

    nrows = len(mapresult.listofarrays)
    ncols = len(mapresult.listofarrays[0]) 
    if fancyplot:
        subplot_kw = dict(projection=ccrs.Mollweide() if (projection is None) else projection) # The desired projection
        array_crs = ccrs.PlateCarree() # what projection the data is in (regular lat lon grid)
    else:
        subplot_kw = dict()

    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, squeeze = False, sharex = True, sharey = True, figsize = (4*ncols,3.5 * nrows), subplot_kw = subplot_kw)
    for i, rowlist in enumerate(mapresult.listofarrays):
        for j, array in enumerate(rowlist):
            ax = axes[i,j]
            if not array is None:
                armin = array.min() if not match_scales else mapresult.minimums.min()
                armax = array.max() if not match_scales else mapresult.maximums.max()
                absmax = max(abs(armin),armax)
                if armin < 0 and armax > 0: # If positive and negative values are present then we want to center the colorscale
                    kwargs = dict(cmap = plt.get_cmap('RdBu_r'), vmin = -absmax, vmax = absmax)
                else:
                    kwargs = dict(vmin = armin, vmax = armax)
                if not fancyplot:
                    im = ax.pcolormesh(array,**kwargs) 
                else:
                    if not region is None:
                        extent = np.array(region[1:])[[1,3,2,0]]  # drop the name and reorder to x0,x1,y0,y1 
                        ax.set_extent(tuple(extent), crs = array_crs)
                    ax.coastlines()
                    # Define lats and lons at grid corners for pcolormesh + projection
                    lats = array.latitude.values # Interpreted as northwest corners (90 is in there)
                    lons = array.longitude.values # Interpreted as northwest corners (-180 is in there, 180 not)
                    lats = np.concatenate([lats[[0]] - np.diff(lats)[0], lats], axis = 0) # Adding the sourthern edge 
                    lons = np.concatenate([lons, lons[[-1]] + np.diff(lons)[0]], axis = 0)# Adding the eastern edge
                    #im = ax.contourf(array.longitude,array.latitude,array.values, transform = array_crs, **kwargs) #Cannot currently do pcolormesh needs lats and lons at gridcorners 
                    im = ax.pcolormesh(lons,lats,array.values, shading = 'flat', transform = array_crs, **kwargs) #Cannot currently do pcolormesh needs lats and lons at gridcorners 
                cbar = fig.colorbar(im, ax = ax)
                cbar.set_label(f'{array.name} [{array.units}]')
                ax.set_title(array._title_for_slice())
            if j == 0:
                ax.set_ylabel(f'{mapresult.rowkeys[i]}')
    return fig, axes

# Function for a bar plot / beeswarm plot for quick global or local importance without geographical reference and collapsed column names?
def barplot(impdf: Union[pd.Series,pd.DataFrame], n_most_important = 10, ignore_in_names = ['respagg','lag','metric']):
    """
    Creates a horizontal barplot with the (scaled) importance of all input variables (single separation/single timeagg) 
    if a pd.DataFrame with multiple columns is supplied then one bar plot per column (perhaps fold? or sample?) is made
    The function accepts only one forecasting occasion. so single respagg, separation
    The ignore_in_names argument is to get more compact plot labels (it are levels dropped from the multiindex of impdf)
    """
    assert len(impdf.index.get_level_values('separation').unique()) == 1 and len(impdf.index.get_level_values('respagg').unique()) == 1, 'barplot function can only accept (multicolumn) importances of only one forecasting problem'
    if isinstance(impdf, pd.Series):
        impdf = pd.DataFrame(impdf) # Just so we can loop over columns
    
    if impdf.columns.nlevels > 1: # Dropping levels on axis = 1 for easier plot titles
        impdf, oldcolnames, dtypes = collapse_restore_multiindex(impdf, axis = 1)
    if impdf.index.nlevels > 1: # Also dropping on axis = 0 
        impdf, oldrownames, dtypes = collapse_restore_multiindex(impdf, axis = 0, ignore_level = ignore_in_names)

    ncols = impdf.shape[-1]
    fig, axes = plt.subplots(nrows = 1, ncols = ncols, squeeze = False, sharex = True, figsize = (4*ncols,4))

    for i, col in enumerate(impdf.columns):
        sorted_by_col = impdf.loc[:,col].sort_values(ascending = False).iloc[(n_most_important - 1)::-1]
        ax = axes[0,i]
        ax.barh(range(n_most_important), width = sorted_by_col)
        ax.set_yticks(range(n_most_important))
        ax.set_yticklabels(sorted_by_col.index)
        ax.set_title(col)
    return fig, axes

def in_out_kept(base: Union[pd.Series,pd.DataFrame], other: Union[pd.Series,pd.DataFrame], n_most_important = 10, ignore_in_names=['respagg', 'lag']):
    """
    Function to display which predictors are new in the n_most important
    of other, compared to base. Also the ones that are kept, and the ones that went out
    Runs per fold. 
    """
    if isinstance(base, pd.Series):
        base = pd.DataFrame(base) # Just so we can loop over columns
    if isinstance(other, pd.Series):
        other = pd.DataFrame(other) # Just so we can loop over columns
    
    assert (base.shape == other.shape) and all(base.columns == other.columns) ,'Everything except the values of the datasets need to match, columns and index content, and column-order'

    if base.index.nlevels > 1: # Also dropping on axis = 0 
        base, oldrownames, dtypes = collapse_restore_multiindex(base, axis = 0, ignore_level = ignore_in_names)
        other, oldrownames, dtypes = collapse_restore_multiindex(other, axis = 0, ignore_level = ignore_in_names)
    # Initialize all at the maximum possible size, 
    # ordering of kept and out index is actually by importance in base
    # ordering of new index is actually by importance in new
    indf = pd.DataFrame(None, columns = base.columns, index = pd.RangeIndex(n_most_important, name = 'other_imp'))
    outdf = pd.DataFrame(None, columns = base.columns, index = pd.RangeIndex(n_most_important, name = 'base_imp'))
    keptdf = pd.DataFrame(None, columns = base.columns, index = pd.RangeIndex(n_most_important, name = 'base_imp'))
    for col in base.columns:
        base_by_col = base.loc[:,col].sort_values(ascending = False).iloc[:(n_most_important - 1)]
        other_by_col = other.loc[:,col].sort_values(ascending = False).iloc[:(n_most_important - 1)]
        for place in range(len(base_by_col)):
            key = base_by_col.index[place]
            if key in other_by_col.index:
                keptdf.loc[place,col] = key
            else:
                outdf.loc[place,col] = key
        for place in range(len(other_by_col)):
            key = other_by_col.index[place]
            if not key in base_by_col.index:
                indf.loc[place,col] = key

    return indf.dropna(how = 'all'), outdf.dropna(how = 'all'), keptdf.dropna(how = 'all')

def scatterplot(impdata: ImportanceData, selection: pd.DataFrame, alpha = 0.5, quantile: float = None, ignore_in_names = ['respagg','lag','metric']):
    """
    scatterplot of X vs y, X is chosen by the row of impdf 
    Two colors. One for the validation fold, one for the training, if X variables depend on the fold
    The exceedence quantile can be added as a horizontal line
    """
    X_full = impdata.get_matching_X(selection) # columns is the time axis
    y_summer = impdata.get_matching_y(selection) # columns is the time axis
    X_summer = X_full.loc[:,y_summer.columns]

    assert X_summer.shape[0] == 1, 'Currently only accepts a single selected X variable, change selection'
    fig, axes = plt.subplots(nrows = 1, ncols = 1, squeeze = False, sharey = True, figsize = (5,5))
    ax = axes[0,0]

    if hasattr(impdata, 'n_folds'):
        """
        Then we want to map which part of the X and y series is training and which part is not
        """
        selected_fold = int(X_summer.index.get_level_values('fold').values)
        validation_indexer = slice(impdata.lookup.loc[selected_fold,'valstart'],impdata.lookup.loc[selected_fold,'valend'])
        train_indexer = ~y_summer.columns.isin(y_summer.loc[:,validation_indexer].columns) # Inverting the slice
        ax.scatter(x = X_summer.loc[:,train_indexer].values.squeeze(), y = y_summer.loc[:,train_indexer].values.squeeze(), alpha = alpha, label = 'train')
        ax.scatter(x = X_summer.loc[:,validation_indexer].values.squeeze(), y = y_summer.loc[:,validation_indexer].values.squeeze(), alpha = alpha, label = 'validation')
        ax.set_title(f'validation: {validation_indexer.start.strftime("%Y-%m-%d")} - {validation_indexer.stop.strftime("%Y-%m-%d")}, imp: {float(np.round(selection.values,3))}')
    else:
        ax.scatter(x = X_summer.values.squeeze(), y = y_summer.values.squeeze(), alpha = alpha, label = 'full')

    # Some general annotation
    if not quantile is None:
        ax.hlines(y = y_summer.quantile(quantile, axis = 1), xmin = X_summer.min(axis = 1), xmax = X_summer.max(axis =1), label = f'q{quantile}') # y was loaded as trended or detrended. depening on Hybrid model presence at init of impdata
    ax.legend()
    ax.set_xlabel(f'{X_summer.index[0]}')
    ax.set_ylabel(f'response agg: {y_summer.index[0]}')

    return fig, axes

def data_for_shapplot(impdata: ImportanceData, selection: pd.DataFrame, base_too: bool = True, ignore_level = ['lag'], fit_base: bool = True) -> dict:
    """
    Function to prepare data for shap.force_plot and shap.summary_plot
    that selects X-vals accompanying the selection
    Collapses the column names (otherwise not accepted)
    Does a transpose for the numpy arrays to feed to the functions
    shap plot cannot handle an array of base values (e.g. coming from the hybrid method) so checks for that
    """
    assert isinstance(selection, pd.DataFrame), 'Need columns for this, if only a single timeslice e.g. select with .iloc[:,[100]]'
    respagg = selection.index.get_level_values('respagg').unique() # Potentially passing a pd.Index to get_baserate
    assert len(respagg) == 1, 'A shap plot should contain only the contributions for one response time aggregation' 

    if base_too:
        if not fit_base:
            base_value = np.unique(impdata.get_matching(impdata.expvals, selection))
        else:
            base_value = np.unique(impdata.get_baserate(when = selection.columns, respagg = respagg))

        assert len(base_value) == 1, f'the base_value should be unique, your selection potentially contains multiple folds, or if fit_base={fit_base} the base rate of the hybrid model could change with time'
        returndict = dict(base_value = float(base_value))
    else:
        returndict = dict()

    try:
        X_vals = impdata.get_matching_X(selection)
        returndict.update(features = X_vals.values.T)
    except: # In case e.g. clustid is aggregated out
        warnings.warn('Could not find X matching the selection, indexes might not correspond, proceed without feature values')
        returndict.update(features = None)
    if selection.index.nlevels > 1:
        selection, names, dtypes = collapse_restore_multiindex(selection, axis = 0, ignore_level = ignore_level, inplace = False)
    returndict.update(dict(shap_values = selection.values.T, feature_names = selection.index))

    return returndict 

def yplot(impdata: ImportanceData, resp_sep_combinations: List[Tuple[int]] = [(31,-15)], when: pd.Index = None, startdate: pd.Timestamp = None, enddate: pd.Timestamp = None):
    """
    Calls upon the importance data object to fit the desired models and get the desired exceedences
    Then fabricates a plot where each line forms the predicted probability.
    On top of that line dots denote a True binary resulting observation
    Possibility to do this for a custom time range, supplied by when, or from startdate and endate
    """
    if not when is None:
        customrange = when
    elif not ((startdate is None) or (enddate is None)):
        customrange = pd.date_range(startdate, enddate, freq = 'D', name = 'time')
    else:
        customrange = ImportanceData.y.columns # Just take range from the object

    customrange = customrange[customrange.map(lambda stamp: stamp.month in [6,7,8])] # Assert summer only
    
    fig, ax = plt.subplots()
    for respagg, separation in resp_sep_combinations:
        obsdata = impdata.get_exceedences(when = customrange, respagg = respagg) # pd.Series of binaries
        linedata = impdata.get_predictions(when = customrange, respagg = respagg, separation = separation) # Also pd.Series
        ax.plot(linedata, label = f'pred: {respagg} at {separation}')
        ax.scatter(x = obsdata.loc[obsdata].index, y = linedata.loc[obsdata], label = f'obs: {respagg} > {impdata.quantile} = True')

    ax.legend()
    ax.set_xlabel('time')
    ax.set_ylabel('probability')
    return fig, ax
    
