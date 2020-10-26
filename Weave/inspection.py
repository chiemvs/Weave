"""
Structure for opening and wrangling the output
of permutation importance and shapley model interpretation
And to prepare for visualization
"""
import logging
import warnings
import itertools
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union

from .inputoutput import Reader

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

    def __init__(self, basepath: Path, respagg: Union[int, list], separation:Union[int, list]) -> None:
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

    def load_data(self, X_too = False, y_too = False, inputpath: Path = None):
        """
        will automatically load the X values if shap 
        Path discovery according: basepath/respagg/separation/*.parquet
        Path discovery for X and y is inputpath / precursor.multiagg.parquet and inputpath / response.multiagg.trended.parquet
        If you wish a custom y either supply .y attribute directly or operate on it
        conversion of index datatypes (string to integer)
        """
        def read_singlefile(fullpath):
            data = pd.read_parquet(fullpath) 
            if not hasattr(self, 'is_shap'):
                self.is_shap = 'expected_value' in data.columns
            if self.is_shap:
                expected = data.loc[:,'expected_value'].copy() # Copied as series
                data = data.drop('expected_value',level = 0, axis = 1)
                data.columns = data.columns.remove_unused_levels() # Necessary because otherwise '' of the expected value at other levels will remain in it, and conversion to int will fail
                data = data.T # transposing because otherwise multiple separations (files) cannot be concatenated by row (with separation in the columns). Fold will not be present in the zeroth axis (in contrast to permimp)
            for index_name in self.integer_indices:
                try:
                    level = data.index.names.index(index_name)
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
            self.X = pd.read_parquet(X_path).T 
        if self.is_shap:
            # Prepare the expected values and load the X data
            self.expvals = pd.concat(expected_values, keys = keys, axis = 1).T # concatenation as columns, not present because they were series. Want to transpose to match shapley sample format
            self.expvals.index.names = ['respagg','separation'] 
        if y_too:
            y_path = list(inputpath.glob('response.multiagg.trended.parquet'))[0]
            self.y = pd.read_parquet(y_path).T # summer only 
            # We drop the variable (t2m-anom) and clustid labels, as these are meaningless for the matching to importances. And we need to change timeagg to respagg 
            self.y.index = pd.Index(self.y.index.get_level_values('timeagg'), name = 'respagg') 

    def get_matching(self, toreindex: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
        """
        Uses values from toreindex frame according the index of the target
        When for instance the selection index or columns still have 'fold' in them, this won't work, because it is not present in X. This is also the mechanism with which you slice the X's to summer and spatcov values only
        absent levels are therefore dropped
        """
        not_present_in_index = [name for name in target.index.names if not name in toreindex.index.names]
        not_present_in_columns = [name for name in target.columns.names if not name in toreindex.columns.names]
        return toreindex.reindex(index = target.index.droplevel(not_present_in_index), columns = target.columns.droplevel(not_present_in_columns))
    
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

class MapInterface(object):
    """
    Class to couple (i.e. map) importance values to their geographical regions
    cluster arrays are searchable for variable/timeagg/lag combinations
    perhaps I want to cache these things in some internal datastructure?
    also have an efficient mapping to xarray?
    """

    def __init__(self, corclustpath: Path) -> None:
        self.basepath = corclustpath

    def load_one_dataset(self, variable: str, timeagg: int) -> None:
        """
        Load one clustid file (multi-lag) into the internal structure 
        Replaces lag with the separation axis (this matches across timescales)
        """
        path = self.basepath / '.'.join([variable,str(timeagg),'corr','nc']) # e.g. clustertest_roll_spearman_varalpha/sst_nhplus.31.corr.nc
        ds = xr.open_dataset(path, decode_times = False) # Not using my own Reader (only handles one var)
        ds.coords.update({'separation': ds.coords['lag'].astype(int) + timeagg, 'timeagg':timeagg}) # create an alternative set of labels for the lag axis, these should match over the timescales
        ds = ds.swap_dims({'lag':'separation'}).drop('lag')[{'separation':slice(None,0,-1)}] # We want to drop non-matching lag. And deselect the simulataneous field (positive separation, zero lag)
        if not hasattr(self, variable): # Then we put the array in place
            setattr(self, variable, ds)         
        else: # This might become quite inefficient when in the end every timeagg is needed. But it can be quite efficient when only one thing is needed
            setattr(self, variable, xr.concat([getattr(self,variable), ds], dim = 'timeagg')) 


def dotplot(df: pd.Series, custom_order: list = None, sizescaler = 50, alphascaler = 1, nlegend_items = 4):
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
    fig, axes = plt.subplots(ncols = ncols, nrows = nrows, squeeze = False, figsize = (4*ncols,3.5 * nrows), sharex = True, sharey = True)
    plotdf = df.reset_index([y_var,x_var], name = imp_var) # We need the x and y values easily accesible
    global_min = plotdf[imp_var].max() # Needs updating to the selected variables
    global_max = plotdf[imp_var].min() 
    for i, variable in enumerate(custom_order):
        paneldf = plotdf.loc[df.index.get_loc_level(key = variable, level = 'variable')[0],:] # Nice, now you don't have to know where in the levels 'variable' is to match the amount of required slice(None) in the slicing tuple
        global_min = min(global_min, paneldf[imp_var].min())
        global_max = max(global_max, paneldf[imp_var].max())
        rgba_colors = np.zeros((len(paneldf),4),dtype = 'float64')
        rgba_colors[:,0] = 1 # Makes it red
        rgba_colors[:,-1] = paneldf.loc[:,imp_var] * alphascaler # first three columns: rgb color values, 4th one: alpha
        ax = axes[int(np.ceil((i+1)/max_per_row)) - 1,(i % max_per_row)]
        ax.scatter(x = paneldf[x_var], y = paneldf[y_var], s = paneldf[imp_var] * sizescaler, color = rgba_colors)
        if (i % max_per_row) == 0:
            ax.set_ylabel('important timeagg [days]')
        if i >= (len(custom_order) - max_per_row):
            ax.set_xlabel(f'{x_var} [days]')
        ax.set_title(f'var: {variable[:8]}')
    fig.suptitle(title)
    # Setting up a custom legend (need to hand make the alpha/size of the labels) based on min/max of the selected variables
    if nlegend_items >= 1:
        imprange = np.round(np.linspace(global_min, global_max, num = nlegend_items), 3)
        items = [None] * len(imprange)
        for j, impval in enumerate(np.linspace(global_min, global_max, num = nlegend_items)):
            items[j] = plt.scatter([],[], s = impval * sizescaler, color = [1,0,0,impval*alphascaler])
        axes[-1,0].legend(items,imprange)
    return fig, axes


def mapplot(df):
    pass
# Think up an interface to clusterids/correlation patterns from a basedir? opening in parallel, what takes the time?

