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
from pathlib import Path
from typing import Union

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

# Think up an interface to clusterids/correlation patterns from a basedir? opening in parallel, what takes the time?

if __name__ == '__main__':
    perm = ImportanceData(Path('/scistor/ivm/jsn295/importance_spatcov_q08_nf5'), 7, -31) 
    perm.load_data()
    shap = ImportanceData(Path('/scistor/ivm/jsn295/shaptest_negative_train'), [7], [-1,-21,-31]) 
    shap.load_data(inputpath = Path('/scistor/ivm/jsn295/clusterpar3_roll_spearman_varalpha'), y_too = True)
    # Example of scaling the permutation importances as I did
    #perm.scale_within(fill_na = True) # Multirank score remains useless as always
    #perm.reduce_over()

    # Want e.g. to map global shap and perm imp dataframe selections to clustids
