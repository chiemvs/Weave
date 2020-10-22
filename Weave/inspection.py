"""
Structure for opening and wrangling the output
of permutation importance and shapley model interpretation
And to prepare for visualization
"""
import logging
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

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

    def load_data(self, X_too = False):
        """
        will automatically load the X values if shap 
        Path discovery according: basepath/respagg/separation/*.parquet
        conversion of index datatypes (string to integer)
        """
        def read_singlefile(fullpath):
            data = pd.read_parquet(fullpath) 
            if not hasattr(self, 'is_shap'):
                self.is_shap = 'expected_value' in data.columns
            if self.is_shap:
                expected = data['expected_value'].copy()
                data.drop(columns = ('expected_value',slice(None)), inplace = True) 
                data = data.T # Transposing because otherwise multiple separations (files) cannot be concatenated by row (with separation in the columns). Fold will not be present in the zeroth axis (in contrast to permimp)
            for index_name in self.integer_indices:
                try:
                    level = data.index.names.index(index_name)
                    data.index.set_levels(data.index.get_level_values(level).astype(int), level = level, inplace = True)
                    print(data.index)
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

        self.df = pd.concat(results, axis = 0, keys = pd.MultiIndex.from_tuples(keys, names = ['respagg','separation2']))
        # Remove double separation?



if __name__ == '__main__':
    perm = ImportanceData(Path('/scistor/ivm/jsn295/importance_spatcov'), 7, -31) 
    perm.load_data()
    shap = ImportanceData(Path('/scistor/ivm/jsn295/shaptest_negative'), [7,11], -31) 
    shap.load_data()
