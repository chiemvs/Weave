#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import netCDF4 as nc
import xarray as xr
import logging
from pathlib import Path
from datetime import datetime
from .utils import Region

std_dimension_formats = {
    'time':{'encoding':{'datatype':'i4', 'fill_value':nc.default_fillvals['i4']},
        'attrs':{'units':'days since 1900-01-01 00:00:00', 'calendar':'standard', },'size':None},
    'latitude': {'encoding':{'datatype':'f4', 'fill_value':nc.default_fillvals['f4']},
        'attrs':{'units':'degrees_north'}},
    'longitude':{'encoding':{'datatype':'f4', 'fill_value':nc.default_fillvals['f4']},
        'attrs':{'units':'degrees_east'}},
    'doy':{'encoding':{'datatype':'i2', 'fill_value':nc.default_fillvals['i2']},
        'attrs':{'units':'dayofyear'}, 'size':366}}

variable_formats = pd.DataFrame(data = {
    'spacing':[0.25,0.25,0.25,0.25,0.25,0.1,0.1,0.1,0.1,0.1,0.1,0.25],
    'datatype':['i2','i2','i1','i2','i2','i2','i1','i1','i1','i1','i1','i1'],
    'scale_factor':[10,10,0.01,0.01,0.02,0.000005,None,0.01,0.01,0.01,0.01,None],
    }, index = pd.Index(['z500','z300','siconc','sst','t2m','transp','snowc','swvl1','swvl2','swvl3','swvl4','tcc'], name = 'varname'))
variable_formats['fill_value'] = variable_formats['datatype'].apply(lambda s: nc.default_fillvals[s])

class Writer(object):
    """
    Netcdf interface to create a dataset and write to it, using a custom encoding.
    """
    def __init__(self, datapath: Path, varname: str, groupname: str = None, ncvarname: str = None, region: Region = None):
        """
        Possibility to supply a hierarchical group for inside the netCDF4 file. Also a possibility to supply a different ncvarname
        than the base varname to which the encoding is linked.
        """
        self.datapath = datapath
        self.varname = varname
        self.groupname = groupname
        self.ncvarname = varname if ncvarname is None else ncvarname
        self.region = region
        self.formats = variable_formats.loc[varname]

    def create_dataset(self, dimensions: tuple = None, example: xr.DataArray = None) -> None:
        """
        Can create a set by supplying the desired standard dimensions (in order of the array axis),
        will then read the standard spacing for the variable
        Alternatively the xarray object can be supplied. Reads the spacing and dimensions from those
        The group created is named 'operation' and the variable created is named 'varname-operation'
        Lat lon dimensions need to be known upon creation, taken from the region box and the spacing.
        """
        if not self.datapath.exists():
            rootset = nc.Dataset(self.datapath, mode='w', format = 'NETCDF4')
            rootset.close()
            logging.info(f'Writer created {self.varname} data file')
        
        if dimensions is None:
            dimensions = example.dims

        with nc.Dataset(self.datapath, mode='a') as presentset:
            if isinstance(self.groupname, str):
                if not (self.groupname in presentset.groups):
                    presentset.createGroup(self.groupname)
                presentset = presentset[self.groupname] # Move one level down
            if not self.ncvarname in presentset.variables:
                for dim in dimensions:
                    try: # Try to extract from a possible supplied example, time axis will be unlimited
                        values = example.coords[dim].values
                        if dim == 'time': # Inferring netcdf writeable coordinates from requires extra effort. xarray gives np.datetime64[ns]
                            values = [pd.Timestamp(x).to_pydatetime() for x in values.tolist()]
                            values = nc.date2num(values, **std_dimension_formats[dim]['attrs'])
                    except AttributeError: # Create all from scratch, in this case time axis will also be unlimited but not filled
                        spacing = self.formats.loc['spacing']
                        if dim == 'latitude':
                            values = np.arange(self.region[3], self.region[1] + spacing, spacing)
                            values = np.round(values, decimals = 2)
                        elif dim == 'longitude':
                            values = np.arange(self.region[2], self.region[4] + spacing, spacing) 
                            values = np.round(values, decimals = 2)
                            values = values[~(values >= 180)] # If wrapping round the full globe (beyond 180) then don't count these cells. In CDS these cells should be given as -180 and above. So remove everything >= 180
                        elif dim == 'doy':
                            values = np.arange(1,std_dimension_formats[dim]['size'] + 1)
                        else:
                            values = None # In this case time dimension. Making sure that nothing will be filled
                    presentset.createDimension(dimname = dim, size = std_dimension_formats[dim]['size'] if dim == 'time' else len(values))
                    presentset.createVariable(varname = dim, dimensions = (dim,), **std_dimension_formats[dim]['encoding'])
                    presentset[dim].setncatts(std_dimension_formats[dim]['attrs'])
                    if values is not None:
                        presentset[dim][:] = values
                # Creating the actual variable, can have a special name that is different from the base varname
                presentset.createVariable(varname = self.ncvarname, dimensions = dimensions, **self.formats.loc[['datatype','fill_value']].to_dict())
                if not self.formats.loc[['scale_factor']].isnull().bool():
                    setattr(presentset[self.ncvarname], 'scale_factor', self.formats.loc['scale_factor']) #set the scaling
                logging.info(f'Writer created group: {self.groupname} and variable: {self.ncvarname}')

    def append_one_day(self, writedate: datetime, dayfield: np.ndarray, index: int, units: str = None):
        """
        Places a one day field at the desired index along the zeroth timeaxis. 
        And stamps it with the writedate
        Can be currently out of range because axis is unlimited
        Optionally adds units (in case it is the firstday)
        """
        with nc.Dataset(self.datapath, mode='a') as presentset:
            if isinstance(self.groupname, str):
                presentset = presentset[self.groupname] # Move one level down
            # Start appending along the time axis
            presentset[self.ncvarname][index,:,:] = dayfield
            presentset['time'][index] = nc.date2num(writedate, units = presentset['time'].units, calendar = presentset['time'].calendar)
            logging.debug(f'Writer has succesfully appended {date} to the netcdf')
            
            if not hasattr(presentset[self.ncvarname], 'units') and not (units is None):
                setattr(presentset[self.ncvarname], 'units',units)

    def write(self, array, blocksize: int = 1000, units: str = None):
        """
        Fully writes an array to the created netcdf dataset.
        Can be numpy maskedarray or an xarray, is written in parts, along the first axis
        Array needs to be masked to correctly write missing values
        So conversion to np.ma will be done when xarray is signalled 
        """
        if isinstance(array, xr.DataArray):
            def convert(subarray):
                return subarray.to_masked_array(copy = False)
        else:
            assert isinstance(array, np.ma.MaskedArray)
            def convert(subarray):
                return subarray

        with nc.Dataset(self.datapath, mode='a') as presentset:
            if isinstance(self.groupname, str):
                presentset = presentset[self.groupname] # Move one level down
            assert presentset[self.ncvarname].shape == array.shape
            starts = np.arange(0,array.shape[0],blocksize)
            for count, start in enumerate(starts):
                if count == len(starts) - 1: # Last start, write everything that remains
                    presentset[self.ncvarname][start:,...] = convert(array[start:,...])
                else:
                    presentset[self.ncvarname][start:(start + blocksize),...] = convert(array[start:(start + blocksize),...])
                logging.debug(f'Writer succesfully appended wrote {count} size {blocksize} to the netcdf')
            if not hasattr(presentset[self.ncvarname], 'units') and not (units is None):
                setattr(presentset[self.ncvarname], 'units',units)

        
