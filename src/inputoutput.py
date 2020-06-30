#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import netCDF4 as nc
import xarray as xr
import multiprocessing as mp
import logging
from pathlib import Path
from datetime import datetime
from .utils import Region, get_corresponding_ctype

std_dimension_formats = {
    'time':{'encoding':{'datatype':'i4', 'fill_value':nc.default_fillvals['i4']},
        'attrs':{'units':'days since 1900-01-01 00:00:00', 'calendar':'standard', },'size':None},
    'latitude': {'encoding':{'datatype':'f4', 'fill_value':nc.default_fillvals['f4']},
        'attrs':{'units':'degrees_north'}},
    'longitude':{'encoding':{'datatype':'f4', 'fill_value':nc.default_fillvals['f4']},
        'attrs':{'units':'degrees_east'}},
    'doy':{'encoding':{'datatype':'i2', 'fill_value':nc.default_fillvals['i2']},
        'attrs':{'units':'dayofyear'}, 'size':366},
    'nclusters':{'encoding':{'datatype':'i4', 'fill_value':nc.default_fillvals['i4']},
        'attrs':{'units':''}},
    'clustid':{'encoding':{'datatype':'i4', 'fill_value':nc.default_fillvals['i4']},
        'attrs':{'units':''}},
    'lag':{'encoding':{'datatype':'i1', 'fill_value':nc.default_fillvals['i1']},
        'attrs':{'units':'days'}},
    'quantile':{'encoding':{'datatype':'f4', 'fill_value':nc.default_fillvals['f4']},
        'attrs':{'units':''}}}

variable_formats = pd.DataFrame(data = {
    'spacing':[0.25,0.25,0.25,0.25,0.25,0.25,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.25,None,None],
    'datatype':['i2','i2','i2','i1','i2','i2','i2','i1','i1','i1','i1','i1','i1','i1','i4','i2'],
    'scale_factor':[10,10,0.02,0.01,0.01,0.02,0.000005,None,0.01,0.01,0.01,0.01,0.01,0.01,None,0.0001],
    }, index = pd.Index(['z500','z300','t850','siconc','sst','t2m','transp','snowc','swvl1','swvl2','swvl3','swvl4','swvl13','tcc','clustid','correlation'], name = 'varname'))
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

    def __repr__(self):
        attrs = ['datapath','ncvarname','formats']
        return '\n'.join([str(getattr(self,attr)) for attr in attrs if hasattr(self,attr)])

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
            logging.info(f'Writer created {self.varname} data file at {self.datapath}')
        
        if dimensions is None:
            dimensions = example.dims

        with nc.Dataset(self.datapath, mode='a') as presentset:
            if isinstance(self.groupname, str):
                if not (self.groupname in presentset.groups):
                    presentset.createGroup(self.groupname)
                presentset = presentset[self.groupname] # Move one level down
            if not self.ncvarname in presentset.variables:
                for dim in dimensions:
                    if not dim in presentset.variables:
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
            logging.debug(f'Writer has succesfully appended {writedate} to the netcdf')
            
            if not hasattr(presentset[self.ncvarname], 'units') and not (units is None):
                setattr(presentset[self.ncvarname], 'units',units)

    def write(self, array, blocksize: int = 1000, units: str = None, attrs: dict = {}):
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
                logging.debug(f'Writer succesfully wrote block {count} size {blocksize} to the netcdf')
            if not hasattr(presentset[self.ncvarname], 'units') and not (units is None):
                setattr(presentset[self.ncvarname], 'units',units)
            # Setting additional attributes (if not yet present)
            if attrs:
                self.write_attrs(attrs = attrs)

    def write_attrs(self, attrs: dict):
        with nc.Dataset(self.datapath, mode='a') as presentset:
            if isinstance(self.groupname, str):
                presentset = presentset[self.groupname] # Move one level down
        
            for key in attrs.keys():
                if not hasattr(presentset[self.ncvarname], key):
                    setattr(presentset[self.ncvarname], key, attrs[key])


class Reader(object):
    """
    Netcdf object to read a netcdf file with a desired precision
    xarray open_dataarray standard provides .data as float 32. Here I want to possibly go to float16.
    Also I want to have the option to flatten the spatial dimensions into one,
    discarding nans, for instance ocean cells in ERA5-Land datasets
    Then there is the option of reading only a sub-set region, but this excludes the flattening
    """
    def __init__(self, datapath: Path, ncvarname: str = None, groupname: str = None, blocksize: int = 1000, region: Region = None):
        """
        Possibility to supply a hierarchical group for inside the netCDF4 file. 
        ncvarname is not a neccesity if it is a dataarray and not a dataset, it is discovered through xarray otherwise
        """
        self.datapath = datapath
        self.groupname = groupname
        self.ncvarname = ncvarname
        self.blocksize = blocksize
        self.region = region

    def __repr__(self):
        attrs = ['datapath','ncvarname','name','dims','shape','dtype','coords']
        return '\n'.join([str(getattr(self,attr)) for attr in attrs if hasattr(self,attr)])

    def get_info(self, flatten: bool = False, decode_lag: bool = None):
        """
        Get additional information by using xarray
        If spatial dims are flattened, then coordinates and size change.
        """
        if not self.ncvarname is None:
            temp = xr.open_dataset(self.datapath, group = self.groupname, decode_times = decode_lag)[self.ncvarname]
        else:
            temp = xr.open_dataarray(self.datapath, group = self.groupname, decode_times = decode_lag)
        if not self.region is None:
            latsubset = temp.latitude.sel(latitude = slice(self.region[3],self.region[1])) # Done with xarray's sel method to get the buildin tolerance measures.
            lonsubset = temp.longitude.sel(longitude = slice(self.region[2],self.region[4]))
            self.latidx = np.nonzero(np.logical_and(temp.latitude.values >= latsubset.values.min(), temp.latitude.values <= latsubset.values.max()))[0]
            self.lonidx = np.nonzero(np.logical_and(temp.longitude.values >= lonsubset.values.min(), temp.longitude.values <= lonsubset.values.max()))[0]
            temp = temp.isel(latitude = self.latidx, longitude = self.lonidx)
        for attrname in ['dims','attrs','size','coords','encoding','name','shape']:
            setattr(self, attrname, getattr(temp, attrname))
        if (flatten and (len(self.dims) > 2)):
            assert self.region is None, 'region subset and flattening cannot be combined'
            block = temp[:self.blocksize,...].stack({'stacked':temp.dims[1:]}).dropna(dim = 'stacked', how = 'all')
            self.dims = block.dims
            bds = block.coords.to_dataset() # Gets the stacked coordinates in there, but zeroth timeaxis is incomplete due to the blocksize, therefore we take time from former
            self.coords = bds.assign_coords(**{self.dims[0]:temp.coords[self.dims[0]]}).coords
            self.size = len(self.coords[self.dims[0]]) * len(self.coords[self.dims[1]])
            self.shape = self.shape[0:1] + (len(self.coords[self.dims[1]]),)

        if self.ncvarname is None:
            self.ncvarname = self.name
        else:
            self.name = self.ncvarname

        logging.debug(f'Reader retrieved info from netcdf at {self.datapath}, flatten: {flatten}, region: {self.region}')
        
    def read_one_day(self, index: int) -> tuple:
        """
        Reads a one day field at the desired index along the zeroth timeaxis. 
        This field can be of type masked array
        And returns it with associated datestamp
        """
        with nc.Dataset(self.datapath, mode='r') as presentset:
            if isinstance(self.groupname, str):
                presentset = presentset[self.groupname] # Move one level down
            # Start appending along the time axis
            presentset[self.ncvarname].set_auto_maskandscale(True) # Default but nice to have explicit that masked arrays are returned (at least, when missing values are present)
            try:
                dayfield = presentset[self.ncvarname][index,...]
                datestamp = nc.num2date(presentset['time'][index], units = presentset['time'].units, calendar = presentset['time'].calendar)
                logging.debug(f'Reader has succesfully read {datestamp} from the netcdf')
                return (datestamp, dayfield)
            except IndexError: # Index out of range
                logging.debug(f'Reader could not find index {index} along zeroth axis')
                return (None, None)

    def read(self, into_shared: bool = True, flatten: bool = False, dtype: type = np.float32, decode_lag: bool = None):
        """
        Only the data reading is taken over from xarray. Coords, dims and encoding not, these become class attributes
        Either reads into a numpy array, and returns that
        or reads into a shared memory object, and returns that, other info becomes attributes
        Flattening is based on dropping completely empty time slices in the first block
        Only possible when 2+ dimensions
        """
        # Storing additional information. By using xarray
        self.get_info(flatten = flatten, decode_lag = decode_lag)
        self.dtype = dtype
        
        with nc.Dataset(self.datapath, mode='r') as presentset:
            if isinstance(self.groupname, str):
                presentset = presentset[self.groupname] # Move one level down

            presentset[self.ncvarname].set_auto_maskandscale(True) # Default but nice to have explicit that masked arrays are returned (at least, when missing values are present, see also comment below)
            if not self.region is None:
                non_nans_ind = tuple(slice(getattr(self,dim[:3] + 'idx').min(), getattr(self,dim[:3] + 'idx').max() + 1) for dim in self.dims[1:]) # for loop is to get the order right of the slices
            elif flatten and (len(presentset[self.ncvarname].shape) > 2):
                testblock = presentset[self.ncvarname][:self.blocksize,...] # Not sure that always masked arrays are returned (see comment below)
                try:
                    non_nans_ind = np.where(~testblock.mask.all(axis = 0))# Tuple with arrays of indices for the to-be-flattened spatial dimensions
                except AttributeError:
                    non_nans_ind = np.where(~np.isnan(testblock).all(axis = 0))
                assert len(self.coords['stacked']) == len(non_nans_ind[0]) # Make sure that the amount of retained coordinates in the same as the amount of retained cells
            else:
                non_nans_ind = (slice(None),) * len(self.dims[1:]) # (slice(None),slice(None)) for two spatial dims
            
            # Prepare the array to be returned (possibly in already flattened shape)
            if into_shared:
                sharedvalues = mp.RawArray(get_corresponding_ctype(dtype), size_or_initializer=self.size)
                values = np.frombuffer(sharedvalues, dtype=self.dtype).reshape(self.shape)
            else:
                values = np.full(shape = self.shape, fill_value = np.nan, dtype = dtype)
            starts = np.arange(0,self.shape[0], self.blocksize)
            for count, start in enumerate(starts):
                # Read a scaled block by slice
                if count == len(starts) - 1: # Last start, write everything that remains
                    blockslice = slice(start,None,None)
                else:
                    blockslice = slice(start,(start+self.blocksize),None)
                block = presentset[self.ncvarname][blockslice,...][(slice(None),) + non_nans_ind].astype(dtype)
                # This block is either a masked array is returned or a regular numpy array when no missing values are found (Unfortunately only in netCDF4 version of 1.4+ we can regulate the behaviour of always returning a masked array)
                # We only use the non_nans_ind on the numpy block, with a possible flattening as the result. netCDF4 cannot handle the tuples with arrays
                # Therefore we convert
                try:
                    block = block.filled(np.nan)
                except AttributeError:
                    pass
                values[blockslice,...] = block
                logging.debug(f'Reader succesfully read block {count} size {self.blocksize} from the netcdf')
            
        if into_shared:
            return sharedvalues
        else:
            return values
