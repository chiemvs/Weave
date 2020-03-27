#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERA5-Land 0.1 degree resolution, hourly. 4 * volumetric soil water content, 1 * transpiration, 1 * snow cover, 1* t2m
ERA5-Land will have about a 9 times larger data volume than a single level of ERA5 in the same domain.
Hourly is not needed so reworking to daily averages/sums. 
ERA5 0.25 degree resolution. Single levels. 1 * Sea Surface Temperatures, 1 * Sea Ice cover. Pressure levels: 1* Z500
"""
import netCDF4 as nc
import numpy as np
import pandas as pd
import cdsapi
import eccodes as ec
import logging
import multiprocessing as mp
import time
from pathlib import Path
from collections import OrderedDict
from typing import Tuple, List, Dict
from .inputoutput import Writer, variable_formats
from .utils import Region

# Construction that downloads raw files (in parallel)
# Supply a region object, variable name, raw-data-folder, 
class CDSDownloader(object):
    
    def __init__(self) -> None:
        """
        Translation table between ERA5 storage, formats in the CDS and my own variable names and storage format
        Evaporation from transformation has an ERA-Land issue and is stored under a different alias 
        """
        self.era_formats = pd.DataFrame(data = {
            'variable':['geopotential','geopotential','sea_ice_cover','sea_surface_temperature','2m_temperature', 'evaporation_from_bare_soil', 'snow_cover','volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4','total_cloud_cover',],
            'pressure_level':[500,300,None,None,None,None,None,None,None,None,None,None],
            'setname':['reanalysis-era5-pressure-levels','reanalysis-era5-pressure-levels','reanalysis-era5-single-levels','reanalysis-era5-single-levels','reanalysis-era5-single-levels','reanalysis-era5-land','reanalysis-era5-land','reanalysis-era5-land','reanalysis-era5-land','reanalysis-era5-land','reanalysis-era5-land','reanalysis-era5-single-levels',],
            'timevariable':['dataTime','dataTime','dataTime','dataTime','dataTime','endStep','forecastTime','dataTime','dataTime','dataTime','dataTime','dataTime',], # The relevant grib parameter in the files, depends on accumulated variable and ERA-Land or not
            }, index = pd.Index(['z500','z300','siconc','sst','t2m','transp','snowc','swvl1','swvl2','swvl3','swvl4','tcc'], name = 'varname'))
        self.era_formats = self.era_formats.join(variable_formats)
    
    def create_requests_and_populate_queue(self, downloaddates: pd.DatetimeIndex, request_kwds: dict) -> list:
        """
        For all variables but the transpiration we want to download the full 24 hrs of a single day
        the request_kwds should contain the varname and the region raw directory path
        """
        def create_request(date: pd.Timestamp, varname: str, region:Region, rawdir:Path) -> Tuple[pd.Timestamp,Tuple[str, dict, str]]:
            cdsname = self.era_formats.loc[varname,'setname']
            rawfilepath = rawdir / ('_'.join([varname, region[0],date.strftime('%Y-%m-%d')]) + '.grib')
            if varname in ['transp','snowc']: # For transp we want the accumulation at 00 the next day. For the other ERA-Land variable that is stored weirdly, we take the end-of-the day snapshot (+00 next day in the ERA-Land api convention, prev day + 24h in the grib encoding) 
                newdate = date + pd.DateOffset(1,'days')
                hours = '00:00'
            else:
                newdate = date
                hours = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']
            specification = {
                    'format':'grib',
                    'variable':self.era_formats.loc[varname,'variable'],
                    'year':newdate.strftime('%Y'),
                    'month':newdate.strftime('%m'),
                    'day':newdate.strftime('%d'),
                    'time':hours,
                    'area':[str(bound) for bound in region[1:]],
                    }
            if not self.era_formats.loc[varname,['pressure_level']].isnull().bool():
                specification.update({'pressure_level':str(int(self.era_formats.loc[varname,'pressure_level']))})
            if not 'land' in cdsname:
                specification.update({'product_type':'reanalysis'})

            return (date, (cdsname, specification, str(rawfilepath)))
        

        self.task_queue = mp.Queue()
        for date in downloaddates:
            req = create_request(date = date, **request_kwds) # returns tuples 
            self.task_queue.put(req)

    def start_queue(self, n_par_requests: int = 1) -> mp.Queue:
        """
        Returns the done queue such that the Organizer can see the downloads coming in
        """
        done_queue = mp.Queue()
        # Start the worker processes and put in their final STOP command
        for i in range(n_par_requests):
            p = mp.Process(target = self.execute_request, args=(self.task_queue, done_queue))
            p.start()
            logging.debug(f'created downloading process at {p.pid}')
            self.task_queue.put(('STOP','STOP'))

        return done_queue
    
    def execute_request(self, inqueue, outqueue) -> None:
        """
        Is called in a subprocess and creates its own Client each time
        When succesful will write a grib file and return a tuple with timestamp and filepath to a queue
        """
        while True:
            date, request = inqueue.get()
            if date == 'STOP':
                logging.debug('terminating this download subprocess')
                break
            else:
                logging.info(f'starting download: {request}')
                c = cdsapi.Client()
                c.retrieve(*request)
                outqueue.put((date,Path(request[-1])))
        
class PreProcessor(object):
    """
    Construction that reworks the raw hourly data to daily. Not parallel, because shared file access to the goal file is hard to do.
    For variables accumulated over 24 hrs only the last timestep needs to be read.
    """
    def __init__(self, operation: str, ncvarname: str, datapath: Path, encoding: pd.Series, region: Region) -> None:
        """
        Operation determines what is done on the grib file containing the timesteps of that day
        to form the single field that will be written for a day.
        Possibilities are ['mean','min','max','hhUTC'] of which hh is between '00' and '24'
        """
        self.operation = operation
        self.ncvarname = ncvarname
        self.datapath = datapath
        self.encoding = encoding # Dataframe defined in the downloader class, subset series for the variable.
        self.writer = Writer(datapath= self.datapath, varname = self.encoding.name, groupname = self.operation, ncvarname = self.ncvarname, region = region)

    def add_to_preprocessing_queue(self, unprocessed_dates: pd.DatetimeIndex, unprocessed_paths: List[Path]) -> None:
        """
        Do a sort before putting into the queue. This increases the chance of a nice monotonic time axis after sequential writing by self.preprocess
        """
        reordering = unprocessed_dates.values.argsort() # Reorder template according to the time index
        unprocessed_dates = unprocessed_dates[reordering]
        unprocessed_paths = np.array(unprocessed_paths)[reordering].tolist()
        
        for date, path in zip(unprocessed_dates, unprocessed_paths):
            self.task_queue.put((date,path)) # Put the info.
        
        logging.debug(f'Added rawfiles for {unprocessed_dates} to preprocess queue.')
    
    def start(self, consecutive: bool = True) -> None:
        """
        This is the function that starts the preprocess-childprocess and supplies the queue to it.
        Consecutive determines whether this child can only append consecutive days to the netcdf file. 
        Potentially leaving the queue full of requests when one intermediate raw day remains missing.
        """
        self.task_queue = mp.Queue()
        p = mp.Process(target = self.preprocess, args=(self.task_queue, consecutive))
        p.start()
        logging.info(f'Fired up the pre-process child process at {p.pid} with consecutive set to {consecutive}')
    
    def end(self):
        """
        Send termination command to child
        """
        self.task_queue.put(('STOP','STOP'))
        logging.info(f'Termination command send to pre-process child process')
    
    def preprocess(self, inqueue: mp.Queue, consecutive: bool) -> None:
        """
        Will read a request with rawfilepath and date from the queue. 
        If consecutive is set to true then it checks whether the requested date is consecutive
        to the content of the netcdf file and puts the request back if it is not.
        If set to False, the eventual netcdf time axis can have missing days and be non-monotonic.
        This function then decodes the raw grib file, extracts hourly fields, 
        does the operation (even if just a simple hhUTC timeslice)
        And will write to the netcdf.
        """
        def extract_from_message(messageid: int) -> Tuple[Dict[str,np.ma.MaskedArray], str]:
            """
            Eccodes tools to extract the field and do a sanity check on the name or cfVarName
            Returns a dictionary of the field, with endStep key, and additionally the units string
            EndStep is important in case the self.operation wants a certain timestep extracted.
            """
            name = ec.codes_get(messageid, 'name').lower().split(' ')
            datestamp = str(ec.codes_get(messageid, 'dataDate'))
            assert ('_'.join(name) == self.encoding.loc['variable']) or (ec.codes_get(messageid, 'cfVarName') == self.encoding.name)
            assert pd.Timestamp(year = int(datestamp[:4]), month = int(datestamp[4:6]), day = int(datestamp[6:])) == date # Date comes from one level up
   
            # Extract the gridded values, reshape and mask.
            values = ec.codes_get_values(messageid) # One dimensional array
            lat_fastest_changing = (ec.codes_get(messageid, 'jPointsAreConsecutive') == 1)
            values = values.reshape((ec.codes_get(messageid, 'Nj'),ec.codes_get(messageid, 'Ni')), order = 'F' if lat_fastest_changing else 'C') # order C means last index fastest changing
            if ec.codes_get(messageid,'latitudeOfFirstGridPointInDegrees') > ec.codes_get(messageid,'latitudeOfLastGridPointInDegrees'):
                values = values[::-1,:] # In my eventual netcdf storage I want the latitudes to increase with index
                
            masked_values = np.ma.MaskedArray(data = values, mask = (values == ec.codes_get(messageid, 'missingValue')))
            units = ec.codes_get(messageid, 'units')
            timeinfo = str(ec.codes_get(messageid, self.encoding.loc['timevariable']))
            if timeinfo[-2:] == '00':
                timeinfo = timeinfo[:-2] # Remove the trailing zeros of the minutes
            if len(timeinfo) == 1:
                timeinfo = '0' + timeinfo[0] # Prepending to match the hhUTC codes
            return ({timeinfo:masked_values},units)
        
        while True:
            # Handling the queued tuples and the termination command
            request = inqueue.get()
            if (request[0] == 'STOP'):
                logging.debug('terminating this preprocess subprocess')
                break
            date, rawfilepath = request # A valid request contains a pd.DatetimeIndex and a Path
            
            # Check file time content, and make sure that we are not requested to write an existing date
            with nc.Dataset(self.datapath, mode='a') as ds:
                times = ds[self.operation]['time'][:]
                if times.size == 0: # Empty timeaxis
                    presentdates = []
                    lastdate = date - pd.Timedelta(1, 'D') # Set because of possible consecitive criterium
                else:
                    presentdates = nc.num2date(times, units = ds[self.operation]['time'].units, calendar = ds[self.operation]['time'].calendar)
                    lastdate = pd.Timestamp(presentdates[-1])
                writedate = date.to_pydatetime()
                assert not (writedate in presentdates)
            
             # Check whether the writedate does not follow the presentdates, given consecutive is set to True
             # Because then the request should be given back until the consecutive turns up
            if consecutive and (lastdate + pd.Timedelta(1, 'D') != date):
                inqueue.put(request)
                try: # Checking if we have one single cycling request, if not or if the comparison fails then log and set a new previous request
                    if request != oldreq:
                        logging.debug(f'Placed back request for {date} as it does not follow the files last day {pd.Timestamp(presentdates[-1])}')
                        oldreq = request
                except NameError: 
                    oldreq = request
                    logging.debug(f'Placed back request for {date} as it does not follow the files last day {pd.Timestamp(presentdates[-1])}')
            else:
                # Let ECcodes loop through as many messages as there are in the raw file of this date (maximum 24)
                content = OrderedDict()
                with open(rawfilepath, mode = 'rb') as f:
                    while True:
                        gid = ec.codes_grib_new_from_file(f)
                        if gid is None:
                            break
                        stepfield, units = extract_from_message(gid)
                        content.update(stepfield)
                        ec.codes_release(gid)
                
                # Do the actual operations
                if self.operation.endswith('UTC'):
                    try:
                        dayfield = content[self.operation[:2]]
                    except KeyError:
                        logging.error(f'field to extract at: {self.operation} is not present in {rawfilepath}')
                else:
                    if len(content) < 24:
                        logging.warning(f'less than 24 hours present in {rawfilepath} for {self.operation}')
                    stacked_array = np.ma.stack(content.values(), axis = 0)
                    operation_method = getattr(stacked_array, self.operation)
                    dayfield = operation_method(axis = 0)

                self.writer.append_one_day(writedate= writedate, dayfield= dayfield, index= len(presentdates), units= units)
                

class DataOrganizer(object):
    """
    One data Organizer per variable. And needs the operation to go from hourly raw to the daily values
    Goal is one file with daily data according to a given range. Defined here.
    Manages a folder with raw datafiles and makes sure that these are processed into by means of the PreProcessor. 
    Populates a Pipe with download requests if files are missing.
    """
    
    def __init__(self, varname: str, region: Region, operation: str = 'avg') -> None:
        """
        operation should be one of ['mean','min','max','hhUTC'] of which hh is between '00' and '24'
        """
        self.vardir = Path('/nobackup_1/users/straaten/ERA5/' + varname)
        self.rawdir = self.vardir / 'raw'
        if not self.rawdir.exists():
            self.rawdir.mkdir(parents=True)
        self.varname = varname
        self.operation = operation
        self.ncvarname = '-'.join([self.varname, self.operation])
        self.region = region
        self.region_name = region[0]
        self.datapath = self.vardir / ('_'.join([varname, self.region_name]) + '.nc')
        self.downloader = CDSDownloader()
        self.preprocessor = PreProcessor(self.operation, self.ncvarname, self.datapath, encoding = self.downloader.era_formats.loc[self.varname], region = region)
    
    def __repr__(self) -> str:
        return(f'DataOrganizer({self.varname},{self.region_name},{self.operation})')

    def missing_dataset_days(self, desireddays: pd.DatetimeIndex ) -> pd.DatetimeIndex:
        """
        Checks if the desired daily timerange is present in the dataset, for the variable + operation of interest. Returns the difference.
        """
        with nc.Dataset(self.datapath, mode='r') as ds:
            try:
                datearray = nc.num2date(ds[self.operation]['time'][:], units = ds[self.operation]['time'].units, calendar = ds[self.operation]['time'].calendar)
                presentdays = pd.DatetimeIndex(datearray)
            except IndexError: # then the time dimension is completely empty
                presentdays = pd.DatetimeIndex([])
            #for var in presentset.variables.keys():
            #    print(presentset[var])
        return(desireddays.difference(presentdays))
    
    def present_raw_days(self, index : pd.DatetimeIndex) -> Tuple[pd.DatetimeIndex,list]:
        """
        Checks whether complete (24hr together, except for accumulated evaporation) raw files are present
        for the dates that were missing in the dataset.
        The raw file format is varname_region_yyyy-mm-dd.grib
        Returns a tuple containing containing the present raw time indices and a list of their paths
        """
        unprocessed_dates = []
        unprocessed_paths = []
        for item in self.rawdir.iterdir():
            if item.is_file():
                try:
                    date = item.name[-15:-5]
                    stamp = pd.Timestamp(date)
                    if stamp in index:
                        unprocessed_dates.append(stamp)
                        unprocessed_paths.append(item)
                except ValueError:
                    pass
        unprocessed_dates = pd.DatetimeIndex(unprocessed_dates)
        return((unprocessed_dates,unprocessed_paths))
    
    def main(self, start: str, end: str):
        
        logging.info(repr(self))
        self.preprocessor.writer.create_dataset(self, dimensions = ('time','latitude','longitude'))
        desireddays = pd.date_range(start = start, end = end, freq = 'D')
        logging.debug(f'desired days: {desireddays}')
        dif = self.missing_dataset_days(desireddays = desireddays)

        if not dif.empty:
            
            logging.debug(f'not yet present: {dif}')
            self.preprocessor.start(consecutive=True)
            
            # Starting a potential first batch of processing, based on already downloaded files
            foundrawdates, foundrawfiles = self.present_raw_days(index = dif)
            logging.debug(f'found these raw files: {foundrawfiles}')
            if not foundrawdates.empty:
                self.preprocessor.add_to_preprocessing_queue(foundrawdates,foundrawfiles)

            to_download = dif.difference(foundrawdates)
            if not to_download.empty:
                self.downloader.create_requests_and_populate_queue(downloaddates=to_download, request_kwds = {'varname':self.varname, 'region':self.region,'rawdir':self.rawdir})
                self.results = self.downloader.start_queue(n_par_requests=5)
                
                # Add downloaded files to the preprocessor as enter the results queue
                while True:
                    time.sleep(60)
                    if self.results.qsize() == 0:
                        break
                    downloaded = [self.results.get() for i in range(self.results.qsize())]
                    rawdates = pd.DatetimeIndex([tup[0] for tup in downloaded])
                    rawpaths = [tup[1] for tup in downloaded]
                    self.preprocessor.add_to_preprocessing_queue(unprocessed_dates=rawdates, unprocessed_paths=rawpaths)
                
            time.sleep(200)
            self.preprocessor.end()

