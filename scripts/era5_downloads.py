#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERA5-Land 0.1 degree resolution, hourly. 4 * volumetric soil water content, 1 * transpiration, 1 * snow cover, 1* t2m
ERA5-Land will have about a 9 times larger data volume than a single level of ERA5 in the same domain.
Hourly is not needed so reworking to daily averages/sums. 
"""
import os
import sys
import logging
import eccodes as ec
from pathlib import Path

sys.path.append('..')
from Weave.utils import get_europe, get_nhplus, get_nhmin, get_nhblock, get_nhnorm
from Weave.downloaders import CDSDownloader, DataOrganizer, DataMerger

logging.basicConfig(filename='create_sst.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(levelname)s-%(message)s')
#d = DataOrganizer(varname = 'z300', region = get_nhnorm(), operation = '12UTC')
#d = DataOrganizer(varname = 't850', region = get_nhblock(), operation = '12UTC')
#d = DataOrganizer(varname = 'transp', region = get_europe(), operation = '24UTC')
d = DataOrganizer(varname = 'sst', region = get_nhplus(), operation = 'mean')
#d = DataOrganizer(varname = 't2m', region = get_europe(), operation = 'mean')
#d = DataOrganizer(varname = 'siconc', region = get_nhmin(), operation = 'mean')
#d = DataOrganizer(varname = 'snowc', region = get_nhmin(), operation = '24UTC')
#d = DataOrganizer(varname = 'swvl4', region = get_europe(), operation = 'mean')
#d = DataOrganizer(varname = 'tcc', region = get_europe(), operation = 'mean')
#d = DataOrganizer(varname = 'u300', region = get_nhnorm(), operation = '12UTC')
#d = DataOrganizer(varname = 'v300', region = get_nhnorm(), operation = '12UTC')

"""
Replacing the name to get the preliminary back extension (for the non era5-land)
"""
if False:
    d.downloader.era_formats['setname'] = d.downloader.era_formats['setname'] + '-preliminary-back-extension'
    d.main(start = '1950-03-01', end = '1978-12-31')
elif False:
    d.main(start = '1979-01-01', end = '2020-10-25')
else:
    d.main(start = '1950-03-01', end = '2021-10-25')

"""
Merging of 'fast' soil moisture layers. based on a depth-weighted average
"""
#paths = paths = [Path('/nobackup_1/users/straaten/ERA5/swvl1/swvl1_europe.nc'), Path('/nobackup_1/users/straaten/ERA5/swvl2/swvl2_europe.nc'), Path('/nobackup_1/users/straaten/ERA5/swvl3/swvl3_europe.nc')]
#groups = [None,None,None]
#weights = [7, 21, 72] # Depth of the soil layers in cm according to https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview
#outpath = Path('/nobackup_1/users/straaten/ERA5/swvl13/swvl13_europe.nc')
#self = DataMerger(inpaths = paths, ingroups = groups, outpath = outpath, outvarname = 'swvl13', region = get_europe(), weights = weights)
#self.merge()

"""
Investigating gribfile content.
"""
#rawfilepath = '/nobackup_1/users/straaten/ERA5/transp/raw/transp_europe_2019-10-01.grib'
#rawfilepath2 = '/nobackup_1/users/straaten/ERA5/transp/raw/transp_europe_2019-10-02.grib'
def gribcontent(rawfilepath):
    with open(rawfilepath, 'rb') as f:
        messages = []
        while True:
            gid = ec.codes_grib_new_from_file(f)
            if gid is None:
                break
            content = {}
            iterid = ec.codes_keys_iterator_new(gid,'ls')
            while ec.codes_keys_iterator_next(iterid):
                keyname = ec.codes_keys_iterator_get_name(iterid)
                content.update({keyname:ec.codes_get(gid, keyname)})
            ec.codes_keys_iterator_delete(iterid)
            ec.codes_release(gid)
            messages.append(content)
    return messages

def strip_redundant_messages(rawdir: Path):
    """
    Used in the transpiration case where
    post 2019-09 three instead of 1 message was downloaded when hours == 00
    """
    for rawfilepath in list(rawdir.glob(f'*.grib')):
        with open(rawfilepath, mode = 'rb') as f:
            nmessages = 0
            while True:
                gid = ec.codes_grib_new_from_file(f)
                if gid is None:
                    break
                nmessages += 1
                ec.codes_release(gid)
        if nmessages > 1:
            print(f'replacing: {rawfilepath}')
            os.system(f'grib_copy -w count=1 {rawfilepath} temp')
            os.system(f'mv temp {rawfilepath}')

