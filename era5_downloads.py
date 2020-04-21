#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERA5-Land 0.1 degree resolution, hourly. 4 * volumetric soil water content, 1 * transpiration, 1 * snow cover, 1* t2m
ERA5-Land will have about a 9 times larger data volume than a single level of ERA5 in the same domain.
Hourly is not needed so reworking to daily averages/sums. 
"""

import logging
from pathlib import Path
from src.utils import get_europe, get_nhplus, get_nhmin
from src.downloaders import CDSDownloader, DataOrganizer, DataMerger

logging.basicConfig(filename='merge_soil.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(levelname)s-%(message)s')
#d = DataOrganizer(varname = 'z500', region = get_nhmin(), operation = '12UTC')
#d = DataOrganizer(varname = 'z300', region = get_nhmin(), operation = '12UTC')
#d = DataOrganizer(varname = 'transp', region = get_europe(), operation = '24UTC')
#d = DataOrganizer(varname = 'sst', region = get_nhplus(), operation = 'mean')
#d = DataOrganizer(varname = 't2m', region = get_europe(), operation = 'mean')
#d = DataOrganizer(varname = 'siconc', region = get_nhmin(), operation = 'mean')
#d = DataOrganizer(varname = 'snowc', region = get_nhmin(), operation = '24UTC')
#d = DataOrganizer(varname = 'swvl4', region = get_europe(), operation = 'mean')
#d = DataOrganizer(varname = 'tcc', region = get_europe(), operation = 'mean')
#d.main(start = '1979-01-01', end = '2019-12-31')

paths = paths = [Path('/nobackup_1/users/straaten/ERA5/swvl1/swvl1_europe.nc'), Path('/nobackup_1/users/straaten/ERA5/swvl2/swvl2_europe.nc'), Path('/nobackup_1/users/straaten/ERA5/swvl3/swvl3_europe.nc')]
groups = ['mean','mean','mean']
weights = [7, 21, 72] # Depth of the soil layers in cm according to https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview
outpath = Path('/nobackup_1/users/straaten/ERA5/swvl13/swvl13_europe.nc')
self = DataMerger(inpaths = paths, ingroups = groups, outpath = outpath, outvarname = 'swvl13', region = get_europe(), weights = weights)
