#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERA5-Land 0.1 degree resolution, hourly. 4 * volumetric soil water content, 1 * transpiration, 1 * snow cover, 1* t2m
ERA5-Land will have about a 9 times larger data volume than a single level of ERA5 in the same domain.
Hourly is not needed so reworking to daily averages/sums. 
"""

from src.utils import get_europe, get_nhplus, get_nhmin
from src.downloaders import CDSDownloader, DataOrganizer

#logging.basicConfig(filename='swvl4.log', filemode='w', level=logging.DEBUG, format='%(process)d-%(levelname)s-%(message)s')
d = DataOrganizer(varname = 'z500', region = get_europe(), operation = '12UTC')
#d = DataOrganizer(varname = 'z300', region = get_europe(), operation = '12UTC')
#d = DataOrganizer(varname = 'transp', region = get_europe(), operation = '24UTC')
#d = DataOrganizer(varname = 'sst', region = get_nhplus(), operation = 'mean')
#d = DataOrganizer(varname = 't2m', region = get_europe(), operation = 'mean')
#d = DataOrganizer(varname = 'siconc', region = get_nhmin(), operation = 'mean')
#d = DataOrganizer(varname = 'snowc', region = get_nhmin(), operation = '24UTC')
#d = DataOrganizer(varname = 'swvl4', region = get_europe(), operation = 'mean')
#d = DataOrganizer(varname = 'tcc', region = get_europe(), operation = 'mean')
#d.main(start = '1981-01-01', end = '2019-09-30')