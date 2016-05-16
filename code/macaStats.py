"""
Created by: Nick Silverman
Date: April 13th, 2016
Description: Functions to calculate statistics in from raster data and 
shapefiles.
"""

import numpy as np
from rasterstats import zonal_stats
from netCDF4 import Dataset
from affine import Affine


def zstats(shp_path, net_path, var, year):
    """
    Returns zonal stats of netcdf raster data from shapefile for a specified
    year.

    :param shp_path: path to shapefile (str)

    :param net_path: path to netcdf file (str)

    :param var: netcdf variable name (str)

    :param year: index for specified year in netcdf file (int)

    :return: the grid-cell count, min, max, and mean of shapefile object
    """
    net_data = Dataset(net_path, 'r')
    data = net_data.variables[var][0, :, :]

    # Affine transformation information:
    # a = width of a pixel
    # b = row rotation (typically zero)
    # c = x-coordinate of the upper-left corner of the upper-left pixel
    # d = column rotation (typically zero)
    # e = height of a pixel (typically negative)
    # f = y-coordinate of the of the upper-left corner of the upper-left pixel

    a = float(getattr(net_data, "geospatial_lon_resolution"))
    b = 0
    c = float(getattr(net_data, "geospatial_lon_min")) - a
    d = 0
    e = -float(getattr(net_data, "geospatial_lat_resolution"))
    f = float(getattr(net_data, "geospatial_lat_max")) - e 

    aff = Affine(a, b, c, d, e, f)
    net_data.close()
    zs = zonal_stats(shp_path, data, affine=aff)

    return zs


class AggStats(object):
    """
    This class will take a list of file names and aggregate them to compute
    statistics (e.g. annual average of all models). 
    """
    def __init__(self, hist_list, fut_list):
        self.hist_list = hist_list
        self.fut_list = fut_list

    def mod_diff(self):
        for fut_file in self.fut_list:
            # get name of the model
            mod_name = fut_file.split("_")[5]

            # find historic file that matches future file
            hist_file = [s for s in self.hist_list if mod_name in s][0]

        print hist_file, fut_file


