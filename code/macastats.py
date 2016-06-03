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


def temp_average(tmin, tmax):
    """
    Calculates averages of tmin and tmax to get tavg
    :param tmin: numpy array of tmin values from mod_diff()
    :param tmax: numpy array of tmax values from mod_diff()
    :return: numpy array of average temperature.
    """
    return np.mean((tmin, tmax), axis=0)


class AggStats(object):
    """
    This class will take a list of file names and aggregate them to compute
    statistics (e.g. annual average of all models). 
    """
    def __init__(self, hist_list, fut_list):
        self.hist_list = hist_list
        self.fut_list = fut_list

    def get_latlon(self):
        """
        Returns lat and lon from the dataset.
        """
        data = Dataset(self.hist_list[0])
        lat = data.variables['lat'][:]
        lon = data.variables['lon'][:]
        data.close()
        return lat, lon

    def mod_diff(self, save=False, dpath="./"):
        """
        Find the projected change for each model in list. Returns a list of the model
        names and an array of the results.

        save (bool) -- do you want to save numpy array?
        dpath (str) -- destination directory for saving file
        """

        # Set dimensions of output
        mod_dim = len(self.hist_list)
        lat, lon = self.get_latlon()
        lat_dim = lat.shape[0]
        lon_dim = lon.shape[0]
        print("Hold on a few minutes while I process " + str(mod_dim) + " models...")
        
        # Find name of variable and change to netcdf name
        vname = self.hist_list[0].split("_")[4]
        if vname == "pr":
            netname = "precipitation"
        elif vname == "tasmin" or vname == "tasmax":
            netname = "air_temperature"

        # For each model in the list find the projected change
        diff_arr = np.zeros((mod_dim, lat_dim, lon_dim))
        counter = 0
        for fut_file in self.fut_list:
            # Get name of the model
            mod_name = fut_file.split("_")[5]

            # Find historic file that matches future file
            hist_file = [s for s in self.hist_list if mod_name in s][0]
            
            # Open datasets
            fut_data = Dataset(fut_file)
            hist_data = Dataset(hist_file)
            
            # Find average over time span
            fut_avg = fut_data.variables[netname][:].mean(axis=0)
            hist_avg = hist_data.variables[netname][:].mean(axis=0)
            diff = fut_avg - hist_avg

            # Add diff to numpy array
            diff_arr[counter, :, :] = diff

            # Complete loop
            counter += 1
            fut_data.close()
            hist_data.close()
            print("Done processing " + mod_name)

        if save:
            print("Saving file...")
            np.save(dpath + "model_diffs_" + vname, diff_arr)
        print("Processing is complete. Thanks for your patience.")
        return diff_arr


