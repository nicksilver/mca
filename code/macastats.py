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
import json
import pandas as pd
import collections

def invert_dict(dic):
    """
    Converts list of dictionaries to dictionary of lists.
    """
    result = collections.defaultdict(list)
    for d in dic:
        for k, v in d.items():
            result[k].append(v)
    return result


def zstats(gis_path, net_data):
    """
    Returns zonal stats of netcdf raster data from shapefile for a specified
    year.

    :param gis_path: path to shapefile (str)

    :param net_data: numpy array of data

    :return: the grid-cell count, min, max, and mean of shapefile object
    """

    # Get names of climate divisions from json file
    with open(gis_path + ".json") as jfile:
        jdata = json.load(jfile)

        cd_names = []
        for cd in jdata['objects']['MT_CLIM_DIVISIONS']['geometries']:
            cd_name = cd['properties']['CLIMDIV']
            cd_names.append(cd_name)

    # Affine transformation information:
    # a = width of a pixel
    # b = row rotation (typically zero)
    # c = x-coordinate of the upper-left corner of the upper-left pixel
    # d = column rotation (typically zero)
    # e = height of a pixel (typically negative)
    # f = y-coordinate of the of the upper-left corner of the upper-left pixel

    # These were taken from the MACA netcdf file
    a = 0.0417
    b = 0
    c = -116.6056 - a
    d = 0
    e = -0.0417
    f = 49.3127 - e
    aff = Affine(a, b, c, d, e, f)

    # Get zone stats for climate divisions
    stats=['min', 'max', 'mean', 'median', 'count', 'std']

    if len(net_data.shape)==3:
        ndata = net_data[0, :, :]
    else:
        ndata = net_data

    zs = zonal_stats(gis_path + ".shp", ndata, affine=aff,
                     stats=stats)
    zs = pd.DataFrame(invert_dict(zs))
    zs['climdiv'] = cd_names

    if len(net_data.shape)==3:
        mth = np.repeat(1, len(zs))
        zs['month'] = mth
        for m in range(net_data.shape[0]-1):
            tzs = zonal_stats(gis_path+'.shp', net_data[m+1, :, :], affine=aff,
                              stats=stats)
            tzs = pd.DataFrame(invert_dict(tzs))
            tzs['climdiv'] = cd_names
            tzs['month'] = mth + m + 1
            zs = zs.append(tzs)

    return zs


def clim_div_names(json_path):
    """
    Returns a list of the climate division names to be used with zstats()
    json_path (str) - path to json file
    """
    with open(json_path) as jfile:
        jdata = json.load(jfile)

    cd_names = []
    for cd in jdata['objects']['MT_CLIM_DIVISIONS']['geometries']:
        cd_name = cd['properties']['CLIMDIV']
        cd_names.append(cd_name)

    return cd_names


def temp_average(tmin, tmax, save=False, dpath="./tavg.npy"):
    """
    Calculates averages of tmin and tmax to get tavg
    :param tmin: path to numpy array of tmin values from mod_diff()
    :param tmax: path to numpy array of tmax values from mod_diff()
    :param save: boolean for whether to save
    :param dpath: directory path for saving
    :return: numpy array of average temperature.
    """
    tmin = np.load(tmin)
    tmax = np.load(tmax)
    tavg = np.mean((tmin, tmax), axis=0)
    if save:
        np.save(dpath, tavg)
    return tavg


def zstats_range(data, gis_path, mod_list, stat='median'):
    """
    Finds the minimum and maximum projection within ensemble and the model name for each.
    :param data: Array of differences
    :param gis_path: Path to shapefile to get zonal_stats
    :param mod_list: List of the names of the model in order of data[0, :, :, :]
    :param stat: Statistic to use for range from data
    :return: Returns tuplet of (min, max, min_name, max_name)
    """
    # Process zonal stats for each model
    df = pd.DataFrame()
    for m in range(data.shape[0]):
        zs = zstats(gis_path, data[m, :, :, :])
        mn = np.repeat(mod_list[m], len(zs))
        zs['model'] = mn
        df = df.append(zs)
        print "I am done processing zonal stats for " + mod_list[m]

    # Find min and max for each climdiv and month
    print "Now I am going to find the min and max..."
    dfs = pd.DataFrame()
    for m in range(1, 13):
        for cd in range(2401, 2408):
            cdm = df[[stat, 'model']][(df['climdiv'] == cd) & (df['month']==m)]
            cdm.index = range(len(cdm))
            cdmax = cdm.iloc[cdm[stat].idxmax()]
            cdmin = cdm.iloc[cdm[stat].idxmin()]
            temp_df = pd.DataFrame()
            temp_df['climdiv'] = [cd]
            temp_df['month'] = [m]
            temp_df['max'] = [cdmax[stat]]
            temp_df['model_max'] = [cdmax['model']]
            temp_df['min'] = [cdmin[stat]]
            temp_df['model_min'] = [cdmin['model']]
            dfs = dfs.append(temp_df)
    return dfs


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


    def mod_names(self):
        """
        Returns a list of the model names in the order used in mod_diff()
        """
        mod_list = []
        for fut_file in self.fut_list:
            # Get name of the model
            mod_name = fut_file.split("_")[5]
            mod_list.append(mod_name)
        return mod_list


    def mod_diff_ann(self, save=False, dpath="./"):
        """
        Find the projected annual change for each model in list. Returns a list
        of the model names and an array of the results.

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

        # Find end year and scenario
        end_yr = self.fut_list[0].split("_")[9]
        rcp = self.fut_list[0].split("_")[6]

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
            name = dpath + "model_diffs_" + vname + "_" + rcp + "_" + end_yr
            np.save(name, diff_arr)
        print("Processing is complete. Thanks for your patience.")
        return diff_arr


    def timestamp(self, historical=True):
        """
        Returns datetime list from time variable in netcdf file.
        """
        days_offset = -25567
        sec_day = 24*60*60
        if historical:
            data = Dataset(self.hist_list[0])
        else:
            data = Dataset(self.fut_list[0])
        t = data.variables['time'][:]
        data.close()
        x = pd.to_datetime(t + days_offset, unit='D')
        return x


    def mod_diff_mon(self, save=False, dpath="./"):
        """
        Find the projected monthly change for each model in list. Returns a list
        of the model names and an array of the results.

        save (bool) -- do you want to save numpy array?
        dpath (str) -- destination directory for saving file
        """

        # Set dimensions of output
        mod_dim = len(self.hist_list)
        lat, lon = self.get_latlon()
        lat_dim = lat.shape[0]
        lon_dim = lon.shape[0]
        time_dim =  12  # number of months
        print("Hold on a few minutes while I process " + str(mod_dim) + " models...")

        # Find name of variable and change to netcdf name
        vname = self.hist_list[0].split("_")[4]
        if vname == "pr":
            netname = "precipitation"
        elif vname == "tasmin" or vname == "tasmax":
            netname = "air_temperature"

        # Find end year and scenario
        end_yr = self.fut_list[0].split("_")[9]
        rcp = self.fut_list[0].split("_")[6]

        # For each model in the list find the projected change
        diff_arr = np.zeros((mod_dim, time_dim, lat_dim, lon_dim))
        counter = 0
        for fut_file in self.fut_list:
            # Get name of the model
            mod_name = fut_file.split("_")[5]

            # Find historic file that matches future file
            hist_file = [s for s in self.hist_list if mod_name in s][0]

            # Open datasets
            fut_data = Dataset(fut_file)
            hist_data = Dataset(hist_file)
            fut_var = fut_data.variables[netname][:]
            hist_var = hist_data.variables[netname][:]

            # Find average for each month
            fut_ts = self.timestamp(historical=False)
            hist_ts = self.timestamp(historical=True)
            fut_mth = np.zeros((time_dim, lat_dim, lon_dim))
            hist_mth = np.zeros((time_dim, lat_dim, lon_dim))
            for m in range(time_dim):
                fut_hold = fut_var[fut_ts.month==m+1, :, :].mean(axis=0)
                fut_mth[m, :, :] = fut_hold
                hist_hold = hist_var[hist_ts.month==m+1, :, :].mean(axis=0)
                hist_mth[m, :, :] = hist_hold

            diff = fut_mth - hist_mth

            # Add diff to numpy array
            diff_arr[counter, :, :, :] = diff

            # Complete loop
            counter += 1
            fut_data.close()
            hist_data.close()
            print("Done processing " + mod_name)

        if save:
            print("Saving file...")
            name = dpath + "model_diffs_mth_" + vname + "_" + rcp + "_" + end_yr
            np.save(name, diff_arr)
        print("Processing is complete. Thanks for your patience.")
        return diff_arr
