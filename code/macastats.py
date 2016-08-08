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

def zstats(gis_path, net_data, metric=True, precip=False):
    """
    Returns zonal stats of netcdf raster data from shapefile for a specified
    year.

    :param gis_path: path to shapefile (str)

    :param net_data: numpy array of data

    :param metric: should the data be in metric (mm and C) or US (in and F) units

    :param precip: is the data precip?

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
    stats = ['min', 'max', 'mean', 'median', 'count', 'std']

    if len(net_data.shape) == 3:
        ndata = net_data[0, :, :]
    else:
        ndata = net_data

    zs = zonal_stats(gis_path + ".shp", ndata, affine=aff,
                     stats=stats)
    zs = pd.DataFrame(invert_dict(zs))
    zs['climdiv'] = cd_names

    if len(net_data.shape) == 3:
        mth = np.repeat(1, len(zs))
        zs['month'] = mth
        for m in range(net_data.shape[0]-1):
            tzs = zonal_stats(gis_path+'.shp', net_data[m+1, :, :], affine=aff,
                              stats=stats)
            tzs = pd.DataFrame(invert_dict(tzs))
            tzs['climdiv'] = cd_names
            tzs['month'] = mth + m + 1
            zs = zs.append(tzs)

    # Convert to inches
    if precip and not metric:
        zs[['max', 'min', 'mean', 'median', 'std']] = zs[['max',
                                                          'min',
                                                          'mean',
                                                          'median',
                                                          'std']]*0.0393701

    # Convert to Fahrenheit
    elif not precip and not metric:
        zs[['max', 'min', 'mean', 'median', 'std']] = zs[['max',
                                                          'min',
                                                          'mean',
                                                          'median',
                                                          'std']]*1.8

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


def ffd(tmin):
    """
    Returns the number of freeze free days per year.
    """
    return (tmin > 273.15)


def gdd(tmin, tmax, base=273.15):
    """
    Returns growing degree day from numpy arrays. Base value should be in Kelvin.
    """
    gdd = (tmin + tmax)/2. - base
    gdd[gdd < 0] = 0
    return gdd


def zstats_range(data, gis_path, zs_data, mod_list, stat='median', precip=False,
                 metric=True):
    """
    Finds the minimum and maximum projection within ensemble and the model name 
    for each. It also finds the percent agreement of the direction of change 
    with the median.
    :param data: Array of differences
    :param gis_path: Path to shapefile to get zonal_stats
    :param zs_data: Data returned from zstats()
    :param mod_list: List of the names of the model in order of data[0, :, :, :]
    :param stat: Statistic to use for range from data
    :param precip: Is the variable precipitation?
    :param metric: Should we return values in metric (mm and C) or US (in. and F)
    :return: Returns dataframe of min, min_model, max, max_name, % agreement
    """

    # Process zonal stats for each model
    df = pd.DataFrame()
    for m in range(data.shape[0]):
        if len(data.shape) > 3:
            zs = zstats(gis_path, data[m, :, :, :])
        else:
            zs = zstats(gis_path, data[m, :, :])
        mn = np.repeat(mod_list[m], len(zs))
        zs['model'] = mn
        df = df.append(zs)
        print "I am done processing zonal stats for " + mod_list[m]

    # Find min and max for each climdiv and month
    print "Now I am going to find the min and max..."

    dfs = pd.DataFrame()
    for m in range(1, 13):
        for cd in range(2401, 2408):
            if len(data.shape) > 3:
                cdm = df[[stat, 'model']][(df['climdiv'] == cd) & (df['month'] == m)]
                med_bool = list(zs_data[stat][(zs_data['climdiv'] == cd) &
                                              (zs_data['month'] == m)] > 0)[0]
            else:
                cdm = df[[stat, 'model']][(df['climdiv'] == cd)]
                med_bool = list(zs_data[stat][(zs_data['climdiv'] == cd)] > 0)[0]

            # Is the current model median value positive
            cdm_bool = (cdm[stat] > 0)

            # Record if the signs are alike
            agr_bool = (med_bool == list(cdm_bool))
            perc_agree = 100 * agr_bool.sum()/float(len(agr_bool))

            # Find index of min and max models
            cdm.index = range(len(cdm))
            cdmax = cdm.iloc[cdm[stat].idxmax()]
            cdmin = cdm.iloc[cdm[stat].idxmin()]

            # Add tp dataframe
            temp_df = pd.DataFrame()
            temp_df['perc_agree'] = [perc_agree]
            temp_df['climdiv'] = [cd]
            temp_df['max'] = [cdmax[stat]]
            temp_df['model_max'] = [cdmax['model']]
            temp_df['min'] = [cdmin[stat]]
            temp_df['model_min'] = [cdmin['model']]
            if len(data.shape) > 3:
                temp_df['month'] = [m]
            dfs = dfs.append(temp_df)
        if len(data.shape) == 3:
            break

    # Convert to inches
    if precip and not metric:
        dfs[['max', 'min']] = dfs[['max', 'min']]*0.0393701

    # Convert to Fahrenheit
    elif not precip and not metric:
        dfs[['max', 'min']] = dfs[['max', 'min']]*1.8
    return dfs


class MacaStats(object):
    """
    This is the base class containing generic functions used in all subclasses.
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

    def agg_time(self, data, freq='annual', historical=True, stat='sum'):
        """
        Returns numpy array aggregated to specified frequency and stat

        data (array) -- numpy array containing data
        freq (str) -- 'annual' or 'monthly'
        historical (bool) -- is the data historical or future?
        stat (str) -- do you want to find the sum or the mean
        """
        #TODO Might be faster to utilize Pandas Panels instead of for loop.

        ts = self.timestamp(historical=historical)
        yrs_uni = np.unique(ts.year)
        lat_dim = data.shape[1]
        lon_dim = data.shape[2]
        yr_dim = len(yrs_uni)
        mth_dim = 12

        if freq == 'monthly':
            agg_data = np.zeros((mth_dim, lat_dim, lon_dim))
            for m in range(mth_dim):
                if stat == 'sum':
                    hold = data[ts.month == m + 1, :, :].sum(axis=0)
                elif stat == 'mean':
                    hold = data[ts.month == m + 1, :, :].mean(axis=0)
                agg_data[m, :, :] = hold/yr_dim

        elif freq == 'annual':
            agg_data = np.zeros((yr_dim, lat_dim, lon_dim))
            counter = 0
            for y in yrs_uni:
                if stat == 'sum':
                    hold = data[ts.year == y, :, :].sum(axis=0)
                elif stat == 'mean':
                    hold = data[ts.year == y, :, :].mean(axis=0)
                agg_data[counter, :, :] = hold
                counter += 1
        return agg_data

    def timestamp(self, historical=True):
        """
        Returns datetime list from time variable in netcdf file.
        """
        days_offset = -25567
        if historical:
            data = Dataset(self.hist_list[0])
        else:
            data = Dataset(self.fut_list[0])
        t = data.variables['time'][:]
        data.close()
        x = pd.to_datetime(t + days_offset, unit='D')
        return x


class MacaPrecip(MacaStats):
    """
    This class handles precipitation statistics.
    """

    def __init__(self, hist_list, fut_list):
        MacaStats.__init__(self, hist_list, fut_list)

    def ens_diff_ann(self, save=False, dpath="./", stat='mean', ctype='absolute'):
        """
        Find the projected annual change for each model in list. Returns a list
        of the model names and an array of the results.

        save (bool) -- do you want to save numpy array?
        dpath (str) -- destination directory for saving file
        stat (str) -- statistic across the time domain (i.e. mean or standard deviation)
        ctype (str) -- absolute or percent change
        """

        # Set dimensions of output
        mod_dim = len(self.hist_list)
        lat, lon = self.get_latlon()
        lat_dim = lat.shape[0]
        lon_dim = lon.shape[0]
        print("Hold on a few minutes while I process " + str(mod_dim) + " models...")

        # Set names for variable
        netname = "precipitation"

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
            fut_var = fut_data.variables[netname][:]
            hist_data = Dataset(hist_file)
            hist_var = hist_data.variables[netname][:]

            fut = self.agg_time(fut_var, freq='annual',
                                historical=False, stat='sum')
            hist = self.agg_time(hist_var, freq='annual',
                                 historical=True, stat='sum')
            if stat == 'mean':
                hist_stat = hist.mean(axis=0)  # average over all years
                fut_stat = fut.mean(axis=0)  # average over all years
                name = dpath + "model_diffs_pr_" + rcp + "_" + end_yr
            elif stat == 'std':
                hist_stat = hist.std(axis=0)  # std over all years
                fut_stat = fut.std(axis=0)  # std over all years
                name = dpath + "model_vars_pr_" + rcp + "_" + end_yr

            if ctype == 'absolute':
                diff = fut_stat - hist_stat
            elif ctype == 'percent' and netname == 'precipitation':
                diff = (fut_stat - hist_stat)/hist_stat
                name = dpath + "model_vars_perc_pr_" + rcp + "_" + end_yr

            # Add diff to numpy array
            diff_arr[counter, :, :] = diff

            # Complete loop
            counter += 1
            fut_data.close()
            hist_data.close()
            print("Done processing " + mod_name)

        if save:
            print("Saving file...")
            np.save(name, diff_arr)
        print("Processing is complete. Thanks for your patience.")
        return diff_arr

    def ens_diff_mon(self, save=False, dpath="./"):
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
        time_dim = 12  # number of months
        print("Hold on a few minutes while I process " + str(mod_dim) + " models...")

        # Set variable names
        netname = "precipitation"

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

            # Find the monthly sum
            fut_mth = self.agg_time(fut_var, freq='monthly',
                                    historical=False, stat='sum')
            hist_mth = self.agg_time(hist_var, freq='monthly',
                                     historical=True, stat='sum')

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
            name = dpath + "model_diffs_mth_pr_" + rcp + "_" + end_yr
            np.save(name, diff_arr)
        print("Processing is complete. Thanks for your patience.")
        return diff_arr


class MacaTemp(MacaStats):
    """
    This class handles temperature statistics.
    """

    def __init__(self, hist_list, fut_list, hist_list_tmax=None, fut_list_tmax=None):
        MacaStats.__init__(self, hist_list, fut_list)
        if hist_list_tmax:
            self.hist_list_tmax = hist_list_tmax
            self.fut_list_tmax = fut_list_tmax

    def list_loop(self, stat='mean'):
        """
        Returns array of differences (future minus historical) from list of models
        for the specified statistic.

        hist_list (list) -- list of historical files (this variable becomes tmin if
        tmax is specified (i.e. for GDD calculations).

        fut_list (list) -- list of future files (this variable becomes tmin if tmax
        is specified (i.e. for GDD calculations).

        hist_list_tmax (list) -- list of historical tmax files

        fut_list_tmax (list) -- list of future tmax files

        stat (str) -- 'mean', 'std', 'gdd', 'ffd'
        """

        mod_dim = len(self.hist_list)
        lat, lon = self.get_latlon()
        lat_dim = lat.shape[0]
        lon_dim = lon.shape[0]
        netname = 'air_temperature'

        diff_arr = np.zeros((mod_dim, lat_dim, lon_dim))
        counter = 0
        for fut_file in self.fut_list:
            # Get name of the model
            mod_name = fut_file.split("_")[5]

            # Find historic file that matches future file
            hist_file = [s for s in self.hist_list if mod_name in s][0]

            # If temperature, find annual average
            if stat == 'mean':
                fut_data = Dataset(fut_file)
                fut_var = fut_data.variables[netname][:]
                hist_data = Dataset(hist_file)
                hist_var = hist_data.variables[netname][:]
                fut_data.close()
                hist_data.close()
                fut_stat = fut_var.mean(axis=0)
                hist_stat = hist_var.mean(axis=0)
            elif stat == 'std':
                fut_data = Dataset(fut_file)
                fut_var = fut_data.variables[netname][:]
                hist_data = Dataset(hist_file)
                hist_var = hist_data.variables[netname][:]
                fut_data.close()
                hist_data.close()
                fut_stat = fut_var.std(axis=0)
                hist_stat = hist_var.std(axis=0)
            elif stat == 'gdd':
                fut_tmin_data = Dataset(fut_file)
                fut_tmin_var = fut_tmin_data.variables[netname][:]
                fut_tmin_data.close()
                fut_tmax_file = [s for s in self.fut_list_tmax if mod_name in s][0]
                fut_tmax_data = Dataset(fut_tmax_file)
                fut_tmax_var = fut_tmax_data.variables[netname][:]
                fut_tmax_data.close()
                fut_gdd = gdd(fut_tmin_var, fut_tmax_var)
                del fut_tmin_var  # Delete to save memory
                del fut_tmax_var  # Delete to save memory
                hist_tmin_data = Dataset(hist_file)
                hist_tmin_var = hist_tmin_data.variables[netname][:]
                hist_tmin_data.close()
                hist_tmax_file = [s for s in self.hist_list_tmax if mod_name in s][0]
                hist_tmax_data = Dataset(hist_tmax_file)
                hist_tmax_var = hist_tmax_data.variables[netname][:]
                hist_tmax_data.close()
                hist_gdd = gdd(hist_tmin_var, hist_tmax_var)
                del hist_tmin_var  # Delete to save memory
                del hist_tmax_var  # Delete to save memory
                hist_stat = self.agg_time(hist_gdd, freq='annual',
                                          historical=True,
                                          stat='sum').mean(axis=0)
                fut_stat = self.agg_time(fut_gdd, freq='annual',
                                         historical=False,
                                         stat='sum').mean(axis=0)
            elif stat == 'ffd':
                fut_data = Dataset(fut_file)
                fut_var = fut_data.variables[netname][:]
                fut_data.close()
                hist_data = Dataset(hist_file)
                hist_var = hist_data.variables[netname][:]
                hist_data.close()
                fut_ffd = ffd(fut_var)
                hist_ffd = ffd(hist_var)
                del fut_var
                del hist_var
                fut_stat = self.agg_time(fut_ffd, freq='annual',
                                         historical=False,
                                         stat='sum').mean(axis=0)
                hist_stat = self.agg_time(hist_ffd, freq='annual',
                                          historical=True,
                                          stat='sum').mean(axis=0)

            diff = fut_stat - hist_stat
            del fut_stat
            del hist_stat

            # Add diff to numpy array
            diff_arr[counter, :, :] = diff

            # Complete loop
            counter += 1
            print("Done processing " + mod_name)

        return diff_arr


    def ens_diff_ann(self, save=False, dpath="./", stat='mean'):
        """
        Find the projected annual change for each model in list. Returns a list
        of the model names and an array of the results.

        save (bool) -- do you want to save numpy array?
        dpath (str) -- destination directory for saving file
        stat (str) -- statistic across the time domain (i.e. mean or standard
        deviation)
        """

        # Set dimensions of output
        mod_dim = len(self.hist_list)
        print("Hold on a few minutes while I process " + str(mod_dim) + " models...")

        # Find name of variable and change to netcdf name
        vname = self.hist_list[0].split("_")[4]

        # Find end year and scenario
        end_yr = self.fut_list[0].split("_")[9]
        rcp = self.fut_list[0].split("_")[6]

        # For each model in the list find the projected change
        if stat == 'mean':
            diff_arr = self.list_loop(stat='mean')
            name = dpath + "model_diffs_" + vname + "_" + rcp + "_" + end_yr
        elif stat == 'std':
            diff_arr = self.list_loop(stat='std')
            name = dpath + "model_vars_" + vname + "_" + rcp + "_" + end_yr
        elif stat == 'gdd':
            diff_arr = self.list_loop(stat='gdd')
            name = dpath + "model_gdd_" + rcp + "_" + end_yr
        elif stat == 'ffd':
            diff_arr = self.list_loop(stat='ffd')
            name = dpath + "model_ffd_" + rcp + "_" + end_yr

        if save:
            print("Saving file...")
            np.save(name, diff_arr)
        print("Processing is complete. Thanks for your patience.")
        return diff_arr

    def ens_diff_mon(self, save=False, dpath="./"):
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
        time_dim = 12  # number of months
        print("Hold on a few minutes while I process " + str(mod_dim) + " models...")

        # Find name of variable and change to netcdf name
        vname = self.hist_list[0].split("_")[4]
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

            # Find the monthly average
            fut_mth = self.agg_time(fut_var, freq='monthly',
                                    historical=False, stat='mean')
            hist_mth = self.agg_time(hist_var, freq='monthly',
                                     historical=True, stat='mean')

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
