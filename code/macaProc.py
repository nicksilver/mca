"""
Created by: Nick Silverman
Date: September 20th, 2015
Description: Functions to process MACA data
"""

import numpy as np
from netCDF4 import Dataset
from netcdftime import utime
from osgeo import gdal, ogr


# Define function to find index of nearest grid cell
def find_nearest(array, val):
    """
    Function to find index of nearest grid-cell in array.

    :param array: array of lats or lons

    :param val: lat or lon value

    :return: array index of nearest specified value.
    """

    idx = np.abs(array - val).argmin()
    return idx


def clip_data(data, var, bottom, top, right, left):
    """
    Function to clip data to a rectangular domain.

    :param data: data to be clipped (netcdf4 object)

    :param var: variable name to be clipped (str)

    :param bottom: minimum lat value (float)

    :param top: maximum lat value (float)

    :param right: maximum lon value (float)

    :param left: minimum lon value (float)

    :return: clipped data
    """

    # Find nearest lat and lon
    l = find_nearest(data.variables['lon'][:], left)
    b = find_nearest(data.variables['lat'][:], bottom)
    r = find_nearest(data.variables['lon'][:], right)
    t = find_nearest(data.variables['lat'][:], top)

    # Clip data
    if var=='lat':
        cdata = data.variables[var][t:b]
    elif var=='lon':
        cdata = data.variables[var][l:r]
    elif len(data.variables[var].shape)==2:
        cdata = data.variables[var][t:b, l:r]
    else:
        cdata = data.variables[var][:, t:b, l:r]

    return cdata


def year_clip(data, var, begin, end):
    tcon = utime('days since 1900-01-01', calendar='noleap')
    dates = tcon.num2date(data.variables['time'][:])
    yvals = np.zeros((len(dates)))  # create array to hold year values

    # year array
    for j in range(len(dates)):
        yvals[j] = dates[j].year

    # aggregate by year
    var_data = data.variables[var][((yvals>begin-1) & (yvals<end+1)), :, :]

    return var_data


def createMask(shp_path, x_min, y_min, x_max, y_max, ncols, nrows):
    """
    Creates mask clipped to shapefile for numpy masked array.

    :param shp_path: Path to shapefile

    :param x_min: Minimum longitude in grid

    :param y_min: Minimum latitude in grid

    :param x_max: Maximum longitude in grid

    :param y_max: Maximum latitude in grid

    :param ncols: Number of horizontal gridcells

    :param nrows: Number of vertical gridcells

    :return: Clipped mask to be applied to numpy masked array.
    """

    xres = (x_max - x_min)/float(ncols)
    yres = (y_max - y_min)/float(nrows)
    sf = ogr.Open(shp_path)
    maskvalue = 1
    geotransform = (x_min, xres, 0, y_max, 0, -yres)
    sf_lyr = sf.GetLayer()
    dst_ds = gdal.GetDriverByName('MEM').Create('', ncols, nrows, 1, gdal.GDT_Byte)
    dst_rb = dst_ds.GetRasterBand(1)
    dst_rb.Fill(0)  # initialize raster with zeros
    dst_rb.SetNoDataValue(0)
    dst_ds.SetGeoTransform(geotransform)
    err = gdal.RasterizeLayer(dst_ds, [maskvalue], sf_lyr)
    dst_ds.FlushCache()
    mask_arr = dst_ds.GetRasterBand(1).ReadAsArray() == 0
    return mask_arr




