import macaproc as mp
import macastats as ms
import numpy as np
import gdal, osr

file_path = "/data/maca_mt/"
data_path = "/home/nick/workspace/data/monthly/"
mod_list = ['IPSL-CM5B-LR']
rcp = "rcp45"
yr = "2099"
# dname = "beetle_thresholds_mth_fut_" + rcp + "_" + yr + ".npy"
dname = "beetle_thresholds_mth_hist.npy"

# Get lists of files
hist_rcp = mp.select_rcp(file_path, 'historical')
hist_pr = mp.select_mod(hist_rcp, var='pr', mod=mod_list)
fut_rcp = mp.select_rcp(file_path, rcp)
fut_pr = mp.select_mod(fut_rcp, var='pr', mod=mod_list, yr=yr)

mstats = ms.MacaStats(hist_pr, fut_pr)
lat, lon = mstats.get_latlon()

# Bring in beetle data and find ensemble median
beetle = np.median(np.load(data_path+dname), axis=0)

# Build Geotiff
xmin = lon.min() - 360
ymin = lat.min()
xmax = lon.max() - 360
ymax = lat.max()
nrows = beetle.shape[1]
ncols = beetle.shape[2]
nbands = beetle.shape[0]
xres = (xmax - xmin)/float(ncols)
yres = (ymax - ymin)/float(nrows)
# dst_filename = "beetle_thresholds_mth_fut_"+ rcp + "_" + yr + ".tif"
dst_filename = "beetle_thresholds_mth_hist.tif"
fmt = "GTiff"
driver = gdal.GetDriverByName(fmt)
dst_ds = driver.Create(dst_filename, ncols, nrows, nbands, gdal.GDT_Float32)
dst_ds.SetGeoTransform([xmin, xres, 0, ymax, 0, -yres])
srs = osr.SpatialReference()
srs.SetWellKnownGeogCS("WGS84")
dst_ds.SetProjection(srs.ExportToWkt())
for band in range(beetle.shape[0]):
    print band
    dst_ds.GetRasterBand(band+1).WriteArray(beetle[band, :, :])

dst_ds = None





