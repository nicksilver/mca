import macaproc as mp
import macastats as ms
import macaplots as mplt
import numpy as np


# data_path = "/data/maca_mt/"
gis_path = "/home/nick/workspace/shapefiles/"
data_path = "/home/nick/workspace/data/"
save_path = None

# Load data
mod_delta_tmin = np.load(data_path + "model_diffs_tasmin_rcp45_2099.npy")
mod_delta_tmax = np.load(data_path + "model_diffs_tasmax_rcp45_2099.npy")
mod_delta_tavg = ms.temp_average(mod_delta_tmin, mod_delta_tmax,
                                 save=False, dpath=save_path)

# Calculate clim div stats
shpfile = gis_path + "MT_CLIM_DIVISIONS"
tavg_zstats = ms.zstats(shpfile, mod_delta_tavg.mean(axis=0))


# Plot
cd_plot = mplt.clim_divs(shpfile + '.shp')
cd_plot.temp_plot(tavg_zstats, "Temperature Change (C)")
