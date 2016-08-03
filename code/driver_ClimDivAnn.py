import macastats as ms
import macaplots as mplt
import numpy as np


# gis_path = "/home/nick/workspace/shapefiles/"
gis_path = "/home/nick/MEGA/workspace/mca/data/shapefiles/"
# data_path = "/home/nick/workspace/data/annually/"
data_path = "/home/nick/MEGA/workspace/mca/data/model_diffs/annually/"
# data_path = "./"
save_path = None

# Load data
# mod_delta_tmin = np.load(data_path + "model_diffs_tasmin_rcp45_2099.npy")
# mod_delta_tmax = np.load(data_path + "model_diffs_tasmax_rcp45_2099.npy")
# mod_delta_tavg = ms.temp_average(mod_delta_tmin, mod_delta_tmax,
#                                  save=False, dpath=save_path)
ann_var = np.load(data_path+"model_vars_perc_pr_rcp45_2099.npy")
plot_data = ann_var

# Calculate clim div stats
shpfile = gis_path + "MT_CLIM_DIVISIONS"
zstats = ms.zstats(shpfile, plot_data.mean(axis=0))

# Plot
cd_plot = mplt.clim_divs(shpfile + '.shp')
cd_plot.temp_plot(zstats, "Temperature Change (C)")
