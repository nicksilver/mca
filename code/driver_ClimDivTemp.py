import macaproc as mp
import macastats as ms
import macaplots as mplt
import numpy as np


# data_path = "/data/maca_mt/"
gis_path = "/home/nick/workspace/shapefiles/"
data_path = "/home/nick/workspace/data/"
save_path = None

######## PROCESSING ###########
# mod_list = ['IPSL-CM5B-LR', 'MIROC-ESM-CHEM']
# rcp_scen = "rcp45"

# Create list of files for historical data
# hist_rcp = mp.select_rcp(data_path, 'historical')
# hist_tmin = mp.select_mod(hist_rcp, var='tasmin', mod=mod_list)
# hist_tmax = mp.select_mod(hist_rcp, var='tasmax', mod=mod_list)

# Create list of files for future data
# fut_rcp = mp.select_rcp(data_path, rcp_scen)
# fut_tmin = mp.select_mod(fut_rcp, var='tasmin', mod=mod_list, yr='2070_2099')
# fut_tmax = mp.select_mod(fut_rcp, var='tasmax', mod=mod_list, yr='2070_2099')

# Calculate differences between historical and future for each model
# agstats_tmin = ms.AggStats(hist_tmin, fut_tmin)
# mod_delta_tmin = agstats_tmin.mod_diff(save=False, dpath=save_path)
# agstats_tmax = ms.AggStats(hist_tmax, fut_tmax)
# mod_delta_tmax = agstats_tmax.mod_diff(save=False, dpath=save_path)

# Load data
mod_delta_tmin = np.load(data_path + "model_diffs_tasmin_rcp45_2099.npy")
mod_delta_tmax = np.load(data_path + "/model_diffs_tasmax_rcp45_2099.npy")
mod_delta_tavg = ms.temp_average(mod_delta_tmin, mod_delta_tmax,
                                 save=False, dpath=save_path)

# Calculate clim div stats
shpfile = gis_path + "MT_CLIM_DIVISIONS"
tavg_zstats = ms.zstats(shpfile, mod_delta_tavg.mean(axis=0), cd_list=True)


# Plot
cd_plot = mplt.clim_divs(shpfile + '.shp')
cd_plot.temp_plot(tavg_zstats, "Temperature Change (C)")
