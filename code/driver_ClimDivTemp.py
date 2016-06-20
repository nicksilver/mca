import macaproc as mp
import macastats as ms
import macaplots as mplt
import numpy as np


# data_path = "/data/maca_mt/"
gis_path = "/home/nick/workspace/shapefiles/"
save_path = None
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
mod_delta_tmin = np.load("/home/nick/workspace/data/model_diffs_tasmin_rcp45_2099.npy")

# agstats_tmax = ms.AggStats(hist_tmax, fut_tmax)
# mod_delta_tmax = agstats_tmax.mod_diff(save=False, dpath=save_path)
mod_delta_tmax = np.load("/home/nick/workspace/data/model_diffs_tasmax_rcp45_2099.npy")


mod_delta_tavg = ms.temp_average(mod_delta_tmin, mod_delta_tmax,
                                 save=False, dpath=save_path)


# Calculate clim div stats
tavg_zstats = ms.zstats(gis_path + "MT_CLIM_DIVISIONS",
                        mod_delta_tavg.mean(axis=0),
                        cd_list=True)


# Plot
