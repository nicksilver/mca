import macaproc as mp
import macastats as ms
import macaplots as mplt
import numpy as np

data_path = "/data/maca_mt/"
# data_path = "/home/nick/workspace/data/"
# data_path = '/media/nick/Seagate Backup Plus Drive/data/MCA_data/'
# gis_path = "/home/nick/MEGA/workspace/mca/data/shapefiles/"
gis_path = "/home/nick/workspace/shapefiles/"
save_path = None

mod_list = ['IPSL-CM5B-LR', 'MIROC-ESM-CHEM']
rcp_scen = "rcp85"

# Create list of files for historical data
hist_rcp = mp.select_rcp(data_path, 'historical')
hist_tmin = mp.select_mod(hist_rcp, var='tasmin', mod=mod_list)
hist_tmax = mp.select_mod(hist_rcp, var='tasmax', mod=mod_list)

# Create list of files for future data
fut_rcp = mp.select_rcp(data_path, rcp_scen)
fut_tmin = mp.select_mod(fut_rcp, var='tasmin', mod=mod_list, yr='2040_2069')
fut_tmax = mp.select_mod(fut_rcp, var='tasmax', mod=mod_list, yr='2040_2069')

######## Annual Ensemble Differences ##########
# agstats_tmin = ms.AggStats(hist_tmin, fut_tmin)
# mod_delta_tmin = agstats_tmin.mod_diff_ann(save=False, dpath=save_path)
# agstats_tmax = ms.AggStats(hist_tmax, fut_tmax)
# mod_delta_tmax = agstats_tmax.mod_diff_ann(save=False, dpath=save_path)

######## Monthly Ensemble Differences ###########
aggstats_tmax = ms.AggStats(hist_tmax, fut_tmax)
mod_delta_tmax_mth = aggstats_tmax.mod_diff_mon(save=False)

# Plot
mdtm = mod_delta_tmax_mth.mean(axis=0)
m = mdtm[0, :, :]
zs = ms.zstats(gis_path+'MT_CLIM_DIVISIONS', m)
