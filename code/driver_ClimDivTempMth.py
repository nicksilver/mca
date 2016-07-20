import macastats as ms
import macaplots as mplt
import numpy as np
import macaproc as mp

# data_path = "/home/nick/workspace/data/monthly/"
data_path = "/home/nick/MEGA/workspace/mca/data/model_diffs/monthly/"
gis_path = "/home/nick/MEGA/workspace/mca/data/shapefiles/"
# gis_path = "/home/nick/workspace/shapefiles/"
save_path = "/home/nick/MEGA/workspace/mca/figures/monthly_temp/"
mod_list = mp.load_pickle(data_path+"model_list.p")

# Load data
mod_delta_tmax_mth = np.load(data_path+"model_diffs_mth_pr_rcp45_2069.npy")

# Plot data
data = mod_delta_tmax_mth.mean(axis=0)
zs = ms.zstats(gis_path+'MT_CLIM_DIVISIONS', data)
zs_range = ms.zstats_range(mod_delta_tmax_mth, gis_path+"MT_CLIM_DIVISIONS", mod_list)
title = "Change in Monthly Precipitation (mm/day) RCP 4.5 2040-2069"
mplt.clim_div_grid(zs, stat='median', title=title, r_data=zs_range, var='precip')

