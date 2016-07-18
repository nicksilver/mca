import macastats as ms
import macaplots as mplt
import numpy as np
import macaproc as mp

# data_path = "/data/maca_mt/"
data_path = "/home/nick/workspace/data/monthly/"
# data_path = '/media/nick/Seagate Backup Plus Drive/data/MCA_data/'
# gis_path = "/home/nick/MEGA/workspace/mca/data/shapefiles/"
gis_path = "/home/nick/workspace/shapefiles/"
save_path = "/home/nick/workspace/figures/"

# Load data
mod_delta_tmax_mth = np.load(data_path+"model_diffs_mth_tasmax_rcp85_2069.npy")
mod_list = mp.load_pickle(data_path+"model_list.p")

# Plot data
data = mod_delta_tmax_mth.mean(axis=0)
zs = ms.zstats(gis_path+'MT_CLIM_DIVISIONS', data)
zs_range = ms.zstats_range(mod_delta_tmax_mth, gis_path+"MT_CLIM_DIVISIONS", mod_list)
title = "Change in Monthly Max. Temp. RCP 8.5 2040-2069"
mplt.clim_div_temp_grid(zs, stat='median', title=title, r_data=zs_range,
                        save_path=save_path+"temp_monthly_rcp85_2069.html")

