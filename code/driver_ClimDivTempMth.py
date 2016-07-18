import macastats as ms
import macaplots as mplt
import numpy as np

# data_path = "/data/maca_mt/"
data_path = "/home/nick/workspace/mca/"
# data_path = '/media/nick/Seagate Backup Plus Drive/data/MCA_data/'
gis_path = "/home/nick/MEGA/workspace/mca/data/shapefiles/"
# gis_path = "/home/nick/workspace/shapefiles/"

# Load data
mod_delta_tmax_mth = np.load(data_path + "model_diffs_mth_test.npy")
mod_list = ['IPSL-CM5B-LR', 'MIROC-ESM-CHEM']
ms.find_range(mod_delta_tmax_mth, mod_list)

mdtm = mod_delta_tmax_mth.mean(axis=0)
zs = ms.zstats(gis_path+'MT_CLIM_DIVISIONS', mdtm)
title = "Change in Monthly Temperature"
mplt.clim_div_temp_grid(zs, stat='median', title=title)