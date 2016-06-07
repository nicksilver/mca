import macaproc as mp
import macastats as ms
import macaplots as mplt
import numpy as np
import pickle

data_path = "/data/maca_mt/"
# save_path = "/home/nick/MEGA/business/MCO/MCA/data/"
save_path = "/home/nick/workspace/data/"
# data_path = "/media/nick/Seagate Backup Plus Drive/data/MCA_data/"
mod_list = None  # ['IPSL-CM5B-LR', 'MIROC-ESM-CHEM']

# Create list of files for historical data
hist_rcp = mp.select_rcp(data_path, 'historical')
hist_pr = mp.select_mod(hist_rcp, var='pr', mod=mod_list)
hist_tmin = mp.select_mod(hist_rcp, var='tasmin', mod=mod_list)
hist_tmax = mp.select_mod(hist_rcp, var='tasmax', mod=mod_list)

# Create list of files for future data
fut_rcp = mp.select_rcp(data_path, 'rcp85')
fut_pr = mp.select_mod(fut_rcp, var='pr', mod=mod_list, yr='2070_2099')
fut_tmin = mp.select_mod(fut_rcp, var='tasmin', mod=mod_list, yr='2070_2099')
fut_tmax = mp.select_mod(fut_rcp, var='tasmax', mod=mod_list, yr='2070_2099')

# Calculate differences between historical and future for each model
agstats_pr = ms.AggStats(hist_pr, fut_pr)
mod_delta_pr = agstats_pr.mod_diff(save=True, dpath=save_path)
mod_names = agstats_pr.mod_names()

agstats_tmin = ms.AggStats(hist_tmin, fut_tmin)
mod_delta_tmin = agstats_tmin.mod_diff(save=True, dpath=save_path)

agstats_tmax = ms.AggStats(hist_tmax, fut_tmax)
mod_delta_tmax = agstats_tmax.mod_diff(save=True, dpath=save_path)

mod_delta_tavg = ms.temp_average(mod_delta_tmin, mod_delta_tmax, 
                                 save=True, dpath=save_path)

# Plot differences
mod_delta_tavg_85 = np.load(save_path + "model_diffs_tavg_rcp85_2099.npy")
mod_delta_tavg_45 = np.load(save_path + "model_diffs_tavg_rcp45_2099.npy")
mod_delta_pr_85 = np.load(save_path + "model_diffs_pr_rcp85_2099.npy")
mod_delta_pr_45 = np.load(save_path + "model_diffs_pr_rcp45_2099.npy")
mod_names = pickle.load(open(save_path + "model_list.p", 'rb'))

mplt.mod_diff_comp(mod_delta_pr_45, mod_delta_tavg_45,
                   precip2=mod_delta_pr_85, temp2=mod_delta_tavg_85,
                   title="CMIP5 Ensemble for 2070-2099",
                   mod_names=mod_names, annotate=True)

