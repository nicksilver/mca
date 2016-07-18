import macaplots as mplt
import numpy as np
import pickle
import macaproc as mp
import macastats as ms

# Set paths
# data_path = "/data/maca_mt/"
# save_path = "/home/nick/workspace/data/"
data_path = "/media/nick/Seagate Backup Plus Drive/data/MCA_data/"
save_path = "/home/nick/MEGA/workspace/mca/data/model_diffs/"

# Load data
mod_delta_tavg_85 = np.load(save_path + "model_diffs_tavg_rcp85_2069.npy")
mod_delta_tavg_45 = np.load(save_path + "model_diffs_tavg_rcp45_2069.npy")
mod_delta_pr_85 = np.load(save_path + "model_diffs_pr_rcp85_2069.npy")
mod_delta_pr_45 = np.load(save_path + "model_diffs_pr_rcp45_2069.npy")
mod_names = pickle.load(open(save_path + "model_list.p", 'rb'))

# Plot
filepath = "cmip5_mt_model_comparison_2069.html"

mplt.mod_diff_comp_bok(mod_delta_pr_45, mod_delta_tavg_45, filepath=filepath,
                       precip2=mod_delta_pr_85, temp2=mod_delta_tavg_85,
                       title="CMIP5 Ensemble for 2070-2069",
                       mod_names=mod_names)
