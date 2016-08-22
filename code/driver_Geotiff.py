import macaproc as mp
import macastats as ms
import numpy as np

file_path = "/data/maca_mt/"
data_path = "/home/nick/workspace/data/monthly/"
mod_list = ['IPSL-CM5B-LR']

# Get lists of files
hist_rcp = mp.select_rcp(file_path, 'historical')
hist_pr = mp.select_mod(hist_rcp, var='pr', mod=mod_list)
fut_rcp = mp.select_rcp(file_path, "rcp45")
fut_pr = mp.select_mod(fut_rcp, var='pr', mod=mod_list, yr="2040_2069")

mstats = ms.MacaStats(hist_pr, fut_pr)
lat, lon = mstats.get_latlon()

beetle = np.load(data_path+"beetle_thresholds")


