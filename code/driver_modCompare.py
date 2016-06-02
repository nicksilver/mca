import macaProc as mp
import macaStats as ms

# data_path = "/data/maca_mt/"
data_path = "/media/nick/Seagate Backup Plus Drive/data/MCA_data/"
mod_list = ['IPSL-CM5B-LR', 'MIROC-ESM-CHEM']

hist = mp.select_rcp(data_path, 'historical')
hist_pr = mp.select_mod(hist, var='pr', mod=mod_list)

fut = mp.select_rcp(data_path, 'rcp45')
fut_pr = mp.select_mod(fut, var='pr', mod=mod_list, yr='2070_2099')

ModDiff = ms.AggStats(hist_pr, fut_pr)
lat, lon = ModDiff.get_latlon()
mod_delta = ModDiff.mod_diff()
