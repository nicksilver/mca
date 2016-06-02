import macaProc as mp
import macaStats as ms

data_path = "/data/maca_mt/"

hist = mp.select_rcp(data_path, 'historical')
hist_pr = mp.select_str(hist, var='pr', mod=['HadGEM2-ES365', 'MIROC-ESM'])

fut = mp.select_rcp(data_path, 'rcp45')
fut_pr = mp.select_str(fut, ['_pr_', '2040_2069', 'HadGEM2-ES365', 'MIROC-ESM'])


ModDiff = ms.AggStats(hist_pr, fut_pr)
lat, lon = ModDiff.get_latlon()
mod_names = ModDiff.mod_diff()
