import macaProc as mp
import macaStats as ms

data_path = "/data/maca_mt/"

hist = mp.selectRCP(data_path, 'historical')
hist_pr = mp.selectStr(hist, '_pr_')

fut = mp.selectRCP(data_path, 'rcp45')
fut_pr = mp.selectStr(fut, ['_pr_', '2040_2069'])


ModDiff = ms.AggStats(hist_pr, fut_pr)
lat, lon = ModDiff.get_latlon()
mod_names = ModDiff.mod_diff()
