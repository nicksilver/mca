import macaProc as mp

data_path = "/data/maca_mt/"

hist = mp.selectRCP(data_path, 'historical')
hist_pr = mp.selectStr(hist, '_pr_')
hist_tmin = mp.selectStr(hist, '_tasmin_')
hist_tmax = mp.selectStr(hist, '_tasmax_')

