import macastats as ms
import macaplots as mplt
import numpy as np
import macaproc as mp
import glob

# Data paths
# data_path = "/home/nick/workspace/data/monthly/"
data_path = "/home/nick/MEGA/workspace/mca/data/model_diffs/monthly/"
gis_path = "/home/nick/MEGA/workspace/mca/data/shapefiles/"
# gis_path = "/home/nick/workspace/shapefiles/"
save_path = "/home/nick/MEGA/workspace/mca/figures/monthly_temp/"

# Specify files to load
mod_list = mp.load_pickle(data_path+"model_list.p")
tdict = {
    '45': 'RCP 4.5',
    '85': 'RCP 8.5',
    '2069': ' 2040-2069',
    '2099': ' 2070-2099'
}
syms = ['*_tavg_*', '*_tasmin_*', '*_tasmax_*']
tbeg = ['Change in Monthly Average Temp (F) ',
        'Change in Monthly Min Temp (F) ',
        'Change in Monthly Max Temp (F) ']

# Loop through specified files and plot
for i, sym in enumerate(syms):
    print("Beginning to process " + sym[2:-2])
    fnames = glob.glob(data_path + sym)
    for f in fnames:
        rcp = f[-11:-9]
        rcpt = tdict[rcp]
        drange = f[-8:-4]
        dranget = tdict[drange]
        mod_delta = np.load(f)

        # Plot data
        shp_name = 'MT_CLIM_DIVISIONS'
        ens_mean = mod_delta.mean(axis=0)
        zs = ms.zstats(gis_path+shp_name, ens_mean, precip=False, metric=False)
        zs_range = ms.zstats_range(mod_delta, gis_path+shp_name, zs, mod_list,
                                   precip=False, metric=False)
        title = tbeg[i]+rcpt+dranget
        fname = sym[2:-2]+"_rcp"+rcp+"_"+drange+"_mthly.html"
        mplt.clim_div_grid(zs, stat='median', title=title, browser=False,
                           r_data=zs_range, var='temp',
                           save_path=save_path+fname)

