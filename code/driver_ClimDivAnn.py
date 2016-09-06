import macastats as ms
import macaplots as mplt
import numpy as np
import macaproc as mp
import glob

gis_path = "/home/nick/workspace/shapefiles/"
# gis_path = "/home/nick/MEGA/workspace/mca/data/shapefiles/"
# data_path = "/home/nick/workspace/data/annually/"
# data_path = "/home/nick/MEGA/workspace/mca/data/processed_metrics/annually/"
data_path = "/home/nick/workspace/data/annually/"
save_path = "./"

# Specify files to load
mod_list = mp.load_pickle(data_path+"model_list.p")
tdict = {
    '45': 'RCP 4.5',
    '85': 'RCP 8.5',
    '2069': ' 2040-2069',
    '2099': ' 2070-2099'
}
# syms = ['*_tavg_*', '*_tasmin_*', '*_tasmax_*']
# syms = ['*_diffs_pr_*']
# syms = ['*_perc_pr_*', '*_vars_perc_pr_*']
# syms = ['*_vars_pr_*']
# syms = ['*_tmax90F_*']
# syms = ['*_consecDD_*']
syms = ['*_consecWD_*']

# tbeg = ['Change in Annual Average Temp (F) ',
#         'Change in Annual Min Temp (F) ',
#         'Change in Annual Max Temp (F) ']
# tbeg = ['Change in Annual Accumulated Precip. (in.) ']
# tbeg = ['Percent Change of Annual Precipitation (%) ',
#         'Percent Change of Precipitation Interannual Variability  (%) ']
# tbeg = ['Change in Precipitation Interannual Variability (in.) ']
# tbeg = ['Change in Annual Number of Days above 90F ']
# tbeg = ['Change in Length of Dry Spells ']
tbeg = ['Change in Length of Wet Spells ']

# Loop through specified files and plot
# i = 0
# sym = syms[0]
# f = fnames[0]
for i, sym in enumerate(syms):
    print("Beginning to process " + sym[2:-2])
    fnames = glob.glob(data_path + sym)
    for f in fnames:
        rcp = f[-11:-9]
        rcpt = tdict[rcp]
        drange = f[-8:-4]
        dranget = tdict[drange]
        mod_delta = np.load(f)

        # Calculate clim div stats
        shpfile = gis_path + "MT_CLIM_DIVISIONS"
        zs = ms.zstats(shpfile, mod_delta.mean(axis=0), units='metric', precip=True)
        zs_range = ms.zstats_range(mod_delta, shpfile, zs, mod_list, precip=True,
                                   units='metric')

        # Plot
        title = tbeg[i]+rcpt+dranget
        fname = sym[2:-2]+"_rcp"+rcp+"_"+drange+"_ann.html"
        mplt.clim_div_ann(shpfile, zs, zs_range, title=title, var='precip',
                          savepath=save_path+fname, browser=False)
