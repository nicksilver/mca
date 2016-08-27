import macastats as ms
import numpy as np
import macaproc as mp
import glob
import pandas as pd

# Data paths
data_path = "/home/nick/MEGA/workspace/mca/data/processed_metrics/monthly/"
gis_path = "/home/nick/MEGA/workspace/mca/gis/WaterSector/"
save_path = "./"
shp_name = 'Marias_Chester2.shp'

# Specify files to load
mod_list = mp.load_pickle(data_path+"model_list.p")
tdict = {
    '45': 'RCP 4.5',
    '85': 'RCP 8.5',
    '2069': ' 2040-2069',
    '2099': ' 2070-2099'
}
var = 'tavg'

fnames = glob.glob(data_path + '*_' + var + '_*')
# f = fnames[0]
for f in fnames:
    rcp = f[-14:-9]
    drange = f[-8:-4]
    mod_delta = np.load(f)
    zs = ms.zs_h2o_range(gis_path + shp_name, mod_delta, mod_list)
    name = shp_name[:-5] + "_" + var + "_" + rcp + "_" + drange + ".csv"
    zs.to_csv("./" + name)
    print "File " + name + " is complete!"
