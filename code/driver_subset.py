from netCDF4 import Dataset
import numpy as np
import glob
import macaProc as mp
from netcdftime import utime
import datetime

datapath = "/data/maca_wusa/"
varname = "precipitation"
fns = glob.glob(datapath + "*_pr_*.nc")
bottomleft = [360 + -116.6, 44]
topright = [360. + -103.5, 49.5] 

fn = fns[0]
data = Dataset(fn, 'a')
v = data.variables

# Clip historical data to 1970-2000
if fn[-47:-35] == '_historical_':
    start_time = datetime.datetime(1970, 12, 31, 0, 0)
    end_time = datetime.datetime(2000, 12, 31, 0, 0)
    tcon = utime('days since 1900-01-01')
    t_list = tcon.num2date(v['time'][:])
    bool_time = (t_list > start_time) & (t_list < end_time)
else:
    bool_time = np.ones(v['time'][:].shape).astype(bool)

# Find index of nearest gridcell to bounding box
l = mp.find_nearest(v['lon'][:], bottomleft[0])
b = mp.find_nearest(v['lat'][:], bottomleft[1])
r = mp.find_nearest(v['lon'][:], topright[0])
t = mp.find_nearest(v['lat'][:], topright[1])

# Clip vars to bounding box
lons = v['lon'][l:r]
lats = v['lat'][b:t]
var = v[varname][bool_time, b:t, l:r]
time = v['time'][bool_time]

