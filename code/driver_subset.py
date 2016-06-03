from netCDF4 import Dataset
import numpy as np
import glob
import macaproc as mp
from netcdftime import utime
import datetime

src_path = "/data/maca_wusa/"
dst_path = "/data/maca_mt/"
varname = "air_temperature"
fns = glob.glob(src_path + "*_tasmax_*.nc")
bottomleft = [360 + -116.6, 44]
topright = [360. + -103.5, 49.5] 

# fn = fns[0]
for fn in fns:
    data = Dataset(fn, 'r')
    v = data.variables
    dst_name = fn[16:-3] + "_MT.nc"

    # Clip historical data to 1970-2000
    if fn[-47:-35] == '_historical_':
        start_time = datetime.datetime(1970, 12, 31, 0, 0)
        end_time = datetime.datetime(2000, 12, 31, 0, 0)
        tcon = utime('days since 1900-01-01')
        t_list = tcon.num2date(v['time'][:])
        bool_time = (t_list > start_time) & (t_list < end_time)
        dst_name = dst_name.replace("1950_2005", "1971_2000")
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
    print "Done clipping..."

    # Create netcdf file
    dst = Dataset(dst_path + dst_name, 'w')

    # Copy global attributes
    for att in data.ncattrs():
        setattr(dst, att, getattr(data, att))

    geo_bounds = "POLYGON((-116.6056 43.9794, -116.6056 49.3127, -103.5225 43.9794, -103.5225 49.3127))"
    setattr(dst, "geospatial_bounds", geo_bounds)
    setattr(dst, "geospatial_lat_min", "43.9794")
    setattr(dst, "geospatial_lat_max", "49.3127")
    setattr(dst, "geospatial_lon_min", "-116.6056")
    setattr(dst, "geospatial_lon_max", "-103.5225")

    if fn[-47:-35] == '_historical_':
        setattr(dst, "time_coverage_start", "1971-01-01T00:0")
        setattr(dst, "time_coverage_end", "2000-12-31T00:0")

    # Create dimensions
    dst.createDimension('time', size=None)
    dst.createDimension('lon', size=len(lons))
    dst.createDimension('lat', size=len(lats))
    dst.createDimension('crs', size=1)

    # Copy variables
    for v_name, varin in data.variables.iteritems():
        outVar = dst.createVariable(v_name, varin.datatype, varin.dimensions)
        outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
        if v_name == "crs":
            outVar[:] = varin[:]

    dst.variables['lat'][:] = lats
    dst.variables['lon'][:] = lons
    dst.variables['time'][:] = time
    dst.variables[varname][:] = var

    dst.close()
    data.close()
    print "Done processing " + dst_name


