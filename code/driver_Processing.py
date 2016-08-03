import macaproc as mp
import macastats as ms

data_path = "/data/maca_mt/"
# data_path = "/home/nick/workspace/data/"
# data_path = '/media/nick/Seagate Backup Plus Drive/data/MCA_data/'
# gis_path = "/home/nick/MEGA/workspace/mca/data/shapefiles/"
gis_path = "/home/nick/workspace/shapefiles/"
save_path = "/home/nick/workspace/data/annually/"

mod_list = None  # ['IPSL-CM5B-LR', 'MIROC-ESM-CHEM']
rcp_scen = ["rcp45", "rcp85"]
time_range = ["2040_2069", "2070_2099"]

# Create list of files for historical data
hist_rcp = mp.select_rcp(data_path, 'historical')
hist_tmin = mp.select_mod(hist_rcp, var='tasmin', mod=mod_list)
hist_tmax = mp.select_mod(hist_rcp, var='tasmax', mod=mod_list)
hist_pr = mp.select_mod(hist_rcp, var='pr', mod=mod_list)

# rcp = rcp_scen[1]
# tr = time_range[1]
for rcp in rcp_scen:
    for tr in time_range:

        # Create list of files for future data
        fut_rcp = mp.select_rcp(data_path, rcp)
        fut_tmin = mp.select_mod(fut_rcp, var='tasmin', mod=mod_list, yr=tr)
        fut_tmax = mp.select_mod(fut_rcp, var='tasmax', mod=mod_list, yr=tr)
        fut_pr = mp.select_mod(fut_rcp, var='pr', mod=mod_list, yr=tr)

        ######## Annual Ensemble Differences ##########
        # agstats_tmin = ms.MacaStats(hist_tmin, fut_tmin)
        # mod_delta_tmin = agstats_tmin.ens_diff_ann(save=False, dpath=save_path)
        # agstats_tmax = ms.MacaStats(hist_tmax, fut_tmax)
        # mod_delta_tmax = agstats_tmax.ens_diff_ann(save=False, dpath=save_path)
        # agstats_pr = ms.MacaStats(hist_pr, fut_pr)
        # mod_delta_pr = agstats_pr.ens_diff_ann(save=True, dpath="./")

        ######## Monthly Ensemble Differences ###########
        # aggstats_tmax = ms.MacaStats(hist_tmax, fut_tmax)
        # mod_delta_tmax_mth = aggstats_tmax.ens_diff_mon(save=True, dpath=save_path)
        # aggstats_tmin = ms.MacaStats(hist_tmin, fut_tmin)
        # mod_delta_tmin_mth = aggstats_tmin.ens_diff_mon(save=True, dpath=save_path)
        # aggstats_pr = ms.MacaStats(hist_pr, fut_pr)
        # mod_delta_pr_mth = aggstats_pr.ens_diff_mon(save=True, dpath="./")
        # tmin = save_path+"model_diffs_mth_tasmin_rcp85_2069.npy"
        # tmax = save_path+"model_diffs_mth_tasmax_rcp85_2069.npy"
        # tavg = save_path+"model_diffs_mth_tavg_rcp85_2069.npy"
        # temp_avg = ms.temp_average(tmin, tmax, save=True, dpath=tavg)

        ######### Annual Ensemble Variability ############
        # aggstats_pr = ms.MacaPrecip(hist_pr, fut_pr)
        # var_pr = aggstats_pr.ens_diff_ann(save=True, dpath=save_path,
        #                                   stat='std', ctype='absolute')

        ######### Annual Ensemble GDD #####################
        aggstats_t = ms.MacaTemp(hist_tmin, fut_tmin, hist_tmax, fut_tmax)
        gdd_diff = aggstats_t.ens_diff_ann(save=True, dpath=save_path, stat='gdd')

