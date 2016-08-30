import macaproc as mp
import macastats as ms

data_path = "/data/maca_mt/"
# data_path = "/home/nick/workspace/data/"
# data_path = '/media/nick/Seagate Backup Plus Drive/data/MCA_data/'
# gis_path = "/home/nick/MEGA/workspace/mca/data/shapefiles/"
gis_path = "/home/nick/workspace/shapefiles/"
# save_path = "/home/nick/workspace/data/monthly/"
save_path = "./"

mod_list = None  # ['IPSL-CM5B-LR', 'MIROC-ESM-CHEM']
rcp_scen = ["rcp45", "rcp85"]
time_range = ["2040_2069", "2070_2099"]

# Create list of files for historical data
hist_rcp = mp.select_rcp(data_path, 'historical')
hist_tmin = mp.select_mod(hist_rcp, var='tasmin', mod=mod_list)
hist_tmax = mp.select_mod(hist_rcp, var='tasmax', mod=mod_list)
hist_pr = mp.select_mod(hist_rcp, var='pr', mod=mod_list)

rcp = rcp_scen[1]
tr = time_range[1]
for rcp in rcp_scen:
    for tr in time_range:

        # Create list of files for future data
        fut_rcp = mp.select_rcp(data_path, rcp)
        fut_tmin = mp.select_mod(fut_rcp, var='tasmin', mod=mod_list, yr=tr)
        fut_tmax = mp.select_mod(fut_rcp, var='tasmax', mod=mod_list, yr=tr)
        fut_pr = mp.select_mod(fut_rcp, var='pr', mod=mod_list, yr=tr)

        ######### Annual Ensemble Variability ############
        # aggstats_pr = ms.MacaPrecip(hist_pr, fut_pr)
        # var_pr = aggstats_pr.ens_diff_ann(save=True, dpath=save_path,
        #                                   stat='std', ctype='absolute')

        ######### Annual Ensemble GDD #####################
        # aggstats_t = ms.MacaTemp(hist_tmin, fut_tmin, hist_tmax, fut_tmax)
        # gdd_diff = aggstats_t.ens_diff_ann(save=True, dpath=save_path, stat='gdd')

        ######### Annual Ensemble FFD #####################
        # aggstats_t = ms.MacaTemp(hist_tmin, fut_tmin)
        # ffd_diff = aggstats_t.ens_diff_ann(save=False, dpath=save_path, stat='ffd')

        ############ Beetle Kill Threshold ################
        # aggstats = ms.MacaTemp(hist_tmin, fut_tmin)
        # b_arr = aggstats.beetle_mon(timeperiod='future', save=True, dpath=save_path)

        ############ Days above 90F ########################
        # aggstats = ms.MacaTemp(hist_tmax, fut_tmax)
        # t90_arr = aggstats.ens_diff_ann(save=True, dpath=save_path, stat='tmax90F')

        ############# Monthly Precip Percentage ############
        # aggstats = ms.MacaPrecip(hist_pr, fut_pr)
        # pr_perc_arr = aggstats.ens_diff_mon(save=True, dpath=save_path, ctype='percent')

        ############ Annual Consecutive Dry Days ############

