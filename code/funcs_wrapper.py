import xarray as xr
import xesmf as xe
import numpy as np
import pandas as pd
import os
import re
import string
import glob
import warnings
from matplotlib import pyplot as plt
import cmocean

from funcs_plot import (figure_climatology,figure_iv_boxplots,figure_scatter,
                        figure_seasmaps,figure_seasmaps_multivar,
                        figure_pr_mse_trends,
                        wrapper_prhsat_figure)
from funcs_support import (get_params,subset_to_srat,utility_print,area_mean)
from funcs_process import (create_season_mask)
from funcs_load import (load_raw)

dir_list = get_params()

def wrapper_figure3(var='hdiff',
                    mod_a='MERRA2',mod_p='CHIRPS',
                    kind='dunning_local',
                    subset_params = {'lat':slice(-3,12.5),'lon':slice(32,55)},
                    plev=650,
                    titles = ['Long rains','Short rains','Long - short rains'],
                    save_fig = False,
                    output_fn = None):

    """ Precipitation by h_s-h^* values figure
    
    Parameters
    -------------
    var : str, by default 'hdiff'
        variable to bin 
        
    mod_a : str, by default 'MERRA2'
        which data product to use for variable `var`
        
    mod_p : str, by defulat 'CHIRPS'
        which data product to use for rainfall
        
    kind : str, by default 'dunning_local'
        how to define seasons: 
            'dunning_local': by whatever season a particular 
                             grid cell is in
            'dunning':       by the average seasonal onset/
                             demise across the double-peaked 
                             region
            'month':         by MAM / OND
            'month_alt':     by MAM / SON
            
    subset_params : dict, by default GHA 
        geograhpic subset parameters
    
    plev : float, by default 650
        what pressure level to use `var` at
        if None, variable `var` is not subset by plev
        
    save_fig : bool, by default False
    
    output_fn : str, by default None
    
    """

    #--------------- Setup ---------------
    # Piped into vardict in the figure functions
    vardict = {'hist':var,'cond':'pr'}
    
    stats_suffix='19810101-20211231_Africa'

    #--------------- Load data ---------------
    # Load hdiff and P data
    if plev is not None:
        subset_params_var = {**subset_params,'plev':[plev]}
    dss = {var:load_raw(mod_a+'/'+var+'_day*_HoA.nc',search_dir=dir_list['proc'],
                            subset_params=subset_params_var,
                            show_filenames=False),
           'pr':load_raw(mod_p+'/pr_day*',subset_params=subset_params,
                         show_filenames=False)}

    # Regrid hdiff to precip grid
    rgrd = xe.Regridder(dss[var],dss['pr'],method='bilinear')
    dss[var] = rgrd(dss[var])

    # Make sure == 0 are nan (xesmf changes nans to 0s upon regridding)
    dss[var] = dss[var].where(dss[var]!=0)

    # Merge hdiff and P data
    dss = xr.merge([v for k,v in dss.items()])

    # Subset to double-peaked region
    dss = subset_to_srat(dss)

    #--------------- Get seasonal booleans ---------------
    print('calculating or loading seasonal booleans...')
    # This rather cludgily creates booleans for all four seasons
    # to reuse some existing code
    seasons = ['long_dry','long_rains','short_dry','short_rains']
    #-------- Load seasonal stats
    # Load seasonal stats
    stats_fn = dir_list['proc']+mod_p+'/pr_ann_'+mod_p+'_historical_seasstats_dunning_'+stats_suffix+'.nc'
    stats = xr.open_dataset(stats_fn)
    # NaN out areas outside double-peaked region
    stats = subset_to_srat(stats,drop=True)
    # Remove singleton 'method' dimension
    stats = stats.isel(method=0,drop=True)
    # Make demise onset + duration
    stats['demise'] = stats['onset']+stats['duration']

    # Get timeframe of the stats ... 
    timeframe = pd.date_range(str(stats.year.min().values)+'-01-01',
                              str(stats.year.max().values)+'-12-31')

    if kind == 'dunning_local':
        #-------- Locally-defined seasons
        bool_fn = ('/dx01/kschwarz/aux_data/seasidxs_day_'+mod_p+'_historical_'+
                     re.sub('\-','',str(dss.time[0].values)[0:10])+'-'+
                     re.sub('\-','',str(dss.time.max().values)[0:10])+'_HoA.nc')

        if not os.path.exists(bool_fn):
            ts = xr.zeros_like(dss.pr).expand_dims({'season':stats['season']})
            yrs = dss.time.dt.year
            doys = dss.time.dt.dayofyear

            # Make sure ts lat/lon order matches up with stats lat/lon order
            ts.transpose('season','time',*[dim for dim in stats.dims if dim in ['lat','lon']])
            # Stack ts and stats
            ts = ts.stack(loc=('lat','lon')).copy()
            stats_tmp = stats.stack(loc=('lat','lon')).copy()

            for yr in tqdm(stats_tmp.year):
                yr_idx0 = np.where(yrs==yr)[0][0]
                for seas_idx in np.arange(0,2):
                    for loc_idx in np.where(~np.isnan(stats_tmp.isel(season=seas_idx).onset).all('year'))[0]:
                        ts[seas_idx,int(yr_idx0+stats_tmp.sel(year=yr).isel(season=seas_idx,loc=loc_idx).onset.values):
                           int(yr_idx0+stats_tmp.sel(year=yr).isel(season=seas_idx,loc=loc_idx).onset.values + 
                               stats_tmp.sel(year=yr).isel(season=seas_idx,loc=loc_idx).duration.values),loc_idx] = True

            ts = ts.unstack().copy()

            ts = ts.rename({'hdiff':'idx'})
            # Make sure it's an integer (maybe should've been bool?)
            ts['idx'] = ts.idx.astype(int)

            ts.attrs['SOURCE'] = 'wrapper_figure3()'
            ts.attrs['DESCRIPTION'] = 'boolean for when a given grid-cell-day is in a given season'
            ts.attrs['STATS_SOURCE'] = stats_fn
            ts.to_netcdf(bool_fn)
            print(bool_fn+' saved!')
        else:
            print(bool_fn+' exists, loaded!')
            ts = xr.open_dataset(bool_fn)
    else:
        if kind == 'dunning':
            #-------- Regional average seasons
            # Create empty dataset
            ts = xr.Dataset({'ts':(('time','season'),np.zeros((len(timeframe),
                                                                          len(seasons))))},
                                        coords={'time':timeframe,
                                                        'season':seasons})
            
            # Calculate regional mean of the rainfall statistics 
            statsm = area_mean(stats)
            ts = create_season_mask(ts,statsm).rename({'ts':'idx'})

        elif kind == 'month':
            #-------- MAM/OND Months
            seas_idxs = [[1,3],[3,6],[6,10],[10,1]]
            
            # Create empty dataset
            ts = xr.Dataset({'idx':(('time','season'),np.zeros((len(timeframe),
                                                                          len(seasons))))},
                                        coords={'time':timeframe,
                                                'season':seasons})
            # Create boolean that's true when within a certain month
            for seas_idx in np.arange(0,len(seas_idxs)):
                if seas_idxs[seas_idx][0]>seas_idxs[seas_idx][1]:
                    ts['idx'][((ts.time.dt.month>=seas_idxs[seas_idx][0]) |
                              (ts.time.dt.month<seas_idxs[seas_idx][1])).values,
                                 seas_idx] = True

                else:
                    ts['idx'][((ts.time.dt.month>=seas_idxs[seas_idx][0]) &
                              (ts.time.dt.month<seas_idxs[seas_idx][1])).values,
                                 seas_idx] = True

        elif kind == 'month_alt':
            #-------- MAM/SON Months
            seas_idxs = [[12,3],[3,6],[6,9],[9,12]]

            # Create empty dataset
            ts = xr.Dataset({'idx':(('time','season'),np.zeros((len(timeframe),
                                                                          len(seasons))))},
                                        coords={'time':timeframe,
                                                        'season':seasons})
            # Create boolean that's true when within a certain month
            for seas_idx in np.arange(0,len(seas_idxs)):
                if seas_idxs[seas_idx][0]>seas_idxs[seas_idx][1]:
                    ts['idx'][((ts.time.dt.month>=seas_idxs[seas_idx][0]) |
                              (ts.time.dt.month<seas_idxs[seas_idx][1])).values,
                                 seas_idx] = True

                else:
                    ts['idx'][((ts.time.dt.month>=seas_idxs[seas_idx][0]) &
                              (ts.time.dt.month<seas_idxs[seas_idx][1])).values,
                                 seas_idx] = True

        #-------- Broadcast to full grid, and subset to just two seasons to fit
        # the requirements of the Figure 3 code 
        ts = xr.broadcast(ts,dss)[0].sel(season=['long_rains','short_rains'])

    
    #--------------- Plot ---------------
    print('plotting...')
    if vardict['hist'] == 'hdiffmax':
        xlabel_add = 'Max daily '
    else:
        xlabel_add = ''
    
    wrapper_prhsat_figure(dss,ts.idx,
                          xlabel_add = xlabel_add,
                          titles = titles,
                          save_fig = save_fig,
                          output_fn = output_fn)


def wrapper_figure4(mod_h='MERRA2',mod_p='CHIRPS',
                    suffix_h = 'HoA',
                    suffix_p = '19810101-20220630',
                    subset_params = {'lat':slice(-3,12.5),
                                     'lon':slice(32,55)},
                    plev = 650,
                    use_surf = False,
                    save_fig = False,
                    output_fn='',
                    ylims = [[0,0.6],[0,6]],
                    dir_list=dir_list):
    
    # Set output_fn
    output_fn = dir_list['figs']+'figure4_'+mod_h
    
    # Load unstable days boolean
    if use_surf:
        load_var = 'unstable-s'
    else:
        load_var = 'unstable'
        
    
    file_h = glob.glob(dir_list['proc']+mod_h+'/'+load_var+'_day_'+mod_h+'_*_'+suffix_h+'.nc')
        
    if len(file_h) > 1:
        warnings.warn('More than one file found for search: '+dir_list['proc']+mod_h+'/'+load_var+'_day_'+mod_h+'_*'+suffix_h+'.nc; '+
                      'the first one is used in the plot.')
    dsh = xr.open_dataset(file_h[0]).sel(**subset_params)
    dsh = dsh.sel(plev=plev)

    # Load precipitation 
    file_p = glob.glob(dir_list['raw']+mod_p+'/pr_day_'+mod_p+'_*'+suffix_p+'.nc')
    if len(file_p) > 1:
        warnings.warn('More than one file found for search: '+dir_list['raw']+mod_p+'/pr_day_'+mod_p+'_*'+suffix_p+'.nc; '+
                      'the first one is used in the plot.')
    dsp = (xr.open_dataset(file_p[0]).sel(**subset_params))
    
    # Get average of both over double-peaked region
    # (really should do area-weighting...) 
    dsh = area_mean(subset_to_srat(dsh.unstable))
    dsp = area_mean(subset_to_srat(dsp.pr))
    
    # Plot figure
    fig = plt.figure(figsize=(8,5))
    fig,ax = figure_climatology([dsh,dsp],
                               ylabel=[r'% of grid cells with $h_s-h^*>0$',
                                                r'Precipitation [mm/day]'],
                               colors = ['k','tab:blue'],
                               plot_axes = 'diff',
                               ylims = ylims,
                               title = r'Seasonal cycle of P, frequency of $(h_s-h^*>0)$',
                               fig = fig)
    
    # Print
    if save_fig:
        plt.tight_layout()
        utility_print(output_fn)
        
        
def wrapper_figure5(mod_h='MERRA2',mod_p='CHIRPS',
                    suffix_h = 'HoA',
                    suffix_p = '19810101-20220630',
                    subset_params = {'lat':slice(-3,12.5),
                                     'lon':slice(32,55)},
                    use_surf = False,
                    plev=650,plev_name = 'plev',
                    save_fig=False,output_fn='',
                    ):
    
    c_p = 1004.6
    L_v = 2.257e6
    
    var_params = {'hsat':{'sp_add':{plev_name:plev},'suffix':suffix_h,'model':mod_h,'source_dir':'proc'},
                  'pr':{'suffix':suffix_p,'model':mod_p,'source_dir':'raw'}}
    
    if use_surf:
        var_params = {**var_params,
                      'hsdiff':{'sp_add':{plev_name:plev},'suffix':suffix_h,'model':mod_h,'source_dir':'proc'},
                      'tas':{'suffix':suffix_h,'model':mod_h,'source_dir':'proc'},
                      'huss':{'suffix':suffix_h,'model':mod_h,'source_dir':'proc'}}
                      
        file_vars = {'hdiff':'hsdiff',
                     'ta':'tas',
                     'hus':'huss'}

    else:
        var_params = {**var_params,
                      'hdiff':{'sp_add':{plev_name:plev},'suffix':suffix_h,'model':mod_h,'source_dir':'proc'},
                      'ta-nsurf':{'suffix':suffix_h,'model':mod_h,'source_dir':'proc'},
                      'hus-nsurf':{'suffix':suffix_h,'model':mod_h,'source_dir':'proc'}}
                      
        file_vars = {'hdiff':'hdiff',
                     'ta':'ta-nsurf',
                     'hus':'hus-nsurf'}
               
    #----------- Load data -----------
    # Stats currently hardcoded, should also be outsourced to loading code
    stats = xr.open_dataset('/dx01/kschwarz/climate_proc/CHIRPS/pr_doyavg_CHIRPS_historical_seasstats_dunning_19810101-20221231_Africa.nc')
    stats = stats.isel(method=0)
    stats['demise'] = stats['onset'] + stats['duration']
    stats = subset_to_srat(stats).mean(('lat','lon'))
    
    # Variables
    dss = dict()
    for var in var_params:
        fn_search = (dir_list[var_params[var]['source_dir']]+var_params[var]['model']+'/'+
                     var+'_day_'+var_params[var]['model']+'_*_'+var_params[var]['suffix']+'.nc')
        fn = glob.glob(fn_search)
        if len(fn) > 1:
            warnings.warn('More than one file found for search: '+fn_search+'; '+
                      'the first one is used in the plot.')
        elif len(fn) == 0:
            raise KeyError('No files found for search: '+fn_search+' for variable '+var)

        if 'sp_add' in var_params[var]:
            sp_tmp = {**subset_params,**var_params[var]['sp_add']}
        else: 
            sp_tmp = subset_params
        dss[var] = xr.open_dataset(fn[0]).sel(**sp_tmp)


    # Get average of both over double-peaked region
    # (really should do area-weighting...) 
    dss = {var:area_mean(subset_to_srat(dss[var][re.split('\-',var)[0]])) for var in var_params}
    
    #----------- Plot -----------
    fig,axs = plt.subplots(2,1,figsize=(8,8))

    # Plot top panel
    fig,axs[0] = figure_climatology([dss[file_vars['hdiff']]/1000,dss['pr']],
                                    ylabel=[r'$h_s-h^*$ [kJ/kg]',
                                             r'Precipitation [mm/day]'],
                                    axv_shading={r'$\mathit{Gu}$'+'\nLong rains':[stats.isel(season=0).onset,stats.isel(season=0).demise],
                                                 r'$\mathit{Deyr}$'+'\nShort rains':[stats.isel(season=1).onset,stats.isel(season=1).demise]},
                                    axv_shading_color='tan',
                                    shade_quantiles = [0.1,0.9],
                                    colors = ['k','tab:blue'],
                                    plot_axes = 'diff',
                                    ylims = [[-17,2],[0,6]],
                                    title = r'Seasonal cycle of P, $h_s-h^*$',
                                    fig = fig,ax = axs[0])

    # Plot bottom panel
    fig,axs[1] = figure_climatology([dss[file_vars['hdiff']]/1000,
                                     dss[file_vars['ta']]*c_p/1000,
                                     dss[file_vars['hus']]*L_v/1000,
                                     dss['hsat']/1000],
                                    axv_shading={r'$\mathit{Gu}$'+'\nLong rains':[stats.isel(season=0).onset,stats.isel(season=0).demise],
                                                 r'$\mathit{Deyr}$'+'\nShort rains':[stats.isel(season=1).onset,stats.isel(season=1).demise]},
                                    axv_shading_color='tan',
                                    shade_quantiles = [0.1,0.9],
                                    ylabel=[r'$h_s-h^*$ component anomalies [kJ/kg]'],
                                    colors = ['k','tab:red','tab:blue','tab:green'],
                                    labels = [r"$(h_s-h^*)'$",r"$c_pT_s'$",r"$L_vq_s'$",r"$h^*$'"],
                                    styles = ['-','-.','--','-'],
                                    anomaly = True,
                                    plot_axes = 'same',
                                    title = r'Seasonal cycle of $h_s-h^*$ anomalies',
                                    fig = fig,ax = axs[1])
    
    # Subplot lettering
    for ax_idx in np.arange(0,len(axs)):
        if type(axs[ax_idx]) == list:
            axs[ax_idx][0].annotate(string.ascii_lowercase[ax_idx]+'.',
                            [0.01,1.01],xycoords='axes fraction',
                            va='bottom',ha='left',fontsize=13,fontweight='bold')
        else:
            axs[ax_idx].annotate(string.ascii_lowercase[ax_idx]+'.',
                            [0.01,1.01],xycoords='axes fraction',
                            va='bottom',ha='left',fontsize=13,fontweight='bold')

    
    #----------- Print -----------    
    if save_fig:
        plt.tight_layout()
        utility_print(output_fn)
        
def wrapper_figure6(mod='MERRA2',
                    suffix = 'HoA',
                    subset_params = {'lat':slice(-3,12.5),
                                     'lon':slice(32,55)},
                    mean_kind='dunning_local',
                    plev=650,
                    plev_name='plev',
                    save_fig=False,
                    output_fn=''):
    
    c_p = 1004.6
    L_v = 2.257e6

    file_vars = {'hsat':'hsat-anom',
                 'h':'h-nsurf-anom',
                 'ta':'ta-nsurf-anom',
                 'hus':'hus-nsurf-anom'}

    var_params = {file_vars['hsat']:{'sp_add':{plev_name:plev},'suffix':suffix,'model':mod,'source_dir':'proc'},
                  file_vars['h']:{'suffix':suffix,'model':mod,'source_dir':'proc'},
                  file_vars['ta']:{'suffix':suffix,'model':mod,'source_dir':'proc'},
                  file_vars['hus']:{'suffix':suffix,'model':mod,'source_dir':'proc'}}

    #----------- Load data -----------
    # Variables
    dss = dict()
    for var in var_params:
        fn_search = (dir_list[var_params[var]['source_dir']]+var_params[var]['model']+'/'+
                     var+'_seasavg_'+var_params[var]['model']+'_*_'+var_params[var]['suffix']+'.nc')
        fn = glob.glob(fn_search)
        if len(fn) > 1:
            warnings.warn('More than one file found for search: '+fn_search+'; '+
                      'the first one is used in the plot.')
        elif len(fn) == 0:
            raise KeyError('No files found for search: '+fn_search+' for variable '+var)

        if 'sp_add' in var_params[var]:
            sp_tmp = {**subset_params,**var_params[var]['sp_add']}
        else: 
            sp_tmp = subset_params
        dss[var] = xr.open_dataset(fn[0]).sel(**sp_tmp)

    # Get average of both over double-peaked region
    dss = {var:area_mean(subset_to_srat(dss[var][re.split('\-',var)[0]])) for var in var_params}

    #----------- Put data in right form -----------
    plot_data = {'hsat':dss[file_vars['hsat']].sel(kind=mean_kind)/1000,
                 'h':dss[file_vars['h']].sel(kind=mean_kind)/1000,
                 'hus':dss[file_vars['hus']].sel(kind=mean_kind)*L_v/1000,
                 'ta':dss[file_vars['ta']].sel(kind=mean_kind)*c_p/1000}
    
    #----------- Plot -----------
    fig,ax = figure_iv_boxplots(plot_data)
    
    #----------- Print -----------
    if save_fig:
        plt.tight_layout()
        utility_print(output_fn)

def wrapper_figure7(mod_a = 'MERRA2',mod_o = 'OISST',
                save_fig=False,output_fn='../figures/figure7'):
    """ Wrapper for Figure 7
    
    Parameters
    -------------
    mod_a : str, by default 'MERRA2'
        which data product to load for atmopsheric variables
    
    mod_o : str, by default 'OISST'
        which data product to load for SSTs
        
    save_fig : bool, by default False
    
    output_fn : str, by default None
    
    """
    
    #---------- Load data ----------
    # Levels variables
    ds = xr.merge([xr.concat([(xr.open_dataset(dir_list['proc']+mod_a+'/'+
                            var+'-nsurf_seasavg_MERRA2_historical_reanalysis_1981-2021_eq-IO.nc').
                             sel(lat=slice(-15,14.9))),
                          (xr.open_dataset(dir_list['proc']+mod_a+'/'+
                            var+'-nsurf_seasavg_MERRA2_historical_reanalysis_1981-2021_subtrop-AfrSAsia.nc').
                             sel(lon=slice(31.5,110)))
                         ],dim='lat') for var in ['uq','vq','ua','va']])

    # Anom variables, on pressure levels
    plev = 650
    ds = xr.merge([ds,
               *[xr.concat([(xr.open_dataset(dir_list['proc']+mod_a+'/'+
                            var+'-anom_seasavg_MERRA2_historical_reanalysis_1981-2021_eq-IO.nc').
                             sel(lat=slice(-15,14.9),plev=plev)),
                          (xr.open_dataset(dir_list['proc']+mod_a+'/'+
                            var+'-anom_seasavg_MERRA2_historical_reanalysis_1981-2021_subtrop-AfrSAsia.nc').
                             sel(lon=slice(31.5,110),plev=plev))
                         ],dim='lat').drop('plev') for var in ['hdiff']]])

    # SST data
    dso = xr.concat([xr.open_dataset(dir_list['proc']+mod_o+'/tos_seasavg_OISST_historical_avhrr_1981-2021_'+suffix+'.nc')
                     for suffix in ['WIndOcean','EIndOcean']],dim='lon')
    dso = dso.sel(lon=slice(31.5,110))

    
    with warnings.catch_warnings():
        # Removing mainly a warning that the regridded arrays aren't
        # C-continuous.
        warnings.simplefilter("ignore")
        
        # Get landmask to cut off atmopsheric data at oceans with
        mask_rgrd = xe.Regridder(dso,ds,method='bilinear')
        landmask = np.isnan(mask_rgrd(dso.isel(season=0,year=0))).sst

        # Regrid SST data to atmopsheric data to make 
        # sure the contourf plot works without overlaps
        ds['tos'] = mask_rgrd(dso.sst)
        ds['tos'] = ds['tos'].where(ds['tos']!=0)

    # Load full stats file (to get bimodal pixels)
    bimod = xr.open_dataset(dir_list['proc']+'CHIRPS/pr_doyavg_CHIRPS_historical_seasstats_dunning_19810101-20141231_HoAfrica.nc')
    bimod = bimod.seas_ratio

    #---------- Plot ----------
    figure_seasmaps_multivar(ds,bimod=bimod,
                              fill_vars = {'hdiff':{'type':'all',
                                                    'params':{'vmin':-15,'vmax':15,'levels':15,'cmap':cmocean.cm.balance_r},
                                                    'label':r'$h_s-h^*$ anomalies [kJ/kg]',
                                                    'scale':1/1000,
                                                    'cbar_loc':'left'},
                                           'tos':{'type':'all',
                                                      'params':{'vmin':21,'vmax':32,'levels':12,'cmap':cmocean.cm.thermal},
                                                      'label':r'SST [K]',
                                                      'scale':1,
                                                  'cbar_loc':'right'}},
                              seas_label_x = -0.125,
                              cbar_right_adjust=0.825,
                              cbar_left_adjust=0.2,
                              save_fig=save_fig,
                              output_fn=output_fn)

    
def wrapper_figure89(mod='MERRA2',var = 'hsat',
                     clabels = {'hsat':r' $h^*$ anomaly [kJ/kg]','h-nsurf':r' $h_s$ anomaly [kJ/kg]'},
                     titles = {'hsat':r'650 hPa $h^*$','h-nsurf':r'Surface $h$'},
                     cmap_params = None,
                     save_fig=False,output_fn=None):
    """ Wrapper for Figures 8, 9 
    
    Parameters
    -------------
    mod : str, by default 'MERRA2'
        which data product to load
        
    var : str, by default 'hsat'
        use 'h-nsurf' for Figure 8, 'hsat' for Figure 9
        
    clabels : dict
        colorbar label; requires at least an entry `var` 
        from above (with '-nsurf' removed, if relevant)
    
    titles : dict
        figure title; requires at least an entry `var` 
        from above (with '-nsurf' removed, if relevant)
    
    cmap_params : dict, by default None
        piped into `figure_seasmaps()`
        
    save_fig : bool, by default False
    
    output_fn : str, by default None
    
    """
    
    freq = 'seasavg'
    
    if cmap_params is None:
        if 'surf' in var:
            cmap_params = {'vmin':-14,'vmax':14}
        else:
            cmap_params = {'vmin':-4,'vmax':4}
    
    
    #---------- Load data ----------
    subset_params = {'lat':slice(-15,15),
                     'year':slice(1981,2021)}
    
    if 'surf' in var:
        subset_params_tmp = subset_params
        var_list = [var,'ua-nsurf','va-nsurf'] 
    else:
        subset_params_tmp = {**subset_params,'plev':[650]}
        var_list = [var,'ua','va'] 

    
    dss = {var:load_raw(var+'-anom_'+freq+'*19810101*.nc',
                     search_dir=dir_list['proc']+mod+'/',
                     subset_params = subset_params_tmp,
                    show_filenames=False,
                    fn_ignore_regexs = 'HoA')
      for var in var_list}

    dss = xr.merge([v.drop_duplicates('lon') for k,v in dss.items()])
    dss = dss.sel(kind='dunning')
    
    # Load full stats file (to get bimodal pixels)
    bimod = xr.open_dataset(dir_list['proc']+'CHIRPS/pr_doyavg_CHIRPS_historical_seasstats_dunning_19810101-20141231_HoAfrica.nc')
    bimod = bimod.seas_ratio
    
    #---------- Plot ----------
    figure_seasmaps(dss[re.sub('\-nsurf','',var)].mean('year')/1000,
                    bimod,
                    save_fig=save_fig,output_fn=output_fn,
                    quiver_data = dss[['ua','va']].coarsen({'lat':10,'lon':10},boundary='trim').mean(),
                    qscale=0.8,
                    **cmap_params,cmap=cmocean.cm.balance,bimod_color='tab:green',
                    clabel=clabels[var],title=titles[var],
                    levels=21,title_suffix=r' and $\vec{u}$ anomalies')
    
def wrapper_figure10(mod_a = 'MERRA2',
                     mod_p = 'CHIRPS',
                     plev = 650,
                     kind = 'month',
                     subset_params = {'lat':slice(-3,12.5),'lon':slice(32,55)},
                     save_fig = False,
                     output_fn = None):
    
    """ Wrapper for scatterplot panel 
    
    Note: by default, data is loaded and averaged over double-peaked 
    region using `area_mean(subset_to_srat(load_raw(..))` 
    
    Parameters
    -----------------
    mod_a : str, by default 'MERRA2'
        Data product for 'hdiff' and 'unstable' variables
        
    mod_p : str, by default 'CHIRPS'
        Data product for 'pr' variable
        
    plev : float, by default 650
        Presuure level for 'hdiff' and 'unstable' data
        
    kind : str, by default 'month'
        Which seasonal definition to use, from: 
            'month': MAM / OND
            'month_alt': MAM / SON
            'dunning': mean GHA seasonal extent
            'dunning_local': local seasonal extent 
        (see `calculates_seasmeans()` for details 
        
    subset_params : dict
        Geographic subset of variables, used in `load_raw()`
        
    save_fig : bool, by default False
    
    output_fn : str, by default None
    
    """

    # Load data
    dss = xr.merge([area_mean(subset_to_srat(xr.merge([load_raw(var+'_seasavg*HoA.nc',search_dir=dir_list['proc']+mod_a+'/',
                                                 subset_params = {**subset_params,'plev':[plev]}) for var in ['hdiff','unstable']]))).drop('plev'),
                     area_mean(subset_to_srat(load_raw('pr_seasavg*-202*Africa.nc',search_dir=dir_list['proc']+mod_p+'/',
                                                        subset_params = subset_params).drop(['lat_bounds','lon_bounds'],errors='ignore'),
                                              srat_mod='CHIRPS'))])
    dss['hdiff'] = dss['hdiff']/1000

    # Plot
    fig,ax = figure_scatter(dss.sel(kind=kind),
                            xlims = [[-7,0],[0,0.45]],
                            ylims = [0,5],show_corrs = False,
                            seas_names = {'long_rains':'MAM',
                                          'short_rains':'OND'},
                            year_subsets = {'long_rains':[2017,2018]},
                            y_label_str = r'$\overline{P}$ [mm/day]',
                            x_label_strs = [r'$\overline{(h_s-h^*)}$ [kJ/kg]',
                                            r'$\overline{(frac.\ unstable)}$'],
                            save_fig = save_fig,
                            output_fn = output_fn)
    
    
def wrapper_figure11(mod_a = 'MERRA2',
                     mod_p = 'CHIRPS',
                     plev = 650,
                     kind = 'month',
                     subset_params = {'lat':slice(-3,12.5),'lon':slice(32,55)},
                     save_fig = False,
                     output_fn = None):
    """ Wrapper for scatterplot panel of year-on-year changes
    
    Note: by default, data is loaded and averaged over double-peaked 
    region using `area_mean(subset_to_srat(load_raw(..))` 
    
    Parameters
    -----------------
    mod_a : str, by default 'MERRA2'
        Data product for 'hdiff' and 'unstable' variables
        
    mod_p : str, by default 'CHIRPS'
        Data product for 'pr' variable
        
    plev : float, by default 650
        Presuure level for 'hdiff' and 'unstable' data
        
    kind : str, by default 'month'
        Which seasonal definition to use, from: 
            'month': MAM / OND
            'month_alt': MAM / SON
            'dunning': mean GHA seasonal extent
            'dunning_local': local seasonal extent 
        (see `calculates_seasmeans()` for details 
        
    subset_params : dict
        Geographic subset of variables, used in `load_raw()`
        
    save_fig : bool, by default False
    
    output_fn : str, by default None
    
    """

    # Load data
    dss = xr.merge([area_mean(subset_to_srat(xr.merge([load_raw(var+'_seasavg*HoA.nc',search_dir=dir_list['proc']+mod_a+'/',
                                                 subset_params = {**subset_params,'plev':[plev]}) for var in ['hdiff','unstable']]))).drop('plev'),
                     area_mean(subset_to_srat(load_raw('pr_seasavg*-202*Africa.nc',search_dir=dir_list['proc']+mod_p+'/',
                                                        subset_params = subset_params).drop(['lat_bounds','lon_bounds'],errors='ignore'),
                                              srat_mod='CHIRPS'))])
    # Convert from J/kg to kJ/kg 
    dss['hdiff'] = dss['hdiff']/1000

    # Plot
    fig,ax = figure_scatter(dss.sel(kind=kind),
                            plot_type = 'changes',
                            xlims = [[-5,5],[-0.3,0.3]],
                            ylims = [-5,5],
                            year_subsets = {'long_rains':[[2017,2017,'tab:red']]},
                            seas_names = {'long_rains':'MAM',
                                          'short_rains':'OND'},
                            y_label_str = r'$\Delta \overline{P}$ [mm/day]',
                            x_label_strs = [r'$\Delta \overline{(h_s-h^*)}$ [kJ/kg]',
                                            r'$\Delta \overline{(frac.\ unstable)}$'],
                            label_add = 'Year-on-year',
                            save_fig=save_fig,
                            output_fn = output_fn)
    
def wrapper_figure12(mod_a = 'MERRA2',
                     mod_p = 'CHIRPS',
                     seas_idx = 1, 
                     yrs = [[2017],[2018]],
                     plev = 650,
                     kind = 'month',
                     subset_params = {'lat':slice(-3,12.5),
                                      'lon':slice(32,55)},
                     save_fig=False,
                     output_fn = None):
    """ Wrapper for Figure 12
    
    Parameters
    -------------
    mod_a : str, by default 'MERRA2'
        which data product to load for atmopsheric variables
    
    mod_p : str, by default 'CHIRPS'
        which data product to load for rainfall
        
    save_fig : bool, by default False
    
    output_fn : str, by default None
    
    """
    
    #------------- Load -------------
    dss = {var:load_raw(var+'_seasavg_*_'+suffix+'.nc',search_dir = dir_list['proc']+mod+'/',
                 subset_params = {**subset_params,**{k:v for k,v in {'plev':[plev]}.items() 
                                                     if not re.search('(\-nsurf)|(pr)',var)}}).drop(['plev'],errors='ignore')
          for var,mod,suffix in zip(['hdiff','hsat','h-nsurf','ta-nsurf','hus-nsurf','pr'],
                                    [*[mod_a]*5,mod_p],
                                    [*['HoA']*5,'19810101-20211231_Africa'])}

    # Regrid MSE variables to precip grid
    rgrd = xe.Regridder(dss['hdiff'],dss['pr'],method='bilinear')

    # Make sure == 0 are nan because these are regridder issues
    for var in [var for var in dss if var not in ['pr']]:
        dss[var] = rgrd(dss[var])
        dss[var] = dss[var].where(dss[var]!=0)

    # Merge
    dss = xr.merge([v for k,v in dss.items()])
    
    # Subset to double-peaked region
    dss = subset_to_srat(dss)

    # Convert T, q to MSE units
    c_p = 1004.6
    L_v = 2.257e6

    dss['ta'] = dss['ta']*c_p
    dss['hus'] = dss['hus']*L_v

    # Convert from J to kJ
    for v in [v for v in dss if v not in ['pr']]:
        dss[v] = dss[v]/1000
    
    #------------- Plot -------------
    figure_pr_mse_trends(dss,yrs = yrs,mean_kind = kind,
                         seas_idx=seas_idx,
                         clims = {'mse':{'vmin':-4,'vmax':4,'levels':11},
                                 'pr':{'vmin':-4,'vmax':4,'levels':11}},
                         save_fig=save_fig,output_fn = output_fn)
