import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import xesmf as xe
import os
import glob
import re
import string
from scipy import stats as sstats
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import lines as mlines
from matplotlib import patches as mpatches
from matplotlib import text as mtext
from matplotlib import ticker as mticker
from matplotlib.collections import PatchCollection
import cartopy
from cartopy import crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
import cmocean
from shapely.geometry import box

from funcs_load import load_raw
from funcs_support import get_params, area_grid, utility_print, area_mean, subset_to_srat, earth_radius
dir_list = get_params()



def figure_climatology(plot_data,
                       roll_w = 40,
                       shade_quantiles = [0.25,0.75],
                       axv_shading = None,
                       axv_shading_color = 'powderblue',
                       plot_axes = 'same',
                       anomaly = False,
                       colors = None,
                       styles = None,
                       labels = None,
                       ylabel = '',
                       ylims = None,
                       show_0line = True,
                       title = '',
                       show_legend = True,
                       fig=None,
                       ax=None):
    """ A general climatology plot of plot_data
    
    Parameters
    -----------------
    plot_data : xr.DataArray or list
        with a time dimension; .groupby('time.dayofyear') will 
        be called on it. If list, can be list of multiple DataArrays.
        If there are two list members, they can alternatively be 
        plotted on the same or on different y-axes, based on the 
        parameter `plot_axes`
        
    roll_w : int, by default 40 (days)
        width (in days) of gaussian running average used to smooth 
        data. If None, no smoothing is done. 
        
    shade_quantiles : list or None, by default [0.25,0.75]
        if not None, the range between two listed quantiles will be
        drawn on the map with alpha=0.1 in the same color as the 
        associated line
        
    plot_axes : str, by default 'same'
        if 'diff': if `plot_data` has two items, then they will
                   be plotted on different y-axes (ignored if 
                   `plot_data` has less or more than two items
        if `same`: all `plot_data` items will be plotted on the 
                   same y-axis
                   
    anomaly : bool, by default False
        if True, then the climatological mean is subtracted from 
        the climatology before plotting. Can be list with the 
        same length as plot_data.
                   
    colors : str or list, by default None
    
    
    styles : str or list, by default None
    
    
    labels : str or list, by default None
        (necessary for legend construction; unless two axes are used,
        if no labels are given, no legend is shown)
    
    ylabel : str or list, by default ''
    
    
    ylims : list/array or list of lists/arrays, by default None
    
    
    show_0line : bool, by default True
    
    
    title : str, by default ''
    
    
    show_legend : bool, by default True
    
    
    fig : mpl.figure.Figure, by default None
        if provided, plotted on existing figure
    
    ax : mpl.axes.Axis, by default None
        if provided, plotted on existing axis
    
    
    """
    
    #----------- Setup -----------
    if fig is None:
        fig = plt.figure()
    
    if ax is None:
        ax = [plt.subplot()]
    elif type(ax) != list:
        ax = [ax]
        
    # Make plot_data a list if it's a dataarray
    if type(plot_data) == xr.core.dataarray.DataArray:
        plot_data = [plot_data]
        
    if type(anomaly) == bool:
        anomaly = [anomaly]*len(plot_data)
        
    # Graph annotation setup
    if type(ylabel) != list:
        ylabel = [ylabel]*len(plot_data)
        
    if type(ylims) != list:
        ylims = [ylims]*len(plot_data)
        
    if type(labels) != list:
        labels = [labels]*len(plot_data)
        
    if styles is None:
        styles = ['-']*len(plot_data)
        
    if colors is None:
        colors = ['k']*len(plot_data)
        
    # Subtract climatological mean if desired
    plot_data = [(data - data.mean('time')) if anom else data
                 for data,anom in zip(plot_data,anomaly)]
    
    # Calculate dayofyear average
    if shade_quantiles is not None:
        # Add quantiles if desired
        plot_data = [[pda.groupby('time.dayofyear').mean(),
                     *[pda.groupby('time.dayofyear').quantile(q) for q in shade_quantiles]]
                     for pda in plot_data]
    else:
        plot_data = [[pd.groupby('time.dayofyear').mean()] for pd in plot_data]
        
    # Get rolling average if desired
    if roll_w is not None: 
        # Get gaussian weight vector for rolling avg
        weight = xr.DataArray(sstats.norm(0, 1).pdf(np.arange(-5,5,10/roll_w)), dims=['window'])
        # Normalize by the weight sum
        weight = weight/weight.sum()
        
        # Smooth using gaussian rolling average
        plot_data = [[(data.pad({'dayofyear':int(roll_w/2)},mode='wrap').
                       rolling(dayofyear=roll_w,center=True).construct('window').
                       dot(weight))[int(roll_w/2):-int(roll_w/2)]
                  for data in pda]
                 for pda in plot_data]

    #----------- Underlying annotations ------------
    if axv_shading is not None:
        for shade_name in axv_shading:
            ax[0].axvspan(*axv_shading[shade_name],
                          alpha=0.2, color=axv_shading_color)

    #----------- Plot -----------
    if (len(plot_data) != 2) or (plot_axes == 'same'):
        # Plot shaded quantiles
        if shade_quantiles is not None:
            for pda,color in zip(plot_data,colors):
                ax[0].fill_between(pda[0].dayofyear,
                         pda[1],pda[2],
                         color=color,alpha=0.1)
        
        # Plot main lines
        for pda,color,style in zip(plot_data,colors,styles):
            pda[0].plot(color=color,linestyle=style,ax=ax[0]) 
        
    elif (len(plot_data) == 2) and (plot_axes == 'diff'):
        ax = [ax[0],ax[0].twinx()]
        
        for ax_tmp,pda,color,style in zip(ax,plot_data,colors,styles):
            # Plot shaded quantiles
            if shade_quantiles is not None:
                ax_tmp.fill_between(pda[0].dayofyear,
                                 pda[1],pda[2],
                                 color=color,alpha=0.1)
            
            # Plot main lines   
            pda[0].plot(color=color,ax=ax_tmp,linestyle=style) 
            
            
        # Annotations for second y-axis
        ax[1].set_ylabel(ylabel[1],color=colors[1],fontsize=13)
        ax[1].spines['right'].set_color(colors[1])
        ax[1].tick_params(axis='y', colors=colors[1]) 
        if ylims[1] is not None:
            ax[1].set_ylim(*ylims[1])
        ax[1].set_title('')

        
    else:
        raise KeyError('`plot_axes` must be "same" or "diff", not '+str(plot_axes))
        
        
    #----------- Further annotations -----------
    # Annotate primary y-axis and xaxis
    ax[0].set_ylabel(ylabel[0],fontsize=13)
    if ylims[0] is not None:
        ax[0].set_ylim(*ylims[0])
    ax[0].set_xlabel('')
    
    # Set xlim
    ax[0].set_xlim([0,366])
    
    # Line at 0 if ylims cross 0
    for ax_tmp in ax:
        if show_0line and (ax_tmp.get_ylim()[0]<0) and (ax_tmp.get_ylim()[1]>0):
            ax_tmp.axhline(0,color='k',linestyle='--')
    
    # Change x axis to calendar months
    doys = np.array(pd.date_range('2001-01-01','2002-02-01',freq='2MS').dayofyear)
    doys[-1] = doys[-1]+365
    ax[0].set_xticks(doys,pd.date_range('2001-01-01','2002-02-01',freq='2MS').strftime('%b'))

    # Annotate title
    ax[0].set_title(title)
    
    # Annotate axis shading, if it exists
    if axv_shading is not None:
        for shade_name in axv_shading:
            if re.search('[(a-z)|(0-9)]',shade_name):
                ax[0].annotate('',
                        xy=[axv_shading[shade_name][0],ax[0].get_ylim()[0]+np.diff(ax[0].get_ylim())[0]*0.025],
                        xytext=[axv_shading[shade_name][1],ax[0].get_ylim()[0]+np.diff(ax[0].get_ylim())[0]*0.025],
                        arrowprops={'arrowstyle':'<->'})
                ax[0].text(axv_shading[shade_name][0]+
                            (axv_shading[shade_name][1] - 
                             axv_shading[shade_name][0])/2,
                        ax[0].get_ylim()[0]+np.diff(ax[0].get_ylim())[0]*0.05,
                        shade_name,va='bottom',ha='center')
    
    # Legend
    if show_legend:
        if (len(plot_data) == 2) and (plot_axes == 'diff') and (shade_quantiles is not None):
            lgd = ax[0].legend(handles=[mlines.Line2D([0], [0], color=colors[0], lw=2),
                                       mpatches.Patch(facecolor=colors[0], edgecolor=colors[0],alpha=0.1),
                                       mlines.Line2D([0], [0], color=colors[1], lw=2),
                                       mpatches.Patch(facecolor=colors[1], edgecolor=colors[1],alpha=0.1)],
                            labels=('','','mean','-'.join([str(q) for q in shade_quantiles])+' IQR'),
                            loc='upper left',numpoints=1,ncol=2,
                            handletextpad=0.5,handlelength=1.0,columnspacing=-0.5,
                            borderaxespad=0.5, 
                            bbox_to_anchor=(1.15,0.7))
        elif not np.all([k is None for k in labels]):
            # Show legend only if there are line labels provided
            lgd = ax[0].legend(handles = [mlines.Line2D([0],[0],color=color,lw=2,label=label,linestyle=style)
                                       for color,label,style in zip(colors,labels,styles)],
                            borderaxespad=0.5, 
                            bbox_to_anchor=(1,0.7),
                           loc="upper left")

        
    
    #----------- Return -----------
    if len(ax) == 1:
        return fig,ax[0]
    else:
        return fig,ax
    
    

def figure_climatology_panel(plot_vars,
                             mod = 'MERRA2',
                             freq = 'day',
                             suffix = 'HoA',
                             mod_p = 'CHIRPS',
                             comp_var = 'ta-nsurf',
                             ds = None,
                             stats = None,
                             stats_suffix = '19810101-20221231_HoA',
                             labels = None,
                             KtoC = False,
                             add_equinox = False,
                             add_season_shading = True,
                             save_fig = False,
                             output_fn = '',
                             return_handles=False):
    """ Plot a panel of climatology figures for multiple variables
    
    Parameters
    --------------------
    plot_vars : list
        Which variables to plot. Note, in addition to standard variables,
        code supports *-nsurf versions and a few extra, calculated 
        variables:
            - 'advT': near-surface T advection, requires `ua-nsurf`, `va-nsurf`
            - 'nadvT': 'advT' x -1
            - 'nadvTday': 'nadvT' * 60 * 60 * 24 to convert from K/s to K/day
            
    comp_var : str or None, by default 'ta-nsurf'
        Which variable to plot on every sub-panel. If None, then only
        `plot_vars` are plotted, one per sub-panel
            
    ds : xr.Dataset, by default None
        If a dataset, then those data are used to plot. If None, then 
        data is loaded, searched for by: 
            
      mod : str, by default 'MERRA2'

      freq : str, by default 'day'
      
      suffix : str, by default 'HoA'
      
    add_season_shading: bool, by default True
        If True, then seasons are shaded in the final panel, using seasonal 
        stats from: 
      
      stats : xr.Dataset, by default None  
        If a dataset, then those seasonal stats are used to plot. If None, 
        then data is loaded, searched for by: 
        
        mod_p : str, by default 'CHIRPS'
        
        stats_suffix : str, by default '19810101-20221231_HoA'
        
    labels : dict, by default None
        Use for custom axis labels (of the form: {var : label})
        
    KtoC : bool, by default False
        If True, then variables that start with 'ta' are converted from
        K to C by subtracting 273.15; labels are changed from C to K as 
        well
        
    add_equinox : bool, by default False
        If True, then vertical lines are inputted at the equinoxes
        
    save_fig : bool, by default False
    
    output_fn : str, by default None 
    
    return_handles : bool, by default False
        If True, then returns fig,axs
    
    """

    if labels is None:
        labels = {'rsdt':r'TOA down shortwave [W/m$^2$]',
                  'rsns':r'Surface net shortwave [W/m$^2$]',
                  'rsds':r'Surface down shortwave [W/m$^2$]',
                  'rlus':r'Surface up longwave [W/m$^2$]',
                  'rlns':r'Surface net longwave [W/m$^2$]',
                  'rlds':r'Surface down longwave [W/m$^2$]',
                  'ts':r'Skin temperature [$^\circ$C]',
                  'cllow':'Low cloud fraction',
                  'clmid':'Mid cloud fraction',
                  'clhgh':'High cloud fraction',
                  'cl':'Total cloud area fraction',
                  'advT':r'$\mathbf{u}\cdot\nabla T$ [K/s]',
                  'nadvT':r'$-(\mathbf{u}\cdot\nabla T)$ [K/s]',
                  'nadvTday':r'$-(\mathbf{u}\cdot\nabla T)$ [K/day]',
                  'hfls':r'Surface latent flux [W/m$^2$]',
                  'hfss':r'Surface sensible flux [W/m$^2$]',
                  'ta-nsurf':r'$T_s$ [K]',
                  'pr':r'$P$ [mm/day]',
                  'hdiff':r'$h_s-h^*$ [kJ/kg]',
                  'hsat':r'$h^*$ [kJ/kg]',
                  'h':r'$h$ [kJ/kg]',
                  'sst_coast':'Coastal SSTs [K]',
                  'sst_wio':'W. Indian Ocean SSTs [K]'}

    #------------- Load data -------------
    # Load variables to plot
    if ds is None:
        # Determine which variables need to be loaded (tas for advection variables)
        load_vars = [['ta-nsurf','ua-nsurf','va-nsurf'] if re.search('advT',v) else [v] for v in plot_vars]
        # Flatten
        load_vars = [l0 for l1 in load_vars for l0 in l1]
        # Add variable to plot in every subplot
        if comp_var is not None:
            load_vars = [comp_var,*load_vars]


        # Determine which directories the variables are in 
        load_dirs = [('raw' if len(glob.glob(dir_list['raw']+mod+'/'+v+'_'+freq+'_*'+suffix+'.nc'))>0 
                            else 'proc' if len(glob.glob(dir_list['proc']+mod+'/'+v+'_'+freq+'_*'+suffix+'.nc'))>0
                            else None)
                     for v in load_vars]
        
        if np.any([d is None for d in load_dirs]):
            raise FileNotFoundError('Variables '+', '.join([v for v,d in zip(load_vars,load_dirs) if d is None])+
                                    ' not found in "raw" or "proc" dirs for model '+mod+'.')

        # Get unique load variables, to not double-load anything
        unique_idxs = np.unique(load_vars,return_index=True)    
        load_dirs = list(np.array(load_dirs)[unique_idxs[1]])
        load_vars = list(unique_idxs[0])

        # Load 
        ds = xr.merge([load_raw(v+'_'+freq+'_*'+suffix+'.nc',
                                search_dir = dir_list[d]+mod+'/') for v,d in zip(load_vars,load_dirs)])

    # Load seasonal stats
    if (add_season_shading) and (stats is None):
        if stats_suffix is None:
            stats_fn = glob.glob(dir_list['proc']+mod_p+'/pr_doyavg_'+mod_p+'_*_seasstats_*.nc')
            if len(stats_fn) > 1:
                raise NotUniqueFile('More than one possible stats file found to load:'+'\n'.join(stats_fn)+'\nTo specify a stats file, use the `stats_suffix` parameter.')
            else:
                stats_fn = stats_fn[0]
        else:
            stats_fn = dir_list['proc']+mod_p+'/pr_doyavg_'+mod_p+'_historical_seasstats_dunning_'+stats_suffix+'.nc'
        stats = xr.open_dataset(stats_fn)
        # NaN out areas outside double-peaked region
        stats = subset_to_srat(stats,drop=True)
        # Remove singleton 'method' dimension
        stats = stats.isel(method=0,drop=True)
        # Make demise onset + duration
        stats['demise'] = stats['onset']+stats['duration']
        # Get area mean
        stats = area_mean(stats)

    #------------- Preprocess -------------
    ## Calculate advection
    if np.any([re.search('advT',v) for v in plot_vars]):
        # Get dy/dx in m
        xlon, ylat = np.meshgrid(ds.lon, ds.lat)
        R = earth_radius(ylat)

        dlat = np.deg2rad(np.gradient(ylat, axis=0))
        dlon = np.deg2rad(np.gradient(xlon, axis=1))

        dy = dlat * R
        dx = dlon * R * np.cos(np.deg2rad(ylat))

        # Get T gradient
        ds['dtdy'] = ds.ta.differentiate('lat')/dy
        ds['dtdx'] = ds.ta.differentiate('lon')/dx

        # Calculate T advection
        ds['advT'] = ds.ua*ds.dtdx+ds.va*ds.dtdy

        if np.any([re.search('nadvT',v) for v in plot_vars]):
            # Get negative advection (make adv _into_
            # region positive) for ease of interpretation
            ds['nadvT'] = -ds['advT']

            if 'nadvTday' in plot_vars:
                # Convert from K/s to K/day for ease 
                # of interpretation
                ds['nadvTday'] = ds['nadvT']*60*60*24

    ## Subset to only variables needed to plot
    ds = ds[[re.sub('\-nsurf','',v) for v in [comp_var,*plot_vars]]] 

    ## Calculate HoA double-peaked region mean
    ds = area_mean(subset_to_srat(ds,srat_mod = mod_p))

    ## Convert all temperature variables to Celsius, if desired
    if KtoC:
        for var in [v for v in [comp_var,*plot_vars] if re.search('^ta',v)]:
            ds[re.sub('\-nsurf','',var)] = ds[re.sub('\-nsurf','',var)]-273.15

        labels = {k:re.sub(r'\[\$\^\\circ\$ C\]','[K]',v) for k,v in labels.items()}


    #------------- Plot -------------
    fig = plt.figure(figsize=(10,((len(plot_vars)+1)//2)*4))
    axs = [None]*len(plot_vars)
    
    for var,plt_idx in zip(plot_vars,np.arange(0,len(plot_vars))): #rsds
        axs[plt_idx] = plt.subplot(len(plot_vars) // 2,2,plt_idx + 1)

        # Set data to plot
        if comp_var is not None:
            plot_data = [ds[var],ds[re.sub('\-nsurf','',comp_var)]]
        else:
            plot_data = ds[var]

        # Set annotations
        title = labels[var]
        ylabels = ['',labels[comp_var]]

        # Set seasonal shading 
        if add_season_shading:
            if plt_idx == 0:
                axv_shading={r'$\mathit{Gu}$'+'\nLong r.':[stats.isel(season=0).onset,stats.isel(season=0).demise],
                             r'$\mathit{Deyr}$'+'\nShort r.':[stats.isel(season=1).onset,stats.isel(season=1).demise]}
            else:
                # Hack, but `figure_climatology()` is currently set to not
                # annotate shading if there are no alphanumeric characters in
                # the shading title - but the keys still need to be different
                # so you can index the dictionary
                axv_shading={'':[stats.isel(season=0).onset,stats.isel(season=0).demise],
                             ' ':[stats.isel(season=1).onset,stats.isel(season=1).demise]}
        else:
            axv_shading=None


        # Plot climatology
        fig,axs[plt_idx]=figure_climatology(plot_data,
                           ax=axs[plt_idx],fig=fig,
                           show_legend=False,
                           plot_axes='diff',
                           colors=['tab:blue','tab:red'],
                           ylabel=ylabels,
                           axv_shading=axv_shading,
                           axv_shading_color='tan',)

        # Set title
        axs[plt_idx][0].set_title(labels[var],color='tab:blue')

        # Add equinox lines, if desired
        if add_equinox:
            for d in [79,266]:
                axs[plt_idx][0].axvline(d,color='grey',linewidth=0.5)
                if plt_idx == 0:
                    axs[plt_idx][0].annotate('equinox',(d,axs[plt_idx][0].get_ylim()[0]+1),va='bottom',ha='right',color='grey',
                                 xycoords='data',rotation=90)  

        # Subplot lettering
        axs[plt_idx][0].annotate(string.ascii_lowercase[plt_idx]+'.',
                        [0.01,1.01],xycoords='axes fraction',
                        va='bottom',ha='left',fontsize=13,fontweight='bold')


    plt.subplots_adjust(wspace=0.4,hspace=0.25)

    #------------- Print -------------
    if save_fig:
        utility_print(output_fn)
        
    #------------- Return -------------
    if return_handles:
        return fig,axs
        
        
def figure_iv_boxplots(plot_data,
                       seas_titles = [r'$\mathit{Jilaal}$'+'\nLong dry period',
                                       r'$\mathit{Gu}$'+'\nLong rains',
                                       r'$\mathit{Xagaa}$'+'\nShort dry period',
                                       r'$\mathit{Deyr}$'+'\nShort rains'],
                       colors = {'hsat':'tab:green',
                                 'h':'tab:grey',
                                 'hus':'tab:blue',
                                 'ta':'tab:red'},
                       labels = {'hsat':r"$h^{*'}$",
                                 'h':r"$h_s'$",
                                 'hus':r"$L_vq_s'$",
                                 'ta':r"$c_pT_s'$"},
                       fig=None,
                       ax=None):
    
    """ A general climatology plot of plot_data
    
    Parameters
    -----------------
    plot_data : dict
        `dict` with keys:
            - 'hsat'
            - 'h'
            - 'hus'
            - 'ta'
        each containing a `year` x `season` xr.DataArray
        
    
    fig : mpl.figure.Figure, by default None
        if provided, plotted on existing figure
    
    ax : mpl.axes.Axis, by default None
        if provided, plotted on existing axis
        
    """


    #----------- Setup -----------
    ref_v = [k for k in plot_data][0]
    
    if fig is None:
        fig = plt.figure(figsize=(plot_data[ref_v].sizes['season']*2,5))

    if ax is None:
        ax = plt.subplot()

    #------------- Plot -------------
    # Boxplot for future scenarios
    for seas_idx in np.arange(0,plot_data[ref_v].sizes['season']):

        bx = [None]*len([k for k in plot_data])
        locs = [seas_idx-0.325,*[seas_idx-0.125+0.15*data_idx for data_idx in np.arange(1,len([k for k in plot_data]))]]
        for v,data_idx in zip(plot_data,np.arange(0,len([k for k in plot_data]))):
            bx[data_idx] = plt.boxplot(plot_data[v].isel(season=seas_idx)[~np.isnan(plot_data[v].isel(season=seas_idx))],
                                      positions=[locs[data_idx]],notch=False,#bootstrap=100000,
                                    patch_artist=True,widths=0.1)
            for item in ['boxes', 'whiskers', 'fliers', 'caps']:
                plt.setp(bx[data_idx][item],color=colors[v])
            plt.setp(bx[data_idx]['medians'],color='white')


    #------------- Annotations -------------
    # Grid and 0 lines
    for x in np.arange(0.5,plot_data[ref_v].sizes['season']+0.5):
        ax.axvline(x,color='k')
    for x in np.arange(-0.15,plot_data[ref_v].sizes['season']+0.3):
        ax.axvline(x,color='grey',linestyle=':')
    ax.axhline(0,color='k',linestyle='--')

    ax.set_xlim(-0.5,plot_data[ref_v].sizes['season']-0.5)

    # Axis labels and annotations
    plt.xticks(np.arange(0,plot_data[ref_v].sizes['season']),seas_titles,fontsize=13)
    ax.set_ylabel(r'$h^*$ or $h_s$ component anomaly [kJ/kg]',fontsize=13)
    ax.set_title(r'Interannual variability in $h_s - h^*$ components',
                fontsize=15)             

    # Legend
    leg_patches = [mpatches.Patch(color=colors[v],label=labels[v])
                  for v in colors]
    ax.legend(handles=leg_patches,fontsize=13,
              loc='upper left',bbox_to_anchor=(1,0.7))

    # Add hsat level / surface annotations, using a mixture of data units 
    # (to get the position right regardless of the number of seasons used
    # and axis units (to get the relative position at the top of the plot)
    ax.text(-0.475,
            (ax.transAxes+ax.transData.inverted()).transform((0,0.99))[1],
            str(int(plot_data['hsat'].plev.values))+'mb',ha='left',va='top')
    ax.text(0.475,
            (ax.transAxes+ax.transData.inverted()).transform((0,0.99))[1],
            'surface',ha='right',va='top')

    #----------- Return -----------
    return fig, ax

# Source: test_wrapper_figure89.ipynb
def figure_seasmaps(plot_data,bimod=None,subset=None,
                    vmin=276,vmax=282,levels=None,
                    quiver_data=None,qscale=1,
                    arrow_props = {'headwidth':2,'headlength':8,'minlength':0.25},
                    extra_seasnames = ['Jilaal','Gu','Xagaa','Deyr'],
                    cmap = cmocean.cm.thermal,bimod_color='white',
                    clabel = r'T [K]',title=None,title_suffix='',seas_label_x = -0.04,
                    figsize=(18,10),
                    save_fig=False,output_fn=None):
    """ Figure with seasonal composites 
   
    4-season panel plot of seasonal composites of a base variable and
    an optional arrow/quiver plot. 
    
    
    Parameters
    ---------------
    plot_data : xr.DataArray
        requires dimensions `lat`, `lon`, `season` (and possibly others, 
        if subset using `subset`). Subset by `subset`, if desired. 
        
    bimod : xr.DataArray, by default None
        if not None, plots a contour at [1] of the variable; intended
        to show the `seas_ratio` boundaries between single- and double-
        peaked rainfall regimes. Assumes it's HoA and will fill in a 
        few gaps. 
    
    bimod_color : str, by default 'white'
        color to outline the [1] contour at, if desired (see above)
        
    subset : dict, by default None
        if not None, then data is susbset using the dict here (it's 
        piped into `da.sel(**subset)`). If 'plev' is a key, then the 
        plev value is added to the colorbar label
        
    vmin,vmax,levels,cmap : colormap parameters
    
    quiver_data : xr.Dataset, by default None
        requires dimensions `lat`, `lon`, `season`, and variables `ua` 
        and `va`. Make sure to coarsen first before inputting. Subset by
        `subset`, if desired. 
        
    arrow_props : dict, by default {'headwidth':2,'headlength':8,'minlength':0.25}
        arror props for quiver plot
        
    extra_seasnames : list, by default ['Jilaal','Gu','Hhagaa','Deyr']
        if not None, then these names are printed in italics above 
        the seasonal name from the input `xr.DataArray` to the left 
        of each panel
        
    clabel : str, by default `r'T [K]'`
        colorbar label
        
    title : str, by default None
        if None, default title is shown as ' T' 
        
    title_suffix : str, by default '' 
        second part of title, attached to `title` above
        
    seas_label_x : float, by default -0.04
        how far away from the left part of the axis to place seasonal titles
        
    figsize : tuple, by default (18,10)
        figure size, in inches
        
    save_fig : bool, by default False
    
    output_fn : str, by default None
        
    """ 
    
    fig = plt.figure(figsize=figsize,facecolor='white')

    if levels is None:
        levels=(vmax-vmin)*2
        
    if title is None:
        title = ' T'

    clabel_string = ''
    if subset is not None:
        plot_data = plot_data.sel(**subset)
        if 'plev' in subset:
            plev = subset['plev']
            clabel_string = str(plev)+' hPa'
        if quiver_data is not None:
            quiver_data = quiver_data.sel(**subset)
        
        
    for seas_idx in np.arange(0,4):
        ax = plt.subplot(4,1,seas_idx+1,projection=ccrs.PlateCarree(central_longitude=105))

        plot_data.isel(season=seas_idx).plot.contourf(transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax,
                                                   cmap=cmap,levels=levels,add_colorbar=False)
        c = ax.coastlines()
        
        if quiver_data is not None:
            q = plt.quiver(quiver_data.lon.transpose().values,quiver_data.lat.transpose().values,
                       quiver_data.isel(season=seas_idx).mean('year').ua.values,
                       quiver_data.isel(season=seas_idx).mean('year').va.values,
                       angles='uv',transform=ccrs.PlateCarree(),
                       scale=qscale,scale_units='x',**arrow_props,
                       pivot='tail')
            if seas_idx == 0:
                ax.quiverkey(q,0.1,1.1,5,'5 m/s',coordinates='axes')

        #-------------- Plot bimodal region --------------
        if bimod is not None:
            a = bimod.plot.contour(levels=[1],colors=[bimod_color],transform=ccrs.PlateCarree())
            # Get paths of the bimod region contour
            bimod_paths = a.collections[0].get_paths()
            # Connect western vertices 
            min_bbxs = np.array([np.array(p.get_extents())[0][0] for p in bimod_paths])
            to_connect = np.where(min_bbxs==min_bbxs.min())[0]
            to_connect = [bimod_paths[x] for x in to_connect]
            to_connect = np.array([to_connect[x].vertices[to_connect[x].vertices[:,0].argmin(),:] for x in np.arange(0,len(to_connect))])
            plt.plot(to_connect[:,0],to_connect[:,1],color=bimod_color,transform=ccrs.PlateCarree())

            # Get gdf of the coastlines paths (this isn't strictly right, but it's too 
            # zoomed out to see the difference by Kismaayo, etc. anyways)
            gdf = gpd.GeoDataFrame(geometry=[k for k in cfeature.COASTLINE.geometries()])
            # Get subset within bounding box
            bbox = np.array([bimod.lat.min(),bimod.lat.max().values,bimod.lon.min().values,bimod.lon.max().values])
            bbox = box(bbox[2],bbox[0],bbox[3],bbox[1])
            gdf = gpd.clip(gdf,mask=bbox)
            gdf.plot(color=bimod_color,transform=ccrs.PlateCarree(),ax=ax)

        #--------------- Annotate --------------
        if seas_idx == 0:
            ax.set_title(clabel_string+title+title_suffix,fontsize=16,fontweight='bold')
        else:
            ax.set_title('')
        ax.axhline(0,linestyle='--',color='k')

        if extra_seasnames is not None:
            ax.annotate(r'$\mathit{'+extra_seasnames[seas_idx]+'}$'+'\n'+
                        re.sub('\_',' ',str(plot_data.season[seas_idx].values)).capitalize(),
                        (seas_label_x,0.5),va='center',ha='center',rotation='vertical',
                        xycoords='axes fraction',fontsize=16)
        else:
            ax.annotate(re.sub('\_',' ',str(plot_data.season[seas_idx].values)).capitalize(),
                        (seas_label_x,0.5),va='center',ha='center',rotation='vertical',
                        xycoords='axes fraction',fontsize=16)

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='-')
        if seas_idx > 0:
            gl.top_labels = False
        if seas_idx < 3:
            gl.bottom_labels=False
        gl.left_labels = False
        
        # Subplot annotation
        ax.annotate(string.ascii_lowercase[seas_idx]+'.',
                        [0.01,1.01],xycoords='axes fraction',
                        va='bottom',ha='left',fontsize=13,fontweight='bold')

    fig.subplots_adjust(right=0.825)
    cax = fig.add_axes([0.875, 0.15, 0.025, 0.7])
    levels = mpl.ticker.MaxNLocator(nbins=levels).tick_values(vmin,vmax)
    norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    
    cbar=plt.colorbar(sm,cax=cax)
    cbar.set_label(label=clabel_string+clabel,size=16)
    cbar.ax.tick_params(labelsize=15)
    
    if save_fig:
        utility_print(output_fn)
        
# Source: test_wrapper_figure7.ipynb
def figure_seasmaps_multivar(ds,
                             landmask = None,
                             bimod = None,
                             bimod_color = 'tab:red',
                             coarsen = {'lat':5,'lon':5},
                             col_params = [{'vars':['ua','va'],'qscale':2,'title':r'Surface $\vec{u}$',
                                             'qkey_value':10,'qkey_label':r'10 $\frac{m}{s}$','qkey_y':1.1,'qkey_x':0},
                                            {'vars':['uq','vq'],'qscale':0.02,'title':r'Surface $\vec{u}q$',
                                             'qkey_value':0.1,'qkey_label':r'10 $\frac{m\cdot g}{s\cdot kg}$','qkey_y':1.1,'qkey_x':1}],
                             seas_names = [['Jilaal','Long dry period'],
                                            ['Gu','Long rains'],
                                            ['Xagaa','Short dry period'],
                                            ['Deyr','Short rains']],
                             fill_vars = {'hus':{'type':'all',
                                                  'params':{'vmin':0,'vmax':30,'levels':10,'cmap':cmocean.cm.rain},
                                                  'label':r'Surface $q$ [g/kg]',
                                                  'scale':1000}},
                             cbar_right_adjust = 0.825,
                             cbar_left_adjust = 0,
                             extent = [32,109,-15,35],
                             seas_label_x = -0.2,
                             save_fig = False,
                             output_fn = None):
    """ Plot multiple field variables on the same N-column plot 
    
    ... this is honestly just a very complex setup with finicky
    input parameters. This function is mainly purpose-built to
    make Figure 7 through `wrapper_figure7()`. 
    
    Parameters
    ---------------
    ds : xr.Dataset
        containing at least the variables listed in `fill_vars` and `col_params`
    
    """
    
    #------------------------- Setup -------------------------
    # Coarsen quiver data
    dsc = ds[list(np.hstack([cp['vars'] for cp in col_params]))].coarsen({'lat':5,'lon':5},boundary='trim').mean()

    #
    fill_var_list = [k for k in fill_vars]
    
    #------------------------- Plot -------------------------
    fig = plt.figure(figsize=(7+len(fill_var_list)*1.2-0.2,10))

    for col_idx in np.arange(0,len(col_params)):
        for seas_idx in np.arange(0,4):

            ax = plt.subplot(4,len(col_params),seas_idx*len(col_params)+col_idx+1,
                             projection=ccrs.PlateCarree())

            #------- Filled plot  -------
            for var in fill_vars:
                if fill_vars[var]['type'] == 'all':
                    fill_data = ds[var]
                elif fill_vars[var]['type'] == 'land':
                    if landmask is None:
                        raise KeyError('a valid `landmask` is required, since the type of variable '+var+' is "land"')
                    else:
                        fill_data = ds[var].where(landmask)
                fill_data = fill_data.isel(season=seas_idx).mean('year')*fill_vars[var]['scale']
                nans = np.isnan(fill_data)
                
                # Replace values outside of cmap bounds with the cmap bound, otherwise you get
                # different colormaps in different plots. Really should be together with changing
                # the colorbar to having arrows at both ends, but that's for another day 
                fill_data = fill_data.where(fill_data>fill_vars[var]['params']['vmin'],fill_vars[var]['params']['vmin']+0.0001)
                fill_data = fill_data.where(fill_data<fill_vars[var]['params']['vmax'],fill_vars[var]['params']['vmax']-0.0001)
                # Now, put the nans back (the where replace also replaces all nans with the replace 
                # value)
                fill_data = fill_data.where(~nans)
                
                
                f = (fill_data.
                     plot.contourf(transform=ccrs.PlateCarree(),
                                    **fill_vars[var]['params'],add_colorbar=False))

            #------- Quiver plot -------
            q = plt.quiver(dsc.lon.transpose().values,dsc.lat.transpose().values,
                           dsc.isel(season=seas_idx).mean('year')[col_params[col_idx]['vars'][0]].values,
                           dsc.isel(season=seas_idx).mean('year')[col_params[col_idx]['vars'][1]].values,
                           angles='uv',transform=ccrs.PlateCarree(),
                           scale=col_params[col_idx]['qscale'],scale_units='x',headwidth=5,headlength=8,minlength=0.25,
                           pivot='tail')

            if seas_idx == 0:
                ax.quiverkey(q,
                             col_params[col_idx]['qkey_x'],col_params[col_idx]['qkey_y'],
                             col_params[col_idx]['qkey_value'],col_params[col_idx]['qkey_label'],
                             coordinates='axes')
                
            #-------------- Plot bimodal region --------------
            if bimod is not None:
                a = bimod.plot.contour(levels=[1],colors=[bimod_color],transform=ccrs.PlateCarree())
                # Get paths of the bimod region contour
                bimod_paths = a.collections[0].get_paths()
                # Connect western vertices 
                min_bbxs = np.array([np.array(p.get_extents())[0][0] for p in bimod_paths])
                to_connect = np.where(min_bbxs==min_bbxs.min())[0]
                to_connect = [bimod_paths[x] for x in to_connect]
                to_connect = np.array([to_connect[x].vertices[to_connect[x].vertices[:,0].argmin(),:] for x in np.arange(0,len(to_connect))])
                plt.plot(to_connect[:,0],to_connect[:,1],color=bimod_color,transform=ccrs.PlateCarree())

                # Get gdf of the coastlines paths (this isn't strictly right, but it's too 
                # zoomed out to see the difference by Kismaayo, etc. anyways)
                gdf = gpd.GeoDataFrame(geometry=[k for k in cfeature.COASTLINE.geometries()])
                # Get subset within bounding box
                bbox = np.array([bimod.lat.min(),bimod.lat.max().values,bimod.lon.min().values,bimod.lon.max().values])
                bbox = box(bbox[2],bbox[0],bbox[3],bbox[1])
                gdf = gpd.clip(gdf,mask=bbox)
                gdf.plot(color=bimod_color,transform=ccrs.PlateCarree(),ax=ax)

            #------- Plot annotations -------
            ax.coastlines(linewidth=2)

            ax.axhline(0,color='k',linestyle='--')
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    color='gray',alpha=0.5)
            gl.xlocator = mticker.FixedLocator(np.arange(30,110,20))
            gl.left_labels = False
            if col_idx == 0:
                gl.right_labels = False

            if seas_idx > 0:
                gl.top_labels = False
            if seas_idx < 3:
                gl.bottom_labels = False

            ax.set_extent(extent,crs=ccrs.PlateCarree())

            #------- Text annotations -------
            if seas_idx == 0:
                ax.set_title(col_params[col_idx]['title'],fontsize=16)
            else:
                ax.set_title('')

            if col_idx == 0:
                ax.annotate((r'$\mathit{'+seas_names[seas_idx][0]+'}$'+'\n'+
                                    seas_names[seas_idx][1]),
                        (seas_label_x,0.5),va='center',ha='center',rotation='vertical',
                        xycoords='axes fraction',fontsize=14)

            # Subplot lettering
            ax.text(0.01,0.98,string.ascii_letters[seas_idx*len(col_params)+col_idx]+'.',
                        transform=ax.transAxes,ha='left',va='top',
                        fontsize=15,fontweight='bold',
                    bbox={'boxstyle':'round','facecolor':'white','alpha':0.8,'edgecolor':'None','pad':0.1})


    plt.subplots_adjust(hspace=0.01,wspace=0.1)

    #------- Colorbar -------
    fig.subplots_adjust(right=cbar_right_adjust,left=cbar_left_adjust)
    for var in fill_vars:
        if fill_vars[var]['cbar_loc'] == 'right':
            cax = fig.add_axes([0.9, 0.15, 0.025, 0.7])
        elif fill_vars[var]['cbar_loc'] == 'left':
            cax = fig.add_axes([0.1, 0.15, 0.025, 0.7])
        
        levels = (mpl.ticker.MaxNLocator(nbins=fill_vars[var]['params']['levels']).
                  tick_values(fill_vars[var]['params']['vmin'],fill_vars[var]['params']['vmax']))
        norm = mpl.colors.BoundaryNorm(levels, ncolors=fill_vars[var]['params']['cmap'].N, clip=True)
        sm = plt.cm.ScalarMappable(cmap=fill_vars[var]['params']['cmap'],norm=norm)
        cbar = plt.colorbar(sm,cax=cax)
        cbar.set_label(fill_vars[var]['label'],fontsize=15)
        cbar.ax.tick_params(labelsize=14)
        
        if fill_vars[var]['cbar_loc'] == 'left':
            cbar.ax.yaxis.set_ticks_position('left')
            cbar.ax.yaxis.set_label_position('left')


    #------- Colorbar -------
    if save_fig:
        utility_print(output_fn)
    
# Source: test_wrapper_figure10.ipynb
def figure_scatter(dss,
                   plot_vars = ['hdiff','unstable'],
                   plot_type = 'levels', # or 'levels'
                   y_var = 'pr',
                   seasons = ['long_rains','short_rains'],
                   seas_names = {'long_dry':r'$\mathit{Jilaal}$'+'\nLong dry period',
                               'long_rains':r'$\mathit{Gu}$'+'\nLong rains',
                               'short_dry':r'$\mathit{Xagaa}$'+'\nShort dry period',
                               'short_rains':r'$\mathit{Deyr}$'+'\nShort rains'},
                   show_corrs = True,
                   subset_color = 'tab:red',
                   year_subsets = None,
                   msize = 10,
                   y_label_str = r'$\Delta \overline{P}$ [mm/day]',
                   x_label_strs = [r'$\Delta \overline{(h_s-h^*)}$ [kJ/kg]',
                                   r'$\Delta \overline{(frac.\ unstable)}$'],
                   label_add = '',
                   save_fig = False,
                   output_fn = '',
                   ylims = [-1.5,1.5],
                   xlims = [[-1.2,1.2],[-0.10,0.10]]
                  ):
    """ Panel of scatterplots 
    
    
    Parameters
    ------------------
    dss : xr.Dataset
        A dataset containing at least the variables in 
        `plot_vars` and `y_var`, with common `year` and
        `season' dimensions
        
    plot_vars : list, by default ['hdiff','unstable']
        A list of variables to plot on the x-axes; each 
        variable will be given its own row in the plot
        
    plot_type : str, by default 'changes'
        if 'levels': plots a simple scatterplot between 
                     the `y_var` and `plot_vars` 
        if 'changes': plots a scatterplot between changes
                      in the `y_var` and changes in the 
                      `plot_vars`
        
    y_var : str, by default 'pr'
        The variable to plot on the y-axis. 
        
    show_corrs : bool, by defualt True
        If True, show linear correlation between plot data
        in top left corner of subplots
        
    seasons : list, by default ['long_rains','short_rains']
        A list of seasons to plot; each season will be 
        given its own column in the plot. The seasons must
        be present in a `season` dimension in `dss`
        
    seas_names : dict, by default based on the default `seasons`
        A dict mapping season names (from `seasons` above) to 
        season names in column headers, to be used as titles
        
    year_subsets : dict, ... a bit complex of a setup
        dict, with keys being season names (for which season you
        want the years highlighted)
    
        If `plot_type == 'levels'`:
            A list of years to highlight. If there are just two,
            and they're only 1 year apart, then an arrow will be
            drawn from the older to the newer one. 
        If `plot_type == 'changes'`:
            A list of lists, for each set of year changes to highlight,
            with the sublists being of the form 
                `[initial_year,end_year,color]`
                (e.g., [2017,2017,'tab:red'])
        
    msize: float, by default 10
        Size of markers, piped into `plt.scatter()`
        
    y_label_str : str, by default "r'$\Delta \overline{P}$ [mm/day]'"
        Second row of y-axis labels, after "Year-on-year"
        
    x_label_strs : list, by default based on `plot_vars` above
        Second half of x-axis labels, after "Year on year", 
        Must be a list the same length as `plot_vars`
        
    save_fig : bool, by default False
        Whether to print figure (using standard `.pdf` and `.png` 
        outputs)
        
    output_fn : str, by default ''
        Filename to use if `save_fig=True`
    
    
    Returns
    ------------------
    fig, ax
        The figure and axis handles
    
    
    """
    
    # Get year-to-year difference
    if plot_type == 'changes':
        dss = xr.merge([dss.isel(year=slice(1,None))[var].values - 
                        dss.isel(year=slice(0,-1))[var] for var in dss])
    elif plot_type != 'levels':
        raise KeyError('`plot_type` must be "levels" or "change"')

    #------------------- Create figure --------------------
    fig = plt.figure(figsize=(4*len(plot_vars),8))

    for var,row_idx in zip(plot_vars,np.arange(0,len(plot_vars))):
        for seas,seas_idx in zip(seasons,np.arange(0,len(seasons))):
            ax = plt.subplot(len(plot_vars),2,row_idx*2+seas_idx+1)

            #--------- Plot ---------
            plot_data = [dss.sel(season=seas)[var],
                         dss.sel(season=seas)[y_var]]

            plt.scatter(*plot_data,s=msize,color='tab:blue')

            if year_subsets is not None:
                if type(year_subsets) == dict:
                    if seas in [k for k in year_subsets]:
                        year_subsets_tmp = year_subsets[seas]
                    else:
                        year_subsets_tmp = []
                else:
                    year_subsets_tmp = year_subsets
                
                if plot_type == 'levels':
                    # If two adjacent years are listed, draw arrows between them
                    if (len(year_subsets_tmp) == 2) and (np.abs(np.diff(year_subsets_tmp))==1):
                        ax.annotate('',
                                   **{k:[pd.sel(year=y).values for pd in plot_data]
                                      for k,y in zip(['xytext','xy'],year_subsets[seas])},
                                    arrowprops={'arrowstyle':'->'},
                                   xycoords='data',textcoords='data')

                    label = ', '.join([str(y) for y in year_subsets_tmp])

                    plt.scatter(*[pd.sel(year=year_subsets_tmp) for pd in plot_data],
                                s=msize,c=subset_color,label=label)
                    
                elif plot_type == 'changes':
                    for ys in year_subsets_tmp:
                        if len(ys) == 2:
                            ys = [ys[0],ys[0],ys[1]]

                        label = str(ys[0])+'-'+str(ys[0]+1)

                        plt.scatter(plot_data[0].isel(year=((plot_data[0].year>=ys[0]) & (plot_data[0].year<=ys[1]))),
                                    plot_data[1].isel(year=((plot_data[1].year>=ys[0]) & (plot_data[1].year<=ys[1]))),
                                    color=ys[2],label=label,s=msize)


            # Calculate and list correlation
            if show_corrs: 
                corr = xr.corr(*plot_data)
                ax.annotate('corr: '+str(np.round(corr.values,2)),
                            (0.05,0.95),
                            ha='left',va='top',xycoords='axes fraction')

            #--------- Annotations ---------
            if row_idx == 0:
                ax.set_title(seas_names[seas],fontsize=13,fontweight='bold')
            else:
                ax.set_title('')

            ax.set_ylim(*ylims)
            if row_idx == 0:
                ax.set_xlim(xlims[0])
            elif row_idx == 1:
                ax.set_xlim(xlims[1])
                
            # Lines at 0 if axis limits cross 0 
            if np.product(np.sign(ax.get_xlim())) == -1:
                ax.axvline(0,color='k',linestyle='--',linewidth=0.5)
            if np.product(np.sign(ax.get_ylim())) == -1:
                ax.axhline(0,color='k',linestyle='--',linewidth=0.5)
            

            if seas_idx == 0:
                ax.set_ylabel(label_add+'\n'+y_label_str,fontsize=13)
            else:
                ax.set_ylabel('')

            ax.set_xlabel(label_add+r' '+x_label_strs[row_idx],fontsize=13)
            
            if (year_subsets is not None):
                if type(year_subsets) is dict:
                    if (seas in [k for k in year_subsets]):
                        ax.legend()
                else:
                     if (row_idx == 0) and (seas_idx == 1):
                        plt.legend(borderaxespad=0.5, 
                               bbox_to_anchor=(1,0.7),
                           loc="upper left")
                        
            # Subplot lettering
            ax.annotate(string.ascii_lowercase[row_idx*2+seas_idx]+'.',
                                    [0.01,1.01],xycoords='axes fraction',
                                    va='bottom',ha='left',fontsize=13,fontweight='bold')


    plt.subplots_adjust(hspace=0.3)
    
    if save_fig:
        utility_print(output_fn)
    
    return fig,ax
    
    
def figure_hdiff_hists(dss,ts_idxs,
                       vardict = {'hist':'hdiff','cond':'pr'},
                       weights=None, 
                       pr_proc = 'density',
                       bins=np.arange(-35000,20000,1000),
                       xscale = 1/1000,
                       titles = ['Long rains','Short rains','Long rains - short rains'],
                       xlabel_add = '',
                       show_titles=True,
                       show_xlabels=True,
                       show_diff_panel=True,
                       show_pos_pct=True,
                       show_legend=True,
                       save_fig=False,output_fn=None,
                       fig=None,axs=None,
                       plot_vars=['hist','cond'],
                       cond_titles = {'density':'Fraction of total rainfall',
                                      'probability':r'Fraction of rainy grid cell days',
                                      'mean':r'$\overline{P}$ [mm/day]'},
                       add_grid=True,
                       return_fig_params=False):
    """ Rainfall by histogram bin of another variable
    
    Part of the Figure 3 workflow
    
    """
    
    
    #---------------- Weights setup ----------------
    if (weights is not None) and (type(weights) == str) and (weights == 'calculate'):
        weights = area_grid(dss.lat,dss.lon)
        weights = weights/weights.mean()

        # Broadcast
        weights = weights.expand_dims({'time':dss.time}).transpose('time','lat','lon')

    
    #---------------- Calculate histograms ----------------
    ##### Stability
    # Subset by season
    stab_tmp = [(dss[vardict['hist']].transpose('time','lat','lon').
                               values.flatten())[(ts_idxs.isel(season=seas_idx).
                                                  transpose('time','lat','lon').
                                                  astype(bool).values.flatten())]
                for seas_idx in np.arange(0,2)]
    
    
    # Calculate histogram for hdiff grid cell days
    if weights is None:
        hist_kwargs = {}
    else:
        hist_kwargs = [{'weights':weights.values.flatten()[(ts_idxs.isel(season=seas_idx).
                                                  transpose('time','lat','lon').
                                                  astype(bool).values.flatten())]}
                       for seas_idx in np.arange(0,2)]
    plot_data = [np.histogram(stab_tmp[stab_idx],
                             bins=bins,**hist_kwargs[stab_idx])[0]
                     for stab_idx in np.arange(0,2)]
   
    # Normalize
    plot_data = [x/np.sum(x) for x in plot_data]
    
    # Get difference in histogram between long and short rains 
    plot_data = [*plot_data,plot_data[0]-plot_data[1]]
    
    ##### Precipitation
    # Calculate histogram by stability for precipitation
    # Subset precipitation to seasons like hdiff
    pr_tmp = [(dss[vardict['cond']].transpose('time','lat','lon').
               values.flatten())[(ts_idxs.isel(season=seas_idx).transpose('time','lat','lon').
                                  astype(bool).values.flatten())]
              for seas_idx in np.arange(0,2)]
    
    if pr_proc == 'density':
        def pr_func(x):
            return np.sum(x)
    elif pr_proc == 'probability':
        def pr_func(x):
            return (np.sum(x>0)/len(x))
    elif pr_proc == 'mean':
        def pr_func(x):
            return (np.mean(x))

    # Get total precipitation that falls on grid cell days of a specific
    # bin of hdiff
    if weights is None:
        pr_tmp = [(xr.DataArray(pr_tmp[seas_idx],dims=['allv']).
                   groupby(xr.DataArray(np.digitize(stab_tmp[seas_idx],bins),
                                        dims=['allv']).where(~np.isnan(stab_tmp[seas_idx]))).
                   map(pr_func))
                   for seas_idx in np.arange(0,2)]
    else:
        # Since normalizing below anyways, this should accurately weight
        # the relative sum 
        pr_tmp = [((xr.DataArray(pr_tmp[seas_idx],dims=['allv'])*hist_kwargs[seas_idx]['weights']).
                   groupby(xr.DataArray(np.digitize(stab_tmp[seas_idx],bins),
                                        dims=['allv']).where(~np.isnan(stab_tmp[seas_idx]))).
                   map(pr_func))
                   for seas_idx in np.arange(0,2)] 
    
    # Normalize
    if pr_proc == 'density':
        pr_tmp = [x/x.sum() for x in pr_tmp]

    # Get difference in histogram between long and short rains 
    pr_tmp = [*pr_tmp,pr_tmp[0]-pr_tmp[1]]
    
    
    #---------------- Plot ----------------
    plot_bins = (bins[0:-1]+np.diff(bins)/2)*xscale
    
    if fig is None:
        fig = plt.figure(figsize=(12,4))
        
    if show_diff_panel:
        plt_idxs = np.arange(0,3)
    else:
        plt_idxs = np.arange(0,2)
        
    for plt_idx in plt_idxs:
        if axs is None:
            ax = plt.subplot(1,3,plt_idx+1)
        else:
            ax = axs[plt_idx]

        # Plot bars (it seems that digitize starts counting at 
        # 1 instead of 0, so the "-1" fixes the off-by-one error)
        if pr_proc == 'density':
            br = ax.bar(plot_bins[pr_tmp[plt_idx].group.astype(int)-1],
                     pr_tmp[plt_idx],color='#9DE0AD',label = r'$P$ fraction') 
        elif len(plot_vars) == 2:
            # If the precip variable isn't density, then the two 
            # plot variables are technically on different y-axes. 
            # Make it so. 
            axr = ax.twinx()
            br = axr.bar(plot_bins[pr_tmp[plt_idx].group.astype(int)-1],
                         pr_tmp[plt_idx],color='#9DE0AD',label = r'$P$')
            
            # Make 'older' axis be up front so the bar chart doesn't 
            # overplot everything. 
            ax.set_zorder(10)
            ax.patch.set_visible(False)

        # Add reference lines
        ax.axhline(0,color='k',linestyle=':')
        ax.axvline(0,color='k',linestyle='-')
       
        # Plot stability histogram as a line
        if 'hist' in plot_vars:
            ax.plot(plot_bins,plot_data[plt_idx],color='tab:red')
        # Or, if only one variability is being plotted and it's not the stability
        # plot as a line as well instead of a bar plot
        else:
            if 'cond' in plot_vars:
                ax.plot(plot_bins[pr_tmp[plt_idx].group.astype(int)-1],pr_tmp[plt_idx],color='tab:green')
        
        #------- Annotate ------
        # X-axis labels
        if show_xlabels:
            ax.set_xlabel(xlabel_add + r'$\Delta($'*int(plt_idx==2)+r'$h_s - h^*$'+r'$)$'*int(plt_idx==2)+' [kJ / kg]',
                          fontsize = 15)
        else:
            ax.set_xlabel('')
        if (pr_proc != 'density') and (len(plot_vars) == 2):
            axr.set_xlabel('')
            
        # Y-axis labels
        if plt_idx == 0:
            if weights is not None:
                #ax.set_ylabel('Day-area fraction')
                ax.set_ylabel('Density')
            else:
                ax.set_ylabel('Fraction of grid cell days')
        if (pr_proc != 'density') and (len(plot_vars) == 2) and (plt_idx == 0):
            axr.set_ylabel('')
            
        # Secondary y-axis labels
        if 'cond' in plot_vars:
            if (pr_proc != 'density') and (len(plot_vars) == 2) and (plt_idx == 1):
                axr.set_ylabel(cond_titles[pr_proc])
            elif (len(plot_vars) == 1) and (plt_idx == 0):
                ax.set_ylabel(cond_titles[pr_proc])
            
        # Titles
        if show_titles:
            ax.set_title(titles[plt_idx])
            
        # Y-axis existence
        if plt_idx == 1:
            ax.tick_params(axis='y', which='both',left=False,labelleft=False)
        if (pr_proc != 'density') and (len(plot_vars) == 2):
            if plt_idx == 0:
                axr.tick_params(axis='y',which='both',right=False,labelright=False)
            
            
        # X-axis limits, as the bins that have non-zero density
        xlims = (plot_bins[np.max([0,
                                  np.min([np.nonzero(x>0)[0][0] for x in plot_data])-1])],
                      plot_bins[np.min([len(bins),
                                  np.max([np.nonzero(x>0)[0][-1] for x in plot_data])-1])])
        ax.set_xlim(xlims)
        if (pr_proc != 'density') and (len(plot_vars) == 2):
            axr.set_xlim(xlims)
        
        # Y axis limits
        if plt_idx < 2:
            if (pr_proc == 'density') or (len(plot_vars) < 2):
                ax.set_ylim(0,
                        np.max([*[np.max(pr).values for pr in pr_tmp],*[np.max(pd) for pd in plot_data[0:2]]])+0.025)
            elif len(plot_vars) < 2:
                if 'hist' in plot_vars:
                    ax.set_ylim(0,
                            np.max([np.max(pd) for pd in plot_data[0:2]])+0.025)
                elif 'cond' in plot_vars:
                    ax.set_ylim(0,
                             np.max([np.max(pr).values for pr in pr_tmp])+0.025)
            else:
                ax.set_ylim(0,
                            np.max([np.max(pd) for pd in plot_data[0:2]])+0.025)
                axr.set_ylim(0,
                             np.max([np.max(pr).values for pr in pr_tmp])+0.025)
                
        # Add axis grid
        if add_grid:
            ax.grid()
        
            
        # Add percent of unstable grid cell days    
        if show_pos_pct and (plt_idx < 2):
            frac_value = np.sum(plot_data[plt_idx][plot_bins>0])
            #frac_value = np.sum((stab_tmp[plt_idx]>0)*hist_kwargs[plt_idx]['weights'])/np.sum(hist_kwargs[plt_idx]['weights'])
            #frac_value = frac_value / (np.sum((~np.isnan(stab_tmp[plt_idx]))*hist_kwargs[plt_idx]['weights'])/
            #                           np.sum(hist_kwargs[plt_idx]['weights']))
            
            ax.annotate(str(int(np.round(frac_value*100)))+'%',
                        [0.98,0.98],xycoords='axes fraction',ha='right',va='top')
            
        if show_legend and (plt_idx == 0):
            ax.legend(handles=[br],loc='upper left')
            #ax.legend(handles=[vl,br])
            
    #---------------- Save ----------------
    if save_fig:
        plt.savefig(output_fn+'.png',dpi=300)
        print(output_fn+'.png saved!')
        plt.savefig(output_fn+'.pdf')
        print(output_fn+'.pdf saved!')
        
    #---------------- Return ----------------
    if return_fig_params:
        return fig
    
def figure_hdiff_boxplots(dss,ts_idxs,
                          ylims=[0,50],
                          vardict = {'hist':'hdiff','cond':'pr'},
                          bins=np.arange(-35000,20000,1000),
                           xscale = 1/1000,
                          axs=None,fig=None,
                          xlabel_add = '',
                          titles = ['Long rains','Short rains','Long rains - short rains'],
                          show_titles=True,
                          show_xlabels=True,
                          show_legend=True,
                          xlims=None,
                          add_grid=True):

    """ Distribution of rainfall within histogram bins of another variable
    
    Part of the Figure 3 workflow
    
    """
    
    #----------------- Setup -----------------
    # Calculate histogram by stability for precipitation
    # Subset precipitation to seasons like hdiff
    pr_tmp = [(dss[vardict['cond']].transpose('time','lat','lon').
               values.flatten())[(ts_idxs.isel(season=seas_idx).transpose('time','lat','lon').
                                  astype(bool).values.flatten())]
              for seas_idx in np.arange(0,2)]
    
    # Subset by season
    stab_tmp = [(dss[vardict['hist']].transpose('time','lat','lon').
                               values.flatten())[(ts_idxs.isel(season=seas_idx).
                                                  transpose('time','lat','lon').
                                                  astype(bool).values.flatten())]
                for seas_idx in np.arange(0,2)]

    
    plot_bins = (bins[0:-1]+np.diff(bins)/2)/1000

    #---------------- Plot -----------------
    
    if fig is None:
        fig = plt.figure(figsize=(10,4))
    
    for seas_idx in np.arange(0,2):
        if axs is None:
            ax = plt.subplot(1,2,seas_idx+1)
        else:
            ax = axs[seas_idx]

        bin_idxs = xr.DataArray(np.digitize(stab_tmp[seas_idx],bins),
                                    dims=['allv']).where(~np.isnan(stab_tmp[seas_idx]))

        #----------- Quantiles -----------
        qs = [((xr.DataArray(pr_tmp[seas_idx],dims=['allv'])).
                           groupby(xr.DataArray(np.digitize(stab_tmp[seas_idx],bins),
                                                dims=['allv']).where(~np.isnan(stab_tmp[seas_idx])))).quantile(q) 
                          for q in [0.05,0.95,0.99,0.999]]

        ax.fill_between(plot_bins[np.unique(bin_idxs)[0:-1].astype(int)],
                         *qs[0:2],
                        facecolor='#9DE0AD',
                        alpha=0.3,label='0.05-0.95')

        ax.plot(plot_bins[np.unique(bin_idxs)[0:-1].astype(int)],
                qs[2],color='#9DE0AD',linestyle='--',label='0.99')

        ax.plot(plot_bins[np.unique(bin_idxs)[0:-1].astype(int)],
                qs[3],color='#9DE0AD',linestyle=':',label='0.999')

        #----------- Boxplots -----------
        for bin_idx in np.unique(bin_idxs):
            if ~np.isnan(bin_idx):
                plot_data = pr_tmp[seas_idx][bin_idxs == bin_idx]
                plot_data = plot_data[~np.isnan(plot_data)]
                bx = ax.boxplot(plot_data,
                           positions=[plot_bins[int(bin_idx)]],showfliers=False,
                           widths=[np.mean(np.diff(plot_bins))-0.5],patch_artist=True)

                for item in ['boxes', 'whiskers', 'fliers', 'caps']:
                    plt.setp(bx[item], color='tab:green')
                plt.setp(bx['medians'],color='k')

        ax.set_ylim(*ylims)


        #----------- Maximum rainfall -----------

        text_kwargs = {'color':'darkgreen','ha':'center','rotation':90,
                       'fontsize':9}

        # Calculate max rainfall in each bin
        maxs = ((xr.DataArray(pr_tmp[seas_idx],dims=['allv'])).
                           groupby(xr.DataArray(np.digitize(stab_tmp[seas_idx],bins),
                                                dims=['allv']).where(~np.isnan(stab_tmp[seas_idx])))).max()

        for m_idx in np.arange(0,maxs.sizes['group']):
            #if (m_idx % 2) == 0:
            m = np.round(maxs)[m_idx]
            if m > 0:
                if m<ylims[1]-5:
                    ax.annotate(str(int(m.values)),
                                (plot_bins[m.group.astype(int)],
                                 m.values),
                                 **text_kwargs,va='bottom')
                else:
                    ax.annotate(str(int(m.values)),
                                (plot_bins[m.group.astype(int)],
                                 ylims[1]),
                                 **text_kwargs,va='top')


        #----------- Annotations -----------
        ax.set_xticks(np.arange(-30,15,10),
                   [str(k) for k in np.arange(-30,15,10)])

        # Add reference lines
        ax.axvline(0,color='k')
        
        # X-axis limits, as the bins that have non-zero density
        if xlims is None:
            xlims = (plot_bins[np.max([0,
                                      np.min([np.nonzero(x>0)[0][0] for x in plot_data])-1])],
                          plot_bins[np.min([len(bins),
                                      np.max([np.nonzero(x>0)[0][-1] for x in plot_data])-1])])
        ax.set_xlim(xlims)
                
        # Titles
        if show_titles:
            ax.set_title(titles[seas_idx])
            
        # X-axis labels
        if show_xlabels:
            ax.set_xlabel(xlabel_add+r'$\Delta($'*int(seas_idx==2)+r'$h_s - h^*$'+r'$)$'*int(seas_idx==2)+' [kJ / kg]',
                          fontsize=15)
        else:
            ax.set_xlabel('')
            
        # Y-axis labels
        if seas_idx == 0:
            ax.set_ylabel(r'$P$ [mm/day]')
            
        # Y-axis existence
        if seas_idx == 1:
            ax.tick_params(axis='y', which='both',left=False,labelleft=False)

        if add_grid:
            ax.grid()

        if show_legend and (seas_idx == 0):
            ax.legend()
    
    
def wrapper_prhdiff_figure(dss,ts_idxs,
                          vardict = {'hist':'hdiff','cond':'pr'},
                          save_fig=False,
                          output_fn=None,
                          use_boxes=True,
                          use_hdiffmax=False,
                          xlabel_add = '',
                          titles = ['Long rains','Short rains','Long - short rains'],
                          box_bins=np.arange(-35000,20000,2000)):
    """ Creates multipanel rainfall-by-histogram-bin figures
    
    Part of the Figure 3 workflow
    
    """
    
    #------- Plot -------
    fig,axs = plt.subplots(3,3,figsize=(12,12))
    
    # Total rainfall by bin
    figure_hdiff_hists(dss,ts_idxs,
                       vardict = vardict,
                       weights='calculate',
                       fig=fig,axs=axs[0],
                       show_xlabels=False,
                       titles = titles)
    
    # Distribution of rainfall within bins
    if use_boxes:
        figure_hdiff_boxplots(dss,ts_idxs,
                              vardict = vardict,
                              fig=fig,axs=axs[1],
                              bins=box_bins,
                              xlims=[-33.5,13.5],
                              xlabel_add = xlabel_add,
                              show_legend=False,
                              show_titles=False,show_xlabels=False)
    else:
        figure_hdiff_hists(dss,ts_idxs,
                           vardict = vardict,
                           weights='calculate',
                           pr_proc='mean',
                           fig=fig,axs=axs[1],
                           xlabel_add = xlabel_add,
                           show_titles=False,show_xlabels=False,
                           show_legend=False,show_pos_pct=False,
                           show_diff_panel=False,
                           plot_vars = ['cond'])

    # Fraction of bins with rain
    figure_hdiff_hists(dss,ts_idxs,
                       vardict = vardict,
                       weights='calculate',
                       pr_proc='probability',
                       fig=fig,axs=axs[2],
                       xlabel_add = xlabel_add,
                       show_titles=False,
                       show_legend=False,show_pos_pct=False,
                       show_diff_panel=False,
                       plot_vars = ['cond'])


    #------- Further processing -------
    # Remove unneeded axes
    for ax in [axs[1][2],axs[2][2]]:
        ax.set_visible(False)

    # Fill in missing xlabel
    #axs[0][2].set_xlabel(r'$h_s-h^*_{650}$ [kJ / kg]',fontsize=13)
    axs[0][2].set_xlabel(xlabel_add + r'$h_s-h^*$ [kJ / kg]',fontsize=15)


    # Bolden titles
    for ax in axs[0]:
        ax.title.set_fontweight('bold')
        ax.title.set_fontsize(15)

    # Enlarge y axes labels
    for ax in [x[0] for x in axs]:
        ax.yaxis.label.set_fontsize(13)

    # Enlarge x axes labels
    for ax in axs[2]:
        ax.xaxis.label.set_fontsize(15)

    # Subplot lettering
    ax_idxs = [[0,0],[0,1],[0,2],
               [1,0],[1,1],
               [2,0],[2,1]]
    for ax_idx in np.arange(0,len(ax_idxs)):
        axs[ax_idxs[ax_idx][0]][ax_idxs[ax_idx][1]].annotate(string.ascii_lowercase[ax_idx]+'.',
                                      [0.01,1.01],xycoords='axes fraction',
                                      va='bottom',ha='left',fontsize=13,fontweight='bold')
        
    # Add readable legend to side of second panel for boxplots
    if use_boxes:
        # From https://stackoverflow.com/questions/27174425/how-to-add-a-string-as-the-artist-in-matplotlib-legend 
        # to allow text as a legend patch 
        class AnyObject(object):
            def __init__(self, text, color,label):
                self.text = text
                self.color = color
                self.label = label

            def get_label(self):
                return self.label

        class AnyObjectHandler(object):
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                x0, y0 = handlebox.xdescent, handlebox.ydescent
                width, height = handlebox.width, handlebox.height
                patch = mtext.Text(x=0, y=0, text=orig_handle.text, color=orig_handle.color, verticalalignment=u'baseline', 
                                        horizontalalignment=u'left', multialignment=None, 
                                        fontproperties=None, rotation=90, linespacing=None, 
                                        rotation_mode=None)
                handlebox.add_artist(patch)
                return patch
    
        text_patch = AnyObject('12','tab:green',r'max $P$ [mm/day]')

        # Build legend elements
        legend_elements = [mpatches.Patch(facecolor='#9DE0AD',alpha=0.3, edgecolor='none',
                                         label='0.05-0.95'),
                           mlines.Line2D([0], [0], linestyle='--',color='#9DE0AD', label='0.99'),
                           mlines.Line2D([0], [0], linestyle=':', color='#9DE0AD', label='0.999'),
                           text_patch]
        
        axs[1][1].legend(handles=legend_elements,
                         handler_map={text_patch:AnyObjectHandler()},title='Quantile',
                         borderaxespad=0.5, 
                         bbox_to_anchor=(1.15,0.7),loc="upper left")

    #------- Save figure -------  
    if save_fig:
        utility_print(output_fn)
        
def figure_pr_mse_trends(dsy,seas_idx=0,mean_kind='month',yrs=[1998,2008],
                         seas_names = {'season':['Long dry period','Long rains','Short dry period','Short rains'],
                                       'month':['JF','MAM','JJAS','OND']},
                         titles = {'pr':r'$P$','hdiff':r'$h_s-h^*$',
                                      'hus':r'$L_vq_s$','ta':r'$c_pT_s$','hsat':r'$-h^*$',
                                  'unstable':r'$\%$ days of $h_s-h^*>0$'},
                         scales = None,
                         mse_vars = ['hdiff','hus','ta','hsat'],
                         cmaps = {'mse':cmocean.cm.curl_r,'pr':cmocean.cm.balance_r},
                         clims = {'mse':{'vmin':-0.5,'vmax':0.5,'levels':11},'pr':{'vmin':-0.5,'vmax':0.5,'levels':11}},
                         dir_list = dir_list,
                         save_fig=False,output_fn=None,
                        ):
    """ Maps of changes or trends in P, h_s-h* components
    Primarily used for Main Text Figure 12
    
    Parameters
    ---------------
    dsy : xr.Dataset
        Dataset with `P`, `hsat`, `h`, `ta`, `hus` variables 
        and season, year, lat, lon dimensions (output from 
        `calculate_seasmeans()`) 
    
    """

    if scales is None:
        scales = {k:(-1 if k == 'hsat' else 1) for k in titles}
    
    if yrs is None:
        yrs = [dsy.year.min(),dsy.year.max()]

    if type(yrs[0]) == list:
        if len(yrs[0])==2:
            subset_params = [{'year':slice(yrs[0],yrs[0]+1)},
                             {'year':slice(yrs[1],yrs[1]+1)}]
        else:
            subset_params = [{'year':yr} for yr in yrs]
    else:
        subset_params = {'year':slice(*yrs)}

    #----------------- Figure -----------------
    fig,axs = plt.subplots(nrows=2,ncols=3,figsize=(12,6),
                           subplot_kw={'projection': ccrs.PlateCarree()})


    #------ Precipitation plot ------
    # Plot trends
    if type(yrs[0]) == list:
        ((dsy['pr'].isel(season=seas_idx).sel(kind=mean_kind,**subset_params[1]).mean('year') - 
          dsy['pr'].isel(season=seas_idx).sel(kind=mean_kind,**subset_params[0]).mean('year')).
         plot.contourf(ax=axs[0,0],transform=ccrs.PlateCarree(),
                       cmap=cmaps['pr'],**clims['pr'],add_colorbar=False,
                       zorder=1))
    else:
        ((dsy['pr'].isel(season=seas_idx).sel(kind=mean_kind,**subset_params).
         polyfit(dim='year',deg=1)['polyfit_coefficients'].isel(degree=0)).
         plot.contourf(ax=axs[0,0],transform=ccrs.PlateCarree(),
                       cmap=cmaps['pr'],**clims['pr'],add_colorbar=False,
                       zorder=1))

    # Title
    axs[0,0].set_title(titles['pr'],fontsize=15)

    # Make colorbar
    cax = fig.add_axes([0.1, 0.535, 0.015, 0.35],zorder=2)
    levels = mpl.ticker.MaxNLocator(nbins=clims['pr']['levels']).tick_values(clims['pr']['vmin'],clims['pr']['vmax'])
    norm = mpl.colors.BoundaryNorm(levels, ncolors=cmaps['pr'].N, clip=True)
    sm = plt.cm.ScalarMappable(cmap=cmaps['pr'],norm=norm)
    cb = plt.colorbar(sm,cax=cax)
    if type(yrs[0]) == list:
        cb.set_label(label=r'$\Delta P$ [mm/day]',fontsize=13)
    else:
        cb.set_label(label=r'$P$ trends [mm/day/year]',fontsize=13)
    # Flip side of colorbar text
    cax.yaxis.set_ticks_position('left')
    cax.yaxis.set_label_position('left')

    # Make box around subplot
    fig.patches.extend([mpatches.FancyBboxPatch((0.03,0.52),0.345,0.4,
                                                edgecolor='none',
                                                 boxstyle='round',
                                                facecolor='lightgrey',alpha=0.2,zorder=0,
                                      transform=fig.transFigure, figure=fig,
                                               mutation_scale=0.025)])
    # Make sure plot axis is still on top of grey box
    axs[0,0].set_zorder(1)
    
    # Subplot lettering
    axs[0,0].annotate('a.',(0.025,0.975),xycoords='axes fraction',
                      ha='left',va='top',fontsize=15,fontweight='bold')



    #------ MSE plots ------
    for var,loc,label in zip(mse_vars,[[0,1],[1,0],[1,1],[1,2]],['b.','c.','d.','e.']):
        if type(yrs[0]) == list:
            (((dsy[var].isel(season=seas_idx).sel(kind=mean_kind,**subset_params[1]).mean('year') - 
              dsy[var].isel(season=seas_idx).sel(kind=mean_kind,**subset_params[0]).mean('year')) * 
              scales[var]).
             plot.contourf(ax=axs[loc[0],loc[1]],transform=ccrs.PlateCarree(),
                           cmap=cmaps['mse'],**clims['mse'],add_colorbar=False))
        else:
            ((dsy[var].isel(season=seas_idx).sel(kind=mean_kind,**subset_params).
             polyfit(dim='year',deg=1)['polyfit_coefficients'].isel(degree=0) * 
             scales[var]).
             plot.contourf(ax=axs[loc[0],loc[1]],transform=ccrs.PlateCarree(),
                           cmap=cmaps['mse'],**clims['mse'],add_colorbar=False))

        # Title
        axs[loc[0],loc[1]].set_title(titles[var],fontsize=15)
        
        # Subplot lettering
        axs[loc[0],loc[1]].annotate(label,(0.025,0.975),xycoords='axes fraction',
                                    ha='left',va='top',fontsize=15,fontweight='bold')


    # Colorbar
    fig.subplots_adjust(bottom=0.125)
    cax = fig.add_axes([0.25,0.08,0.5,0.03])
    levels = mpl.ticker.MaxNLocator(nbins=clims['mse']['levels']).tick_values(clims['mse']['vmin'],clims['mse']['vmax'])
    norm = mpl.colors.BoundaryNorm(levels, ncolors=cmaps['mse'].N, clip=True)
    sm = plt.cm.ScalarMappable(cmap=cmaps['mse'],norm=norm)
    cb = plt.colorbar(sm,cax=cax,orientation='horizontal')
    if type(yrs[0]) == list:
        cb.set_label(label=r'$\Delta$MSE [kJ/kg]',fontsize=13)
    else:
        cb.set_label(label='MSE trends [kJ/kg/year]',fontsize=13)


    #------ Annotations and figure adjustments ------
    # Remove top right axis
    axs[0,2].remove()

    # Put in text in that spot instead
    if type(yrs[0]) == list:
        axs[0,1].annotate('Change in '+seas_names[mean_kind][seas_idx]+'\n'+
                          '-'.join([str(y) for y in yrs[0]]) + ' vs. ' + '-'.join([str(y) for y in yrs[1]]),(1.3,0.7),xycoords='axes fraction',
                      ha='left',va='center',fontsize=15,fontweight='bold')

    else:
        axs[0,1].annotate(seas_names[mean_kind][seas_idx]+' trends\n'+'-'.join([str(y) for y in yrs]),(1.3,0.7),xycoords='axes fraction',
                      ha='left',va='center',fontsize=15,fontweight='bold')

    # Add borders, coastlines
    for ax in axs.flatten():
        if ax != axs[0,2]:
            ax.coastlines()
            # Get ISO-standard borders
            gdf = gpd.read_file(dir_list['aux']+'ne_10m_admin_0_countries_iso/ne_10m_admin_0_countries_iso.shp')
            axlims = [*ax.get_xlim(),*ax.get_ylim()]
            gdf.cx[axlims[0]:axlims[1],
                   axlims[2]:axlims[3]].plot(ax=ax,facecolor='none',edgecolor='k',
                                                                   linestyle='-',linewidth=0.15,
                                                                   transform=ccrs.PlateCarree())
            ax.set_extent(axlims)

            #ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
        
    #----------------- Export -----------------
    if save_fig:
        utility_print(output_fn)
        
