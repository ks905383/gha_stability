# Plot Figure S7 - seasonal correlations between 
# GHA hsat and hsat everywhere
import xarray as xr
import xskillscore as xs
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import re
import glob
import matplotlib as mpl
from matplotlib import pyplot as plt
import cartopy
import cartopy.feature as cfeature
from cartopy import crs as ccrs
from shapely.geometry import box
import cmocean
from funcs_load import load_raw
from funcs_support import get_params,area_mean,subset_to_srat,sig_fdr,utility_print
from funcs_process import create_season_mask
import warnings

dir_list = get_params()

#--------------- Setup ----------------
mod = 'MERRA2'
mod_p = 'CHIRPS'
stats_suffix = None
plev = 650
seasons = ['long_dry','long_rains','short_dry','short_rains']
cmap = cmocean.cm.balance

save_fig = True
output_fn = dir_list['figs']+'figure_s7'

bimod_color = 'tab:green'

titles = {'all':'All days',
          'long_dry':'Boreal winter',
          'long_rains':'Long rains',
          'short_dry':'Boreal summer',
          'short_rains':'Short rains'}

#--------------- Load data ----------------
# Load and concatenate hsat files 
dss = [load_raw('hsat_seasavg_*_'+suffix+'*',search_dir = dir_list['proc']+mod+'/',
                load_single=False,subset_params = {'plev':[plev],'kind':['dunning']}) for suffix in ['eq','subtrop']]
dss = xr.concat([dss[0],dss[1].sel(lat=slice(15.01,40))],dim='lat') 

#-------- Load seasonal stats
# Get seasonal ratio
srat = load_raw('pr_doyavg_*_seasstats_*HoA.nc',
                           search_dir=dir_list['proc']+mod_p+'/')
srat = srat.seas_ratio.drop('method')

# Get mean hsat value over Horn of Africa
dsm = area_mean(subset_to_srat(dss))

#--------------- Plot ----------------
fig = plt.figure(figsize=(8.5,8))
for season,ax_idx in zip(seasons,np.arange(0,len(seasons))):
    #-------- Calculate correlations 
    # Make sure all-nan years are removed, these mess up the p-value
    # calculation below (mainly only the first year for the long_dry
    # season, since it spans the new year gap)
    eval_data = [dsm.sel(season=season).hsat,dss.sel(season=season).hsat]
    eval_data = [d.isel(year=(~np.isnan(eval_data[0]))) for d in eval_data]
    
    with warnings.catch_warnings():
        # Ignore the nan slice and dimension warnings for 
        # p-value calculations
        warnings.simplefilter("ignore")
        
        # Calculate correlation between mean hsat over HoA and 
        # hsat everywhere else
        corrs = xr.corr(*eval_data,dim='year')
        # Calculate p-values for correlation
        ps = xs.pearson_r_eff_p_value(*eval_data,dim='year')

        # Get significant locations under multiple testing correction
        locs_sig = sig_fdr(ps)
    
    #-------- Plot data
    ax = plt.subplot(len(seasons),1,ax_idx+1,
                     projection=ccrs.PlateCarree(central_longitude=110))
    
    # Plot correlations
    corrs.plot.contourf(transform=ccrs.PlateCarree(),vmin=-1,vmax=1,levels=21,
                        cmap=cmap,add_colorbar=False)
    # Mask out insignificant locations
    locs_sig.plot.contourf(levels = [-0.1,0.9],colors = 'none',hatches = ['/////',None],
                           transform=ccrs.PlateCarree(),add_colorbar=False)
    
    #-------- Mark bimodal region 
    a = srat.plot.contour(levels=[1],colors=[bimod_color],transform=ccrs.PlateCarree())
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
    bbox = np.array([srat.lat.min(),srat.lat.max().values,srat.lon.min().values,srat.lon.max().values])
    bbox = box(bbox[2],bbox[0],bbox[3],bbox[1])
    gdf = gpd.clip(gdf,mask=bbox)
    gdf.plot(color=bimod_color,transform=ccrs.PlateCarree(),ax=ax)
    
    #-------- Further annotations
    # Coastlines
    ax.coastlines()
    
    # Grid
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                          linewidth=1, color='gray', alpha=0.5, linestyle='-')
    
    # Equatorial line
    ax.axhline(0,color='k',linestyle='--')
    
    # Subplot titles
    ax.set_title(titles[season],fontweight='bold',fontsize=11)
    
# Vertical colorbar
fig.subplots_adjust(hspace=0.25,bottom=0.1)
cax = fig.add_axes([0.21,0.095,0.6,0.025])
levels = mpl.ticker.MaxNLocator(nbins=21).tick_values(-1,1)
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
plt.colorbar(sm,cax=cax,orientation='horizontal',
             label='Interannual correlation between double-peaked region\n'+r'seasonal mean $h^*$ and $h^*$ at all grid cells')


plt.subplots_adjust(hspace=-0.1)

#--------------- Print ----------------
if save_fig:
    utility_print(output_fn)