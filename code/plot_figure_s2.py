# Code to plot supp docs Figure S2, analyzing h_s-h^* across
# pressure levels
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import cmocean
import seaborn as sns

from funcs_support import (get_params,subset_to_srat,area_mean,utility_print)
from funcs_load import load_raw
dir_list = get_params()

#--------------- Setup ----------------
save_fig = True
output_fn = '../figures/figure_s2'

# Which seasonal definitions to use
kind = 'dunning_local'

# Colormap
cmap = cmocean.cm.amp
nbins = 15

# Titles of seasonal subplots
seas_titles = [r'$\mathit{Jilaal}$'+'\nLong dry period',
                                       r'$\mathit{Gu}$'+'\nLong rains',
                                       r'$\mathit{Xagaa}$'+'\nShort dry period',
                                       r'$\mathit{Deyr}$'+'\nShort rains']

#--------------- Load and setup data ----------------
# Load seasonal average h_s-h^*
ds = load_raw('hdiff_seasavg*HoA.nc',
         search_dir=dir_list['proc']+'MERRA2/')

# Calculate area mean 
ds = area_mean(subset_to_srat(ds))

#--------------- Plot --------------
fig = plt.figure(figsize=(15,5))

for seas_idx in np.arange(0,ds.sizes['season']):
    ax = plt.subplot(1,4,seas_idx+1)
    
    # Plot density across years 
    df = ds.hdiff.sel(kind=kind).isel(season=seas_idx).drop(['season','kind']).to_dataframe().reset_index()
    sns.histplot(x=df['hdiff']/1000, y=df['plev'],
             cmap=cmap,ax=ax,
                 vmin=0,vmax=ds.sizes['year'],bins=nbins)
    
    # Plot interannual mean 
    ((ds.hdiff.sel(kind=kind).isel(season=seas_idx)/1000).
     mean('year').plot(y='plev',color='grey',linewidth=4.5,label='mean across years'))
    
    # Invert y axis to make up higher in the atmopshere
    ax.invert_yaxis()
    
    ## Annotations
    ax.set_title(seas_titles[seas_idx])
    
    ax.set_xlabel(r'$h_s-h^*$ [kJ/kg]')
    
    ax.set_xlim(-25,2.5)
    ax.axvline(0,color='k',linestyle='--')
    
    # Reference line at 650 hPa, at which most of the 
    # paper's analysis is conducted
    ax.axhline(650,color='grey',linestyle=':')
    
    if seas_idx != 0:
        ax.tick_params(axis='y', which='both',left=False,right=False,labelleft=False)
        ax.set_ylabel('')
    else:
        ax.set_ylabel('Pressure level [hPa]')
        ax.legend()
    ax.grid()
    
    
# Vertical colorbar
fig.subplots_adjust(right=0.825)
cax = fig.add_axes([0.875, 0.15, 0.025, 0.7])
levels = mpl.ticker.MaxNLocator(nbins=nbins).tick_values(0,ds.sizes['year'])
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
plt.colorbar(sm,cax=cax,label='Number of years')
#--------------- Save --------------
if save_fig:
    utility_print(output_fn)