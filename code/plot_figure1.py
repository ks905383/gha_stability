# Code to generate main text Figure 1
import xarray as xr
import xagg as xa
import numpy as np
from matplotlib import pyplot as plt
import cartopy.mpl.ticker as cticker
import matplotlib as mpl
from cartopy import crs as ccrs
import cmocean

from funcs_support import get_params
from funcs_load import load_raw
dir_list = get_params()

#--------------- Setup ----------------
output_fn = dir_list['figs']+'figure1'
save_fig=True

map_data = 'GPCP'
cmap = cmocean.cm.rain

line_styles = {'GPCP':'-','CHIRPS':':','TRMM':':','GPCC':':'}
line_colors = {'GPCP':'tab:blue','CHIRPS':'tab:orange','GPCC':'tab:purple','TRMM':'tab:green'}

#--------------- Load and set up data ----------------
# Load datasets
dss = {'GPCP':xa.fix_ds(load_raw('pr_Amon*.nc',search_dir = dir_list['raw']+'GPCP/').
              drop(('time_bnds','lat_bnds','lon_bnds')).rename({'precip':'pr'})),
       'CHIRPS':xa.fix_ds(load_raw('pr_Amon*.nc',search_dir = dir_list['raw']+'CHIRPS/').
                 rename({'precip':'pr'})),
       'GPCC':xa.fix_ds(load_raw('pr_Amon*.nc',search_dir = dir_list['raw']+'GPCC/'))
      }

dss = {k:v.sel(lat=slice(-20,20)) for k,v in dss.items()}

# Convert from mm/month to mm/day for GPCC
dss['GPCC'] = dss['GPCC']/dss['GPCC'].time.dt.daysinmonth
dss['CHIRPS'] = dss['CHIRPS']/dss['CHIRPS'].time.dt.daysinmonth

# Get meridional mean
subset_params = {'lat':slice(-3,12.5),'time':slice('1982-01-01','2021-12-31')}
ds = xr.concat([v.sel(**subset_params).mean(('lat','time')).interp({'lon':np.arange(-180,180)}) for k,v in dss.items()],
               dim='dataset')
ds['dataset'] = [k for k in dss]

#----------------------------------------------------------
#-------------------------- Plot --------------------------
#----------------------------------------------------------

axs = mpl.gridspec.GridSpec(2,1,hspace=0,height_ratios=[2,1])

fig = plt.figure(figsize=(12,5))

#--------------- Meridional mean rainfall ----------------
### Plot
ax = plt.subplot(axs[0])
for dataset_idx in np.arange(0,ds.dims['dataset']):
    ds.isel(dataset=dataset_idx).pr.plot(label=(str(ds['dataset'][dataset_idx].values) + 
                                                ((ds['dataset'][dataset_idx].values not in ['GPCP','TRMM'])*' (land only)')),
                                         linestyle=line_styles[str(ds['dataset'][dataset_idx].values)],
                                         color=line_colors[str(ds['dataset'][dataset_idx].values)])
### Annotate
ax.legend()

ax.set_ylabel('Precipitation [mm/day]',fontsize=13)
ax.set_xlabel('')
# Title
lat_str = str(np.abs(subset_params['lat'].start))
if subset_params['lat'].start<0:
    lat_str = lat_str+r'$^\circ$S to '
else:
    lat_str = lat_str+r'$^\circ$N to '
lat_str = lat_str + str(np.abs(subset_params['lat'].stop))
if subset_params['lat'].stop<0:
    lat_str = lat_str+r'$^\circ$S'
else:
    lat_str = lat_str+r'$^\circ$N'
    
ax.set_title(subset_params['time'].start[0:4]+'-'+subset_params['time'].stop[0:4]+
             ' mean rainfall, avg. over '+lat_str,fontsize=15)

ax.grid()
ax.axvline(32,color='tab:red',linestyle='-')
ax.axvline(55,color='tab:red',linestyle='-')

ax.set_xlim(-178,178)

ax.tick_params(axis='x',which='both',bottom=False,labelbottom=False)

#--------------- Map ----------------
### Plot
ax = plt.subplot(axs[1],projection=ccrs.PlateCarree())
dss[map_data].sel(lat=slice(-20,20)).mean('time').pr.plot.contourf(transform=ccrs.PlateCarree(),
                                                                  cmap=cmap,
                                                                  add_colorbar=False,
                                                                   vmin=0,vmax=15,levels=16)
### Annotate
ax.coastlines()
ax.grid(axis='x')

ax.axhline(subset_params['lat'].start,color='k',linestyle=':')
ax.axhline(subset_params['lat'].stop,color='k',linestyle=':')

ax.plot([32,32],[subset_params['lat'].start,subset_params['lat'].stop],color='tab:red',linestyle='-',transform=ccrs.PlateCarree())
ax.plot([55,55],[subset_params['lat'].start,subset_params['lat'].stop],color='tab:red',linestyle='-',transform=ccrs.PlateCarree())

# Set axis ticks
ax.set_xticks(np.arange(-150,161,50), crs=ccrs.PlateCarree())
ax.set_xticklabels(np.arange(-150,161,50))
ax.set_yticks(np.arange(-15,21,15), crs=ccrs.PlateCarree())
ax.set_yticklabels(np.arange(-15,21,15))
ax.yaxis.tick_left()

# Set axis tick labels
lon_formatter = cticker.LongitudeFormatter()
lat_formatter = cticker.LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

# Add grid
ax.grid(linewidth=2, color='grey',alpha=0.5,linestyle='-')

# Annotate axes
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('')

# Colorbar
fig.subplots_adjust(bottom=0.2)
cax = fig.add_axes([0.336,0.125,0.35,0.025])
levels = mpl.ticker.MaxNLocator(nbins=16).tick_values(0,15)
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
cb = plt.colorbar(sm,cax=cax,orientation='horizontal')
cb.set_label('Precipitation [mm/day]',fontsize=13)

# Add dataset name
ax.annotate(map_data, 
            [5,5],xycoords='axes pixels',
                ha="left", va="bottom",size=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="whitesmoke", ec="k", lw=0.5))

ax.set_extent([-178,178,-19,19],crs=ccrs.PlateCarree())

#--------------- Export ----------------
if save_fig:
    plt.savefig(output_fn+'.png',dpi=300)
    print(output_fn+'.png saved!')
    plt.savefig(output_fn+'.pdf')
    print(output_fn+'.pdf saved!')