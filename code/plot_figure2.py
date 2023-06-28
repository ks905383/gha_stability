# Code to plot main text Figure 2
import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
import cartopy
from cartopy import crs as ccrs
from shapely import geometry
import regionmask
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker
import cmocean
from scipy import stats as sstats

from funcs_support import get_params, get_subset_params, area_mean, utility_print
from funcs_load import load_raw
dir_list = get_params()
subset_params_all = get_subset_params()

#--------------- Setup ----------------
add_geom = True # Add boxes
roll_w = 40 # Window of rolling average for climatology insets
subset_params = subset_params_all['gha_slice'] # Plot extent

save_fig = True
output_fn = dir_list['figs']+'figure2'


#----------------- Load data -----------------
stats = load_raw('pr_doyavg_*_GHA.nc',dir_list['proc']+'CHIRPS/')
ds = load_raw('pr_day_*_GHA.nc',dir_list['raw']+'CHIRPS/')

# Subset
stats = stats.sel(**subset_params)
ds = ds.sel(**subset_params)

# Load using ds.mean(), since the code sometimes gets stuck
# on the day-of-year averaging below if it's not loaded
# (and for some reason, ds.mean() often loads it faster than
# ds.load())
ds.mean('time')

# Make dayofyear average
ds = ds.groupby('time.dayofyear').mean()

# Add seas_ratio to ds, since subset_to_srat keeps 
# crashing the kernel... 
ds['seas_ratio'] = stats.seas_ratio

# Get ISO-standard borders
gdf = gpd.read_file(dir_list['aux']+'ne_10m_admin_0_countries_iso/ne_10m_admin_0_countries_iso.shp')

#----------------- Plot -----------------
fig = plt.figure(figsize=(8,6))


spec = fig.add_gridspec(3, 3,width_ratios=[0.8,1,1],
                        wspace=0.3)


# Get gaussian weight vector for rolling avg
weight = xr.DataArray(sstats.norm(0, 1).pdf(np.arange(-5,5,10/roll_w)), dims=['window'])
# Don't forget to normalize by the weight sum!!!!!!!
weight = weight/weight.sum()


## -------------- Ethiopia sub 
# Get mask for Ethiopia
mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(ds)

# Get subset of Ethiopia, but single-peaked
# (subset_to_srat is causing kernel panics, for some reason, so doing it the old 
# fashioned way using stats.seas_ratio here)
plot_data = (ds.pr.where(mask==(regionmask.defined_regions.natural_earth_v5_0_0.
                                           countries_110.map_keys("Ethiopia"))).
                   where(ds['seas_ratio']>1,drop=True))
# Get area average
plot_data = area_mean(plot_data)

ax = fig.add_subplot(spec[0,0])

# Plot rolling average
((plot_data.pad({'dayofyear':int(roll_w/2)},mode='wrap').
 rolling(dayofyear=roll_w,center=True).construct('window').dot(weight))[int(roll_w/2):-int(roll_w/2)].
 plot(color='tab:red'))


# Get rid of x axis labels
ax.set_xlabel('')
ax.tick_params(axis='x',bottom=False,labelbottom=False)
ax.set_ylabel(r'$P$ [mm/day]')

ax.set_ylim(0,8)
# Add grid
doys = np.array(pd.date_range('2001-01-01','2002-02-01',freq='4MS').dayofyear)
doys[-1] = doys[-1]+365
plt.xticks(doys)
ax.grid(True)

# Add reference line to map 
ar = ax.annotate('',(2.25,-0.25),(1,0.75),xycoords='axes fraction',
            arrowprops={'arrowstyle':'->'},zorder=100)
fig.texts.append(ar)

# Subplot lettering
ax.annotate('a.',
            [0.01,0.99],xycoords='axes fraction',
            va='top',ha='left',fontsize=13,fontweight='bold')

ax.set_title('Representative\nclimatologies',fontsize=13)

## -------------- Study area sub 
# Get subset to double-peaked region
plot_data = area_mean(ds.where(ds['seas_ratio']<1,drop=True).pr)

ax = fig.add_subplot(spec[1,0])

# Plot rolling average
((plot_data.pad({'dayofyear':int(roll_w/2)},mode='wrap').
 rolling(dayofyear=roll_w,center=True).construct('window').dot(weight))[int(roll_w/2):-int(roll_w/2)].
 plot(color='darkgreen'))


# Get rid of x axis labels
ax.set_xlabel('')
ax.tick_params(axis='x',bottom=False,labelbottom=False)
ax.set_ylabel(r'$P$ [mm/day]')
ax.set_title('')

ax.set_ylim(0,5)
# Add grid
doys = np.array(pd.date_range('2001-01-01','2002-02-01',freq='4MS').dayofyear)
doys[-1] = doys[-1]+365
plt.xticks(doys)
ax.grid(True)

# Add reference line to map 
ar = ax.annotate('',(2.75,0.5),(1,0.75),xycoords='axes fraction',
            arrowprops={'arrowstyle':'->'},zorder=100)
# Trick to ensure arrow stays above main map, from 
# https://stackoverflow.com/questions/13831824/how-to-prevent-a-matplotlib-annotation-being-clipped-by-other-axes
fig.texts.append(ar)

# Subplot lettering
ax.annotate('b.',
            [0.01,0.99],xycoords='axes fraction',
            va='top',ha='left',fontsize=13,fontweight='bold')
# Subplot lettering
ax.annotate('Study area',
            [0.99,0.99],xycoords='axes fraction',
            va='top',ha='right',fontsize=13,color='darkgreen')

for s in ['bottom','top','right','left']:
    ax.spines[s].set_color('darkgreen')
    ax.spines[s].set_linewidth(2)


## -------------- Tanzania sub 

# Get subset of Ethiopia, but single-peaked
# (subset_to_srat is causing kernel panics, for some reason, so doing it the old 
# fashioned way using stats.seas_ratio here)
plot_data = (ds.pr.where(mask==(regionmask.defined_regions.natural_earth_v5_0_0.
                                           countries_110.map_keys("Tanzania"))).
                   where(ds['seas_ratio']>1,drop=True))
# Get area average
plot_data = area_mean(plot_data)

ax = fig.add_subplot(spec[2,0])

# Plot rolling average
((plot_data.pad({'dayofyear':int(roll_w/2)},mode='wrap').
 rolling(dayofyear=roll_w,center=True).construct('window').dot(weight))[int(roll_w/2):-int(roll_w/2)].
 plot(color='tab:red'))

# Make x-axis have months 
doys = np.array(pd.date_range('2001-01-01','2002-02-01',freq='4MS').dayofyear)
doys[-1] = doys[-1]+365
plt.xticks(doys,pd.date_range('2001-01-01','2002-02-01',freq='4MS').strftime('%b'))
ax.set_xlabel('')
ax.set_ylabel(r'$P$ [mm/day]')
ax.set_title('')

ax.set_ylim(0,8)
# Add grid
ax.grid(True)

# Add reference line to map 
ar = ax.annotate('',(1.95,0.5),(1,0.25),xycoords='axes fraction',
            arrowprops={'arrowstyle':'->'},zorder=100)
# Trick to ensure arrow stays above main map, from 
# https://stackoverflow.com/questions/13831824/how-to-prevent-a-matplotlib-annotation-being-clipped-by-other-axes
fig.texts.append(ar)

# Subplot lettering
ax.annotate('c.',
            [0.01,0.99],xycoords='axes fraction',
            va='top',ha='left',fontsize=13,fontweight='bold')

## -------------- Main map 
ax = fig.add_subplot(spec[:,1:3],projection=ccrs.PlateCarree())
np.log(stats.sel(lat=slice(-10,20),lon=slice(28,60))).seas_ratio.plot.contourf(transform=ccrs.PlateCarree(),cmap=cmocean.cm.curl,
                 levels=16,
         cbar_kwargs={'label':r'Log ratio of 1st harmonic to 2nd harmonic'})

np.log(stats.sel(lat=slice(-10,20),lon=slice(28,60))).seas_ratio.plot.contour(transform=ccrs.PlateCarree(),levels=[0],
                                                                             colors=['k'])
ax.coastlines()
#ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
gdf.cx[subset_params['lon'],subset_params['lat']].plot(ax=ax,facecolor='none',edgecolor='k',
                                                       linestyle='-',linewidth=0.15,
                                                       transform=ccrs.PlateCarree())
ax.axhline(0,color='k',linestyle='--') # Equator
ax.set_title('"Double-peakedness" of CHIRPS rainfall\n(1981-2021)',fontsize=13)

#{'hoa': {'lat': [-3, 12.5], 'lon': [32, 55]},
# 'gha': {'lat': [-10, 20], 'lon': [28, 52]},
if add_geom:
    geoms = [geometry.box(subset_params_all['gha']['lon'][0],subset_params_all['gha']['lat'][0],
                          subset_params_all['hoa']['lon'][0],subset_params_all['gha']['lon'][1]),
             geometry.box(subset_params_all['hoa']['lon'][0],subset_params_all['gha']['lat'][0],
                          60,subset_params_all['hoa']['lat'][0]),
             geometry.box(subset_params_all['hoa']['lon'][0],subset_params_all['hoa']['lat'][1],
                          60,subset_params_all['gha']['lat'][1]),
             geometry.box(subset_params_all['hoa']['lon'][1],subset_params_all['hoa']['lat'][0],
                          60,subset_params_all['hoa']['lon'][1])]
    ax.add_geometries(geoms, crs=ccrs.PlateCarree(), facecolor='white',edgecolor='none',alpha=0.5)
    
    geom = geometry.box(subset_params_all['hoa']['lon'][0],subset_params_all['hoa']['lat'][0],
                        subset_params_all['hoa']['lon'][1],subset_params_all['hoa']['lat'][1])
    ax.add_geometries([geom],crs=ccrs.PlateCarree(),facecolor='none',edgecolor='k')
    
    ax.text(subset_params_all['gha']['lon'][1]-0.5,
            subset_params_all['hoa']['lat'][0]+0.25,
            'Study area',ha='right',va='bottom',color='darkgreen',fontsize=18,
                transform=ccrs.PlateCarree())
    
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                color='gray',alpha=0.5)
gl.xlocator = mticker.FixedLocator(np.arange(30,55,5))
gl.right_labels = False

ax.set_extent((subset_params['lon'].start,subset_params['lon'].stop,
              subset_params['lat'].start,subset_params['lat'].stop))


# Subplot lettering
ax.annotate('d.',
            [0.01,0.99],xycoords='axes fraction',
            va='top',ha='left',fontsize=13,fontweight='bold')



## -------------- Output
if save_fig:
    utility_print(output_fn)