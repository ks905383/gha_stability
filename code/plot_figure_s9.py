import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
import cmocean
import os
import re
import glob
from matplotlib import pyplot as plt
from funcs_load import load_raw
from funcs_support import get_params,get_subset_params,area_mean,subset_to_srat,utility_print
from funcs_process import create_season_mask

dir_list = get_params()
subset_params_all = get_subset_params()

#--------------- Setup ----------------
save_fig = True
output_fn = dir_list['figs']+'figure_s9'

# Reanalysis and precip data sources
mods_r = ['MERRA2','JRA-55','ERA5']
mods_p = ['CHIRPS','GPCP','IMERG']

# What seasonal averaging to use
kind = 'month' #MAM, OND

#--------------- Load data ----------------
# Load reanalysis data
dss = xr.concat([area_mean(subset_to_srat(xr.merge([load_raw(var+'_seasavg*HoA.nc',search_dir=dir_list['proc']+mod+'/',
                                             subset_params = {'plev':[650],'kind':[kind]}) 
                                                    for var in ['hdiff','unstable']]))).drop('plev')
                 for mod in mods_r],
              dim=pd.Index(mods_r,name='model_r'))
# Convert from J/kg to kJ/kg
dss['hdiff'] = dss['hdiff']/1000

# Load rainfall data
dssp = xr.concat([area_mean(subset_to_srat(load_raw('pr_seasavg*HoA.nc',
                                                    search_dir=dir_list['proc']+mod+'/',
                                                    subset_params={'kind':[kind],**subset_params_all['hoa_slice']}).
                                           drop(['lat_bounds','lon_bounds'],errors='ignore'),
                                          srat_mod='CHIRPS'))
                  for mod in mods_p],
                 dim=pd.Index(mods_p,name='model_p'))
# NB: this code uses the CHIRPS definition of the double-peaked 
# region (`srat_mod = 'CHIRPS'`). It is functionally identical 
# over land to that of IMERG and GPCP, so for ease of coding, 
# the CHIRPS region is used.
                                                              
# Broadcast and merge
dsc = xr.merge(xr.broadcast(dssp,dss))

dsc = dsc.drop('kind')


#--------------- Calculate correlations ----------------
# Calculate year-on-year differences
ddsc = xr.merge([dsc.isel(year=slice(1,None))[var].values - 
                    dsc.isel(year=slice(0,-1))[var] for var in dsc])

# Calculate between vars and rainfall 
corrs = xr.merge([xr.corr(ddsc['pr'],ddsc[var],dim='year').to_dataset(name=var)
                  for var in [v for v in ddsc if v not in ['pr']]])

# Transfer correlations to dataframe
df = corrs.sel(season=['long_rains','short_rains']).to_dataframe()
df = df.reset_index()
df = pd.melt(df,id_vars=['season','model_p','model_r'])

#--------------- Plot figure ----------------
titles = {'vars':{'hdiff':r'$\Delta\overline{(h_s-h^*)}$',
          'unstable':r'$\Delta\overline{(frac.\ unstable)}$'},
          'seasons':{'long_rains':r'$\mathbf{MAM}$',
                     'short_rains':r'$\mathbf{OND}$'}}

#-------- Plot --------
dfp = pd.pivot(df,
                    columns=['season','model_p'],index=['variable','model_r'])

ax = plt.subplot()
ax = sns.heatmap(dfp,
                cmap=cmocean.cm.balance,vmin=-1,vmax=1,
                 annot=True,square=True,ax=ax,
                cbar=False)

#-------- Axis annotations --------
# Remove all existing ticks / labels
plt.tick_params(axis='x',bottom=False,labelbottom=False)
plt.tick_params(axis='y',left=False,labelleft=False)
# Remove labels
ax.set_xlabel('')
ax.set_ylabel('')

## X axis 
# Reanalysis models across top
mod_names = np.asarray(list(dfp.columns))[:,-1]
for mod,mod_idx in zip(mod_names,np.arange(0,len(mod_names))+0.5):
    ax.annotate(mod,(mod_idx,-0.1),xycoords='data',color='k',
                ha='center',va='bottom',annotation_clip=False,fontsize=11)
    
# Seasons at the top 
seas_names = np.unique(np.asarray(list(dfp.columns))[:,-2])
for seas,seas_idx in zip(seas_names,np.arange(0,len(seas_names))):
    ax.annotate(titles['seasons'][seas],(len(np.unique(mod_names))*(0.5+seas_idx),-0.5),
                xycoords='data',color='k',ha='center',annotation_clip=False,fontsize=13)
    
    # Major line 
    ax.axvline((seas_idx+1)*len(np.unique(mod_names)),color='k',linestyle='-',linewidth=2)

## Y axis         
# Obs sources on the left
mod_names = np.asarray(list(dfp.index))[:,-1]
for mod,mod_idx in zip(mod_names,np.arange(0,len(mod_names))+0.5):
    ax.annotate(mod,(-0.1,mod_idx),xycoords='data',color='k',
                ha='right',va='center',rotation=90,annotation_clip=False,fontsize=11)
    
# Variables on the left
var_names = np.unique(np.asarray(list(dfp.index))[:,-2])
for var,var_idx in zip(var_names,np.arange(0,len(var_names))):
    ax.annotate(titles['vars'][var],(-0.5,len(np.unique(mod_names))*(0.5+var_idx)),xycoords='data',color='k',
                ha='right',va='center',rotation=90,annotation_clip=False,fontsize=13,
                )                   
    
    # Major line 
    ax.axhline((var_idx+1)*len(np.unique(mod_names)),color='k',linestyle='-',linewidth=2)
    
    
#--------------- Output ----------------
if save_fig:
    utility_print(output_fn)