# Code to plot supp docs Figure S4, comparing the climatologies
# of reanalysis hs-h* and components
import xarray as xr
import numpy as np
import pandas as pd
import os
import glob
import re
import string
from matplotlib import pyplot as plt

from funcs_load import load_raw
from funcs_plot import figure_climatology
from funcs_support import (get_params, get_subset_params, area_mean, subset_to_srat,utility_print)
dir_list = get_params()
subset_params_all = get_subset_params()

#--------------- Setup ----------------
# Line colors
colors = {'CHIRPS':'k',
          'MERRA2':'tab:orange',
          'JRA-55':'tab:purple',
          'ERA5':'tab:cyan'}

save_fig = True
output_fn = '../figures/figure_s4'

suffix = 'HoA'
plev = 650

#--------------- Load data ----------------
dss = {mod:xr.merge([load_raw(var+'_day_*_'+suffix+'*.nc',search_dir = dir_list['proc']+mod+'/',
                    subset_params = subset_params_all['hoa_slice'])
                    for var in ['hdiff','hsat','unstable','ta-nsurf','hus-nsurf']]).sel(plev=plev)
       for mod in ['MERRA2','ERA5','JRA-55']}
    
# Get area mean over double-peaked region
dss = {k:area_mean(subset_to_srat(v)) for k,v in dss.items()}

# Concat
dss = xr.concat([v for k,v in dss.items()],
                dim = pd.Index([k for k in dss],name='model'))


#--------------- Plot figure ----------------
fig,axs = plt.subplots(figsize=(10,6),nrows=2,ncols=3)

# Plot setup 
var_list = ['unstable','hdiff','hus','ta','hsat']
scales = [1,1/1000,1000,1,1/1000]
titles = [r'frac. unstable',r'$h_s-h^*$','Near-surface specific humidity','Near-surface air temperature',r'$h^*$']
ylabels = ['frac. of grid cells with\n'+r'$h_s-h^*>0$',r'$h_s-h^*$ [kJ/kg]',r'$q_s$ [g/kg]',r'$T_s$ [K]',r'$h^*$ [kJ/kg]']
ax_list = [axs[0,0],axs[0,1],axs[1,0],axs[1,1],axs[1,2]]

# Plot by variable / subplot
for var,scale,title,ylabel,ax,ax_idx in zip(var_list,scales,titles,ylabels,ax_list,np.arange(0,len(var_list))): 
    if ax == axs[0,1]:
        show_legend = True
    else:
        show_legend = False
    fig,ax = figure_climatology([(dss[var].sel(model=mod)*scale) for mod in dss.model.values],
                                    shade_quantiles=[0.1,0.9],
                                    colors = [colors[mod] for mod in dss.model.values],
                                    labels = [mod for mod in dss.model.values],
                                    title = title,
                                    ylabel = ylabel,
                                    fig = fig, ax = ax,show_legend = show_legend)
    # Subplot lettering
    ax.annotate(string.ascii_lowercase[ax_idx]+'.',
                                  [0.01,0.99],xycoords='axes fraction',
                                  va='top',ha='left',fontsize=13,fontweight='bold')

# Remove axis to be left blank
fig.delaxes(axs[0,2])

# Make sure subplots are far enough apart to read all labels
plt.subplots_adjust(hspace=0.4,wspace=0.45)    


#----------- Print -----------    
if save_fig:
    utility_print(output_fn)
