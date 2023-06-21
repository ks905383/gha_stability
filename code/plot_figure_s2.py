# Code to plot supp docs Figure S2
import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from funcs_support import (get_params,subset_to_srat,area_mean,utility_print)
from funcs_load import load_raw
dir_list = get_params()

#--------------- Setup ----------------
subset_params = {'lat':slice(-3,12.5),'lon':slice(32,55),'time':slice('19810101','20211231')}

# Line colors
colors = {'CHIRPS':'k',
          'MERRA2':'tab:orange',
          'JRA-55':'tab:purple',
          'ERA5':'tab:cyan'}

save_fig = False
output_fn = '../figures/figure_s2'

#--------------- Load data ----------------
dss = {mod:load_raw('pr_day_*'+suffix+'*.nc',search_dir = dir_list['raw']+mod+'/',
                    subset_params = subset_params)
       for mod,suffix in zip(['MERRA2','ERA5','JRA-55','CHIRPS'],
                      [*['HoA']*3,'Africa'])}

# Standardize to mm/day (from kg/m^2/s)
for mod in [k for k in dss if k not in ['JRA-55','CHIRPS']]:
    dss[mod] = dss[mod]*60*60*24
    
# Get area mean over double-peaked region
dss = {k:area_mean(subset_to_srat(v)) for k,v in dss.items()}

# Concat
dss = xr.concat([v for k,v in dss.items()],
                dim = pd.Index([k for k in dss],name='model'))

# Get day of year average
dss = dss.groupby('time.dayofyear').mean()


#---------------Plot ----------------
ax = plt.subplot()
for mod in dss.model.values:
    dss.sel(model=mod).pr.plot(label=str(mod),color=colors[mod])
ax.legend()

# Replace day of year index with month names on x axis labels
doys = np.array(pd.date_range('2001-01-01','2002-02-01',freq='2MS').dayofyear)
doys[-1] = doys[-1]+365
plt.xticks(doys,pd.date_range('2001-01-01','2002-02-01',freq='2MS').strftime('%b'))

ax.set_ylabel('Precipitation [mm/day]')
ax.set_xlabel('')
ax.set_title('Rainfall climatologies by data product')

if save_fig:
    utility_print(output_fn)