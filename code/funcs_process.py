# Data processing code for Schwarzwald et al., GHA stability

import xarray as xr
import xesmf as xe
import numpy as np
import pandas as pd
import os
import re
import glob
from functools import reduce
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm 
import warnings

from funcs_support import (area_mean,get_params,subset_to_srat,NotUniqueFile)
from funcs_load import load_raw

# Get list of file directories
dir_list = get_params()


def calculate_h(freq = 'day',
                mod = 'MERRA2',
                subset_params = {'time':slice('19800101','20211231')},
                source_dir = 'raw',
                comp_vars = ['ta','hus'],
                varname_add = '',#-nsurf
                dir_list=dir_list,overwrite=False,
                c_p = 1004.6,
                L_v = 2.257e6, 
                T_0 = 288.15, # K reference T at sea level
                P_0 = 100000, # 101325 # Pa reference P at sea level
                L = 6.5e-3, #K/m
                g = 9.80665, # m/s^2
                Rs = 287.053 # J/(kg K) specific gas constant
               ):
    """ Calculate $h$, $h^*$
    
    Parameters
    ------------------
    dir_list : list
        output from get_params().
    
    overwrite : bool, by default False
        if True, then exsiting file with same name as output file 
        is removed before saving. 
        
    freq : str, by default 'day'
        which frequency files to look for
    
    mod : str, by default 'MERRA2'
        which data product to use
        
    subset_params : dict, by default {'time':slice('19800101,20211231')}
        whether to subset by in `load_raw()`. NB use only time, since 
        this code cycles through all suffixes... 
        
    c_p : float, by default 1004.6 (J/K)
    
    L_v : float, by default 2.257e6 (J/kg)
    
    T_0 : float, by default 288.15 (K) 
        reference T at sea level
    
    P_0 : float, by default 100000 (Pa)
        reference P at sea level
        
    L : float, by default 6.5e-3 (K/m)
    
    g : float, by default 9.80665 (m/s^2)
        graviational constant
        
    Rs : float, by default 287.053 (J/kg K)
        specific gas constant
        
    NB: T_0, P_0, L, g, Rs are only used to calculate `z` if `z` is 
    not in `comp_vars`
        
    Saves
    ------------------
    
    """
    
    #------------------ Setup ------------------
    # Get all suffixes of `h` components (T and hus)
    suffixes = [[re.split('\_',re.split('\/',fn)[-1])[-1] 
                 for fn in glob.glob(dir_list[source_dir]+mod+'/'+var+varname_add+'_'+freq+'_'+mod+'_*.nc')] 
                for var in comp_vars]
    # Get suffixes that match up across comp_vars
    suffixes = list(reduce(np.intersect1d,suffixes))
    
    #------------------ Process ------------------
    # Process by geographic subset (as identified by suffixes)
    for suffix in suffixes:
        print('\n--------------------------\n'+
              'processing files of the form: "*'+suffix+'"!')
        #---------------- Load ----------------
        # Load files 
        dss = {var:load_raw(mod+'/'+var+varname_add+'_'+freq+'_*'+suffix,
                            subset_params = subset_params,
                            search_dir = dir_list[source_dir])
              for var in comp_vars}
        # Merge into single ds
        dss = xr.merge([v for k,v in dss.items()])
        
        #---------------- Manage output ----------------
        # Get string frequency, model, experiment, run # from 
        # input filenames. This assumes one filename per suffix... 
        output_fn_comps = '_'.join(re.split('\_',re.split('\/',glob.glob(dir_list[source_dir]+mod+'/'+comp_vars[0]+varname_add+'_'+freq+'_*'+suffix)[0])[-1])[1:5])

        # Get date from data 
        if type(dss.time.min().values) == np.datetime64:
            dates = (pd.to_datetime(str(dss.time.min().values)).strftime('%Y%m%d') + '-' + 
                    pd.to_datetime(str(dss.time.max().values)).strftime('%Y%m%d'))

        # Build output filenames for `h`, `hsat`, `qsat`
        output_fns = {var:dir_list['proc']+mod+'/'+var+varname_add+'_'+output_fn_comps+'_'+dates+'_'+suffix
                      for var in ['hsat','h','qsat']}
        
        if overwrite or np.any([not os.path.exists(v) for k,v in output_fns.items()]):
            #---------------- Calculate ----------------
            # --- Calculate `z` -----------
            # (if not included in dss / comp_vars)
            if 'z' not in dss:
                dss['z'] = (T_0/L)*((dss['ta'].plev*100/P_0)**(-L*Rs/g) - 1)

            # --- Calculate `h` -----------
            if overwrite or (not os.path.exists(output_fns['h'])):
                dss['h'] = c_p*dss['ta'] + dss['z']*g + L_v*dss['hus']
            # Delete hus because it's not needed anymore, and we need the memory
            dss = dss.drop('hus')

            # --- Calculate `qsat` -----------
            def q_s_m(T,P,L_v = L_v):
                # From Murray 1969 (this one actually fits closer to Sarachik and Cane than the other ones... 
                # From https://cran.r-project.org/web/packages/humidity/vignettes/humidity-measures.html
                a = 17.2693882
                b = 35.86    

                e_s = 6.1078*np.exp((a*(T-273.16))/(T-b))

                q_s = 0.622*e_s / (P - 0.378*e_s)
                return q_s

            if overwrite or (not os.path.exists(output_fns['qsat'])):
                # Calculation saturation specific humidity
                dss['qsat'] = q_s_m(dss['ta'],dss['plev'])

            # --- Calculate `hsat` -----------
            if overwrite or (not os.path.exists(output_fns['hsat'])):
                if 'qsat' not in dss:
                    dss['qsat'] = xr.open_dataset(output_fns['qsat']).qsat
                # Calculate saturation MSE
                dss['hsat'] = c_p*dss['ta'] + dss['z']*g + L_v*dss['qsat']

            # Drop unneeded variables
            dss = dss.drop(['ta','z'])


            #---------------- Save ----------------
            descs = {'h':('calculated as c_p*T + g*z + L_v*q, for '+
                                        'c_p='+str(c_p)+' J/kgK, '+
                                        'g='+str(g)+' m/s^2, '
                                        'L_v='+str(L_v)),
                     'qsat':'calculated following Murray (1969); from https://cran.r-project.org/web/packages/humidity/vignettes/humidity-measures.html',
                     'hsat':('calculated as c_p*T + g*z + L_v*qsat, for '+
                                            'c_p='+str(c_p)+' J/kgK, '+
                                            'g='+str(g)+' m/s^2, '
                                            'L_v='+str(L_v)+', and qsat from the equivalent qsat_* file')
                    }

            for v in output_fns:
                if overwrite or (not os.path.exists(output_fns[v])):
                    ds = dss[[v]]
                    ds.attrs['SOURCE'] = 'calculate_h() from funcs_process.py'
                    ds.attrs['DESCRIPTION'] = descs[v]

                    if os.path.exists(output_fns[v]):
                        os.remove(output_fns[v])
                        print(output_fns[v]+' removed to allow overwrite!')
                        
                    if not os.path.exists(os.path.dirname(output_fns[v])):
                        os.mkdir(os.path.dirname(output_fns[v]))
                        print(os.path.dirname(output_fns[v])+'/ created!')
                    
                    ds.to_netcdf(output_fns[v])
                    print(output_fns[v]+' saved!')
                else:
                    print(output_fns[v]+' already exists!')
        else:
            print('all files:\n'+
                  '\n'.join('   '+v for k,v in output_fns.items())+'\n'+
                  'already exist!')
            


def subset_files(mod,subset,source_dir,output_dir,
                 var_list = None,
                 skip_list = None,
                 freq='day',dir_list = dir_list,
                 overwrite=False):
    """ Subsets all files of a model in a particular directory, 
    saves original copy in output_dir
    
    To make files smaller for export / sharing. 
    
    Parameters
    ------------------
    mod : str, by default 'MERRA2'
        which data product to use
        
    subset : dict
        piped as `ds.sel(**subset)`
    
    source_dir : str, by default 'raw'
        which `dir_list[source_dir]` directory to search for files 
        to process
        
    output_dir : str, by default 'proc'
        which `dir_list[source_dir]+mod+'/'+output_dir` directory 
        to save the output file in
        
    var_list : None or list, by default None
        if a list, then only files with the variables listed in 
        `var_list` will be processed (based on the [var] slot in 
        their filenames); if None, then all files with a given 
        `freq` and `mod` in `dir_list[source_dir]` will be processed
    
    dir_list : list
        output from get_params().
    
    overwrite : bool, by default False
        if True, then exsiting file with same name as output file 
        is removed before saving. 
        
    freq : str, by default 'day'
        which frequency files to look for
    
    
    Saves
    ------------------
    All files in the directory `dir_list[source_dir]+mod+'/'`, subset
    to `subset`, with the original files placed in 
        `dir_list[source_dir]+mod+'/'
    
    """
    
    # Get all files that match the data frequency, model
    file_list = glob.glob(dir_list[source_dir]+mod+'/*_'+freq+'_'+mod+'_*.nc')
    
    # Subset by variable if desired
    if type(var_list) == list:
        search_str = dir_list[source_dir]+mod+'/'+'('+'\_)|('.join(var_list)+'\_)'+freq+'\_'+mod+'.*\.nc'
        file_list = [fn for fn in file_list if re.search(search_str,fn)]
        
    # Remove files if desired
    if type(skip_list) == list:
        for sl in skip_list:
            file_list = [fn for fn in file_list if re.search(sl,fn) is None]
        
    if len(file_list) == 0:
        if type(var_list) == list:
            warnings.warn('No files found for search '+dir_list[source_dir]+mod+'/*_'+freq+'_'+mod+'_*.nc, '+
                          'subset to variables: '+', '.join(var_list))
        else:
            warnings.warn('No files found for search '+dir_list[source_dir]+mod+'/*_'+freq+'_'+mod+'_*.nc')
        
    # Process by file
    output_dir_full = dir_list[source_dir]+mod+'/'+output_dir+'/'
    if not os.path.exists(output_dir_full):
        os.mkdir(output_dir_full)
        print(output_dir_full+' created!')
    
    for fn in file_list:
        move_fn = output_dir_full+re.split('\/',fn)[-1]
        if overwrite or (not os.path.exists(move_fn)):
            ds = xr.open_dataset(fn)
            
            if np.all([k in ds.sizes for k in subset]):
                ds = ds.sel(**subset)

                ds.attrs['SOURCE'] = ds.attrs['SOURCE'] + '--> subset to '+str(subset)+', with original saved in '+output_dir

                # Move original file
                os.system('mv '+fn+' '+move_fn)
                print('original file moved to '+move_fn)

                # Save subset file
                ds.to_netcdf(fn)
                print('subset '+fn+' saved!')
            else:
                print('not all of '+', '.join([k for k in subset])+' found in ds dims ('+', '.join([k for k in ds.sizes])+'), skipped.')
        else:
            print(fn+' already subset, moved to '+output_dir+'; skipped.')
    
            
def calculate_nearsurface(var_list = 'all',
                          freq = 'day',
                          mod = 'MERRA2',
                          source_dir = 'raw', #into dir_list[]
                          output_dir = 'proc', #into dir_list[]
                          nan_search_subset = {'time':slice(0,365*10)}, #{}
                          dir_list = dir_list,overwrite=False):
    """ Extracts value in pressure level above highest nan level
    
    If a file has a vertical coordinate, the lowest level at each 
    location that is above the highest `nan` level is created in 
    a new `ds` and saved. 
    
    Parameters
    ------------------
    dir_list : list
        output from get_params().
    
    overwrite : bool, by default False
        if True, then exsiting file with same name as output file 
        is removed before saving. 
        
    var_list : str or list, by default 'all'
        if a list, then only files with the variables listed in 
        `var_list` will be processed (based on the [var] slot in 
        their filenames); if 'all', then all files with a given 
        `freq` and `mod` in `dir_list[source_dir]` will be processed
        
    freq : str, by default 'day'
        which frequency files to look for
    
    mod : str, by default 'MERRA2'
        which data product to use
        
    source_dir : str, by default 'raw'
        which `dir_list[x]` directory to search for files to process
        
    output_dir : str, by default 'proc'
        which `dir_list[x]` directory to save the output file in
        
    nan_search_subset : dict, by default `{'time':slice(0,365*10)}`
        the code searches across time for vertical levels that have any
        nans (in time); using `nan_search_subset` limits that search to 
        a subset of timeslices to save time. 
        
    Saves
    ------------------
    Note: the variable name remains unchanged, even as the filename changes. 
    
    
    """
    
    # Get all files that match the data frequency, model
    file_list = glob.glob(dir_list[source_dir]+mod+'/*_'+freq+'_'+mod+'_*.nc')
    
    # Subset by variable if desired
    if type(var_list) == list:
        search_str = dir_list[source_dir]+mod+'/'+'('+'\_)|('.join(var_list)+'\_)'+freq+'\_'+mod+'.*\.nc'
        file_list = [fn for fn in file_list if re.search(search_str,fn)]
        
    if len(file_list) == 0:
        if type(var_list) == list:
            warnings.warn('No files found for search '+dir_list[source_dir]+mod+'/*_'+freq+'_'+mod+'_*.nc, '+
                          'subset to variables: '+', '.join(var_list))
        else:
            warnings.warn('No files found for search '+dir_list[source_dir]+mod+'/*_'+freq+'_'+mod+'_*.nc')
        
    # Process by file
    for fn in file_list:
        # Extract variable name for file renaming purposes
        var = re.split('\_',re.split('\/',fn)[-1])[0]
        
        # Get output filename by appending `-nsurf` to the variable name
        output_fn = re.sub(dir_list[source_dir],dir_list[output_dir],
                           re.sub(mod+'\/'+var+'\_',mod+'/'+var+'-nsurf_',fn))
        
        if overwrite or (not os.path.exists(output_fn)):
            # Load metadata
            ds = xr.open_dataset(fn)

            # Verify that the variable has pressure levels (i.e., has a
            # vertical dimension)
            #if 'vertical' in ds.cf.keys():
            if ('plev' in ds.dims) or ('level' in ds.dims):
                # ^ this uses cf_xarray, but the below still assumes the 
                # vertical coordinate is called 'plev'... but was janky,
                # so going back to the hard-code. 
                
                if 'plev' in ds.dims:
                    icdim = 'plev'
                elif 'level' in ds.dims:
                    icdim = 'level'
                
                # Load
                ds = ds.load()
                
                # Get which plev has no nans (searching a subset to save time) 
                ds_nonas = ~(np.isnan(ds.isel(**nan_search_subset)).any('time'))
                
                # Get if any locations have nans in all vertical levels (this would
                # break the ufunc below) 
                ds_allnas = (~ds_nonas).all(icdim)
                ds_nonas = ds_nonas.where(~ds_allnas,True)

                # Get plev index with no nans
                ds_surfidxs = xr.apply_ufunc(lambda x: np.where(x)[0][0],
                               ds_nonas,input_core_dims=[[icdim]],vectorize=True)

                # Subset to no-nans plev
                ds_out = ds.isel({icdim:ds_surfidxs[var]})
                
                # Set locations with all nans in the vertical to nan (technically not
                # needed since `ds` is already nan at those locations anyways, but
                # worth being robust here)
                ds_out = ds_out.where(~ds_allnas[var])
                ds_out[icdim] = ds_out[icdim].where(~ds_allnas[var])
                

                # Add processing details
                ds_out.attrs['SOURCE'] = ds_out.attrs['SOURCE']+' --> calculate_nearsurface() from funcs_process.py'
                ds_out.attrs['DESCRIPTION'] = (ds_out.attrs['DESCRIPTION']+' Then; subset to the first plev that '
                                              + 'has no nans in the slice "'+str(nan_search_subset)+'"')
                
                if overwrite and (os.path.exists(output_fn)):
                    os.remove(output_fn)
                    print(output_fn+' removed to allow overwrite!')
                
                if not os.path.exists(os.path.dirname(output_fn)):
                    os.mkdir(os.path.dirname(output_fn))
                    print(os.path.dirname(output_fn)+'/ created!')
                
                ds_out.to_netcdf(output_fn)
                print(output_fn+' saved!')

        else:
            print(output_fn+' already exists!')
                                 
    

def calculate_hdiff(freq = 'day',
                    mod = 'MERRA2',
                    use_surf = False,
                    suffix_skip = ['HoA-merid-slice.nc'],
                    dir_list=dir_list,overwrite=False):
    """ Calculate $h_s-h^*$
    
    Looks for all suffixes / areas with `h-nsurf*` and hsat* files, 
    takes the difference between `h` in `h-nsurf*` and `hsat` in 
    `hsat*`, saves output. 
    
    Parameters
    ------------------
    dir_list : list
        output from get_params().
    
    overwrite : bool, by default False
        if True, then exsiting file with same name as output file 
        is removed before saving. 
        
    use_surf : bool, by default False
        if True, then calculation uses surface variables (`hs`); 
        otherwise, calculation uses variables from the 
        first pressure level above the topography (`h-nsurf`)
        
    freq : str, by default 'day'
        which frequency files to look for
    
    mod : str, by default 'MERRA2'
        which data product to use
        
        
    Saves
    ------------------
    hdiff_* files, with the same filename as the input h/hsat files
    
    """
    #------------------ Setup ------------------
    if use_surf:
        fn_vars = ['hsat','hs']
        file_vars = {'hsat':'hsat','h':'hs','hdiff':'hsdiff'}
    else:
        fn_vars = ['hsat','h-nsurf']
        file_vars = {'hsat':'hsat','h':'h','hdiff':'hdiff'}
    
    # Get all suffixes of `hdiff` components (hsat and h)
    suffixes = [[re.split('\_',re.split('\/',fn)[-1])[-1] 
                 for fn in glob.glob(dir_list['proc']+mod+'/'+var+'_'+freq+'_'+mod+'_*.nc')] 
                for var in fn_vars]
    # Get suffixes that match up across variables
    suffixes = list(reduce(np.intersect1d,suffixes))
    # Remove unwanted suffixes
    suffixes = [suffix for suffix in suffixes if suffix not in suffix_skip]
    
    attrs = {'SOURCE':'calculate_hdiff() from funcs_process.py',
             'DESCRIPTION':'h_s - h^*'}
    
    # Process by geographic subset (as identified by suffixes)
    for suffix in suffixes:
        print('\n--------------------------\n'+
              'processing files of the form: "*'+suffix+'"!')
        #---------------- Load ----------------
        # Load files 
        dss = {var:load_raw(mod+'/'+var+'_'+freq+'_*'+suffix,
                            search_dir=dir_list['proc'])
              for var in fn_vars}
        # Merge into single ds
        dss = xr.merge([v for k,v in dss.items()],combine_attrs='drop')
        
        #---------------- Manage output ----------------
        # Get string frequency, model, experiment, run # from 
        # input filenames. This assumes one filename per suffix... 
        output_fn_comps = '_'.join(re.split('\_',re.split('\/',glob.glob(dir_list['proc']+mod+'/'+fn_vars[0]+'_'+freq+'_*'+suffix)[0])[-1])[1:5])

        # Get date from data 
        if type(dss.time.min().values) == np.datetime64:
            dates = (pd.to_datetime(str(dss.time.min().values)).strftime('%Y%m%d') + '-' + 
                    pd.to_datetime(str(dss.time.max().values)).strftime('%Y%m%d'))

        # Build output filenames `hdiff`
        output_fn = dir_list['proc']+mod+'/'+file_vars['hdiff']+'_'+output_fn_comps+'_'+dates+'_'+suffix
        
        #---------------- Calculate and save ----------------
        if overwrite or (not os.path.exists(output_fn)):
            # Calculate hdiff
            dss[file_vars['hdiff']] = dss[file_vars['h']] - dss[file_vars['hsat']]
            
            # Drop constituent parts 
            dss = dss.drop([file_vars['h'],file_vars['hsat']])
            
            # Add attributes
            for k in attrs:
                dss.attrs[k] = attrs[k]
                
            if os.path.exists(output_fn):
                os.remove(output_fn)
                print(output_fn+' removed to allow overwrite!')
                
            if not os.path.exists(os.path.dirname(output_fn)):
                os.mkdir(os.path.dirname(output_fn))
                print(os.path.dirname(output_fn)+'/ created!')
            
            dss.to_netcdf(output_fn)
            print(output_fn+' saved!')
            
        else:
            print(output_fn+' already exists!')


def calculate_unstable(freq = 'day',
                       mod = 'MERRA2',
                       var = 'hdiff', #vs. hdiffmax
                        search_str = '*.nc',
                       use_surf=False,
                       dir_list = dir_list,overwrite=False):
    """ Generate boolean for unstable time periods
    
    Unstable time periods are defined as h_s-h^*>0
    

    Parameters
    ------------------
    dir_list : list
        output from get_params().
    
    overwrite : bool, by default False
        if True, then exsiting file with same name as output file 
        is removed before saving. 
        
    search_str : str, by default "*.nc"
        used to search for files from which to get $h_s-h^*$ 
        data. Files need to contain a variable called "hdiff".
        The full search call is: 
            `glob.glob(dir_list['proc']+mod+'/hdiff_'+freq+'_'+mod+search_str)`
            
        (so, use `glob` standards, not REs!)

    freq : str, by default 'day'
        which frequency files to look for
    
    mod : str, by default 'MERRA2'
        which data product to use
        
    Saves
    ------------------
    From every file found by `glob.glob(search_str)` that has 
    a variable `var`, a dataset with the variable "unstable" = 
    a boolean array of `[var]>0` is saved with the same
    file name, but "[var]" replaced by "unstable". 
    
    If `var = 'hdiffs'`, then the output filename uses the 
    variable "unstable-s" instead. 

    """
    
    # Get list of files to process
    search_str = dir_list['proc']+mod+'/'+var+'_'+freq+'_'+mod+search_str
    fns = glob.glob(search_str)
    
    if len(fns) == 0:
        warnings.warn('No files found for search: '+search_str)
    
    # Process by file 
    for fn in fns: 
        if var == 'hsdiff':
            output_fn = re.sub(var,'unstable-s',fn)
        else:
            output_fn = re.sub(var,'unstable',fn)
        if overwrite or (not os.path.exists(output_fn)):
            ds = xr.open_dataset(fn)

            # Calculate 
            ds[var] = ds[var] > 0 
            ds = ds.rename({var:'unstable'})

            ds.attrs['SOURCE'] = 'calculate_unstable() from funcs_process.py'
            ds.attrs['DESCRIPTION'] = 'boolean for h_s-h^* > 0 (with '+var+' from the file '+fn+')'

            if os.path.exists(output_fn):
                os.remove(output_fn)
                print(output_fn+' removed to allow overwrite!')
                
            if not os.path.exists(os.path.dirname(output_fn)):
                os.mkdir(os.path.dirname(output_fn))
                print(os.path.dirname(output_fn)+'/ created!')

            ds.to_netcdf(output_fn)
            print(output_fn+' saved!')
        else:
            print(output_fn+' already exists!')
            
def calculate_uq(freq = 'day',
                 mod = 'MERRA2',
                 subset_params = None,
                 source_dirs = ['raw','proc'],
                 varstrs_add = ['','-nsurf'],
                 comp_vars = ['ua','va','hus'],
                 dir_list=dir_list,overwrite=False,):
    
    """ Calculate $\vec{u}q$
    
    Finds all file suffix groups that have `ua`, `va`, and
    `hus` files (with all forms of `varstrs_add` added to the
    variable names), and calculates `uq` and `vq` from them. 
    
    Parameters
    ------------------
    source_dirs : list, by default ['raw','proc']
        which directories (of `dir_list`) to cycle through
        to find files
        
    varstrs_add : list, by default ['','-nsurf']
        which variable name additions to cycle through to 
        find files 
        
    comp_vars : list, by defualt ['ua','va',hus']
        the filename variables to search for and load
    
    dir_list : list
        output from get_params().
    
    overwrite : bool, by default False
        if True, then exsiting file with same name as output file 
        is removed before saving. 
        
    freq : str, by default 'day'
        which frequency files to look for
    
    mod : str, by default 'MERRA2'
        which data product to use
        
    subset_params : dict, by default None
        whether to subset by in `load_raw()`. NB use only time, since 
        this code cycles through all suffixes... 
    """

    #------------------ Setup ------------------
    suffixes = dict()
    # Get filename suffixes that match acorss all variables
    for source_dir in source_dirs:
        suffixes[source_dir] = dict()
        for varstr in varstrs_add:
            # Get all suffixes of `h` components (T and hus)
            suffixes[source_dir][varstr] = [[re.split('\_',re.split('\/',fn)[-1])[-1] 
                         for fn in glob.glob(dir_list[source_dir]+mod+'/'+var+varstr+'_'+freq+'_'+mod+'_*.nc')] 
                        for var in comp_vars]
            # Get suffixes that match up across comp_vars
            suffixes[source_dir][varstr] = list(reduce(np.intersect1d,suffixes[source_dir][varstr]))

            # Delete if empty
            if len(suffixes[source_dir][varstr]) == 0:
                del suffixes[source_dir][varstr]

    #------------------ Process ------------------
    # Process by geographic subset (as identified by suffixes)
    for source_dir in suffixes:
        for varstr in suffixes[source_dir]:
            for suffix in suffixes[source_dir][varstr]:
                print('\n--------------------------\n'+
                      'processing files of the form: "*'+suffix+'" in directory '+source_dir+' with variables *'+varstr+'!')
                #---------------- Load ----------------
                # Load files 
                dss = {var:load_raw(mod+'/'+var+varstr+'_'+freq+'_*'+suffix,
                                    subset_params = subset_params,
                                    search_dir = dir_list[source_dir])
                      for var in comp_vars}
                # Merge into single ds
                dss = xr.merge([v for k,v in dss.items()])

                #---------------- Manage output ----------------
                # Get string frequency, model, experiment, run # from 
                # input filenames. This assumes one filename per suffix... 
                output_fn_comps = '_'.join(re.split('\_',re.split('\/',                                                         glob.glob(dir_list[source_dir]+mod+'/'+comp_vars[0]+varstr+'_'+freq+'_*'+suffix)[0])[-1])[1:5])

                # Get date from data 
                if type(dss.time.min().values) == np.datetime64:
                    dates = (pd.to_datetime(str(dss.time.min().values)).strftime('%Y%m%d') + '-' + 
                             pd.to_datetime(str(dss.time.max().values)).strftime('%Y%m%d'))

                # Build output filenames for `h`, `hsat`, `qsat`
                output_fns = {var:dir_list['proc']+mod+'/'+var+varstr+'_'+output_fn_comps+'_'+dates+'_'+suffix
                              for var in ['uq','vq']}

                if overwrite or np.any([not os.path.exists(v) for k,v in output_fns.items()]):
                    #---------------- Calculate ----------------
                    dss['uq'] = dss['ua']*dss['hus']
                    dss['vq'] = dss['va']*dss['hus']

                    #---------------- Save ----------------
                    descs = {l+'q':'Calculated as '+l+'a * q'
                             for l in ['u','v']}

                    for v in output_fns:
                        if overwrite or (not os.path.exists(output_fns[v])):
                            ds = dss[[v]]
                            ds.attrs['SOURCE'] = 'calculate_uq() from funcs_process.py'
                            ds.attrs['DESCRIPTION'] = descs[v]

                            if os.path.exists(output_fns[v]):
                                os.remove(output_fns[v])
                                print(output_fns[v]+' removed to allow overwrite!')

                            if not os.path.exists(os.path.dirname(output_fns[v])):
                                os.mkdir(os.path.dirname(output_fns[v]))
                                print(os.path.dirname(output_fns[v])+'/ created!')

                            ds.to_netcdf(output_fns[v])
                            print(output_fns[v]+' saved!')
                        else:
                            print(os.path.exists(output_fns[v])+' already exists!')
                else:
                    print('all files:\n'+
                          '\n'.join('   '+v for k,v in output_fns.items())+'\n'+
                          'already exist!')

            
def calculate_resampled(var = 'hdiff',resample = {'time':'1D'},
                        freq = '3hr',
                        output_freq = 'day',
                        mod = 'MERRA2',
                        func = 'max',
                        func_str = None,
                        search_str = '*.nc',
                        source_dir = 'proc',
                        use_surf=False,
                        dir_list = dir_list,overwrite=False,
                        save=True,return_ds=False):
    """ Resample by a function and save
    
    Note: currently implicitly mainly intended for _temporal_
    resampling. Resampling spatially (or by another dimension)
    may cause filename issues if `save=True`. 
    
    Parameters
    ---------------
    var : str, by default 'hdiff'
    
    resample : dict, by default {'time':'1D'}
        piped into `ds.resample()`
        
    func : str or function, by default 'max'
        if 'max' or 'min', then 
            `ds.resample(resample).max()` 
        (or `.min()`) is called. Otherwise, 
            `ds.resample(resample).apply(func)
        is called. 
        
    func_str : str or None, by default None
        if `func=='max'` or `'min'`, then that 
        is added to the output variable filename.
        Otherwise, input a str to add to the output
        variable filename. 
        
    freq : str, by default '3hr'
        what frequency data to look for
        
    output_freq : str, by default 'day'
        what frequency to name the output data in the 
        output filename
    
    dir_list : list
        output from get_params().
    
    overwrite : bool, by default False
        if True, then exsiting file with same name as output file 
        is removed before saving. 
        
    search_str : str, by default "*.nc"
        used to search for files from which to get data. 
        The full search call is: 
            `glob.glob(dir_list['proc']+mod+'/hdiff_'+freq+'_'+mod+search_str)`
            
        (so, use `glob` standards, not REs!)

    freq : str, by default 'day'
        which frequency files to look for
    
    mod : str, by default 'MERRA2'
        which data product to use
        
    save : bool, by default True
        whether to save output
        
    return_ds : bool, by default False
        whether to return resampled, calculated dataset
        
    Returns
    ------------------
    if `return_ds==True`, the processed dataset. 
        
    Saves
    ------------------
    From every file found, a dataset with the maximum 
    over the timeframe given by the `resample` dict is 
    saved with the same file name, but "[var]" replaced 
    by "[var]max". 
    
    """
    # Get list of files to process
    search_str = dir_list[source_dir]+mod+'/'+var+'_'+freq+'_'+mod+search_str
    fns = glob.glob(search_str)
    
    if len(fns) == 0:
        warnings.warn('No files found for search: '+search_str)
        
    if (func_str is None):
        if (type(func) == str):
            func_str = func
        else:
            raise Error("Need `func_str` to set output filename (will be put into the form '[var][func_str]_[output_freq]_[mod]_....nc'")
    
    # Process by file 
    for fn in fns: 
        # Get filename to save, replacing [var] with, e.g., [var]max
        # and the frequency with the output frequency
        output_fn = re.sub(var+'\_',var+func_str+'_',
                           re.sub('\_'+freq+'\_','_'+output_freq+'_',fn))
        
        if overwrite or (not os.path.exists(output_fn)):
            ds = xr.open_dataset(fn)
        
            if type(func) == str:
                if func == 'max':
                    ds_out = ds.resample(resample).max()
                elif func == 'min':
                    ds_out = ds.resample(resample).min()
                else:
                    raise KeyError('Only "max" and "min" are supported as string inputs for `func`; if you wish to run another function, input the function directly')
            else:
                ds_out = ds.resample(resample).apply(func)
            
            if save: 
                ds_out.attrs['SOURCE'] = 'calculate_resampled() from funcs_process.py'
                ds_out.attrs['DESCRIPTION'] = 'resampled from '+freq+' to '+output_freq+' using the function '+str(func)

                if os.path.exists(output_fn):
                    os.remove(output_fn)
                    print(output_fn+' removed to allow overwrite!')

                if not os.path.exists(os.path.dirname(output_fn)):
                    os.mkdir(os.path.dirname(output_fn))
                    print(os.path.dirname(output_fn)+'/ created!')

                ds_out.to_netcdf(output_fn)
                print(output_fn+' saved!')
            
            if return_ds:
                return ds_out
        else:
            if return_ds:
                ds_out = xr.open_dataset(output_fn)
                return ds_out
            else:
                print(output_fn+' already exists!')
            
        
    
    
def create_season_mask(ds,stats):
    """ Create boolean masks of belonging to a particular season
    
    
    Parameters
    ---------------
    ds : xr.Dataset
        `ds` MUST have the new seasonal dimension already 
        saved. This makes this code not generally applicable
        but mainly to be used in concert with `caculate_seasmeans`.
        The output dataset has the same shape, but with 
        boolean flags grid cell is in a given season. 
    
    
    stats : xr.Dataset
        seasonal stats file, with at least onset and demise
        for 'long_rains' and 'short_rains'
    
    
    Returns
    ---------------
    ds : xr.Dataset
        Dataset with variable 'ts', which is a boolean flag
        of belonging to a particular season
    
    """
    
    
    seasons = ds.season

    # Process 
    for yr in stats.year:
        # Boreal winter long dry season
        # (assuming that the new year starts with the long dry season everywhere,
        # not a super late short rain season. This tends to be true 'effectively'
        # everywhere). 
        if yr == stats.year[0]:
            idx = ((ds.time.dt.year == yr) & 
                   (ds.time.dt.dayofyear < np.round(stats.sel(year=yr).isel(season=0).onset)))
        else:
            idx = (((ds.time.dt.year == (yr-1)) & 
                         (ds.time.dt.dayofyear >= np.round(stats.sel(year=(yr-1)).isel(season=1).demise))) |  
                       ((ds.time.dt.year == yr) & 
                         (ds.time.dt.dayofyear < np.round(stats.sel(year=yr).isel(season=0).onset))))
            if np.any(stats.sel(year=yr).isel(season=1).demise>=366):
                # If any locations have short rains that spill over into this year, 
                # then delay the onset in this year in those locations
                idx_alt = (((ds.time.dt.year == (yr)) & 
                         (ds.time.dt.dayofyear >= np.round(stats.sel(year=(yr-1)).isel(season=1).demise-365))) |  
                       ((ds.time.dt.year == yr) & 
                         (ds.time.dt.dayofyear < np.round(stats.sel(year=yr).isel(season=0).onset))))
                idx = idx.where((stats.sel(year=yr).isel(season=1).demise<=365),idx_alt)
        ds['ts'].loc[{'season':seasons[0]}] = ds['ts'].loc[{'season':seasons[0]}].where(~idx,True)

        # Long rains
        idx = ((ds.time.dt.year == yr) & 
                (ds.time.dt.dayofyear >= np.round(stats.sel(year=yr).isel(season=0).onset)) & 
                (ds.time.dt.dayofyear < np.round(stats.sel(year=yr).isel(season=0).demise)))
        ds['ts'].loc[{'season':seasons[1]}] = ds['ts'].loc[{'season':seasons[1]}].where(~idx,True)


        # Boreal summer short dry season
        idx = ((ds.time.dt.year == yr) & 
               (ds.time.dt.dayofyear >= np.round(stats.sel(year=yr).isel(season=0).demise)) & 
               (ds.time.dt.dayofyear < np.round(stats.sel(year=yr).isel(season=1).onset)))
        ds['ts'].loc[{'season':seasons[2]}] = ds['ts'].loc[{'season':seasons[2]}].where(~idx,True)

        # Short rains (allowing the demise to span years...)
        idx = ((ds.time.dt.year == yr) & 
                   (ds.time.dt.dayofyear >= np.round(stats.sel(year=yr).isel(season=1).onset)) & 
                   (ds.time.dt.dayofyear < np.round(stats.sel(year=yr).isel(season=1).demise)))
        if np.any(stats.sel(year=yr).isel(season=1).demise>=366):
            if yr < stats.year[-1]:
                idx_alt = (((ds.time.dt.year == yr) & 
                        (ds.time.dt.dayofyear >= np.round(stats.sel(year=yr).isel(season=1).onset))) | 
                       ((ds.time.dt.year == yr + 1) & 
                        (ds.time.dt.dayofyear < np.round(stats.sel(year=yr).isel(season=1).demise-365))))
            else:
                idx_alt = ((ds.time.dt.year == yr) & 
                       (ds.time.dt.dayofyear >= np.round(stats.sel(year=yr).isel(season=1).onset)))
            idx = idx.where((stats.sel(year=yr).isel(season=1).demise<=365),idx_alt)
        ds['ts'].loc[{'season':seasons[3]}] = ds['ts'].loc[{'season':seasons[3]}].where(~idx,True)

    # Add last boreal winter long dry season
    if np.any(stats.sel(year=yr).isel(season=1).demise<366):
        idx = ((ds.time.dt.year == yr) & 
                       (ds.time.dt.dayofyear >= np.round(stats.sel(year=yr).isel(season=1).demise)))
        idx = idx.where(stats.sel(year=yr).isel(season=1).demise<366).astype(bool)
        ds['ts'].loc[{'season':seasons[0]}] = ds['ts'].loc[{'season':seasons[0]}].where(~idx,True)
        
        
    # Put in all-range nans from original stats dataset
    ds = ds.where(~np.isnan(stats.onset).all(('year','season')))
    
    # Return
    return ds

def calculate_seasmeans(dir_list = dir_list,
                        mod = 'MERRA2',
                        search_dir = 'raw', 
                        var = 'pr',
                        freq = 'day',
                        mod_p = 'CHIRPS',
                        suffixes = None,
                        suffix_out = None,
                        subset_params_load = None, #For subsetting the loaded ds{'lat':slice(-3,12.5),'lon':slice(32,55)}
                        calculate_local = True, 
                        anomaly = None,
                        seas_grouping_mean = 'time.dayofyear',
                        stats_suffix=None,
                        overwrite = False,
                        return_output = False):
    """ Calculate seasonal means
    
    Calculate seasonal means using the Horn of Africa long/short
    rain seasonal definitions. 
    
    Means are calculated in up to 4 ways, with the following names
    used as the method coordinate in the dimension 'kind':
    - 'dunning_local' : (only if `calculate_local=True`) onset and 
            demise are locally determined - values are only returned
            in pixels in the double-peaked region (as determined by
            the input seasonal stats file)
    - 'dunning' : onset and demise are the area average of the onset
            and demise over the whole double-peaked region
    - 'month' : long rains are March-May, short rains are Oct - Dec
    - 'month_alt' : long rains are March-May, short rains are Sep - Nov
    
    Note: the code currently calculates the seasonal indices using
    `create_season_mask()` using the `mod_p` (!) grid, and then 
    regrids (bilinearly) those indices to the `mod` grid if necessary 
    (rounding to 0 or 1). I think on the margins this produces less
    distorted results than regridding the onset and demise to the `mod`
    grid and creating the season mask based on those bounds (particularly 
    since on the geographic margins of the double-peaked region, 
    individual misclassified pixels can have onsets/demises _very_ 
    different from each other)
    
     
    Parameters
    ---------------
    dir_list : list
        output from get_params().
    
    overwrite : bool, by default False
        if True, then exsiting file with same name as output file 
        is removed before saving. 
        
    return_output : bool, by default False
        if True, then the `ds` of means is returned
        
    freq : str, by default 'day'
        which frequency files to look for
    
    mod : str, by default 'MERRA2'
        which data product to use
        
    mod_p : str, by default 'CHIRPS'
        which data product's precipitation seasonal stats to use
        
    subset_params_load : dict, by default None
        whether to subset the data whose means are to be calculated,
        piped into `load_raw()`.
        
    calculate_local : bool, by default False
        if True, then, if the geographic ranges overlap with the 
        seasonal stats data, the mean based on a grid cell's local 
        onset/demise is calculated as `kind='dunning_local'`. Only 
        values at locations within the double-peaked region are returned. 
        
    anomaly : str or None, by default None
        if not `None`, then the means of the anomaly are calculated: 
        - if 'clim':
            the means are calculated based on the anomaly vs. the 
            long-term average (over the whole time period of the 
            input file)
        - if 'seas':
            the means are calculated based on the anomaly vs. the 
            seasonal cycle, as determined by the mean over 
            `seas_grouping_mean` (see below)
        
    seas_grouping_mean : str, by default `time.dayofyear`
        if `anomaly == 'seas'`, the mean is calculated of the anomaly
        vs. the seasonal cycle, as determined by this input; the mean
        call is `ds.groupby(seas_grouping_mean).mean()`. 
    
    
    Returns
    ---------------
    if return_output == True, then the calculated mean values are returned
    
    
    Saves
    ---------------
    The calculated seasonal means, saved in `dir_list['proc']`. 
    
    """

    #------------------ Setup ------------------
    kind = ['dunning','month','month_alt']
    if calculate_local:
        kind = ['dunning_local',*kind]

    seasons = ['long_dry','long_rains','short_dry','short_rains']

    #------------------ Load ------------------
    #-------- Load seasonal stats
    # Load seasonal stats
    if stats_suffix is None:
        stats_fn = glob.glob(dir_list['proc']+mod_p+'/pr_ann_'+mod_p+'_*_seasstats_*.nc')
        if len(stats_fn) > 1:
            raise NotUniqueFile('More than one possible stats file found to load:'+'\n'.join(stats_fn)+'\nTo specify a stats file, use the `stats_suffix` parameter.')
        else:
            stats_fn = stats_fn[0]
    else:
        stats_fn = dir_list['proc']+mod_p+'/pr_ann_'+mod_p+'_historical_seasstats_dunning_'+stats_suffix+'.nc'
    stats = xr.open_dataset(stats_fn)
    # NaN out areas outside double-peaked region
    stats = subset_to_srat(stats,drop=True)
    # Remove singleton 'method' dimension
    stats = stats.isel(method=0,drop=True)
    # Make demise onset + duration to avoid wrap-around averaging 
    # issues around the calendrical new year
    stats['demise'] = stats['onset']+stats['duration']

    # Calculate regional mean of the rainfall statistics 
    statsm = area_mean(stats)

    # Get timeframe of the stats ... 
    timeframe = pd.date_range(str(stats.year.min().values)+'-01-01',
                              str(stats.year.max().values)+'-12-31')

    #-------- Load input ds
    if suffixes is None:
        # Get all suffixes of `h` components (T and hus)
        suffixes = [re.split('\_',re.split('\/',fn)[-1])[-1] 
                     for fn in glob.glob(dir_list[search_dir]+mod+'/'+var+'_'+freq+'_'+mod+'_*.nc')]
    else:
        if type(suffixes) != list:
            suffixes = [suffixes]

    if len(suffixes)==0:
        raise FileNotFoundError('No files found for search string: '+
                                dir_list[search_dir]+mod+'/'+var+'_'+freq+'_'+mod+'_*.nc')


    if anomaly is None:
        var_extra = ''
    elif anomaly == 'clim':
        var_extra = '-anom'
    elif anomaly == 'seas':
        var_extra = '-seasanom'
    else:
        raise KeyError(var_extra+' must be one of : ["clim","seas"]')

    for suffix in suffixes:
        # Load file 
        ds = load_raw(mod+'/'+var+'_'+freq+'_*'+suffix,
                      subset_params = subset_params_load,
                      search_dir = dir_list[search_dir])

        # Get anomaly if needed
        if anomaly is not None:
            if anomaly == 'clim':
                ds = ds - ds.mean('time')
            elif anomaly == 'seas':
                ds = ds.groupby(seas_anom_grouping) - ds.groupby(seas_anom_grouping).mean()

        #-------- Further setup based on correspondence between stats, ds
        # Subset to timeframe from season stats
        ds = ds.sel(time=slice(timeframe[0],timeframe[-1]))

        # If there's no geographic overlap between the stats file and the 
        # input ds, then calculate_local makes no sense and is ignored. 
        if calculate_local:
            if ((ds.lat > stats.lat.max()).all() or 
                (ds.lat < stats.lat.min()).all() or
                (ds.lon > stats.lon.max()).all() or
                (ds.lon < stats.lon.min()).all()):
                warnings.warn('no geographic overlap between stats file '+
                              'and input `ds`; no `dunning_local` means will '+
                              'be calculated.')
                calculate_local = False

        #-------- Manage output
        # Get string frequency, model, experiment, run # from 
        # input filenames. This assumes one filename per suffix... 
        output_fn_comps = '_'.join(re.split('\_',
                                            re.split('\/',
                                                    glob.glob(dir_list[search_dir]+mod+'/'+
                                                              var+'_'+freq+'_*'+suffix)[0])[-1])[2:5])

        # Get date from data 
        if type(ds.time.min().values) == np.datetime64:
            dates = (pd.to_datetime(str(ds.time.min().values)).strftime('%Y%m%d') + '-' + 
                     pd.to_datetime(str(ds.time.max().values)).strftime('%Y%m%d'))

        # Build output filename
        if suffix_out is None:
            suffix_out_tmp = suffix
        else:
            suffix_out_tmp = suffix_out
        output_fn = dir_list['proc']+mod+'/'+var+var_extra+'_seasavg_'+output_fn_comps+'_'+dates+'_'+suffix_out_tmp

        #---------------------- Process ----------------------
        if overwrite or (not os.path.exists(output_fn)):
            #------------------ Load data to allow for indexing ------------------
            ds = ds.load()

            #------------------ Setup seasonal booleans ------------------
            tss = dict()
            #-------- Locally-defined seasons
            if calculate_local:
                ts_fn = (dir_list['proc']+mod_p+'/seasidxs_day_'+mod_p+'_historical_'+
                         str(stats.year.min().values)+'0101-'+str(stats.year.max().values)+'1231_HoA.nc')

                if overwrite or (not os.path.exists(ts_fn)):
                    tss['dunning_local'] = xr.Dataset({'ts':(('lat','lon','time','season'),
                                                  np.zeros((stats.sizes['lat'],stats.sizes['lon'],
                                                            len(timeframe),
                                                            len(seasons))))},
                                           coords = {'lat':stats.lat,
                                                     'lon':stats.lon,
                                                     'time':timeframe,
                                                     'season':seasons})
                    # Make sure ts lat/lon order matches up with stats lat/lon order
                    #tss['dunning_local'].transpose('season','time',*[dim for dim in stats.dims if dim in ['lat','lon']])

                    #------------- Dunning, locally defined -------------
                    tss['dunning_local'] = create_season_mask(tss['dunning_local'],stats)

                    # Remove single-season pixels (should already be removed through
                    # subset_to_srat but that is not the case
                    tss['dunning_local'] = tss['dunning_local'].where(~((~np.isnan(stats.onset.isel(season=0))) & 
                                                                        (np.isnan(stats.onset.isel(season=1)))).all('year'))

                    tss['dunning_local'].attrs['SOURCE'] = 'calculate_seasmeans() from funcs_process.py'
                    tss['dunning_local'].attrs['DESCRIPTION'] = 'boolean for when a given grid-cell-day is in a given season'
                    tss['dunning_local'].attrs['STATS_SOURCE'] = stats_fn

                    if os.path.exists(ts_fn):
                        os.remove(ts_fn)
                        print(ts_fn+' removed to allow overwrite!')

                    tss['dunning_local'].to_netcdf(ts_fn)
                    print(ts_fn+' saved!')
                else:
                    print(ts_fn+' exists, loaded!')
                    tss['dunning_local'] = xr.open_dataset(ts_fn)

            #-------- Regional average seasons
            # Average over double-peaked region
            tss['dunning'] = xr.Dataset({'ts':(('time','season'),np.zeros((len(timeframe),
                                                                          len(seasons))))},
                                        coords={'time':timeframe,
                                                        'season':seasons})


            tss['dunning'] = create_season_mask(tss['dunning'],statsm)


            #-------- MAM/OND Months
            seas_idxs = [[1,3],[3,6],[6,10],[10,1]]

            tss['month'] = xr.Dataset({'ts':(('time','season'),np.zeros((len(timeframe),
                                                                          len(seasons))))},
                                        coords={'time':timeframe,
                                                'season':seasons})

            for seas_idx in np.arange(0,len(seas_idxs)):
                if seas_idxs[seas_idx][0]>seas_idxs[seas_idx][1]:
                    tss['month']['ts'][((tss['month'].time.dt.month>=seas_idxs[seas_idx][0]) |
                                  (tss['month'].time.dt.month<seas_idxs[seas_idx][1])).values,
                                 seas_idx] = True

                else:
                    tss['month']['ts'][((tss['month'].time.dt.month>=seas_idxs[seas_idx][0]) &
                                  (tss['month'].time.dt.month<seas_idxs[seas_idx][1])).values,
                                 seas_idx] = True


            #-------- MAM/SON Months
            seas_idxs = [[12,3],[3,6],[6,9],[9,12]]

            tss['month_alt'] = xr.Dataset({'ts':(('time','season'),np.zeros((len(timeframe),
                                                                          len(seasons))))},
                                        coords={'time':timeframe,
                                                        'season':seasons})

            for seas_idx in np.arange(0,len(seas_idxs)):
                if seas_idxs[seas_idx][0]>seas_idxs[seas_idx][1]:
                    tss['month_alt']['ts'][((tss['month_alt'].time.dt.month>=seas_idxs[seas_idx][0]) |
                                  (tss['month_alt'].time.dt.month<seas_idxs[seas_idx][1])).values,
                                 seas_idx] = True

                else:
                    tss['month_alt']['ts'][((tss['month_alt'].time.dt.month>=seas_idxs[seas_idx][0]) &
                                  (tss['month_alt'].time.dt.month<seas_idxs[seas_idx][1])).values,
                                 seas_idx] = True

            #------------------ Calculate means ------------------
            #-------- Locally-defined seasons
            if calculate_local:
                # Regrid if not the same grid
                if not (np.all([l in tss['dunning_local'].lat for l in ds.lat.values]) and 
                        np.all([l in tss['dunning_local'].lon for l in ds.lon.values])):
                    rgrd = xe.Regridder(stats,ds,method='bilinear')
                    with warnings.catch_warnings():
                        # Ignore the FutureWarning that shows up from inside xesmf 
                        # and adds nothing to the conversation
                        warnings.simplefilter("ignore") 
                        tss['dunning_local'] = rgrd(tss['dunning_local'])
                    tss['dunning_local'] = np.round(tss['dunning_local'])
                    # Make sure NaNs stay NaNs (by setting pixels that were never
                    # assigned to a season to NaN) - xesmf sets them to 0 otherwise
                    tss['dunning_local'] = tss['dunning_local'].where((tss['dunning_local'].isel(season=0).mean('time')!=0).ts)
                    # Change nans to False
                    tss['dunning_local'] = tss['dunning_local'].where(~np.isnan(tss['dunning_local']),False)

            # Make sure they're all bools
            tss = {k:v.astype(bool) for k,v in tss.items()}    

            # Process by season
            ds_out = dict()
            for seas in seasons:
                # For seasons that never cross a calendar year boundary, take mean by
                # merely resampling 
                if seas in ['long_rains','short_dry']:
                    ds_out[seas] = xr.concat([ds.where(tss[k].ts.sel(season=seas)==1).resample(time='1Y').mean()
                                               for k in tss],
                                             dim=pd.Index([k for k in tss],name='kind'))

                    # Replace timestring with just the year of the season
                    ds_out[seas]['time'] = ds_out[seas]['time'].dt.year

                else:
                    # Offset to June start date for short rains, long dry period
                    # annual averages... 
                    ds_out[seas] = xr.concat([ds.where(tss[k].ts.sel(season=seas)==1).resample(time='AS-JUN').mean()
                                                for k in tss],
                                             dim=pd.Index([k for k in tss],name='kind'))

                    if seas == 'long_dry':
                        # Remove the last year's season, which is always going to be 
                        # incomplete (since the long rains don't start before the 
                        # New Year)
                        ds_out[seas] = ds_out[seas].isel(time=slice(None,-1))

                        # Replace timestring with the year + 1 of the season
                        ds_out[seas]['time'] = ds_out[seas]['time'].dt.year + 1

                        # NaN out the first long rains, since there's no way of 
                        # verifying if the mean covers the whole season or not
                        ds_out[seas] = ds_out[seas].where(ds_out[seas].time != ds_out[seas].time[0])

                    elif seas == 'short_rains':
                        # Remove the first year's season, which is an artifact of 
                        # starting the resampling in June of the previous year, but
                        # doesn't actually have data
                        ds_out[seas] = ds_out[seas].isel(time=slice(1,None))

                        # Replace timestring with just the year of the season
                        ds_out[seas]['time'] = ds_out[seas]['time'].dt.year

                        # NaN out locations where the demise is past the calendar year, 
                        # i.e., where the seasonal mean doesn't cover a whole season, 
                        # for the local bounds. We can check this by just seeing where 
                        # the short rains are still around on the last day of `tss` 
                        # (technically it's a cut-off of a day early, but it only applies
                        # to edge cases anyways... 
                        if calculate_local:
                            ds_out[seas].loc[{'kind':'dunning_local',
                                              'time':ds_out[seas].time[-1]}] = (ds_out[seas].loc[{'kind':'dunning_local',
                                                                                                 'time':ds_out[seas].time[-1]}].
                                                                                where(~tss['dunning_local'].isel(time=-1,season=3).ts))
                        #~tss['dunning_local'].isel(time=-1,season=3).ts
                        
                        # NaN out the whole last year's short rains for the regional 
                        # average bounds if the average demise is after the new year
                        if statsm.onset.isel(year=-1,season=1) > 365:
                            ds_out[seas].loc[{'kind':'dunning'}] = ds_out[seas].loc[{'kind:dunning'}].where(ds_out[seas].time < 
                                                                                                            ds_out[seas].time[-1])

                # Rename 'time' to 'year'
                ds_out[seas] = ds_out[seas].rename({'time':'year'})

            # Concatenate into single ds
            ds_out = xr.concat([v for k,v in ds_out.items()],
                               dim = pd.Index([k for k in ds_out],name='season'))

            #------------------ Output ------------------
            ds_out.attrs['SOURCE'] = 'calculate_seasmeans() from funcs_process.py'
            ds_out.attrs['DESCRIPTION'] = ('Seasonal mean values calculated from the seasonal stats '+
                                          ' listed in the STATS_FILE below')
            ds_out.attrs['STATS_FILE'] = stats_fn
            if anomaly is not None:
                if anomaly == 'clim':
                    ds_out.attrs['ANOMALY'] = 'Means are of anomalies vs. long-term avg.'
                elif anomaly == 'seas':
                    ds_out.attrs['ANOMALY'] = ('Means are of anomalies vs. seasonal cycle, determined by '+
                                               'grouping string: '+seas_grouping_mean)

            if os.path.exists(output_fn):
                os.remove(output_fn)
                print(output_fns[v]+' removed to allow overwrite!')

            if not os.path.exists(os.path.dirname(output_fn)):
                os.mkdir(os.path.dirname(output_fn))
                print(os.path.dirname(output_fn)+'/ created!')

            ds_out.to_netcdf(output_fn)
            print(output_fn+' saved!')

        else:
            if return_output:
                ds_out = xr.open_dataset(output_fn)
                print(output_fn+' loaded from file!')
            else:
                print(output_fn+' already exists, skipped!')
    
    #------------------ Return ------------------
    if return_output:
        return ds_out