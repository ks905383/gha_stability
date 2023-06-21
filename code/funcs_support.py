import xarray as xr
import xesmf as xe
import xagg as xa
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re
import os
import glob
import warnings

class NotUniqueFile(Exception):
    """ Exception for when one file needs to be loaded, but the search returned multiple files """
    pass


def get_params():
    ''' Get parameters 
    
    Outputs necessary general parameters. 
    
    Parameters:
    ----------------------
    (none)
    
    
    Returns:
    ----------------------
    dir_list : dict()
        a dictionary of directory names for file system 
        managing purposes: 
            - 'raw':   where raw climate files are stored, in 
                        subdirectories by model/product name
            - 'proc':  where processed climate files are stored,
                        in subdirectories by model/product name
            - 'aux':   where aux files (e.g. those that transcend
                        a single data product/model) are stored
    '''
    
    # Dir_list
    dir_list = pd.read_csv('dir_list.csv')
    dir_list = {d:dir_list.set_index('dir_name').loc[d,'dir_path'] for d in dir_list['dir_name']}
    
    
    # Return
    return dir_list


def get_subset_params():
    ''' Get parameters 
    
    Outputs necessary subsetting parameters. 
    
    Parameters:
    ----------------------
    (none)
    
    
    Returns:
    ----------------------
    subset_params : dict()
        a dictionary of subsetting dictionaries: 
        
    '''
    
    # Read subset params
    subset_params = pd.read_table('default_params.csv',delimiter='   ',engine='python')
    subset_params = {sp:eval(subset_params.set_index('variable').loc[sp,'value']) for sp in subset_params['variable']}
    
    # Create slice'd subset params (with an extra list step in the 
    # for-loop, since running `for sp in subset_params` throws 
    # runtime errors for changing the size of the dictionary
    for sp in [k for k in subset_params]:
        subset_params[sp+'_slice'] = {dim:slice(*v) for dim,v in subset_params[sp].items()}
    
    # Return
    return subset_params

def get_varlist(source_dir=None,var=None,varsub='all',
                experiment=None,freq=None,
                empty_warnings=False):
    ''' Get a list of which models have which variables
    
    Searches the filesystem for all models (directory names) and 
    all variables (first part of filenames, before the first 
    underscore), and returns either that information for all 
    models and variables, or an array of models that have 
    files for specified variables. 
    
    NB: if no experiment or frequency is specified, and the
    full dataframe is returned (`var=None`), then the fields
    have True whenever any file with that variable in the filename
    for that model is present (and potentially more than one). 
    In general, the code does not differentiate between multiple
    files for a single model/variable combination. 
    
    Parameters
    ---------------
    source_dir : str; default dir_list['raw']
        a path to the directory with climate data (all 
        subdirectories are assumed to be models, all files in
        these directories are assumed to be climate data files
        in rough CMIP format).
        
    var : str, list; default `None`
        one variable name or a list of variables for which to 
        subset the model list of. If not `None`, then only a list
        of models for which this variable(s) is present is returned
        (instead of the full Dataframe).
        
    varsub : str; default 'all'
        - if 'all', then if `var` has multiple variables, 
          only models that have files for all of the variables 
          are returned
        - if 'any', then if `var` has multiple variables, 
          models that have files for any of the variables are 
          returned
          
    experiment : str; default `None`
        if not None, then only returns models / True if files
        for the given 'experiment' (in CMIP6 parlance, the 
        fourth filename component) are found. If not None, the
        variable is piped into re.search(), allowing for re
        searches for the experiment. 
        
    freq : str; default `None`
        if not None, then only returns models / True if files
        for the given 'frequency' (in CMIP6 parlance, the 
        second filename component) are found. If not None, the
        variable is piped into re.search(), allowing for re
        searches for the frequency. 
        
    empty_warnings : bool; default `False`
        if True, a warning is thrown if no files at all (before 
        subsetting) are found for a model. 
    
    
    Returns
    ---------------
    varindex : pd.DataFrame()
        if `var` is None, then a models x variables pandas
        DataFrame is returned, with `True` if that model has 
        a file with that variable, and `False` otherwise.
        
    mods : list
        if `var` is not None, then a list of model names 
        that have the variables, subject to the subsetting above
    
    
    '''
    if source_dir is None:
        dir_list = get_params()
        source_dir = dir_list['raw']
    
    
    ##### Housekeeping
    # Ensure the var input is a list of strings, and not a string
    if type(var) == str:
        var = [var]
    
    ##### Identify models
    # Figure out in which position of the filename path the model name
    # directory is located (based on how many directory levels there 
    # are in the parent directory)
    modname_idx = len(re.split('/',source_dir)) - 1
    # Get list of all the models (the directory names in source_dir)
    all_mods = [re.split('/',x)[modname_idx] for x in [x[0] for x in os.walk(source_dir)] if re.split('/',x)[modname_idx]!='']
    all_mods = [mod for mod in list(np.unique(all_mods)) if 'ipynb' not in mod]
    
    ##### Identify variables
    # Get list of all variables used and downloaded
    # Make this a pandas dataarray - mod x var
    varlist = []
    for mod in all_mods[:]:
        varlist.append([re.split('\_',fn)[0] for fn in [x for x in os.walk(source_dir+mod+'/')][0][2]])
    varlist = [item for sublist in varlist for item in sublist]

    varlist = list(np.unique(varlist))

    # Remove "README" and ".nc" files 
    varlist = [var for var in [var for var in varlist if 'READ' not in var] if '.nc' not in var]
    
    ##### Populate dataframe
    # Create empty dataframe to populate with file existence
    varindex = pd.DataFrame(columns=['model',*varlist])

    # Populate the model column
    varindex['model'] = all_mods

    # Actually, just set the models as the index
    varindex = varindex.set_index('model')
    
    # Now populate the dataframe with Trues if that model has that variable as a file
    for mod in all_mods:
        # Get variable name of each file 
        file_varlist = [re.split('\_',fn)[0] for fn in [x for x in os.walk(source_dir+mod+'/')][0][2]]

        if len(file_varlist) == 0:
            if empty_warnings:
                warnings.warn('No relevant files found for model '+mod)
            varindex.loc[mod] = False
        else:
            # Subset by frequency, or experiment, if desired
            if freq is not None:
                try:
                    freq_bools = [(re.search(freq,re.split('\_',fn)[1]) != None) for fn in [x for x in os.walk(source_dir+mod+'/')][0][2]]
                except IndexError:
                    freq_bools = [False]*len(file_varlist)
                    if empty_warnings:
                        warnings.warn('Model '+mod+' has files not in CMIP format.')
                    continue
            else:
                freq_bools = [True]*len(file_varlist)

            if experiment is not None:
                try:
                    exp_bools = [(re.search(experiment,re.split('\_',fn)[3]) != None) for fn in [x for x in os.walk(source_dir+mod+'/')][0][2]]
                except IndexError:
                    exp_bools = [False]*len(file_varlist)
                    if empty_warnings:
                        warnings.warn('Model '+mod+' has files not in CMIP format.')
                    continue
            else:
                exp_bools = [True]*len(file_varlist)

            # Remove from list if it doesn't fit the frequency/experiment subset
            file_varlist = list(np.asarray(file_varlist)[np.asarray(freq_bools) & np.asarray(exp_bools)])

            # Add to dataframe
            varindex.loc[mod] = [var in file_varlist for var in varlist]

    # Fill NaNs with False
    varindex = varindex.fillna(False)

    ##### Return
    if var is None: 
        return varindex
    else:
        if type(var) == str:
            var = [var]
        if varsub == 'all':
            # (1) is to ensure the `all` is across variables/columns, not rows/models
            return list(varindex.index[varindex[var].all(1)].values)
        elif varsub == 'any':
            return list(varindex.index[varindex[var].any(1)].values)
        else:
            raise KeyError(str(varsub) + ' is not a supported variable subsetting method, choose "all" or "any".')
            
            
def utility_print(output_fn,formats=['pdf','png']):
    if 'pdf' in formats:
        plt.savefig(output_fn+'.pdf')
        print(output_fn+'.pdf saved!')
    
    if 'png' in formats:
        plt.savefig(output_fn+'.png',dpi=300)
        print(output_fn+'.png saved!')
        
    if 'svg' in formats:
        plt.savefig(output_fn+'.svg')
        print(output_fn+'.svg saved!')
            
# This whole  business is because np.nanargmax (and its 
# derivatives, including ds.argmax in xr) can't deal if 
# the whole column of months is NaNs. So in that case, we 
# just keep the value NaN, and run nanargmax on the columns
# that do have values (land pixels in GPCC, for example)

def nan_argmax_xr(x,val=0,dim='month'):
    """ Get the index of each maximum month in the 'month'
    dimension of an arbitrary dataarray with dimensions
    'month' and others. This spits out NaN for any 
    row of months that's entirely NaNs, and therefore 
    provides a workaround for np.argmax() and 
    np.nanargmax(), which both fail in this situation.
    Furthermore, it automatically stacks/unstacks for 
    the calculation, so the input can have an arbitrary
    number of dimensions. 
    """
    
    # Stack to have [__ x month]
    input_dims = list(x.dims)
    
    if dim not in input_dims:
        raise LookupError("no '"+dim+"' dimension found.")
    
    input_dims.remove(dim)
    
    if len(input_dims)>1:
        x = x.stack(alld=(tuple(input_dims)))
        unstack = True
    else: 
        unstack = False
    
    if x.ndim>1:
        # Pre-build np.nan
        out_vals = np.zeros((np.shape(x)[np.argmax([key!=dim for key in x.dims])]))*np.nan
    
        #out_vals[~np.isnan(x[0,:])] = x[:,~np.isnan(x.values[0,:])].argsort(0).isel({dim:-1-val})
        #out_vals = xr.DataArray(out_vals,dims=x.dims[1],coords={x.dims[1]:x[x.dims[1]]})
        if not np.all(np.isnan(x).all(dim)): #else keep it nan
            out_vals[~np.isnan(x).all(dim)] = x[:,~np.isnan(x).all(dim)].argmax(dim)
        out_vals = xr.DataArray(out_vals,dims=x.dims[1],coords={x.dims[1]:x[x.dims[1]]})
    else:
        if np.all(np.isnan(x)):
            out_vals = np.nan
        else:
            out_vals = x.argmax(dim)
        # Pretty sure this just needs to be one value... to be consistent
        #out_vals[~np.isnan(x)] = x[~np.isnan(x.values)].argsort(0).isel({dim:-1-val})
        #out_vals = xr.DataArray(out_vals,dims=x.dims[0],coords={x.dims[0]:x[x.dims[0]]})
        out_vals = xr.DataArray(out_vals)
    
    if unstack:
        out_vals = out_vals.unstack()
    
    return out_vals

def nan_argmin_xr(x,val=0,dim='month'):
    """ Get the index of each maximum month in the 'month'
    dimension of an arbitrary dataarray with dimensions
    'month' and others. This spits out NaN for any 
    row of months that's entirely NaNs, and therefore 
    provides a workaround for np.argmax() and 
    np.nanargmax(), which both fail in this situation.
    Furthermore, it automatically stacks/unstacks for 
    the calculation, so the input can have an arbitrary
    number of dimensions. 
    """
    
    # Stack to have [__ x month]
    input_dims = list(x.dims)
    
    if dim not in input_dims:
        raise LookupError("no '"+dim+"' dimension found.")
    
    input_dims.remove(dim)
    
    if len(input_dims)>1:
        x = x.stack(alld=(tuple(input_dims)))
        unstack = True
    else: 
        unstack = False
    
    if x.ndim>1:
        # Pre-build np.nan
        out_vals = np.zeros((np.shape(x)[np.argmax([key!=dim for key in x.dims])]))*np.nan
        
        #out_vals[~np.isnan(x[0,:])] = x[:,~np.isnan(x.values[0,:])].argsort(0).isel({dim:-1-val})
        #out_vals = xr.DataArray(out_vals,dims=x.dims[1],coords={x.dims[1]:x[x.dims[1]]})
        if not np.all(np.isnan(x).all(dim)): #else keep it nan
            out_vals[~np.isnan(x).all(dim)] = x[:,~np.isnan(x).all(dim)].argmin(dim)
        out_vals = xr.DataArray(out_vals,dims=x.dims[1],coords={x.dims[1]:x[x.dims[1]]})
    else:
        if np.all(np.isnan(x)):
            out_vals = np.nan
        else:
            out_vals = x.argmin(dim)
        # Pretty sure this just needs to be one value... to be consistent
        #out_vals[~np.isnan(x)] = x[~np.isnan(x.values)].argsort(0).isel({dim:-1-val})
        #out_vals = xr.DataArray(out_vals,dims=x.dims[0],coords={x.dims[0]:x[x.dims[0]]})
        out_vals = xr.DataArray(out_vals)
    
    if unstack:
        out_vals = out_vals.unstack()
    
    return out_vals

def subset_to_srat(da,srat_mod = 'CHIRPS',
                   srat_file = None,
                   print_srat_fn = False,
                   drop = False,
                   subset = 'double_peaked',
                   subset_params = {'lat':slice(-3,12.5),'lon':slice(32,55)},
                   regrid_method = 'bilinear'):
    """ Subset dataarray to double-peaked area
    
    Parameters 
    -------------------
    da : xarray.core.dataarray.DataArray
        The DataArray to subset
        
    srat_mod : str, by default 'CHIRPS'
        The data product from which to find the seas_ratio file; 
        if CHIRPS, then the filename is hardcoded, if a different
        model, then the first file to satisfy the search:
            'pr_doyavg_[mod]_*_seasstats*.nc' 
        in that model's [proc] directory is used; if `srat_file` is 
        not None, then this parameter is ignored.
        
    srat_file : str, by default None
        If not None, then the file with this filename is used as
        the source of the `seas_ratio` variable 
        
    print_srat_fn : bool, by default False
        If True, then the filename used for the `seas_ratio` 
        variable is printed. 
        
    subset : str, by default 'double-peaked'
        If `=='double-peaked'`, then data are subset to all locations
        where `seas_ratio<1`. If `=='single-peaked'`, then data are
        subset to all locations where `seas_ratio>1`. 
        
    subset_params : dict, by default {'lat':slice(-3,12.5),'lon':slice(32,55)}
        If not None, then `seas_ratio` variable is subset using 
        this subset dictionary. 
        
    regrid_method : str, by default 'bilinear'
        Which method used to regrid `seas_ratio` to the input `da`
        grid; piped into `xe.Regridder()`
        
    drop : bool, by default False
        If true, then dimension coords with all nans are dropped
        
        
    Returns
    -------------------
    da : xarray.core.dataarray.DataArray
        The input DataArray, but now subset geographically to 
        areas with a double-peaked rainfall climatology, as 
        defined by the `seas_ratio` variable / file used. 
    
    """
    from funcs_load import load_raw
    
    dir_list = get_params()
        
    # Load seas_ratio from stats file
    if srat_file is not None:
        srat = xr.open_dataset(srat_file).seas_ratio
    else:
        srat, fns_match = load_raw('pr_doyavg_*_seasstats_*HoA.nc',
                                   search_dir=dir_list['proc']+srat_mod+'/',
                                   return_filenames=True)
        srat = srat.seas_ratio.drop('method')
            
    if print_srat_fn:
        print('used '+fns_match[0]+' as source for `seas_ratio` variable.')
            
    # Subset seas_ratio to desired location
    if subset_params is not None:
        srat = srat.sel(**subset_params)
    
    # Get rid of singleton dimensions (e.g., "method")
    srat = srat.squeeze()
    
    # Regrid hdiff to precip grid, if different grids
    if not (np.all([l in srat.lat for l in da.lat.values]) and 
        np.all([l in srat.lon for l in da.lon.values])):
        with warnings.catch_warnings():
            # Ignore the FutureWarning that shows up from inside xesmf 
            # and adds nothing to the conversation
            warnings.simplefilter("ignore") 
            # Regrid 
            rgrd = xe.Regridder(srat,da,method=regrid_method)
            srat = rgrd(srat)

    # Set 0s to nan (artifact of the process) 
    srat = srat.where(srat!=0)
        
    # Subset to double-peaked region
    if subset == 'double_peaked':
        da = da.where(srat<1,drop=drop)
    elif subset == 'single_peaked':
        da = da.where(srat>1,drop=drop)
    else:
        raise KeyError("`subset` must be either 'double-peaked' or 'single-peaked', but was '"+subset+"'")
    
    # Return
    return da


# The next two are from https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7

def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid
    defined by WGS84
    
    Input
    ---------
    lat: vector or latitudes in degrees  
    
    Output
    ----------
    r: vector of radius in meters
    
    Notes
    -----------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''
    from numpy import deg2rad, sin, cos

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)
    
    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    # radius equation
    # see equation 3-107 in WGS84
    r = (
        (a * (1 - e2)**0.5) 
         / (1 - (e2 * np.cos(lat_gc)**2))**0.5 
        )

    return r


def area_grid(lat, lon):
    """
    Calculate the area of each grid cell
    Area is in square meters
    
    Input
    -----------
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees
    
    Output
    -----------
    area: grid-cell area in square-meters with dimensions, [lat,lon]
    
    Notes
    -----------
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """
    from numpy import meshgrid, deg2rad, gradient, cos
    from xarray import DataArray

    xlon, ylat = meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = deg2rad(gradient(ylat, axis=0))
    dlon = deg2rad(gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * cos(deg2rad(ylat))

    area = dy * dx

    xda = DataArray(
        area,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    return xda

def area_mean(ds):
    """ Calculate area-weighted mean of all variables in a  dataset
    
    Mean over lat / lon, weighted by the relative size of each
    pixel, dependent on latitude. Only weights by latitude, does
    not take into account lat/lon bounds, if present. 
    
    Parameters
    ------------------
    ds : xr.Dataset
    
    Returns
    ------------------
    dsm : xr.Dataset
        The input dataset, `ds`, averaged.
    
    """
    
    # Calculate area in each pixel
    weights = area_grid(ds.lat,ds.lon)

    # Remove nans, to make weight sum have the right magnitude
    weights = weights.where(~np.isnan(ds))
    
    # Calculate mean
    ds = ((ds*weights).sum(('lat','lon'))/weights.sum(('lat','lon')))
    
    # Return 
    return ds


def sig_fdr(ps: xr.core.dataarray.DataArray,FDR=0.2,stack_dims=('lat','lon')) -> xr.core.dataarray.DataArray:
    ''' Calculate field significance contingent on a given FDR
    Adapted from Wilks 2016: "'The Stippling Shows Statistically 
    Significant Grid Points': How Research Results are Routinely 
    Overstated and Overinterpreted, and What to Do about it"
    
    Parameters 
    -----------------------
    ps : xr.DataArray
        a DataArray of significance values over some grid containing
        lat / lon, and any optional number of other dimensions
    
    FDR : float (default 0.2) 
        the desired false discovery rate (FDR)
    
    Returns 
    -----------------------
    sigTests : xr.DataArray
        a DataArray of booleans showing whether a pixel is significant
        based on a given FDR
    ''' 
    if type(ps) != xr.core.dataarray.DataArray:
        raise TypeError('`ps` must be an xarray DataArray.')
    
    # stack geographic variables into one
    ps_stack = ps.stack(loc=stack_dims)
    
    # calculate (i/N)*a_fdr, where i is the rank, N is the 
    # number of pixels (highest rank gets it; otherwise would
    # be a count of non-nans, which might be faster but uglier), 
    # and a_fdr is the significance marker
    sigLevel = (ps_stack.rank('loc')/ps_stack.rank('loc').max('loc'))*FDR

    # significant pixels are those where
    # p < p_fdr = max{p(i):p(i) < (i/N)a_fdr}
    sigTests = ps_stack < ps_stack.where(ps_stack < sigLevel).max('loc')
    # restore nans, which are removed through the < 
    sigTests = sigTests.where(~np.isnan(ps_stack))
    
    # unstack and return
    sigTests = sigTests.unstack()
    return sigTests