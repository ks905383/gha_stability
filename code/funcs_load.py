import xarray as xr
import cf_xarray
import numpy as np
import pandas as pd
import os
import re
import glob
import itertools
import warnings
import cftime

from funcs_support import get_params, NotUniqueFile    


def load_raw(search_str,
             search_dir=None,  
             rsearch=False,
             fn_ignore_regexs=[],
             subset_params=None, 
             squeeze=True,
             aggregate=False,    
             aggregate_dims = ['latitude','longitude'], 
             load_single=True,
             show_filenames = False,
             return_filenames = False,
             fix360_subset = True
            ):
    ''' Loads and subsets climate data files
    
    Theoretically takes all the back-end work out of loading raw climate
    data. Loads, subsets, and aggregates, without having to remember which
    file suffix corresponds to which geographic subset. 
    
    Depends on the behavior of xr.combine_by_coords().
    
    NOTE: Currently can't deal with files with overlapping domains...
    
    Parameters:
    ------------------------------
    search_str : str
        The string used by glob.glob to search for files to load. 
        
    search_dir : str, by default get_params()['raw']
        The directory in which to look for files. 
        
    rsearch : bool, default False
        NOT YET IMPLEMENTED. If `True`, looks recursively through
        directories within `search_dir` and concatenates along 
        a dimension [X]. 
        
    fn_ignore_regexs : str or list
        
    subset_params : dict, by default None
        If not `None`, then files will be subset to the slices in 
        this dict. Sample: 
            `subset_params = {'lat':slice(-5,5)}`
        Pro-tip: if instead of a slice you put in a single value, 
        then the code may break. Replace with a list, e.g.:
            `subset_params = {'plev':[650]}`
        which will be squeezed out at the end anyways, if 
        `squeeze=True`. 
            
    squeeze : bool, by default True
        If True, then the returned xr.Dataset is squeezed (i.e., 
        singleton dimensions are removed)
            
    aggregate : bool, by default False
        If `True`, then the mean over `aggregate_dims` is taken
        
    aggregate_dims : list, by default `['latitude','longitude']`
        If `aggregate == True`, then the average over these dimensions
        is taken. The code will first look to see if these dimension
        names exist in the merged dataset, or, if that fails, attempt
        to take the mean treating `aggregate_dims` as `cf_xarray` 
        names. 
        
    load_single : bool, by default True
        If `True`, then if no subset is taken (subset_params), 
        if more than one file is found in `search_str`, the code fails.
        Designed as a failsafe if too many files are unintentionally
        caught with `search_str`
        
    show_filenames : bool, by default False
        If `True`, then the matched filenames are printed.
        
    return_filenames : bool, by default False
        If `True`, then the matched filenames are returned.
        
    fix360_subset : bool, by default True
        If `True`, then if the calendar type of a file is cftime.Datetime360Day
        `subset_params` includes a time subset, and that time subset ends 
        on the 31st day of a month, this is replaced with the 30th 
        day of the month instead     
    
    Returns:
    ------------------------------
    ds : xr.Dataset
        A merged dataset, potentially aggregated, of the desired data.
        
    fns_match : list
        If `return_filenames = True`, the list of matched filenames
    
    '''
    
    
    #----------- Setup -----------
    # If no search_dir provided, assume it's the raw 
    # data directory from get_params()
    if search_dir is None:
        dir_list = get_params()
        search_dir = dir_list['raw']
        
    if type(fn_ignore_regexs) != list:
        fn_ignore_regexs = [fn_ignore_regexs]
    
    
    #----------- Find and load files -----------
    # Get files in [search_dir] that match [search_str]
    if rsearch:
        raise NotYetImplementedError()
    
    fns_match = glob.glob(search_dir+search_str)
    if show_filenames:
        print('Files found from search "'+search_dir+search_str+'":\n  '+
              '\n  '.join(fns_match))
            
    if len(fns_match) == 0:
        raise Exception('No files found using search "'+search_dir+search_str+'"')
    
    if len(fn_ignore_regexs) != 0:
        for fn_ignore_regex in fn_ignore_regexs:
            fns_match = list(filter(lambda item: item is not None,
                                    [fn if (re.search(fn_ignore_regex,fn) is None) else None for fn in fns_match]))
        if show_filenames: 
            print('filenames after subsetting:\n'+'\n'.join(fns_match))
    
    # Load them for their dimensions 
    try: 
        dss = [xr.open_dataset(fn) for fn in fns_match]
    except:
        raise Exception('Issue loading one of the following files:'+'\n'.join(fns_match))
    
    # Subset all using subset_params
    if subset_params is not None: 
        subset_params_tmp = subset_params
        if (('time' in subset_params) and 
            (type(dss[0].time.values[0]) == cftime.Datetime360Day) and 
            fix360_subset):
            subset_params_tmp['time'] = slice(*[re.sub('-31$','-30',subset_params['time'].start),
                                            re.sub('-31$','-30',subset_params['time'].stop)])

        dss = [ds.sel(**subset_params_tmp) for ds in dss]
        
        # Test if this subsetting resulted in empty 
        # dimensions in a particular file
        subset_flags = [np.all([ds.sizes[k]!=0 for k in subset_params_tmp])
                        for ds in dss]
        # Remove empty subsets
        dss = list(itertools.compress(dss,subset_flags))
    else:
        if len(dss)>1:
            warnings.warn('Multiple files found, with no '+
                          'desired subset: \n  '+
                          '\n  '.join(fns_match))
            if load_single:
                raise NotUniqueFile('More than one file found, but since '+
                                'load_single==True, no files are loaded '+ 
                                'to avoid memory overloads.')
                
    #----------- Concatenate -----------
    dss = xr.combine_by_coords(dss,combine_attrs='drop_conflicts')
    
    #----------- Additional processing if desired -----------
    if aggregate:
        try:
            # If subset_params dimensions names aren't in the 
            # dataset dimensions, then try using cf_xarray
            # conventions
            if np.any([dim not in dss.sizes for dim in aggregate_dims]):
                dss = dss.cf.mean(aggregate_dims)
            # If subset_params dimensions names are in the 
            # dataset dimensions, aggregate over those dimensions
            else:
                dss = dss.mean(aggregate_dims)
        except ValueError:
            raise Exception('The dimensions on which to aggregate ('+','.join(aggregate_dims)+') '+
                            'were not all found in the dimension list ('+','.join([dim for dim in dss.sizes])+') '+
                            'or as cf_xarray supported dimension names.')
    
    #----------- Return -----------
    # Remove singleton dimensions if desired
    if squeeze:
        dss = dss.squeeze()
    if return_filenames:
        return dss,fns_match
    else:
        return dss


def load_raws(search_str,
              search_dir,
              dir_skip = None,
              **kwargs):
    ''' Call `load_raw()` over all subdirectories 
    
    
    Parameters:
    ------------------
    search_str : str
        String to search 
        
    search_dir : str
        Directory with subdirectories (e.g., by model) 
        to search through
        
    dir_skip : list, by default None
        Any subdirectories (e.g., models) to skip
        
    **kwargs: 
        Any inputs to `load_raw()`
        
    Returns:
    ------------------
    dss : dict()
        
    
    '''
    
    # Get all subdirectories in the search_dir
    all_subdirs = [re.split('/',x)[-2] for x in glob.glob(search_dir+'/*/')]
    # Skip all that are to be skipped
    if dir_skip is not None:
        all_subdirs = [mod for mod in all_subdirs if mod not in dir_skip]
        
    # Start an empty list of possible subdirectories
    # to search in
    search_subdirs = dict()
    # Start an empty dictionary 
    dss_out = dict()

    # If a match exist for the search string in the subdirectory, 
    # then save it 
    for subdir in all_subdirs[:]: 
        if len(glob.glob(search_dir+subdir+'/'+search_str)) > 0:
            search_subdirs[subdir] = search_dir+subdir+'/'
            dss_out[subdir] = None
            
    # For each subdirectory with a match, run `load_raw`
    for subdir in search_subdirs:
        dss_out[subdir] = load_raw(search_str,
                               search_dir = search_subdirs[subdir],
                               **kwargs)
        
    # Return 
    return dss_out