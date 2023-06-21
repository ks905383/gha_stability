import xarray as xr
import numpy as np
import pandas as pd
import os
import re
import glob
import warnings
from functools import reduce
from scipy import stats
from scipy import interpolate as spi
from tqdm.notebook import tqdm
import copy

from funcs_support import get_varlist,get_params,nan_argmax_xr,nan_argmin_xr


def wrapper_seasonal_stats(subset_params,overwrite=False,
                           override_30day_convert=False,
                           alt_doy_for_ann=None,
                           mod_subset=None,
                           proc_year=True):
    ''' Wrapper function calculating seasonal stats on all precipitation files
    
    1) Get lists of models with precipitation files
    2) Run process_seasonal_stats_dunning() on each of the models
    
    override_30dayconvert : does not convert from 360 to 365-day calendar if 
                            the last day of month is found to be 30 (useful if
                            file ends in the middle of the year)
    '''
    #--------------------------------------------------------------#
    # Figure out which models have precipitation files, and get 
    # filenames
    #--------------------------------------------------------------#
    dir_list = get_params()
    mod_list = get_varlist(source_dir=dir_list['raw'],var=['pr'])
    
    mod_fns = dict()

    if mod_subset is not None:
        mod_list = [mod for mod in mod_list if mod in mod_subset]

    for mod in mod_list[:]: # For some reason without the [:] it randomly skips a few. No clue why. 
        #print(mod)
        hist_fns = [os.path.basename(x) for x in glob.glob(dir_list['raw']+mod+'/pr_day*'+subset_params['experiment_id']+'*.nc')]

        if len(hist_fns)==0:
            mod_list.remove(mod)
        else:
            if len(hist_fns)>1:
                warnings.warn('Model '+mod+' has more than one "'+subset_params['experiment_id']+'" file. Only the first, '+hist_fns[0]+', will be used.')
            mod_fns[mod]=hist_fns[0]
    del hist_fns, mod_list
    
    
    #--------------------------------------------------------------#
    # Calculate seasonal stats for each precipitation file
    #--------------------------------------------------------------#
    #master_params['overwrite']
    for mod in mod_fns.keys():
        process_seasonal_stats_dunning(mod,mod_fns,dir_list,overwrite=overwrite,
                                       subset_params=subset_params,
                                       override_30day_convert=override_30day_convert,
                                       alt_doy_for_ann=alt_doy_for_ann,
                                       proc_year=proc_year)
        


def seas_params_dunning(ds,subset_params,dir_list,num_seasons=2,save_temp=True,output_fn=None,
                        mod_name='',yr_str='',fn_suffix='',ignore_warnings=True,
                        overwrite=False,
                        override_30day_convert=False, # If False, then subset_params timeslices that end in 31 get changed to 30 if the last day of month is 30 instead of 31 (implying, sometimes, a 360-day-calendar). Problematic if input doesn't end on 12-31.
                        diag_mode=False): #diag_mode returns C in addition to ds,bi_idxs
    '''
    Calculate seasonal onsets/demises and associated seasonal statistics on a rainfall climatology
    
    '''
    
    if ignore_warnings:
        warnings.filterwarnings('ignore')
        
    if output_fn is None:
        if yr_str == '':
            try:
                yr_str = re.sub('-','',str(ds.time[0].values)[:10])+'-'+re.sub('-','',str(ds.time[-1].values)[:10])
            except: 
                yr_str == ''

        output_fn = dir_list['proc']+mod_name+'/pr_doyavg_'+mod_name+'_seasstats_dunning'+yr_str+fn_suffix+'.nc'


    if overwrite or (not os.path.exists(output_fn)): 
        if (not override_30day_convert) and (ds.time.max().dt.day==30):
            cal360 = True
        else:
            cal360 = False

        # Remove leap days to not mess up the fft
        ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))

        # If [seasonal defining] 'method' is not a dimension, extend in that dimension
        if 'method' not in ds.dims:
            try:
                ds.expand_dims('method')
            except IndexError: 
                # The following error once showed up: 
                # IndexError: The indexing operation you are attempting to perform is not 
                # valid on netCDF4.Variable object. Try loading your data into memory first 
                # by calling .load().
                # This is an attempt to bypass it. I'm really not sure why it showed up / 
                # what deeper madness caused it. It showed up with MPI-ESM-1-2-HAM ssp370 processing. 
                ds = ds.load()
                ds.expand_dims('method')
            ds['method'] = ['dunning']
        else:
            # If 'cum_acc' is not already a 'method' option, make it so
            if 'dunning' not in ds['method']:
                ds['method'] = list(ds['method'].values)+['dunning']
        # Add seasonal dimension if not present (just 2 seasons max)
        if 'season' not in ds.dims:  
            ds.expand_dims('season')
            ds['season'] = np.arange(1,num_seasons+1)
        # Add variables if not present
        stat_vars = ['onset','demise','duration','peak_timing','peak_amount','integrated_amount']
        for stat_var in stat_vars:
            if stat_var not in ds.variables.keys():
                ds = ds.assign({stat_var:(('lat','lon','season','method'),
                                          np.zeros((ds.dims['lat'],ds.dims['lon'],
                                                    ds.dims['season'],ds.dims['method']))*np.nan)})

        # Stack by creating one master dimension for lat/lon
        # Adding 'bnds', because occasionally this causes a 
        # spurious dimensional add
        dims_tmp = [e for e in ds.pr.dims if e not in ('time','dayofyear','method','season','bnds','year')]
        ds = ds.stack(allv=dims_tmp)

        # Figure out if there are nans, and deal with them 
        # accordingly. If all timesteps are nan for a pixel
        # that's fine (maybe still extract them to avoid
        # trying to fft it). But for those where, say, a year
        # is missing, then we need another way of dealing with it.
        # FFT will (obviously) fail if there are random nans in the 
        # signal. So how about:
        #    1. Check if *any* are nan
        #    2. If yes, get a boolean array for isnan on the whole 
        #       array (this may cause memory issues).
        #    3. Calculate sum of nans by pixel - if sum == width, 
        #       then file it away as pixels we don't have to deal with
        #    4. If sum != width ! then get the index of the first nan
        #       in each of those rows. If they're all the same, then 
        #       subset to the time before that first nan, throw a warning,
        #       and change the resultant filename. 
        #    5. Otherwise error?
        if np.isnan(ds.pr.dot(ds.pr)): # This is the fastest way to figure out if there are any nans
            nan_bool = np.isnan(ds.pr)

            # This is the pixels where every timestep is nan
            nanskips = (nan_bool.sum('time')==ds.dims['time'])

            # If all the nans are just pixels where every 
            # timestep is nan, then we don't have to continue
            if not (nanskips | (nan_bool.sum('time')==0)).all():
                # This gives the first nan idx for those where
                # not all are nan
                nonstandard_nans = nan_bool[:,((~nanskips) & (nan_bool.sum('time')>0) )].argmax(axis=0)
                if len(np.unique(nonstandard_nans))==1: 
                    if (np.unique(nonstandard_nans) % 365) == 0:
                        ds = ds.isel(time=slice(0,np.unique(nonstandard_nans)[0]))
                        warnings.warn('Uniformly missing data; array has been truncated to ['+
                                      str(ds.time[0].values)[:10]+
                                      ', '+str(ds.time[np.unique(nonstandard_nans)[0]-1].values)[:10]+']')
                    else:
                        # Round down to end of previous year
                        new_max_idx = np.unique(nonstandard_nans)[0] - (np.unique(nonstandard_nans)[0] % 365)
                        ds = ds.isel(time=slice(0,new_max_idx))
                        warnings.warn('Uniformly missing data, but not for at a year boundary (at idx '+
                                      str(int(np.unique(nonstandard_nans)[0]))+'); array has been truncated to ['+
                                     str(ds.time[0].values)[:10]+', '+
                                        str(ds.time[new_max_idx-1].values)[:10]+']')
                    if len(yr_str)>0:
                        print('Changing year string from '+yr_str+'...')
                        yr_str = re.sub(yr_str.split('-')[1][0:4],str(int(yr_str[0:4])+int(ds.dims['time']/365)-1),yr_str)
                        print('... to '+yr_str)    

                else:
                    nanskips[nan_bool.sum('time')>0] = True
                    warnings.warn('Inconsistently missing data... putting nans in all pixels with some nans.')
            else:

                warnings.warn('Uniformly missing geographic data; no temporal truncation necessary.')
        else:
            nanskips = xr.DataArray([False]*ds.dims['allv'],dims=['allv'],coords=[ds.allv])

        ## Get the ratio of the strength of the yearly signal and the 
        ## 6-month signal, to figure out if the pixel has an annual or
        ## a biannual precipitation seasonality. 
        # Calculate the FFT
        # (Replace with `xrft` when the issue with 365-day calendars is fixed)
        ds_seas = np.zeros(ds.dims['allv'])*np.nan
        nonnan_idxs = (~nanskips).values.nonzero()[0]


        # Subset to complete years only, otherwise the fft gets weird 
        ds_fft = ds.sel(time=slice(ds.time[0],
                      str(pd.DatetimeIndex([ds.time[-1].values + np.timedelta64(1, 'D')]).year[0]-1)+
                      '-12-'+str(ds.time.dt.day.max().values)))

        try:
            fft_tmp = np.fft.fft(ds_fft.pr.isel(allv=~nanskips),axis=0) #axis=0 is time, since it's (time,lat,lon). Hopefully it stays that way.

            # Get the ratio of the N/365 and N/(2*365) signals
            if cal360: # 360 day
                ds_seas[nonnan_idxs] = np.abs(fft_tmp[int(ds_fft.dims['time']/360),:])/np.abs(fft_tmp[int(2*ds_fft.dims['time']/360),:])
            else: # 365 day
                ds_seas[nonnan_idxs] = np.abs(fft_tmp[int(ds_fft.dims['time']/365),:])/np.abs(fft_tmp[int(2*ds_fft.dims['time']/365),:])
            del fft_tmp
        except MemoryError:
            print('Memory error encountered. Calculating seasonal ratios by groups of 1000 pixels.')

            for proc_idx in np.arange(0,np.ceil(len(nonnan_idxs)/1000)+1,dtype=int):
                max_idx = np.min([((proc_idx+1)*1000),len(nonnan_idxs)])
                fft_tmp = np.fft.fft(ds_fft.pr.isel(allv=nonnan_idxs[(proc_idx*1000):max_idx]))
                # Get the ratio of the N/365 and N/(2*365) signals
                if cal360: # 360 day
                    ds_seas[nonnan_idxs[(proc_idx*1000):max_idx]] = np.abs(fft_tmp[int(ds_fft.dims['time']/360),:])/np.abs(fft_tmp[int(2*ds_fft.dims['time']/360),:])
                else: # 365 day
                    ds_seas[nonnan_idxs[(proc_idx*1000):max_idx]] = np.abs(fft_tmp[int(ds_fft.dims['time']/365),:])/np.abs(fft_tmp[int(2*ds_fft.dims['time']/365),:])

        # Add to [ds]
        ds['seas_ratio'] = (('allv'),ds_seas)
        del ds_seas,ds_fft


        # Get average dayofyear time series and subset temporally 
        # (switching 31 to 30 for 360-day calendars)
        if cal360:
            ds_doy = ds.sel(time=slice(subset_params['time'][0],re.sub('-31','-30',subset_params['time'][1])))
            # Get day-of-year averages
            ds_doy = ds_doy.groupby('time.dayofyear').mean().isel(dayofyear=slice(0,360)) 
            # And regrid to 365 to make the comparisons to 365-day calendars valid (?)
            # Have to put in the compute() because these 
            # are by default dask arrays, chunked along
            # the time dimension, and can't interpolate
            # across dask chunks... 
            ds_doy = ds_doy.compute().interp(dayofyear=(np.arange(1,366)/365)*360)
            # And reset it to 1:365 indexing on day of year
            ds_doy['dayofyear'] = np.arange(1,366)
            # Throw in a warning, too, why not
            warnings.warn(mod_name+' has a 360-day calendar; daily values were interpolated to a 365-day calendar')
        else:
            ds_doy = ds.sel(time=slice(*subset_params['time']))
            # Get day-of-year averages, but skipping day 366
            ds_doy = ds_doy.groupby('time.dayofyear').mean().isel(dayofyear=slice(0,365)) 


        #### Get the overall cumulative anomaly
        # Calculate A as above
        C = (ds_doy.pr.cumsum('dayofyear') - ds_doy.pr.mean('dayofyear')*ds_doy.dayofyear)


        #### ANNUAL CYCLE (DEFAULT, TO ALSO COVER SOME MIS-IDENTIFIED BIANNUAL ONCES)
        # Add to the dataset - with the "+1" because the day *after* the 
        # minmum/maximum should be the onset/demise in this method
        ds.sel({'season':1,'method':'dunning'})['onset'][:] = nan_argmin_xr(C,dim='dayofyear')+1
        ds.sel({'season':1,'method':'dunning'})['demise'][:] = nan_argmax_xr(C,dim='dayofyear')+1


        #### BIANNUAL CYCLE
        # Smooth using 30-day running mean
        C = C.pad(dayofyear=15,mode='wrap').rolling(dayofyear=30,center=True).mean().isel(dayofyear=slice(15,365+15))
        C = C.pad(dayofyear=4,mode='wrap')

        relmins = ((C[4:-4,:].values<C[5:-3,:].values) & (C[4:-4,:].values<C[6:-2,:].values) & (C[4:-4].values<C[7:-1].values) & (C[4:-4].values<C[8:].values) &
        (C[4:-4,:].values<C[0:-8,:].values) & (C[4:-4,:].values<C[1:-7,:].values) & (C[4:-4,:].values<C[2:-6,:].values) & (C[4:-4].values<C[3:-5].values))
        relmaxs = ((C[4:-4,:].values>C[5:-3,:].values) & (C[4:-4,:].values>C[6:-2,:].values) & (C[4:-4].values>C[7:-1].values) & (C[4:-4].values>C[8:].values) &
        (C[4:-4,:].values>C[0:-8,:].values) & (C[4:-4,:].values>C[1:-7,:].values) & (C[4:-4,:].values>C[2:-6,:].values) & (C[4:-4].values>C[3:-5].values))

        # Now, identify the seasons for each pixel 
        # (have to do this with a for loop, because
        # sometimes there are more identified starts
        # than ends or vice-versa. For each season start,
        # we define the season as lasting until the next 
        # end, and pick the longest ones (make sure not to
        # double count, I guess))
        bi_idxs = (ds.seas_ratio<1).values.nonzero()[0]

        for idx in bi_idxs:
            # Pre-generate season array
            season_dates = np.zeros((2,np.max([np.sum(relmins[:,idx]),np.sum(relmaxs[:,idx])])))*np.nan

            # For each onset, find the next highest demise, 
            # going onwards if multiple are found
            onset_idx = 0
            demise_idx = 0


            ## DETERMINE ONSETS
            for season_idx in np.arange(0,np.shape(season_dates)[1]):    
                # If first season, use the first onset, and increase
                # the onset_idx
                if (season_idx == 0):
                    season_dates[0,season_idx] = relmins[:,idx].nonzero()[0][onset_idx]
                    onset_idx = onset_idx+1
                # If we are at the last onset, just fill all remaining ones 
                # with this onset 
                elif onset_idx == np.sum(relmins[:,idx]):
                    season_dates[0,season_idx:] = relmins[:,idx].nonzero()[0][onset_idx-1]
                    break
                # If there are no demises between this onset and the 
                # previous one, use that onset, and increase the onset_idx
                # (this is the case for double onsets) OR if there is one 
                # demise (the sign of this thing actually working perfectly)
                elif np.sum((relmaxs[:,idx].nonzero()[0]>relmins[:,idx].nonzero()[0][onset_idx-1]) & 
                           (relmaxs[:,idx].nonzero()[0]<relmins[:,idx].nonzero()[0][onset_idx]))<=1:
                    season_dates[0,season_idx] = relmins[:,idx].nonzero()[0][onset_idx]
                    onset_idx = onset_idx+1
                # If there is more than one demise between this onset
                # and the previous onset, AND there are fewer than that number
                # of doubled season_dates, then use the previous onset_idx 
                # and do not increase the onset_idx
                elif ((np.sum((relmaxs[:,idx].nonzero()[0]>relmins[:,idx].nonzero()[0][onset_idx-1]) & 
                               (relmaxs[:,idx].nonzero()[0]<relmins[:,idx].nonzero()[0][onset_idx]))>1) & 
                          (np.sum(season_dates[0,:]==relmins[:,idx].nonzero()[0][onset_idx-1]) < 
                           (np.sum(np.logical_and(relmaxs[:,idx].nonzero()[0]>relmins[:,idx].nonzero()[0][onset_idx-1],
                                                  relmaxs[:,idx].nonzero()[0]<relmins[:,idx].nonzero()[0][onset_idx]))))):
                    season_dates[0,season_idx] = relmins[:,idx].nonzero()[0][onset_idx-1]

                else:
                    season_dates[0,season_idx] = relmins[:,idx].nonzero()[0][onset_idx]
                    onset_idx = onset_idx + 1

            if len(np.unique(season_dates[0,:]))==1:
                warnings.warn('Pixel '+str(idx)+' was labeled as biannual, but only one onset was found.')
                continue

            ## DETERMINE DEMISES
            for season_idx in np.arange(0,np.shape(season_dates)[1]):
                # Count up, until the demise_idx shows a demise that is beyond the 
                # season onset

                # First, the case if the season 'wraps around'; i.e. there are no 
                # demises after the onset
                if np.sum(relmaxs[:,idx].nonzero()[0]>season_dates[0,season_idx]) == 0:
                    # If there are mulitple demises before the first onset (i.e. 
                    # multiple demises associated with this onset), then put in 
                    # the demise set by how many nans underneath the multiple 
                    # onsets are left in [season_dates]
                    if np.sum(relmaxs[:,idx].nonzero()[0]<season_dates[0,0]) > 1:
                        season_dates[1,season_idx] = relmaxs[:,idx].nonzero()[0][np.sum(season_dates[0,:]==season_dates[0,season_idx]) - 
                                                                                 np.sum(np.isnan(season_dates[1,season_dates[0,:]==season_dates[0,season_idx]]))]
                    # Otherwise, the demise is just the first demise
                    else:
                        season_dates[1,season_idx] = relmaxs[:,idx].nonzero()[0][0]
                # Now the case if there are demises between this and the next onset
                else:
                    try:
                        # If there's only one (or 0) demise between this onset and the next 
                        # onset just list that demise
                        if np.sum((relmaxs[:,idx].nonzero()[0]>season_dates[0,season_idx]) & (relmaxs[:,idx].nonzero()[0]>season_dates[0,season_idx])) <= 1:
                            season_dates[1,season_idx] = relmaxs[:,idx].nonzero()[0][(relmaxs[:,idx].nonzero()[0]>season_dates[0,season_idx]).nonzero()[0][0]]
                        #elif ([np.sum(season_dates[0,:]==season_dates[0,season_idx]) - np.sum(np.isnan(season_dates[1,season_dates[0,:]==season_dates[0,season_idx]]))] > 
                        #      np.sum(relmaxs[:,idx].nonzero()[0]>season_dates[0,season_idx]).nonzero()[0]): 
                        else:
                            season_dates[1,season_idx] = relmaxs[:,idx].nonzero()[0][(relmaxs[:,idx].nonzero()[0]>
                                                                                      season_dates[0,season_idx]).nonzero()[0][np.sum(season_dates[0,:]==season_dates[0,season_idx]) - 
                                                                                     np.sum(np.isnan(season_dates[1,season_dates[0,:]==season_dates[0,season_idx]]))]]
                    except IndexError:
                        # Keep this error just to make sure nothing else goes wrong. 
                        # This is the case when there are both 'wrap-around' and 'in-year'
                        # demises. So get the next overshoot in the wrapped-around year
                        max_idx = ([np.sum(season_dates[0,:]==season_dates[0,season_idx]) - np.sum(np.isnan(season_dates[1,season_dates[0,:]==season_dates[0,season_idx]]))] - np.sum(relmaxs[:,idx].nonzero()[0]>season_dates[0,season_idx]).nonzero()[0])
                        max_idx = max_idx-len((relmaxs[:,idx].nonzero()[0]>season_dates[0,season_idx]).nonzero()[0])

                        max_idx = (relmaxs[:,idx].nonzero()[0]<season_dates[0,0]).nonzero()[0][max_idx]

                        season_dates[1,season_idx] = relmaxs[:,idx].nonzero()[0][max_idx]
                        #breakpoint()

            # Gah there's also the case where there are multiple onsets in a row followed by multiple demises in a row

            if len(np.unique(season_dates[1,:]))==1:
                warnings.warn('Pixel '+str(idx)+' was labeled as biannual, but only one demise was found.')
                continue


            # Season (periodic) length
            seas_length = np.min([np.abs(season_dates[1,:]-season_dates[0,:]),365-np.abs(season_dates[1,:]-season_dates[0,:])],0)
            sort_idxs = np.argsort(seas_length)[::-1]

            # Make sure not to double count - if the longest two 
            # seasons both have the same onset or the same demise, 
            # then the shorter of the two is deleted (it is assumed
            # that the longer length is the true measure of the season)
            for season_idx in np.arange(0,np.min([num_seasons-1,np.shape(season_dates)[1]-1])):
                while len(np.unique(season_dates[0,sort_idxs[season_idx:(season_idx+2)]])) == 1:
                    sort_idxs = np.delete(sort_idxs,[season_idx+1])

                while len(np.unique(season_dates[1,sort_idxs[season_idx:(season_idx+2)]])) == 1:
                    sort_idxs = np.delete(sort_idxs,[season_idx+1])

            # Now, sort the sorting indices to ensure that the 
            # 'first' season is the one with the earliest onset by 
            # calendar date. 
            sort_idxs = np.sort(sort_idxs[0:num_seasons])

            # Add to the dataset - with the "+1" because the day *after* the 
            # minmum/maximum should be the onset/demise in this method
            ds.sel({'method':'dunning'}).isel(allv=idx)['onset'][0:np.min([num_seasons,np.shape(season_dates)[1]])] = season_dates[0,sort_idxs]+1
            ds.sel({'method':'dunning'}).isel(allv=idx)['demise'][0:np.min([num_seasons,np.shape(season_dates)[1]])] = season_dates[1,sort_idxs]+1


        #### OTHER STATISTICS
        # Calculate duration (with minimum of both directions)
        ds.sel({'method':'dunning'})['duration'][:] = np.minimum(np.abs(ds.sel({'method':'dunning'})['demise'] - ds.sel({'method':'dunning'})['onset']),
                                                             365-np.abs(ds.sel({'method':'dunning'})['demise'] - ds.sel({'method':'dunning'})['onset']))

        for season_idx in np.arange(0,num_seasons):
            #season_belonging=np.zeros((365,ds.dims['allv']))
            season_belonging=np.ones((365,ds.dims['allv']))*np.nan

            # Now, fill with 1's the days of each season
            for loc in np.arange(0,ds_doy.dims['allv']):
                if not np.isnan(ds.isel(season=season_idx,allv=loc).sel({'method':'dunning'}).onset):
                    # Some seasons go across the new year, the indexing here gets complex.
                    # In this case, the demise day of year index is less than the onset 
                    # day of year index (i.e., like, "onset = 300", "demise = 10" or so)
                    if (ds.isel(season=season_idx,allv=loc).sel({'method':'dunning'}).demise < 
                        ds.isel(season=season_idx,allv=loc).sel({'method':'dunning'}).onset):
                        season_belonging[int(ds.isel(season=season_idx,allv=loc).sel({'method':'dunning'}).onset):,
                                         int(loc)]=True
                        season_belonging[0:int(ds.isel(season=season_idx,allv=loc).sel({'method':'dunning'}).demise),
                                         int(loc)]=True
                        #`breakpoint()
                    else:
                        season_belonging[int(ds.isel(season=season_idx,allv=loc).sel({'method':'dunning'}).onset):
                                          int(ds.isel(season=season_idx,allv=loc).sel({'method':'dunning'}).demise),
                                         int(loc)]=True

            # Calculate Integrated Amount
            ds.isel(season=season_idx).sel({'method':'dunning'})['integrated_amount'][:] = (season_belonging*
                     ds_doy.pr).sum('dayofyear')

            # Calculate Peak Timing
            #breakpoint()
            ds.isel(season=season_idx).sel({'method':'dunning'})['peak_timing'][:] = nan_argmax_xr(season_belonging*
                     ds_doy.pr,dim='dayofyear')+1

            # Calculate Peak Amount
            ds.isel(season=season_idx).sel({'method':'dunning'})['peak_amount'][:] = (season_belonging*
                     ds_doy.pr).max(axis=0)

        # Make sure the nans are nans
        for varn in ['integrated_amount','peak_timing','peak_amount']:
            #breakpoint()
            ds[varn] = (ds[varn].where(~np.isnan(ds.onset)))


        #### Unstack
        ds = ds.unstack()

        #### Save as temporary file if desired
        ds.attrs['SOURCE'] = 'seas_params_dunning()'
        ds.attrs['DESCRIPTION'] = 'seasonal parameters calculated using the methods of Dunning et al. 2017'
        if save_temp:
            if overwrite & os.path.exists(output_fn):
                os.remove(output_fn)
                print(output_fn+' removed to allow overwrite.')

            # Don't save the precip again though
            ds.drop('pr').drop('time').to_netcdf(output_fn)
            print(output_fn+' saved!')
        
    else:
        print(output_fn+' already exists; loaded.')
        ds = xr.open_dataset(output_fn)
    
        dims_tmp = [e for e in ds.seas_ratio.dims if e not in ('time','dayofyear','method','season','bnds','year')]
        bi_idxs = (ds.stack(allv=dims_tmp).seas_ratio<1).values.nonzero()[0]
    
    #### Return
    if diag_mode:
        return ds,bi_idxs,C,relmins,relmaxs
    else:
        return ds,bi_idxs    




def seas_params_dunning_byyear(ds,bi_idxs,
                               num_seasons=2,
                               window_width = [50,20],
                               save_temp=True,output_fn=None,
                               mod_name='',yr_str='',fn_suffix='',ignore_warnings=True,
                              overwrite=False):
    '''
    Calculate seasonal onsets/demises and associated seasonal statistics on each year of rainfall data
    
    '''
    
    
    if ignore_warnings:
        warnings.filterwarnings('ignore')
        
        
    # Make demise = onset + duration, to wrap over years
    ds['demise'] = ds['onset'] + ds['duration']

    #------- Make a new dataset with the yearly data ------
    ds_yearly = xr.Dataset({'lat': (['lat'],ds.unstack().lat.values),
                            'lon': (['lon'],ds.unstack().lon.values),
                         'year': (['year'],np.unique(ds.time.dt.year)[0:-1]),
                         'season':(['season'],np.arange(1,num_seasons+1)),
                         'method':(['method'],['dunning'])})
    # Gotta stack it after making the lat/lon; ds.allv doesn't transfer 
    # the full multi-index for some reason
    ds_yearly = ds_yearly.stack(allv=('lat','lon')) 

    stat_vars = ['onset','demise','duration','peak_timing','peak_amount','integrated_amount']
    for stat_var in stat_vars:
        if stat_var not in ds_yearly.variables.keys():
            ds_yearly = ds_yearly.assign({stat_var:(('allv','year','season','method'),
                                      np.zeros((ds_yearly.dims['allv'],
                                                ds_yearly.dims['year'],
                                                ds_yearly.dims['season'],
                                               ds_yearly.dims['method']))*np.nan)})

    # Set the amount of days to wrap around the cumulative
    # sum time series to the next year; to be the latest demise 
    # plus the number of days looking around the climatological
    # onset/demise for the single year one
    wrap_around_days = int(ds.demise.max().values-365+window_width[0])
            
    # Hm. So I think it's possible to calculate the 
    # cumulative anomaly year-by-year for every location 
    # at the same time, vectorized, but then the identification
    # of the minimum might have to be done by for loop 
    # from year 2 to n_year-1 (because some seasons span 
    # calendar years)
    # Honestly, `A` and the original max/mins should be 
    # calculable for all years simultaneously by 
    # reshaping pr as year x day x allv
    print('starting year-by-year seasonal identification...')
    for yr in tqdm(np.arange(0,ds_yearly.dims['year'])):
        # Calculate A as above
        A = (ds.isel(time=slice(yr*365,(yr+1)*365+wrap_around_days)).pr.cumsum('time') - 
             ds.isel(time=slice(yr*365,(yr+1)*365+wrap_around_days)).pr.mean('time')*
              xr.DataArray(np.arange(1,366+wrap_around_days),
                           dims='time',coords=[ds.isel(time=slice(yr*365,(yr+1)*365+wrap_around_days)).time.values]))

        ## SINGLE-PEAKED SEASONS
        # First find the minimum/onset for all of them. 
        # Then isolate those where the minimum doesn't 
        # fall between 50 days from the climatological 
        # minimum (and aren't double-peaked). This is
        # a lot faster than doing the for loop. Probably
        # even faster if I can reshape it above. 
        # Do the same with maximum/demise.
        # Doing it only for single-peaked to save some micro-
        # seconds; the missed values will be overwritten
        # in the double-peaked processing anyways
        #breakpoint()
        mins_tmp = nan_argmin_xr(A,dim='time')
        bad_min_idxs = ((np.abs(mins_tmp-ds.onset.sel({'method':'dunning','season':1}))>window_width[0]) & 
                        (ds.seas_ratio>=1)).values.nonzero()[0]
        for idx in bad_min_idxs:
            search_idxs = ds.onset.sel({'method':'dunning','season':1}).isel(allv=idx)
            search_idxs = np.array([int(search_idxs.values-50),int(search_idxs.values+50)])
            # But the question is, if the onset is 
            # within 50 days of the end/start of the 
            # year, does the search continue in the same
            # year or in the next/previous one? 
            search_idxs[search_idxs<0] = 0
            search_idxs[search_idxs>(364+wrap_around_days)] = (364+wrap_around_days)
            # Changed to line below now that nan_argm**_xr is more robust
            mins_tmp[idx] = nan_argmin_xr(A.isel(allv=idx,time=slice(*search_idxs)),dim='time').values+search_idxs[0]

        # Now for maxs
        maxs_tmp = nan_argmax_xr(A,dim='time')
        bad_max_idxs = ((np.abs(mins_tmp-ds.demise.sel({'method':'dunning','season':1}))>window_width[0]) & 
                        (ds.seas_ratio>=1)).values.nonzero()[0]
        for idx in bad_max_idxs:
            search_idxs = ds.demise.sel({'method':'dunning','season':1}).isel(allv=idx)
            search_idxs = np.array([int(search_idxs.values-50),int(search_idxs.values+50)])
            # But the question is, if the onset is 
            # within 50 days of the end/start of the 
            # year, does the search continue in the same
            # year or in the next/previous one? 
            search_idxs[search_idxs<0] = 0
            search_idxs[search_idxs>(364+wrap_around_days)] = (364+wrap_around_days)
            # Changed to line below now that nan_argm**_xr is more robust
            maxs_tmp[idx] = nan_argmax_xr(A.isel(allv=idx,time=slice(*search_idxs)),dim='time').values+search_idxs[0]


        ## DOUBLE-PEAKED SEASONS (seas_ratio<1)
        for season_idx in np.arange(0,num_seasons):
            ## ONSET
            # I could get the min/max for the first season 
            # across all points, and calculate the maxs in that
            # segment, and then check which don't work here
            # Would probably avoid much of a for loop
            mins_tmp[bi_idxs] = nan_argmin_xr(A.isel(allv=bi_idxs,
                                            time=slice(int(ds.isel(allv=bi_idxs,season=season_idx).sel({'method':'dunning'}).onset.min().values),
                                                       int(ds.isel(allv=bi_idxs,season=season_idx).sel({'method':'dunning'}).onset.max().values))),
                                     dim='time')+int(ds.isel(allv=bi_idxs,season=season_idx).sel({'method':'dunning'}).onset.min().values)
            bad_min_idxs = ((np.abs(mins_tmp[bi_idxs]-ds.onset.
                                                          isel(allv=bi_idxs,season=season_idx).
                                                          sel({'method':'dunning'}))>window_width[1])).values.nonzero()[0]
            for idx in bad_min_idxs:
                # All of these have to have two .isel calls
                # since we're calculating indices relative to
                # the already subset indices of pixels with 
                # double rainy seasons
                search_idxs = ds.onset.isel(allv=bi_idxs,season=season_idx).sel({'method':'dunning'}).isel(allv=idx)
                search_idxs = np.array([int(search_idxs.values-20),int(search_idxs.values+20)])
                # But the question is, if the onset is 
                # within 50 days of the end/start of the 
                # year, does the search continue in the same
                # year or in the next/previous one? 
                search_idxs[search_idxs<0] = 0
                search_idxs[search_idxs>(364+wrap_around_days)] = (364+wrap_around_days)
                # +1 because it's the day *after* we're looking
                # for, and [bi_idxs[idxs]] because we're already
                # calculating indices relative to the double-peaked
                # subset (see above) [NO BI_IDXS[IDX] IS NOT NEEDED BECAUSE MINS_TMP ALREADY IS SUBSET]
                # Changed to line below now that nan_argm**_xr is more robust
                mins_tmp[bi_idxs[idx]] = nan_argmin_xr(A.isel(allv=bi_idxs).
                                                         isel(allv=idx,time=slice(*search_idxs)),
                                                       dim='time').values+search_idxs[0]+1

            ## DEMISE
            # I could get the min/max for the first season 
            # across all points, and calculate the maxs in that
            # segment, and then check which don't work here
            # Would probably avoid much of a for loop
            maxs_tmp[bi_idxs] = nan_argmax_xr(A.isel(allv=bi_idxs,
                                            time=slice(int(ds.isel(allv=bi_idxs,season=season_idx).sel({'method':'dunning'}).onset.min().values),
                                                       int(ds.isel(allv=bi_idxs,season=season_idx).sel({'method':'dunning'}).onset.max().values))),
                                     dim='time')+int(ds.isel(allv=bi_idxs,season=season_idx).sel({'method':'dunning'}).onset.min().values)
            bad_max_idxs = ((np.abs(maxs_tmp[bi_idxs]-ds.demise.
                                                          isel(allv=bi_idxs,season=season_idx).
                                                          sel({'method':'dunning'}))>window_width[1])).values.nonzero()[0]
            for idx in bad_max_idxs:
                # All of these have to have two .isel calls
                # since we're calculating indices relative to
                # the already subset indices of pixels with 
                # double rainy seasons
                search_idxs = ds.demise.isel(allv=bi_idxs,season=season_idx).sel({'method':'dunning'}).isel(allv=idx)
                search_idxs = np.array([int(search_idxs.values-20),int(search_idxs.values+20)])
                # But the question is, if the onset is 
                # within 50 days of the end/start of the 
                # year, does the search continue in the same
                # year or in the next/previous one? 
                search_idxs[search_idxs<0] = 0
                search_idxs[search_idxs>(364+wrap_around_days)] = (364+wrap_around_days)
                # +1 because it's the day *after* we're looking
                # for, and [bi_idxs[idxs]] because we're already
                # calculating indices relative to the double-peaked
                # subset (see above)
                # Changed to line below now that nan_argm**_xr is more robust
                maxs_tmp[bi_idxs[idx]] = nan_argmax_xr(A.isel(allv=bi_idxs).
                                                         isel(allv=idx,time=slice(*search_idxs)),
                                                       dim='time').values+search_idxs[0]+1


            if season_idx == 0:
                ds_yearly.isel(season=season_idx,year=yr).sel({'method':'dunning'})['onset'][:] = mins_tmp
                ds_yearly.isel(season=season_idx,year=yr).sel({'method':'dunning'})['demise'][:] = maxs_tmp
            else:
                ds_yearly.isel(season=season_idx,year=yr).sel({'method':'dunning'})['onset'][bi_idxs] = mins_tmp[bi_idxs]
                ds_yearly.isel(season=season_idx,year=yr).sel({'method':'dunning'})['demise'][bi_idxs] = maxs_tmp[bi_idxs]
                
            ## CALCULATE OTHER STATISTICS
            # Get a zeros the size of one year's worth of data.
            # Not *np.nan because nan_argmax_xr only deals with nans 
            # in the loc dimension, not the time dimension. Maybe 
            # should put in a warning into the nan_arg*_xr functions... 
            season_belonging=np.zeros((365+wrap_around_days,ds_yearly.dims['allv']))*np.nan

            # Now, fill with 1's the days of each season
            for loc in np.arange(0,ds_yearly.dims['allv']):
                if not np.isnan(ds_yearly.isel(season=season_idx,year=yr,allv=loc).onset):
                    # -1 on onset and +- 0 on the demise to fit with python indexing, 
                    # since Day 1 is "1" in onset
                    season_belonging[int(ds_yearly.isel(season=season_idx,year=yr,allv=loc).onset-1):
                                     int(ds_yearly.isel(season=season_idx,year=yr,allv=loc).demise),
                                 int(loc)]=True

            # Integrated Amount
            ds_yearly.isel(season=season_idx,year=yr).sel({'method':'dunning'})['integrated_amount'][:] = (season_belonging*
                     ds.isel(time=slice(yr*365,(yr+1)*365+wrap_around_days)).pr).sum('time')

            # Peak Timing
            ds_yearly.isel(season=season_idx,year=yr).sel({'method':'dunning'})['peak_timing'][:] = nan_argmax_xr(season_belonging*
                     ds.isel(time=slice(yr*365,(yr+1)*365+wrap_around_days)).pr,dim='time')+1

            # Peak Amount
            ds_yearly.isel(season=season_idx,year=yr).sel({'method':'dunning'})['peak_amount'][:] = (season_belonging*
                     ds.isel(time=slice(yr*365,(yr+1)*365+wrap_around_days)).pr).max(axis=0)

    #### DURATION
    # Calculate duration (with minimum of both directions)
    ds_yearly.sel({'method':'dunning'})['duration'][:] = np.minimum(np.abs(ds_yearly.sel({'method':'dunning'})['demise'] - ds_yearly.sel({'method':'dunning'})['onset']),
                                                         365-np.abs(ds_yearly.sel({'method':'dunning'})['demise'] - ds_yearly.sel({'method':'dunning'})['onset']))
    
    # Make sure the nans are nans
    for varn in ['integrated_amount','peak_timing','peak_amount']:
        ds_yearly[varn] = (ds_yearly[varn].where(~np.isnan(ds_yearly.onset)))
    
    
    #### Unstack
    ds_yearly = ds_yearly.unstack()
    
    #### Save as temporary file if desired
    ds_yearly.attrs['SOURCE'] = 'seas_params_dunning_byyear()'
    ds_yearly.attrs['DESCRIPTION'] = 'seasonal parameters for each year calculated using the methods of Dunning et al. 2017'
    
    if save_temp:
        if output_fn is None:
            output_fn = dir_list['proc']+mod_name+'/'+'pr_ann_'+mod_name+'_seasstats_dunning'+yr_str+fn_suffix+'.nc'
        if overwrite & os.path.exists(output_fn):
            os.remove(output_fn)
            print(output_fn+' removed to allow overwrite.')
        
        ds_yearly.to_netcdf(output_fn)
        print(output_fn+' saved!')
        
    #### Return
    return ds_yearly



def process_seasonal_stats_dunning(mod,mod_fns,dir_list,
                                   subset_params,
                                   proc_year = True, 
                                   overwrite=False,
                                   override_30day_convert=False,
                                   alt_doy_for_ann=None):
    '''
    Calculate both climatological and year-by-year season statistics
    
    '''

    print('processing model '+mod+'!')
    
    dir_list = get_params()
    
    if not os.path.exists(dir_list['proc']+mod+'/'):
            os.mkdir(dir_list['proc']+mod+'/')
            warnings.warn('Directory '+dir_list['proc']+mod+'/ created')
    
    # Load file
    ds_tmp = xr.open_dataset(dir_list['raw']+mod+'/'+mod_fns[mod])

    # Sort by time, if not sorted (this happened with
    # a model; keeping a warning, cuz this seems weird)
    if (ds_tmp.time.values != np.sort(ds_tmp.time)).any():
        warnings.warn('Model '+mod+' has an unsorted time dimension.')
        ds_tmp = ds_tmp.sortby('time')
        
        
    if ds_tmp.time.min().dt.year>pd.to_datetime(subset_params['time'][0]).year:
        print('')
        print('Subset minimum, '+subset_params['time'][0]+
              ', is before the first date ('+str(ds_tmp.time.min().values)[:10]+
              ') of the file of model '+mod+'. Code cannot yet deal with that, '+
              'so processing for '+mod+' is skipped.')
        print('')
        # NOTE THIS JUST CHECKS FOR YEAR< NOT THE WHOLE DATE - GOT A WEIRD ERROR AND I'M TIRED SO I'M KEEPING IT THIS FOR NOW
    else:
    
        # Set filename characteristics
        yr_str_avg = '_'+'-'.join([re.sub('-','',subset_params['time'][x]) for x in np.arange(0,2)])
        yr_str_yr = '_'+'-'.join([re.sub('-','',
                               re.sub(subset_params['time'][0][0:4],
                                      str(int(float(subset_params['time'][0][0:4]))), #+1
                                      subset_params['time'][0])),
                                 re.sub('-','',re.sub(subset_params['time'][1][0:4],
                                      str(int(float(subset_params['time'][1][0:4]))-1),
                                      subset_params['time'][1]))])

        output_fn_avg = dir_list['proc']+mod+'/pr_doyavg_'+mod+'_'+subset_params['experiment_id']+'_seasstats_dunning'+yr_str_avg+subset_params['fn_suffix']+'.nc'
        output_fn_yr = dir_list['proc']+mod+'/pr_ann_'+mod+'_'+subset_params['experiment_id']+'_seasstats_dunning'+yr_str_yr+subset_params['fn_suffix']+'.nc'

        if overwrite | ((not overwrite) & (not os.path.exists(output_fn_avg))) | ((not overwrite) & (not os.path.exists(output_fn_yr))):

            # Subset temporally (switching 31 to 30 for 360-day calendars)
            if (not override_30day_convert) and (ds_tmp.time.max().dt.day==30):
                ds_tmp = ds_tmp.sel(time=slice(subset_params['time'][0],re.sub('-31','-30',subset_params['time'][1])))
                # eh just end this
                # Throw in a warning, too, why not
                #warnings.warn(mod+' has a 360-day calendar; daily values were interpolated to a 365-day calendar')
                warnings.warn("Interpolating from 360 to 365 day calendar")
                
                og_idxs = np.arange(0,365*len(np.unique(ds_tmp.time.dt.year)),365/360)
                og_idxs[-1] = np.round(og_idxs[-1])
                ip = spi.interp1d(og_idxs,ds_tmp.pr,axis=0)
                
                dates_out = pd.date_range(str(ds_tmp.time[0].dt.year.values)+'-01-01',
                                          str(ds_tmp.time[-1].dt.year.values)+'-12-31',
                                          freq='D')

                ds_tmp = xr.Dataset({'pr':(('time','lat','lon',),ip(np.arange(0,365*len(np.unique(ds_tmp.time.dt.year)))))},
                                        coords={'lat':(['lat'],ds_tmp.lat.values),
                                                  'lon':(['lon'],ds_tmp.lon.values),
                                                  'time':(['time'],pd.DatetimeIndex(data=[t for t in dates_out if not ((t.month==2) & (t.day==29))]))})

                ds_tmp.attrs['interpolated'] = 'INTERPOLATED FROM 360-DAY CALENDAR; ARTEFACTS MAY HAVE OCCURED'
                
                
            else:
                ds_tmp = ds_tmp.sel(time=slice(*subset_params['time']))
                
            # Subset spatially
            if ds_tmp.lat[0] < ds_tmp.lat[1]:
                ds_tmp = ds_tmp.sel(lat=slice(*subset_params['lat']))
            else:
                ds_tmp = ds_tmp.sel(lat=slice(*subset_params['lat'][::-1]))
            ds_tmp = ds_tmp.sel(lon=slice(*subset_params['lon']))
                
            # Load to make processing work (otherwise the occasional error shows up for particularly large 
            # files in seas_params_dunning: 
            # IndexError: The indexing operation you are attempting to perform is not 
            # valid on netCDF4.Variable object. Try loading your data into memory first 
            # by calling .load().
            # This is an attempt to bypass it. I'm really not sure why it showed up / 
            # what deeper madness caused it. It showed up with MPI-ESM-1-2-HAM ssp370 processing. 
            # If this doesn't work, then I don't know what to do. Skip it, burn it, whatever. 
            try:
                ds_tmp = ds_tmp.load()
            except:
                breakpoint()
                warnings.warn("Major issue with processing. Can't load data. Something deeper and eldritch is going on here.")
                
            # Save original precip data
            da_pr = ds_tmp.pr
        
            # Calculate seasonal statistics of average year
            print('processing average seasonal statistics')
            ds_tmp,bi_idxs = seas_params_dunning(ds_tmp,subset_params,dir_list,mod_name=mod,yr_str=yr_str_avg.strip('_'),fn_suffix=subset_params['fn_suffix'],
                                                        overwrite=overwrite,output_fn = output_fn_avg)
            if proc_year:
                print('processing year-by-year seasonal statistics')

                # Under certain (?) conditions, the above code 
                # can accidentally add a 'method' dimension to 
                # the 'pr' variable (which should be the same
                # for all, since it's not the seasonal stats
                # but the underlying precipitation). This 
                # takes care of that. (NB: uh, I don't think 
                # seas_params_dunning still outputs pr?) 
                if ('pr' in ds_tmp) and ('method' in ds_tmp.pr.dims):
                    ds_tmp['pr'] = ds_tmp.pr.isel(method=0)

                # Calculate seasonal statistics by year
                # The overwrite thing is an issue because 
                # the filename can be changed within the above
                # code because of nan issues, which can lead 
                # to false filenames here and unnecessary processing
                if overwrite | (not os.path.exists(output_fn_yr)):
                    if alt_doy_for_ann is not None:
                        if type(alt_doy_for_ann) is str:
                            ds_tmp = xr.open_dataset(alt_doy_for_ann)
                        else:
                            ds_tmp = alt_doy_for_ann


                    # Re-add precip variable that's removed by processing 
                    # above
                    if 'pr' not in ds_tmp:
                        ds_tmp['pr'] = da_pr

                    dims_tmp = [e for e in ds_tmp.pr.dims if e not in ('time','dayofyear','method','season','bnds','year')]

                    ds_tmp = ds_tmp.stack(allv=dims_tmp)
                    if alt_doy_for_ann is not None:
                        bi_idxs = (ds_tmp.seas_ratio<1).values.nonzero()[0]

                    ds_tmp_year = seas_params_dunning_byyear(ds_tmp,bi_idxs,#ds.stack(allv=('lat','lon')),
                                                             mod_name=mod,yr_str=yr_str_yr,fn_suffix=subset_params['fn_suffix'],
                                                             overwrite=overwrite,output_fn = output_fn_yr)

                    ds_tmp = ds_tmp.unstack()

            # Return
            if proc_year:
                return ds_tmp,ds_tmp_year
            else:
                return ds_tmp

        else:
            print('files: ')
            print(output_fn_yr)
            print(output_fn_avg)
            print('already exist; skipped.')

        print('')