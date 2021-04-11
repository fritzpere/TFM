#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:03:43 2021

@author: fritz
"""

import numpy as np
import scipy.signal as spsg
#from scipy.spatial import distance_matrix #manera més óptima

def clean(data):
    """
     discard silent channels
    :param data: data to clean (N,T,n_trials,n_blocks)
    :return: cleaned data
    """
    invalid_ch_s0 = np.logical_or(np.abs(data[:, :, 0, 0]).max(axis=1) == 0,np.isnan(data[:, 0, 0, 0]))
    invalid_ch_s1 = np.logical_or(np.abs(data[:, :, 0, 1]).max(axis=1) == 0,np.isnan(data[:, 0, 0, 1]))
    invalid_ch = np.logical_or(invalid_ch_s0, invalid_ch_s1)
    #invalid_ch = np.logical_or(np.abs(data[:,:,0,:]).max(axis=1)==0, np.isnan(data[:,0,0,:]))
    #valid_ch_per_block = np.logical_not(invalid_ch)
    #valid_ch=valid_ch_per_block.sum(axis=1)==data.shape[-1]
    valid_ch = np.logical_not(invalid_ch)
    cleaned_data= data[valid_ch,:,:,:]
    N = valid_ch.sum()
    print('there are',N,'clean channels' )
    
    return cleaned_data,N

def get_ts(data,n_block,n_trials,T,N):
    """
   get time series for each block
    :param data: data to group in blocks
    :param n_block: number of blocks of trials
    :param n_trials: number of trials per block
    :param T: number of miliseconds recorded per trial
    param N: number of channels
    :return: TimeSeries
    """
    # get time series for each block
    ts = np.zeros([3,n_trials*4,T,N])
    ts_dic={}
    for i_block in range(n_block):
        i=i_block%2
        i=n_trials*i
        j=i_block//6
        j=2*n_trials*j
        i_motiv=i_block//2%3
        for i_trial in range(n_trials):
            # swap axes for time and channelsi

            ts[i_motiv,i_trial+i+j,:,:] = data[:,:,i_trial,i_block].T 
    clean_trials= lambda x: np.logical_and(np.isnan(ts[x,:,0,:]).sum(axis=1)==0,ts[x,:,0,:].sum(axis=1)!=0)
    ts_dic[0]=ts[0,clean_trials(0),:,:]
    ts_dic[1]=ts[1,clean_trials(1),:,:]
    ts_dic[2]=ts[2,clean_trials(2),:,:]
    return ts_dic

def freq_filter(ts,n_motiv,n_trials,T,N): #dont need all variables if using dict
    """
   filters the frequency into 'alpha', 'betta' and 'gamma' waves.
    :param ts: TimeSeries obtained from the EEGs
    :param n_motiv: number of motivational states
    :param n_trials: number of trials per block
    :param T: number of miliseconds recorded per trial
    param N: number of channels
    :return: Filtered TimeSeries
    """
    n_bands=3
    freq_bands = ['alpha','beta','gamma']
    filtered_ts_dic = {}#np.zeros([n_bands,n_motiv,n_trials,T,N])
    for i_band in range(n_bands):

        # select band
        freq_band = freq_bands[i_band]

        # band-pass filtering (alpha, beta, gamma)
        n_order = 3
        sampling_freq = 500. # sampling rate

        if freq_band=='alpha':
            low_f = 8./sampling_freq
            high_f = 15./sampling_freq
        elif freq_band=='beta':   
            # beta
            low_f = 15./sampling_freq
            high_f = 32./sampling_freq
        elif freq_band=='gamma':
            # gamma
            low_f = 32./sampling_freq
            high_f = 80./sampling_freq
        else:
            raise NameError('unknown filter')

        # apply filter ts[n_motiv,n_trials,T,N]
        b,a = spsg.iirfilter(n_order, [low_f,high_f], btype='bandpass', ftype='butter')
        #filtered_ts[i_band,:,:,:,:] = spsg.filtfilt(b, a, ts, axis=2)
        filtered_ts_dic[i_band,0]=spsg.filtfilt(b, a, ts[0], axis=1)
        filtered_ts_dic[i_band,1]=spsg.filtfilt(b, a, ts[1], axis=1)
        filtered_ts_dic[i_band,2]=spsg.filtfilt(b, a, ts[2], axis=1)
    return filtered_ts_dic
