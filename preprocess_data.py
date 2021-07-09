#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:03:43 2021

@author: fritz
"""

import numpy as np
import scipy.signal as spsg
import sklearn.pipeline as skppl



class Preprocessor:
    def __init__(self,data):
        self.data = data

        self.pipeline  = skppl.Pipeline
        self.N,self.T,self.n_trials,self.n_blocks=data.shape
        self.n_motiv=3
        self.n_bands=3
        self.ts_dic={}
        self.trials_total=0
        
    def clean_data(self):
        """
         discard silent channels
        :param data: data to clean (N,T,n_trials,n_blocks)
        :return: cleaned data
        """
        invalid_ch_s0 = np.logical_or(np.abs(self.data[:, :, 0, 0]).max(axis=1) == 0,np.isnan(self.data[:, 0, 0, 0]))
        invalid_ch_s1 = np.logical_or(np.abs(self.data[:, :, 0, 1]).max(axis=1) == 0,np.isnan(self.data[:, 0, 0, 1]))
        invalid_ch = np.logical_or(invalid_ch_s0, invalid_ch_s1)
        #invalid_ch = np.logical_or(np.abs(data[:,:,0,:]).max(axis=1)==0, np.isnan(data[:,0,0,:]))
        #valid_ch_per_block = np.logical_not(invalid_ch)
        #valid_ch=valid_ch_per_block.sum(axis=1)==data.shape[-1]
        valid_ch = np.logical_not(invalid_ch)
        cleaned_data= self.data[valid_ch,:,:,:]
        self.N = valid_ch.sum()
        print('there are',self.N,'clean channels' )
        
        self.data=cleaned_data
        return
    
    def get_ts(self):
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
        temp=np.zeros([3,self.n_trials*4])
        ts = np.zeros([3,self.n_trials*4,self.T,self.N])
        for i_block in range(self.n_blocks):
            i=i_block%2
            i=self.n_trials*i
            j=i_block//6
            j=2*self.n_trials*j
            i_motiv=(i_block//2)%3
            for i_trial in range(self.n_trials):
                # swap axes for time and channelsi
                ts[i_motiv,i_trial+i+j,:,:] = self.data[:,:,i_trial,i_block].T
                temp[i_motiv,i_trial+i+j]=i_block
        clean_trials= lambda x: np.logical_and(np.isnan(ts[x,:,0,:]).sum(axis=1)==0,ts[x,:,0,:].sum(axis=1)!=0)
        self.ts_dic[0]=ts[0,clean_trials(0),:,:]
        self.ts_dic[1]=ts[1,clean_trials(1),:,:]
        self.ts_dic[2]=ts[2,clean_trials(2),:,:]
        
        self.tr2bl_ol=[]
        self.tr2bl_ol.extend(temp[0,clean_trials(0)])
        self.tr2bl_ol.extend(temp[1,clean_trials(1)])
        self.tr2bl_ol.extend(temp[2,clean_trials(2)])
        self.tr2bl_ol=np.array(self.tr2bl_ol)
        return
    
    def freq_filter(self): #dont need all variables if using dict
        """
        filters the frequency into 'alpha', 'betta' and 'gamma' waves.
        :param ts: TimeSeries obtained from the EEGs
        :param n_motiv: number of motivational states
        :param n_trials: number of trials per block
        :param T: number of miliseconds recorded per trial
        param N: number of channels
        :return: Filtered TimeSeries
        """
        freq_bands = ['alpha','beta','gamma']
        filtered_ts_dic = {}#np.zeros([n_bands,n_motiv,n_trials,T,N])
        for i_band in range(self.n_bands):
    
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
            filtered_ts_dic[i_band]={}
            filtered_ts_dic[i_band][0]=spsg.filtfilt(b, a, self.ts_dic[0], axis=1)
            filtered_ts_dic[i_band][1]=spsg.filtfilt(b, a, self.ts_dic[1], axis=1)
            filtered_ts_dic[i_band][2]=spsg.filtfilt(b, a, self.ts_dic[2], axis=1)
            
        filtered_ts_dic[-1]={}
        filtered_ts_dic[-1][0]=self.ts_dic[0]
        filtered_ts_dic[-1][1]=self.ts_dic[1]
        filtered_ts_dic[-1][2]=self.ts_dic[2]
        self.ts_dic=filtered_ts_dic
        self.n_bands=4
        return


    def get_filtered_ts_dic(self):
        if not bool(self.ts_dic):
            self.clean_data()
            self.get_ts()       
            self.freq_filter()
        return self.ts_dic
    
    def get_trials_and_labels(self):
        if not bool(self.ts_dic):
            self.clean_data()
            self.get_ts()       
            self.freq_filter()
            
        
        trials=np.zeros(3,dtype='int')
        
        for i_state in range(3):
            trials[i_state]=self.ts_dic[-1][i_state].shape[0]
            self.trials_total+=trials[i_state]
    
    
        ts_band=np.zeros((self.trials_total,4,self.T,self.N))
        cum_trials=np.concatenate((np.zeros(1,dtype='int'),trials.cumsum(dtype='int')),axis=0)
        for i_band in range(-1,3):
            for i_state in range(3):
                for k in range(trials[i_state]):
                    
                    ts_band[cum_trials[i_state]+k][i_band]=self.ts_dic[i_band][i_state][k]
                    
        labels=np.concatenate((np.zeros(trials[0]),np.ones(trials[1]),np.ones(trials[2])*2),axis=0)         
        #self.tr2bl_ol=self.tr2bl_ol.reshape(-1)
        return ts_band,labels

    def reject_outliers(self,data, labels,m=1):
        norms=np.linalg.norm(data,axis=1)
        no_outliers=abs(norms - np.mean(norms,axis=0)) < m * np.std(norms)
        self.tr2bl=self.tr2bl_ol[no_outliers]
        return data[no_outliers],labels[no_outliers]