#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 12:05:26 2021

@author: fritzpere
"""
import numpy as np

def feature_vector_per_band(avg_life, std_life, entropy, pooling, avg_midlife, std_midlife):
    
    band_dic={-1: 'no_filter', 0:'alpha',1:'betta',2:'gamma'}
    feat_vect_size=8
    feat_vect={}
    labels=[]
    for i_band in range(-1,3):
        feat_vect[band_dic[i_band]]={}
        for i_state in range(3):
            trials=len(avg_life[0][i_band][i_state])
            feat_vect[band_dic[i_band]]['dim0']=np.zeros((trials,feat_vect_size))
            feat_vect[band_dic[i_band]]['dim1']=np.zeros((trials,feat_vect_size))
            feat_vect[band_dic[i_band]]['dim0dim1']=np.zeros((trials,feat_vect_size*2))
            
            feat_vect[band_dic[i_band]]['dim0pooling']=np.zeros((trials,10))
            feat_vect[band_dic[i_band]]['dim1pooling']=np.zeros((trials,10))
            feat_vect[band_dic[i_band]]['dim0dim1pooling']=np.zeros((trials,10*2))
            for k in range(trials):
                feat_vect[band_dic[i_band]]['dim0'][k]=np.concatenate((np.array([avg_life[0][i_band][i_state][k],std_life[0][i_band][i_state][k],avg_midlife[0][i_band][i_state][k],std_midlife[0][i_band][i_state][k],entropy[0][i_band][i_state][k]]),pooling[0][i_band][i_state][k][:3]),axis=0)
                feat_vect[band_dic[i_band]]['dim1'][k]=np.concatenate((np.array([avg_life[1][i_band][i_state][k],std_life[1][i_band][i_state][k],avg_midlife[1][i_band][i_state][k],std_midlife[1][i_band][i_state][k],entropy[1][i_band][i_state][k]]),pooling[1][i_band][i_state][k][:3]),axis=0)
                feat_vect[band_dic[i_band]]['dim0dim1'][k]=np.concatenate((feat_vect[band_dic[i_band]]['dim0'][k],feat_vect[band_dic[i_band]]['dim1'][k]),axis=0)
                
                feat_vect[band_dic[i_band]]['dim0pooling'][k]=pooling[0][i_band][i_state][k]
                feat_vect[band_dic[i_band]]['dim1pooling'][k]=pooling[1][i_band][i_state][k]
                feat_vect[band_dic[i_band]]['dim0dim1pooling'][k]=np.concatenate((feat_vect[band_dic[i_band]]['dim0pooling'][k],feat_vect[band_dic[i_band]]['dim1pooling'][k]),axis=0)
                
                if i_band==-1: #to only do it once
                    labels.append(i_state)
    return feat_vect,labels

#def get_accuracies_per_band(feature_vector_dic,labels):
    #todo 
    


                
    