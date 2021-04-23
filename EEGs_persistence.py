#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:07:20 2021

@author: fritz
"""
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from matplotlib import pyplot
import gudhi as gd
from sklearn.preprocessing import normalize
from preprocess_data import *
from topological_descriptors import *
import os
#plt.style.use(['seaborn-bright'])

def persistency_per_band_and_state(tensor,measure,n_bands=3):
    """
    Computes Persistent Homology for each band of
    frequency and each motivational state
    :param tensor: tensor with form band x motiv_state x trial x channel
    :param measure: string that indicates if one uses distance matrixes
    :param n_bands=3: number of bands of the tensor 
    :return: dictionary with key=(band,state) and value=persistence
    """
    persistence_dic={}
    trials=tensor.shape[0]
    for band in range(-1,n_bands):
        persistence_dic[band]={}
        for i in range (3):#motivational
            persistence_dic[band][i]=[]
            for k in range(trials):
                band_tensor = np.abs(tensor[band][i][k,:,:].T)
            #print('nans?',np.isnan(band_tensor[i]).any())
            #print('cloud shape:',band_tensor.shape)
            
                if measure=='intensities':
                    matrix=distance_matrix(band_tensor,band_tensor)
                   
                else:
                    #TODO
                    '''points=band_tensor.copy()
                    normalized_p=normalize(points-np.mean(points,axis=0),axis=1)
                    matrix= normalized_p @ normalized_p.T
                    matrix=1-matrix'''
                #max_edge=np.max(matrix)
                Rips_complex_sample = gd.RipsComplex(distance_matrix=matrix)#,max_edge_length=max_edge)
                Rips_simplex_tree_sample = Rips_complex_sample.create_simplex_tree(max_dimension=2)
                persistence = Rips_simplex_tree_sample.persistence()
                persistence_dic[band][i].append( persistence) #dictionary with key=(band,state) and value=persistence
    return persistence_dic 

def compute_persistence_from_EEG(data,measure='intensities',reduc=5,subj_dir=None,space=None,save=True,):
    """
    Pipeline that beggins with raw_data and preprocess it
    to later compute Persistent Homology
    :param data: raw_data from electrode/font_space
    :param measure: string that indicates if one uses distance matrixes
    :param subj_dir: path to the directory of the subject
    :param space: string that indicates if the data comes 
    from electrode space or font space
    :param save=True: boolean that indicates if the persistence
    should be saved in a txt file
    :return: dictionary with key=(band,state) and value=persistence
    """
    if measure!='intensities' and measure !='correlation':
        print('measure paramater is incorrect')
        measure='intensities'
    N,T,n_trials,n_blocks=data.shape
    #n_blocks=12
    cleaned_data,N=clean(data)
    
    n_motiv=3
    n_band=3
    ts_dic=get_ts(cleaned_data,n_blocks,n_trials,T,N)
    #n_trials=n_trials*4
    #ts.shape=  (3, 432, 1200, 48) motivational state, trial, time, channel
    filtered_ts_dic=freq_filter(ts_dic,n_motiv,n_trials,T,N)
    
    filtered_ts_dic[-1,0]=ts_dic[0]
    filtered_ts_dic[-1,1]=ts_dic[1]
    filtered_ts_dic[-1,2]=ts_dic[2]
    

    persistence_dictionary=persistency_per_band_and_state(filtered_ts_dic,measure)
        
    return persistence_dictionary #dictionary with key=(band,state) and value=persistence


    