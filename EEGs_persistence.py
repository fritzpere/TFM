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
import os

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
    for band in range(n_bands):
        band_tensor = np.abs(tensor[band,:,:,:])
        for i in range (3): 
            #print('nans?',np.isnan(band_tensor[i]).any())
            if measure=='distance':
                matrix=distance_matrix(band_tensor[i],band_tensor[i])
               
            else:
                points=band_tensor[i].copy()
                normalized_p=normalize(points-np.mean(points,axis=0),axis=1)
                matrix= normalized_p @ normalized_p.T
                matrix=1-matrix
            max_edge=np.max(matrix)
            Rips_complex_sample = gd.RipsComplex(distance_matrix=matrix,max_edge_length=max_edge)
            Rips_simplex_tree_sample = Rips_complex_sample.create_simplex_tree(max_dimension=2)
            persistence = Rips_simplex_tree_sample.persistence()
            persistence_dic[band,i]= persistence #dictionary with key=(band,state) and value=persistence
    return persistence_dic 

def compute_persistence_from_EEG(data,measure='distance',subj_dir=None,space=None,save=True,):
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
    if measure!='distance' and measure !='correlation':
        print('measure paramater is incorrect')
        measure='distance'
    N,T,n_trials,n_blocks=data.shape
    #n_blocks=12
    cleaned_data,N=clean(data)
    n_motiv=3
    ts=get_ts(cleaned_data,n_blocks,n_trials,T,N)
    n_trials=n_trials*4
    #ts.shape=  (3, 432, 1200, 48) motivational state, trial, time, channel
    filtered_ts=freq_filter(ts,n_motiv,n_trials,T,N)
    vect_features=np.zeros([3,n_motiv,n_trials,N])
    for i_band in range(3):
        vect_features[i_band] = np.abs(filtered_ts[:,i_band,:,:,:]).mean(axis=2) #Mean of time
    #filtered_ts.shape (3, 3, 432,1200 40) band,state, trial, time, channel
    #print(vect_features.shape)
    persistence_dictionary=persistency_per_band_and_state(vect_features,measure)
    if save:
        if not os.path.exists(subj_dir+space+'/'+measure):
            print("create directory:",subj_dir+space+'/'+measure)
            os.makedirs(subj_dir+space+'/'+measure)
        for i in range(3):
            for j in range(3):
                f = open(subj_dir+'/'+space+'/'+measure+'/'+str(i)+str(j)+'persistence.txt', "w")
                for persistence in persistence_dictionary[(i,j)] :
                    f.write(''.join(map(str,persistence))+'\n')
                f.close()
    return persistence_dictionary #dictionary with key=(band,state) and value=persistence

def plot_persistence(persistence_dic,subj_dir,intervals=1000,repre='diagrams',space='',measure='',save=False):
    """
    Plots Persistence diagrams or barcodes for a persistence
    dictionary.
    :param persistence_dic: dictionary with key=(band,state) and value=persistence
    :param subj_dir: path to the directory of the subject
    :param intervals=1000: number of (birth , dead) tuples to plot
    from electrode space or font space
    :param repre='diagrams': string that indicates if one wants to plot
    persistence 'diagrams' or persistent 'barcodes'
    :param save=True: boolean that indicates if the plots should be saved
    """
    if repre!='diagrams' and repre!='barcodes':
        repre='diagrams'
        print('Representation parameter incorrect')
    if repre=='diagrams':
        plot_func=lambda x,axes: gd.plot_persistence_diagram(x,legend=True,max_intervals=intervals,axes=axes)
    else:
        plot_func=lambda x,axes: gd.plot_persistence_barcode(x,legend=True,max_intervals=intervals,axes=axes)
    band_dic={0:'alpha',1:'betta',2:'gamma'}
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 9))
    for i in range(3):
        for j in range(3):
            a=plot_func(persistence_dic[(i,j)],axes=axes[i][j])
            a.set_title('{0} persistence {1} of \n motivational state {2} and band {3}'.format(space,repre,j,band_dic[i]))
    fig.suptitle('Persistence {0} of the {1} for\n different frequency bands and motivational space'.format(repre,space),fontsize=24)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    if save:
        if not os.path.exists(subj_dir+space+'/'+measure):
            print("create directory(plot):",subj_dir+space+'/'+measure)
            os.makedirs(subj_dir+'/'+space+'/'+measure)
        pyplot.savefig(subj_dir+space+'/'+measure+'/'+repre+'.png')
    plt.show()
