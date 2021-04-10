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
        persistence_dic[band]={}
        for i in range (3):#motivational
            band_tensor = np.abs(tensor[band][i][:,:])
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
            max_edge=np.max(matrix)
            Rips_complex_sample = gd.RipsComplex(distance_matrix=matrix,max_edge_length=max_edge)
            Rips_simplex_tree_sample = Rips_complex_sample.create_simplex_tree(max_dimension=2)
            persistence = Rips_simplex_tree_sample.persistence()
            persistence_dic[band][i]= persistence #dictionary with key=(band,state) and value=persistence
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
    
    vect_features_dic={}
    for i_band in range(n_band):
        vect_features_dic[i_band]={}
        for i_state in range(n_motiv):
            #n_trials=filtered_ts_dic[i_state,i_band].shape[0]
            vect_features_dic[i_band][i_state]= filtered_ts_dic[i_band,i_state][13,:,:].T

    #print(vect_features.shape)
    persistence_dictionary=persistency_per_band_and_state(vect_features_dic,measure)
        
    if save:
        band_dic={0:'alpha',1:'betta',2:'gamma'}
        if not os.path.exists(subj_dir+space+'/'+measure+'/'+'persistencies'):
            print("create directory:",subj_dir+space+'/'+measure+'/'+'persistencies')
            os.makedirs(subj_dir+space+'/'+measure+'/'+'persistencies')

        for i in range(3):
            for j in range(3):
                f = open(subj_dir+'/'+space+'/'+measure+'/'+'persistencies'+'/'+str(j)+band_dic[i]+'persistence.txt', "w")
                for persistence in persistence_dictionary[i][j] :
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
        plot_func=lambda x,axes: gd.plot_persistence_diagram(x,legend=True,max_intervals=intervals,axes=axes)#,inf_delta=0.5)
    else:
        plot_func=lambda x,axes: gd.plot_persistence_barcode(x,legend=True,max_intervals=intervals,axes=axes)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 12))
    band_dic={0:'alpha',1:'betta',2:'gamma'}
    for i in range(3):
        aux_lis=np.array([persistence_dic[i][0],persistence_dic[i][1],persistence_dic[i][2]], dtype=object)
        x_max=np.amax(list(map(lambda y: np.amax(list(map(lambda x: x[1][0],y))),aux_lis)))+0.05
        y_max=np.amax(list(map(lambda y: np.amax(list(map(lambda x: x[1][1] if x[1][1]!=np.inf  else 0 ,y))),aux_lis)))*1.2
        for j in range(3):
            a=plot_func(persistence_dic[i][j],axes=axes[i][j])
            a.set_title('{0} persistence {1} of \n motivational state {2} and band {3}'.format(space,repre,j,band_dic[i]))
            a.set_xlim(-0.05,x_max)
            a.set_ylim(0,y_max)
    fig.suptitle('Persistence {0} of the {1} for\n different frequency bands and motivational state'.format(repre,space),fontsize=24)
    fig.tight_layout(pad=1.00)
    fig.subplots_adjust(top=0.8)
    
    if save:
        if not os.path.exists(subj_dir+space+'/'+measure):
            print("create directory(plot):",subj_dir+space+'/'+measure)
            os.makedirs(subj_dir+'/'+space+'/'+measure)
        pyplot.savefig(subj_dir+space+'/'+measure+'/'+repre+'.png')
    plt.show()
    
    descriptors=compute_topological_descriptors(persistence_dic,subj_dir,space,measure)
  
    
    if not os.path.exists(subj_dir+space+'/'+measure+'/'+'descriptor_tables'):
        print("create directory(plot):",subj_dir+space+'/'+measure+'/'+'descriptor_tables')
        os.makedirs(subj_dir+'/'+space+'/'+measure+'/'+'descriptor_tables')
    descriptors[0].to_csv(subj_dir+space+'/'+measure+'/descriptor_tables/'+'bottleneck_tables.csv')
    descriptors[1].to_csv(subj_dir+space+'/'+measure+'/descriptor_tables/'+'top_descriptors_tables.csv')
    
    '''
def save_tables(descriptors,subj_dir,space,measure):
    band_dic={0:'alpha',1:'betta',2:'gamma'}
    bottleneck_alpha,alpha_descriptors=descriptors[0]
    bottleneck_betta,betta_descriptors=descriptors[0]
    bottleneck_gamma,gamma_descriptors=descriptors[0]
        
    if not os.path.exists(subj_dir+space+'/'+measure+'/'+'bottleneck_tables'):
        print("create directory(plot):",subj_dir+space+'/'+measure+'/'+'bottleneck_tables')
        os.makedirs(subj_dir+'/'+space+'/'+measure+'/'+'bottleneck_tables')
    bottleneck_alpha.to_csv(subj_dir+space+'/'+measure+'/'+'bottleneck_tables/'+band_dic[0]+'.csv')
    bottleneck_betta.to_csv(subj_dir+space+'/'+measure+'/'+'bottleneck_tables/'+band_dic[1]+'.csv')
    bottleneck_gamma.to_csv(subj_dir+space+'/'+measure+'/'+'bottleneck_tables/'+band_dic[2]+'.csv')
    
    if not os.path.exists(subj_dir+space+'/'+measure+'/'+'descriptor_tables'):
        print("create directory(plot):",subj_dir+space+'/'+measure+'/'+'descriptor_tables')
        os.makedirs(subj_dir+'/'+space+'/'+measure+'/'+'descriptor_tables')
    alpha_descriptors.to_csv(subj_dir+space+'/'+measure+'/'+'descriptor_tables/'+band_dic[0]+'.csv')
    betta_descriptors.to_csv(subj_dir+space+'/'+measure+'/'+'descriptor_tables/'+band_dic[1]+'.csv')
    gamma_descriptors.to_csv(subj_dir+space+'/'+measure+'/'+'descriptor_tables/'+band_dic[2]+'.csv')'''