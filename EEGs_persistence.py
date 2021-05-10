#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:07:20 2021

@author: fritz
"""
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import gudhi as gd
import gudhi.representations
from sklearn.preprocessing import normalize
from preprocess_data import *
from topological_descriptors import *
import os
#plt.style.use(['seaborn-bright'])
import time  
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
    for band in range(-1,n_bands):
        persistence_dic[band]={}
        for i in range (3):#motivational
            persistence_dic[band][i]=[]
            trials=tensor[band][i].shape[0]
            for k in range(trials):
                band_tensor = tensor[band][i][k,:,:].T
            #print('nans?',np.isnan(band_tensor[i]).any())
            #print('cloud shape:',band_tensor.shape)
            
                if measure=='intensities':
                    band_tensor = np.abs(tensor[band][i][k,:,:].T)
                    matrix=distance_matrix(band_tensor,band_tensor)
                    
                    
                    '''
                    from scipy.spatial.distance import pdist
                    
                    n_coor = band_tensor.shape[0]
                    dist = np.zeros((n_coor, n_coor))
                    row,col = np.tril_indices(n_coor,1)
                    dist[row,col] = pdist(band_tensor)'''
                else:
                    '''
                    points=band_tensor.copy()
                    normalized_p=normalize(points-np.mean(points,axis=0),axis=1)
                    matrix= normalized_p @ normalized_p.T
                    matrix=1-matrix
                    '''
                    T=1200
                    ts_tmp = band_tensor.T.copy()
                    ts_tmp -= np.outer(np.ones(T),ts_tmp.mean(0))
                    matrix= np.tensordot(ts_tmp,ts_tmp,axes=(0,0)) / float(T-1)
                    
                    matrix/= np.sqrt(np.outer(matrix.diagonal(),matrix.diagonal()))
                    matrix=np.arccos(matrix)
                #max_edge=np.max(matrix)
                Rips_complex_sample = gd.RipsComplex(distance_matrix=matrix)#,max_edge_length=max_edge)
                #Rips_complex_sample = gd.AlphaComplex(distance_matrix=matrix)#,max_edge_length=max_edge)
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
    filtered_ts_dic[-1]={}
    filtered_ts_dic[-1][0]=ts_dic[0]
    filtered_ts_dic[-1][1]=ts_dic[1]
    filtered_ts_dic[-1][2]=ts_dic[2]
    

    persistence_dictionary=persistency_per_band_and_state(filtered_ts_dic,measure)
    
    
    
    
    
    
    
    
    plot_landscapes(persistence_dictionary,subj_dir,space, measure,100)
    return persistence_dictionary #dictionary with key=(band,state) and value=persistence


def plot_landscapes(persistences,subj_dir,space='',measure='',resolut=1000):
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(16, 16))
    zero_dim={}
    one_dim={}
    band_dic={-1: 'no_filter', 0:'alpha',1:'betta',2:'gamma'}
    
    LS = gd.representations.Landscape(num_landscapes=2,resolution=resolut)
    L0 = []
    for i in range(-1,3):
        zero_dim[i],one_dim[i]=separate_dimensions(persistences[i])
        L0.append([LS.fit_transform(zero_dim[i][0]),LS.fit_transform(zero_dim[i][1]),LS.fit_transform(zero_dim[i][2])])
        mean_landscape0=np.array([L0[i][0].mean(axis=0),L0[i][1].mean(axis=0),L0[i][2].mean(axis=0)])
        y_max=np.max(mean_landscape0)
        for j in range(3):

            axes[i][j].plot(mean_landscape0[j][:resolut])
            axes[i][j].plot(mean_landscape0[j][resolut:2*resolut])
            '''
            axes[i][j].plot(mean_landscape0[j][2*resolut:3*resolut])
            axes[i][j].plot(mean_landscape0[j][3*resolut:4*resolut])
            axes[i][j].plot(mean_landscape0[j][4*resolut:5*resolut])'''
    
            axes[i][j].set_title('{0} persistence Landscapes of \n motivational state {1} and band {2}'.format(space,j,band_dic[i]))
            axes[i][j].set_xlim(-2,resolut)
            axes[i][j].set_ylim(0,y_max*1.1)
            
    fig.suptitle('Persistence Landscapes of the {0} for\n different frequency bands and motivational state of 0 dimensional features'.format(space),fontsize=24)
    fig.tight_layout(pad=0.5)
    fig.subplots_adjust(top=0.8)
    
    if not os.path.exists(subj_dir+space+'/'+measure):
        print("create directory(plot):",subj_dir+space+'/'+measure)
        os.makedirs(subj_dir+'/'+space+'/'+measure)
    plt.savefig(subj_dir+space+'/'+measure+'/Landscapes_dim0.png')
    plt.close()
    
    
    fig2, axes2 = plt.subplots(nrows=4, ncols=3, figsize=(16, 16))
    L1=[]
    for i in range(-1,3):

        L1.append([LS.fit_transform(one_dim[i][0]),LS.fit_transform(one_dim[i][1]),LS.fit_transform(one_dim[i][2])])
        mean_landscape1=np.array([L1[i][0].mean(axis=0),L1[i][1].mean(axis=0),L1[i][2].mean(axis=0)])
        y_max=np.max(mean_landscape1)
        for j in range(3):

            axes2[i][j].plot(mean_landscape1[j][:resolut])
            axes2[i][j].plot(mean_landscape1[j][resolut:2*resolut])
            '''
            axes2[i][j].plot(mean_landscape1[j][2*resolut:3*resolut])
            axes2[i][j].plot(mean_landscape1[j][3*resolut:4*resolut])
            axes2[i][j].plot(mean_landscape1[j][4*resolut:5*resolut])'''
            
            axes2[i][j].set_title('{0} persistence Landscapes of \n motivational state {1} and band {2}'.format(space,j,band_dic[i]))
            axes[i][j].set_xlim(-2,resolut)
            axes[i][j].set_ylim(0,y_max*1.1)
    fig2.suptitle('Persistence Landscapes of the {0} for\n different frequency bands and motivational state of 1 dimensional features'.format(space),fontsize=24)
    fig2.tight_layout(pad=0.5)
    fig2.subplots_adjust(top=0.8)
    
    plt.savefig(subj_dir+space+'/'+measure+'/Landscapes_dim1.png')
    plt.close()
    
    
    SH = gd.representations.Silhouette(resolution=resolut, weight=lambda x: np.power(x[1]-x[0],1))
    fig3, axes3 = plt.subplots(nrows=4, ncols=3, figsize=(16, 16))
    S0=[]
    for i in range(-1,3):
        S0.append([SH.fit_transform(zero_dim[i][0]),SH.fit_transform(zero_dim[i][1]),SH.fit_transform(zero_dim[i][2])])
        mean_silhouette0=np.array([S0[i][0].mean(axis=0),S0[i][1].mean(axis=0),S0[i][2].mean(axis=0)])
        y_max=np.max(mean_silhouette0)
        for j in range(3):

            axes3[i][j].plot(mean_silhouette0[j])
            axes3[i][j].set_title('{0} persistence Silhouette of \n motivational state {1} and band {2}'.format(space,j,band_dic[i]))
            axes3[i][j].set_xlim(-2,resolut)
            axes3[i][j].set_ylim(0,y_max*1.1)
            
    fig3.suptitle('Persistence Silhouette of the {0} for\n different frequency bands and motivational state of 0 dimensional features'.format(space),fontsize=24)
    fig3.tight_layout(pad=0.5)
    fig3.subplots_adjust(top=0.8)
    
    if not os.path.exists(subj_dir+space+'/'+measure):
        print("create directory(plot):",subj_dir+space+'/'+measure)
        os.makedirs(subj_dir+'/'+space+'/'+measure)
    plt.savefig(subj_dir+space+'/'+measure+'/Silhouette_dim0.png')
    plt.close()
    
    fig4, axes4 = plt.subplots(nrows=4, ncols=3, figsize=(16, 16))
    S1=[]
    for i in range(-1,3):
        S1.append([SH.fit_transform(one_dim[i][0]),SH.fit_transform(one_dim[i][1]),SH.fit_transform(one_dim[i][2])])
        mean_silhouette1=np.array([S1[i][0].mean(axis=0),S1[i][1].mean(axis=0),S1[i][2].mean(axis=0)])
        y_max=np.max(mean_silhouette1[i])
        for j in range(3):

            axes4[i][j].plot(mean_silhouette1[j])
            axes4[i][j].set_title('{0} persistence Silhouette of \n motivational state {1} and band {2}'.format(space,j,band_dic[i]))
            axes4[i][j].set_xlim(-2,resolut)
            axes4[i][j].set_ylim(0,y_max*1.1)
            
    fig4.suptitle('Persistence Silhouette of the {0} for\n different frequency bands and motivational state of 1 dimensional features'.format(space),fontsize=24)
    fig4.tight_layout(pad=0.5)
    fig4.subplots_adjust(top=0.8)
    
    if not os.path.exists(subj_dir+space+'/'+measure):
        print("create directory(plot):",subj_dir+space+'/'+measure)
        os.makedirs(subj_dir+'/'+space+'/'+measure)
    plt.savefig(subj_dir+space+'/'+measure+'/Silhouette_dim1.png')
    plt.close()
    
    return
    

    feat_vect_land,labels=feat_vect_repr(np.array(L0),np.array(L1),'landscapes',resolut*2)
    print('plotting accuracies and confusion matrixes for Landscapes Classification')
    t=time.time()
    get_accuracies_per_band(feat_vect_land,labels,subj_dir,space,measure,500,'landscapes'+str(resolut))
    print((time.time()-t)/60, 'minuts for Landscape classification')
    
    
    feat_vect_sil,labels=feat_vect_repr(np.array(S0),np.array(S1),'silhouetes',resolut)
    print('plotting accuracies and confusion matrixes for Silhouetes Classification')
    t=time.time()
    get_accuracies_per_band(feat_vect_sil,labels,subj_dir,space,measure,500,'silhouettes'+str(resolut))
    print((time.time()-t)/60, 'minuts for silhouettes classification')


    