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



    
    
    
    
    
    
    '''
    
    plot_landscapes(persistence_dictionary,subj_dir,space, measure,100)
    return persistence_dictionary #dictionary with key=(band,state) and value=persistence'''


def plot_landscapes(persistences,subj_dir,space='',measure='',resolut=1000):
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(16, 16))
    zero_dim={}
    one_dim={}
    band_dic={-1: 'no_filter', 0:'alpha',1:'beta',2:'gamma'}
    
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


    