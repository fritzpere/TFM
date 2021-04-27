#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:22:15 2021

@author: fritz
"""


import pandas as pd
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import os
from classification import *

def compute_topological_descriptors(pers_band_dic,subj_dir,space,measure):
    #bottleneck_table={}
    zero_dim={}
    one_dim={}
    for i in range(-1,3):
        zero_dim[i],one_dim[i]=separate_dimensions(pers_band_dic[i])

        
    avg_life, std_life, entropy, pooling=compute_basicstats(zero_dim,one_dim,subj_dir,space,measure,feat='life')
    avg_midlife, std_midlife=compute_basicstats(zero_dim,one_dim,subj_dir,space,measure,feat='midlife')
    
    plot_descriptorBoxplots(avg_life,'Average Life',subj_dir,space,measure)
    plot_descriptorBoxplots(std_life,'Standard Deviation Life',subj_dir,space,measure)
    plot_descriptorBoxplots(avg_midlife,'Average Midlife',subj_dir,space,measure)
    plot_descriptorBoxplots(std_midlife,'Standard Deviation Midlife',subj_dir,space,measure)
    plot_descriptorBoxplots(entropy,'Persistent Entropy',subj_dir,space,measure)

    return feature_vector_per_band(avg_life, std_life, entropy, pooling, avg_midlife, std_midlife)
    
    
    
def separate_dimensions(pers_band_dic):
    zero_dim=[]
    one_dim=[]
    for i in range(3):
        trials=len(pers_band_dic[i])
        zero_dim.append([])
        one_dim.append([])
        for k in range(trials):
            dim_list=np.array(list(map(lambda x: x[0], pers_band_dic[i][k])))
            point_list=np.array(list(map(lambda x: x[1], pers_band_dic[i][k])))
            zero_dim[i].append(point_list[np.logical_and(point_list[:,1]!=float('inf'),dim_list==0)])
            one_dim[i].append(point_list[np.logical_and(point_list[:,1]!=float('inf'),dim_list==1)])

    return zero_dim,one_dim
    
'''
def compute_bottleneck(zero_dim,one_dim,k):
    band_dic={-1: 'no_filter', 0:'alpha',1:'betta',2:'gamma'}
    distances_0_dim=np.zeros((3,3))
    distances_1_dim=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            if i!=j:
                distances_0_dim[i,j]=gd.bottleneck_distance(zero_dim[i],zero_dim[j])
                distances_1_dim[i,j]=gd.bottleneck_distance(one_dim[i],one_dim[j])
            else:
                distances_0_dim[i,j]=0
                distances_1_dim[i,j]=0
    table=pd.DataFrame(np.concatenate((distances_0_dim,distances_1_dim),axis=1),index=[band_dic[k]+' Motivational state 0',band_dic[k]+' Motivational state 1',band_dic[k]+' Motivational state 2'],columns=['M0 dimension 0','M1 dimension 0','M2 dimension 0','M0 dimension 1','M1 dimension 1','M1 dimension 2'])
    return table'''


def compute_basicstats(zero_dim,one_dim,subj_dir,space,measure,feat='life',pool_n=10): ## Falta aixo
    #zero_feat={}
    #one_feat={}feat_vect_size
    zero_pooling_vector={}
    one_pooling_vector={}
    
    zero_avg_feat={}
    one_avg_feat={}
    zero_std_feat={}
    one_std_feat={}
    
    
    zero_persistent_entropies={}
    one_persistent_entropies={}
    for i in range(-1,3):
        zero_avg_feat[i]=[]
        one_avg_feat[i]=[]
        zero_std_feat[i]=[]
        one_std_feat[i]=[]
        #zero_feat[i]=[]
        #one_feat[i]=[]

        if feat=='life':
            fun=lambda x: x[1]-x[0]
            zero_persistent_entropies[i]=[]
            one_persistent_entropies[i]=[]
            zero_pooling_vector[i]={}
            one_pooling_vector[i]={}
        else:
            fun=lambda x: (x[1]+x[0])/2
        for j in range(3):
            trials=len(zero_dim[i][j])
            zero_avg_feat[i].append([])
            one_avg_feat[i].append([])
            zero_std_feat[i].append([])
            one_std_feat[i].append([])

            
            if feat=='life':
                zero_pooling_vector[i][j]=[]
                one_pooling_vector[i][j]=[]
                zero_persistent_entropies[i].append([])
                one_persistent_entropies[i].append([])
            for k in range(trials):
                zero_feat=[]
                one_feat=[]
                zero_feat.append(np.array(list(map(fun, zero_dim[i][j][k]))))
                one_feat.append(np.array(list(map(fun, one_dim[i][j][k]))))
                zero_feat=np.array(zero_feat).flatten()
                zero_L=zero_feat.sum()
                n_zero=zero_feat.shape[0]
                if  n_zero==0:
                    zero_avg_feat[i][j].append(-1)
                    zero_std_feat[i][j].append(-1)
                else:
                    zero_avg_feat[i][j].append(zero_L/ n_zero)
                    zero_std_feat[i][j].append(zero_feat.std())
                one_feat=np.array(one_feat).flatten()
                one_L=one_feat.sum()
                n_one=one_feat.shape[0]
                if  n_one==0:
                    one_avg_feat[i][j].append(-1)
                    one_std_feat[i][j].append(-1)
                else:
                    one_avg_feat[i][j].append(one_L/ n_one)
                    one_std_feat[i][j].append(one_feat.std())

            
                if feat=='life':
                
                    zero_feat.sort()
                    zero_pooling_vector[i][j].append(np.array([zero_feat[-k2] if k2<=n_zero else 0 for k2 in range(1,pool_n+1)],dtype=object))
                    one_feat.sort()
                    one_pooling_vector[i][j].append(np.array([one_feat[-k2] if k2<=n_one else 0 for k2 in range(1,pool_n+1)],dtype=object))
                    zero_persistent_temp=[]
                    one_persistent_temp=[]
                    fun_entr= lambda d: (d/L)*np.log2(d/L if L!=0 else -1)
                    L=zero_L
                    zero_persistent_temp.append(np.array(list(map(fun_entr, zero_feat))))
                    L=one_L
                    one_persistent_temp.append(np.array(list(map(fun_entr, one_feat))))
                    zero_persistent_entropies[i][j].append(-np.array(zero_persistent_temp).sum())
                    one_persistent_entropies[i][j].append(-np.array(one_persistent_temp).sum())
        '''       
        zero_avg_lifes=np.array(zero_avg_lifes,dtype=object).reshape((1,-1))
        one_avg_lifes=np.array(one_avg_lifes,dtype=object).reshape((1,-1))
        zero_std_lifes=np.array(zero_std_lifes,dtype=object).reshape((1,-1))
        one_std_lifes=np.array(one_std_lifes,dtype=object).reshape((1,-1))'''
        


    
    if feat=='life':
        return (zero_avg_feat, one_avg_feat), (zero_std_feat, one_std_feat), (zero_persistent_entropies, one_persistent_entropies), (zero_pooling_vector, one_pooling_vector)
    
    return (zero_avg_feat, one_avg_feat), (zero_std_feat, one_std_feat)


def plot_descriptorBoxplots(feature, name,subj_dir,space,measure):
    feat0,feat1=feature
    band_dic={-1: 'no_filter', 0:'alpha',1:'betta',2:'gamma'}
    
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(19, 14))
    for i in range(-1,3):

        axes[0][i].boxplot(feat0[i],showfliers=False)
        axes[0][i].set_title(name+' BoxPlot dimension 0 of band '+band_dic[i])
        axes[1][i].boxplot(feat1[i],showfliers=False)
        axes[1][i].set_title(name+' BoxPlot dimension 1 of band '+band_dic[i])
        #a.set_xlim(-0.05,x_max)
        #a.set_ylim(0,y_max)
    fig.suptitle('{0} Boxplots of the {1} for\n different frequency bands and motivational state'.format(name,space),fontsize=24)
    fig.tight_layout(pad=1.00)
    fig.subplots_adjust(top=0.8)
    
    
    if not os.path.exists(subj_dir+space+'/'+measure+'/descriptor_tables'):
        print("create directory(plot):",subj_dir+space+'/'+measure+'/'+'descriptor_tables')
        os.makedirs(subj_dir+space+'/'+measure+'/'+'descriptor_tables')
    
    pyplot.savefig(subj_dir+space+'/'+measure+'/descriptor_tables/'+name+'_boxplots''.png')
    plt.clf()
    #plt.show()
    


