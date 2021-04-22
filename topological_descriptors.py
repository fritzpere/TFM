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

def compute_topological_descriptors(pers_band_dic,subj_dir,space,measure):
    bottleneck_table={}
    zero_dim={}
    one_dim={}
    for i in range(-1,3):
        zero_dim[i],one_dim[i]=separate_dimensions(pers_band_dic[i])
        bottleneck_table[i]=compute_bottleneck(zero_dim[i],one_dim[i],i)
        
    avg_life_table=compute_basicstats(zero_dim,one_dim,subj_dir,space,measure,feat='life')
    avg_midlife_table=compute_basicstats(zero_dim,one_dim,subj_dir,space,measure,feat='midlife')

    #add more descriptors
    descriptors_table_0=avg_life_table[0].join( avg_midlife_table[0],on=avg_life_table[0].index)
    descriptors_table_1=avg_life_table[1].join( avg_midlife_table[1],on=avg_life_table[1].index)
    descriptors_table_2=avg_life_table[2].join( avg_midlife_table[2],on=avg_life_table[2].index)
    descriptors_table_3=avg_life_table[-1].join( avg_midlife_table[-1],on=avg_life_table[-1].index)
    descriptors_table=pd.concat([descriptors_table_0,descriptors_table_1,descriptors_table_2,descriptors_table_3])
    
    bottleneck_final_table=pd.concat([bottleneck_table[0],bottleneck_table[1],bottleneck_table[2],bottleneck_table[-1]])
    
    return bottleneck_final_table,descriptors_table
    
    
def separate_dimensions(pers_band_dic):
    zero_dim=[]
    one_dim=[]
    for i in range(3):
        dim_list=np.array(list(map(lambda x: x[0], pers_band_dic[i])))
        point_list=np.array(list(map(lambda x: x[1], pers_band_dic[i])))
        zero_dim.append(point_list[np.logical_and(point_list[:,1]!=float('inf'),dim_list==0)])
        one_dim.append(point_list[np.logical_and(point_list[:,1]!=float('inf'),dim_list==1)])
    return zero_dim,one_dim
    

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
    return table


def compute_basicstats(zero_dim,one_dim,subj_dir,space,measure,feat='life',pool_n=10):
    table={}
    zero_lifes={}
    one_lifes={}
    band_dic={-1: 'no_filter', 0:'alpha',1:'betta',2:'gamma'}
    zero_pooling_vector={}
    one_pooling_vector={}
    for i in range(-1,3):
        zero_avg_lifes=[]
        one_avg_lifes=[]
        zero_std_lifes=[]
        one_std_lifes=[]
        zero_lifes[i]=[]
        one_lifes[i]=[]

        if feat=='life':
            fun=lambda x: x[1]-x[0]
            zero_persistent_entropies=[]
            one_persistent_entropies=[]
        else:
            fun=lambda x: (x[1]+x[0])/2
        for j in range(3):
            zero_lifes[i].append(np.array(list(map(fun, zero_dim[i][j]))))
            one_lifes[i].append(np.array(list(map(fun, one_dim[i][j]))))
            zero_L=zero_lifes[i][j].sum()
            zero_avg_lifes.append(zero_L/len(zero_lifes[i][j]))
            one_L=one_lifes[i][j].sum()
            one_avg_lifes.append(one_L/(len(one_lifes[i][j])))
            zero_std_lifes.append(zero_lifes[i][j].std())
            one_std_lifes.append(one_lifes[i][j].std())
            
            if feat=='life':
                n_zero=len(zero_lifes[i][j])
                zero_lifes[i][j].sort()
                zero_pooling_vector[i,j]=np.array([zero_lifes[i][j][-k] if k<=n_zero else 0 for k in range(1,pool_n+1)],dtype=object)
                n_one=len(one_lifes[i][j])
                one_lifes[i][j].sort()
                one_pooling_vector[i,j]=np.array([one_lifes[i][j][-k] if k<=n_one else 0 for k in range(1,pool_n+1)],dtype=object)
                zero_persistent_temp=[]
                one_persistent_temp=[]
                fun_entr= lambda x: ((x[1]-x[0])/L)*np.log2((x[1]-x[0])/L)
                L=zero_L
                zero_persistent_temp.append(np.array(list(map(fun_entr, zero_dim[i][j]))))
                L=one_L
                one_persistent_temp.append(np.array(list(map(fun_entr, one_dim[i][j]))))
                zero_persistent_entropies.append(-np.array(zero_persistent_temp).sum())
                one_persistent_entropies.append(-np.array(one_persistent_temp).sum())
                
        zero_avg_lifes=np.array(zero_avg_lifes,dtype=object).reshape((1,-1))
        one_avg_lifes=np.array(one_avg_lifes,dtype=object).reshape((1,-1))
        zero_std_lifes=np.array(zero_std_lifes,dtype=object).reshape((1,-1))
        one_std_lifes=np.array(one_std_lifes,dtype=object).reshape((1,-1))
        

        if feat=='life':
            
            zero_persistent_entropies=np.array(zero_std_lifes,dtype=object).reshape((1,-1))
            one_persistent_entropies=np.array(one_persistent_entropies,dtype=object).reshape((1,-1))
            table[i]=pd.DataFrame(np.concatenate((zero_avg_lifes,one_avg_lifes,zero_std_lifes,one_std_lifes,zero_persistent_entropies,one_persistent_entropies),axis=0).T,index=[band_dic[i]+' Motivational state 0',band_dic[i]+' Motivational state 1',band_dic[i]+' Motivational state 2'],columns=['Avg. '+feat+' dim0','Avg. '+feat+' dim1','std '+feat+'dim0','std '+feat+' dim1','persistent entropy dim0','persistent entropy dim1'])
        else:
            table[i]=pd.DataFrame(np.concatenate((zero_avg_lifes,one_avg_lifes,zero_std_lifes,one_std_lifes),axis=0).T,index=[band_dic[i]+' Motivational state 0',band_dic[i]+' Motivational state 1',band_dic[i]+' Motivational state 2'],columns=['Avg. '+feat+' dim0','Avg. '+feat+' dim1','std '+feat+'dim0','std '+feat+' dim1'])
    
    if feat=='life':
        pooling_table_dim0=pd.DataFrame.from_dict(zero_pooling_vector,orient='index',columns=list(range(1,11)))
        pooling_table_dim0.index=list(map(lambda x: ('Dim0 '+band_dic[x[0]]+' Motivational state '+str(x[1])) ,[*zero_pooling_vector]))
        pooling_table_dim1=pd.DataFrame.from_dict(one_pooling_vector,orient='index',columns=list(range(1,11)))#,index=list(map(lambda x: 'Dim1 '+band_dic[x[0]]+' Motivational state '+str(x[1])  ,[*one_pooling_vector])),columns=list(range(1,11)))
        pooling_table_dim1.index=list(map(lambda x: ('Dim1 '+band_dic[x[0]]+' Motivational state '+str(x[1])) ,[*one_pooling_vector]))
        pooling_table=pd.concat([pooling_table_dim0,pooling_table_dim1])

    
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(24, 14))
    for i in range(-1,3):
        for j in range(3):
            zero_lifes[i][j]=zero_lifes[i][j].flatten()
            one_lifes[i][j]=one_lifes[i][j].flatten()
        axes[0][i].boxplot(zero_lifes[i],showfliers=False)
        axes[0][i].set_title(feat+' BoxPlot dimension 0 of band '+band_dic[i])
        axes[1][i].boxplot(np.array(one_lifes[i],dtype=object),showfliers=False)
        axes[1][i].set_title(feat+' BoxPlot dimension 1 of band '+band_dic[i])
        #a.set_xlim(-0.05,x_max)
        #a.set_ylim(0,y_max)
    fig.suptitle('{0} Boxplots of the {1} for\n different frequency bands and motivational state'.format(feat,space),fontsize=24)
    fig.tight_layout(pad=1.00)
    fig.subplots_adjust(top=0.8)
    
    
    if not os.path.exists(subj_dir+space+'/'+measure+'/descriptor_tables'):
        print("create directory(plot):",subj_dir+space+'/'+measure+'/'+'descriptor_tables')
        os.makedirs(subj_dir+space+'/'+measure+'/'+'descriptor_tables')
    
    pyplot.savefig(subj_dir+space+'/'+measure+'/descriptor_tables/'+feat+'_boxplots''.png')
    if feat=='life':
        pooling_table.to_csv(subj_dir+space+'/'+measure+'/descriptor_tables/pooling_vectors.csv')
    #plt.show()
    
    return table

