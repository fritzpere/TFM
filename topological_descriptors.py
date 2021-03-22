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
    zero_dim,one_dim=separate_dimensions(pers_band_dic)
    bottleneck_table=compute_bottleneck(zero_dim,one_dim)
    avg_life_table=compute_stats(zero_dim,one_dim,subj_dir,space,measure,feat='life')
    avg_midlife_table=compute_stats(zero_dim,one_dim,subj_dir,space,measure,feat='midlife')

    #add more descriptors
    descriptors_table=avg_life_table.join( avg_midlife_table,on=avg_life_table.index)
    
    return bottleneck_table,descriptors_table
    
    
def separate_dimensions(pers_band_dic):
    zero_dim=[]
    one_dim=[]
    for i in range(3):
        dim_list=np.array(list(map(lambda x: x[0], pers_band_dic[i])))
        point_list=np.array(list(map(lambda x: x[1], pers_band_dic[i])))
        zero_dim.append(point_list[np.logical_and(point_list[:,1]!=float('inf'),dim_list==0)])
        one_dim.append(point_list[np.logical_and(point_list[:,1]!=float('inf'),dim_list==1)])
    return zero_dim,one_dim
    

def compute_bottleneck(zero_dim,one_dim):
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
    table=pd.DataFrame(np.concatenate((distances_0_dim,distances_1_dim),axis=1),index=['Motivational state 0','Motivational state 1','Motivational state 2'],columns=['M0 dimension 0','M1 dimension 0','M2 dimension 0','M0 dimension 1','M1 dimension 1','M1 dimension 2'])
    return table

def compute_stats(zero_dim,one_dim,subj_dir,space,measure,feat='life'):
    zero_avg_lifes=[]
    one_avg_lifes=[]
    zero_std_lifes=[]
    one_std_lifes=[]
    zero_lifes=[]
    one_lifes=[]
    if feat=='life':
        fun=lambda x: x[1]-x[0]
    else:
        fun=lambda x: (x[1]+x[0])/2
    for i in range(3):
        zero_lifes.append(np.array(list(map(fun, zero_dim[i]))))
        one_lifes.append(np.array(list(map(fun, one_dim[i]))))
        zero_avg_lifes.append(zero_lifes[i].mean())
        one_avg_lifes.append(one_lifes[i].mean())
        zero_std_lifes.append(zero_lifes[i].std())
        one_std_lifes.append(one_lifes[i].std())
    
    zero_avg_lifes=np.array(zero_avg_lifes).reshape((1,-1))
    one_avg_lifes=np.array(one_avg_lifes).reshape((1,-1))
    zero_std_lifes=np.array(zero_std_lifes).reshape((1,-1))
    one_std_lifes=np.array(one_std_lifes).reshape((1,-1))
    
    if not os.path.exists(subj_dir+space+'/'+measure+'/'+'descriptor_tables'):
        print("create directory(plot):",subj_dir+space+'/'+measure+'/'+'descriptor_tables')
        os.makedirs(subj_dir+space+'/'+measure+'/'+'descriptor_tables')
    fig1, ax1 = plt.subplots()
    ax1.set_title('BoxPlot dimension 0')
    ax1.boxplot(zero_lifes)
    #ax1.set_xticks(([1,2,3],['alpha', 'betta', 'gamma']))
    
    pyplot.savefig(subj_dir+space+'/'+measure+'/descriptor_tables/'+feat+'boxplot_dimension_0.png')

    plt.show()
    
    fig2, ax2 = plt.subplots()
    ax2.set_title('BoxPlot dimension 1')
    ax2.boxplot(one_lifes)
    ax2.set_xticks([1, 2, 3], ['alpha', 'betta', 'gamma'])
    
    pyplot.savefig(subj_dir+space+'/'+measure+'/descriptor_tables/'+feat+'boxplot_dimension_1.png')

    table=pd.DataFrame(np.concatenate((zero_avg_lifes,one_avg_lifes,zero_std_lifes,one_std_lifes),axis=0).T,index=['Motivational state 0','Motivational state 1','Motivational state 2'],columns=['Avg. '+feat+' dim0','Avg. '+feat+' dim1','std '+feat+'life dim0','std'+feat+' dim1'])
    return table

