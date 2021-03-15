#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:22:15 2021

@author: fritz
"""


import pandas as pd
import numpy as np
import gudhi as gd

def compute_topological_descriptors(pers_band_dic):
    zero_dim,one_dim=separate_dimensions(pers_band_dic)
    bottleneck_table=compute_bottleneck(zero_dim,one_dim)
    avg_life_table=compute_avg_life(zero_dim,one_dim)
    avg_midlife_table=compute_avg_midlife(zero_dim,one_dim)

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

def compute_avg_life(zero_dim,one_dim):
    zero_avg_lifes=[]
    one_avg_lifes=[]
    for i in range(3):
        n_zero_dim=len(zero_dim[i])
        n_one_dim=len(one_dim[i])
        zero_lifes=np.array(list(map(lambda x: x[1]-x[0], zero_dim[i])))
        one_lifes=np.array(list(map(lambda x: x[1]-x[0], one_dim[i])))
        zero_avg_lifes.append(zero_lifes.sum()/n_zero_dim)
        one_avg_lifes.append(one_lifes.sum()/n_one_dim)
    
    zero_avg_lifes=np.array(zero_avg_lifes).reshape((1,-1))
    one_avg_lifes=np.array(one_avg_lifes).reshape((1,-1))
    table=pd.DataFrame(np.concatenate((zero_avg_lifes,one_avg_lifes),axis=0).T,index=['Motivational state 0','Motivational state 1','Motivational state 2'],columns=['Avg. life dim0','Avg. life dim1'])
    return table

def compute_avg_midlife(zero_dim,one_dim):
    zero_avg_midlifes=[]
    one_avg_midlifes=[]
    for i in range(3):
        n_zero_dim=len(zero_dim[i])
        n_one_dim=len(one_dim[i])
        zero_midlifes=np.array(list(map(lambda x: (x[1]+x[0])/2, zero_dim[i])))
        one_midlifes=np.array(list(map(lambda x: (x[1]+x[0])/2, one_dim[i])))
        zero_avg_midlifes.append(zero_midlifes.sum()/n_zero_dim)
        one_avg_midlifes.append(one_midlifes.sum()/n_one_dim)
    zero_avg_midlifes=np.array(zero_avg_midlifes).reshape((1,-1))
    one_avg_midlifes=np.array(one_avg_midlifes).reshape((1,-1))
    table=pd.DataFrame(np.concatenate((zero_avg_midlifes,one_avg_midlifes),axis=0).T,index=['Motivational state 0','Motivational state 1','Motivational state 2'],columns=['Avg. midlife dim0','Avg. midlife dim1'])
    return table
