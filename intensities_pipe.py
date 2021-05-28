#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 16:39:25 2021

@author: fritzpere
"""
import numpy as np
import sklearn.model_selection as skms

def intensity(ts_band,labels,band):
    cv_schem = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.1)
    ts=np.abs(ts_band[:,band,:,:])
    ts_intens=ts.mean(axis=1)
    persistence=[]
    for ind_train, ind_test in cv_schem.split(ts_intens,labels):
        X_train=ts_intens[ind_train]
        for i_motiv  in range(3):
            X_temp=X_train[labels==i_motiv]
            n_coor=X_tran.shape[0]
            row,col = np.triu_indices(n_coor,1)
            distancies=pdist(band_tensor)
            matrix[row,col] = distancies
            matrix[col,row] = distancies
    
            Rips_complex_sample = gd.RipsComplex(distance_matrix=matrix)#,max_edge_length=max_edge)
            #Rips_complex_sample = gd.AlphaComplex(distance_matrix=matrix)#,max_edge_length=max_edge)
            Rips_simplex_tree_sample = Rips_complex_sample.create_simplex_tree(max_dimension=2)
            persistence.append(Rips_simplex_tree_sample.persistence())
                    
        
    